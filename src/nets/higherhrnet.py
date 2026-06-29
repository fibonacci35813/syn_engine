'''
HigherHRNet for spacecraft keypoint detection on SPEED+.
HRNet-W32 backbone + deconvolution head → heatmaps → soft-argmax coords.

Interface matches KeypointRegressionNet (park2019.py):
  train: model(images, target) → (loss, summary_dict)
  test:  model(images)         → (xc, yc)  both [B, K] normalized [0,1]
'''
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes,   planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,              1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes,   planes,              3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes,   planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        residual = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + residual)


def _make_layer(block, inplanes, planes, num_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# HRNet module: parallel branches + multi-resolution fusion
# ---------------------------------------------------------------------------

class HRModule(nn.Module):
    def __init__(self, num_branches, num_blocks, num_channels):
        super().__init__()
        self.num_branches = num_branches

        # One stack of BasicBlocks per resolution branch
        self.branches = nn.ModuleList([
            _make_layer(BasicBlock, num_channels[i], num_channels[i], num_blocks)
            for i in range(num_branches)
        ])

        # Fusion layers: fuse_layers[i][j] maps branch j → branch i resolution
        # Identity placeholder for i==j (skipped in forward)
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            row = nn.ModuleList()
            for j in range(num_branches):
                if j == i:
                    row.append(nn.Identity())
                elif j > i:
                    # lower-res j → higher-res i: 1×1 conv + upsample
                    row.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest'),
                    ))
                else:
                    # higher-res j → lower-res i: (i-j) stride-2 convs
                    in_ch = num_channels[j]
                    downs = []
                    for k in range(i - j - 1):
                        downs += [
                            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(in_ch, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        ]
                    # final conv changes channel count, no ReLU (added after fusion sum)
                    downs += [
                        nn.Conv2d(in_ch, num_channels[i], 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                    ]
                    row.append(nn.Sequential(*downs))
            self.fuse_layers.append(row)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Process each branch independently
        x = [self.branches[i](x[i]) for i in range(self.num_branches)]
        # Fuse all branches into each resolution
        out = []
        for i in range(self.num_branches):
            y = x[i]
            for j in range(self.num_branches):
                if j != i:
                    y = y + self.fuse_layers[i][j](x[j])
            out.append(self.relu(y))
        return out


# ---------------------------------------------------------------------------
# Full HigherHRNet model
# ---------------------------------------------------------------------------

class HigherHRNet(nn.Module):
    '''
    HRNet-W32 backbone with a transposed-conv upsampling head.
    Outputs keypoint heatmaps at stride 2 (112×112 for 224×224 input).
    Soft-argmax extracts normalised (x,y) ∈ [0,1] coordinates at test time.

    Parameters
    ----------
    num_keypoints : int   – number of spacecraft keypoints (11 for Tango)
    heatmap_sigma : float – Gaussian sigma (in heatmap pixels) for training targets
    '''

    def __init__(self, num_keypoints=11, heatmap_sigma=2.0):
        super().__init__()
        self.nK    = num_keypoints
        self.sigma = heatmap_sigma

        # ------------------------------------------------------------------
        # Stem: 2 stride-2 convs → total stride 4
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        # 4× Bottleneck blocks: 64 → 256 channels, stride stays at 4
        self.layer1 = _make_layer(Bottleneck, 64, 64, 4)  # output: [B, 256, H/4, W/4]

        # ------------------------------------------------------------------
        # Transition 1: 256 ch → two parallel branches [32, 64]
        # ------------------------------------------------------------------
        self.tr1_branch0 = nn.Sequential(   # stride 4,  32 ch
            nn.Conv2d(256, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.tr1_branch1 = nn.Sequential(   # stride 8,  64 ch
            nn.Conv2d(256, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Stage 2: 1 HRModule, 2 resolutions [32, 64]
        # ------------------------------------------------------------------
        self.stage2 = nn.ModuleList([HRModule(2, 4, [32, 64])])

        # Transition 2: add 3rd resolution branch 128 ch (stride 16)
        self.tr2_branch2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Stage 3: 4 HRModules, 3 resolutions [32, 64, 128]
        # ------------------------------------------------------------------
        self.stage3 = nn.ModuleList([HRModule(3, 4, [32, 64, 128]) for _ in range(4)])

        # Transition 3: add 4th resolution branch 256 ch (stride 32)
        self.tr3_branch3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Stage 4: 3 HRModules, 4 resolutions [32, 64, 128, 256]
        # ------------------------------------------------------------------
        self.stage4 = nn.ModuleList([HRModule(4, 4, [32, 64, 128, 256]) for _ in range(3)])

        # ------------------------------------------------------------------
        # Deconvolution head: stride 4 → stride 2 ("higher" resolution)
        # 32 ch → 32 ch feature map at H/2, W/2
        # ------------------------------------------------------------------
        self.deconv_head = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        # Final 1×1 projection → K heatmaps
        self.final_conv = nn.Conv2d(32, num_keypoints, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _backbone(self, x):
        x = self.stem(x)       # [B, 64,  H/4, W/4]
        x = self.layer1(x)     # [B, 256, H/4, W/4]

        # Stage 2
        xl = [self.tr1_branch0(x), self.tr1_branch1(x)]
        for m in self.stage2:
            xl = m(xl)

        # Stage 3
        xl = [xl[0], xl[1], self.tr2_branch2(xl[1])]
        for m in self.stage3:
            xl = m(xl)

        # Stage 4
        xl = [xl[0], xl[1], xl[2], self.tr3_branch3(xl[2])]
        for m in self.stage4:
            xl = m(xl)

        return xl[0]  # highest-resolution branch: [B, 32, H/4, W/4]

    # ------------------------------------------------------------------
    def _make_gaussian_targets(self, keypts, H, W):
        '''
        keypts : [B, 2, K]  normalised coords (x horizontal, y vertical) in [0,1]
        returns: [B, K, H, W] Gaussian heatmaps
        '''
        B, _, K = keypts.shape
        device  = keypts.device

        kx = keypts[:, 0, :] * (W - 1)   # [B, K]  column index
        ky = keypts[:, 1, :] * (H - 1)   # [B, K]  row index

        xs = torch.arange(W, device=device, dtype=torch.float32)
        ys = torch.arange(H, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # [H, W]

        kx = kx[:, :, None, None]  # [B, K, 1, 1]
        ky = ky[:, :, None, None]

        heatmaps = torch.exp(
            -((grid_x - kx) ** 2 + (grid_y - ky) ** 2) / (2.0 * self.sigma ** 2)
        )                                  # [B, K, H, W]
        return heatmaps

    # ------------------------------------------------------------------
    def _soft_argmax(self, heatmaps):
        '''
        heatmaps : [B, K, H, W]
        returns xc [B, K], yc [B, K] normalised [0,1]
        '''
        B, K, H, W = heatmaps.shape
        device = heatmaps.device

        # Flatten spatial dims, apply temperature-scaled softmax
        flat    = heatmaps.view(B, K, -1)           # [B, K, H*W]
        weights = F.softmax(flat * 100.0, dim=-1)   # peaked distribution

        xs = torch.arange(W, device=device, dtype=torch.float32) / max(W - 1, 1)
        ys = torch.arange(H, device=device, dtype=torch.float32) / max(H - 1, 1)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

        grid_x = grid_x.reshape(-1)   # [H*W]
        grid_y = grid_y.reshape(-1)

        xc = (weights * grid_x).sum(dim=-1)   # [B, K]
        yc = (weights * grid_y).sum(dim=-1)

        return xc, yc

    # ------------------------------------------------------------------
    def forward(self, x, y=None):
        feat     = self._backbone(x)         # [B, 32,  H/4,  W/4]
        feat     = self.deconv_head(feat)    # [B, 32,  H/2,  W/2]
        heatmaps = self.final_conv(feat)     # [B, K,   H/2,  W/2]

        if y is not None:
            # ---- TRAINING ----
            # y: [B, 2, K] normalised keypoint coords from dataloader
            _, _, H, W = heatmaps.shape
            targets = self._make_gaussian_targets(y, H, W)   # [B, K, H, W]
            # .mean() stays small in fp16 (no overflow); multiply by H*W as a
            # Python scalar (fp32 op) to get readable ~10-50 range values.
            # Equivalent to sum(spatial).mean(batch,kp) but fp16-safe.
            loss = ((heatmaps - targets) ** 2).mean() * (H * W)
            sm   = {'loss_hm': float(loss.detach())}
            return loss, sm
        else:
            # ---- TESTING ----
            xc, yc = self._soft_argmax(heatmaps)   # [B, K] each
            return xc.cpu(), yc.cpu()


if __name__ == '__main__':
    model = HigherHRNet(num_keypoints=11)
    total = sum(p.numel() for p in model.parameters())
    print('Total parameters: {:,}'.format(total))

    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 2, 11)  # normalised keypoints

    loss, sm = model(x, y)
    print('Train loss:', loss.item(), sm)

    xc, yc = model(x)
    print('Test xc shape:', xc.shape, 'yc shape:', yc.shape)
