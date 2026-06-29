'''
Training loop for HigherHRNet (HRN) — mirrors trainer.py:train_single_epoch_krn
but reports heatmap loss instead of per-axis coordinate losses.
'''
from __future__ import absolute_import, division, print_function

import time
import random
import logging

from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast

from src.utils.utils import AverageMeter, report_progress

logger = logging.getLogger('Training')


def train_single_epoch_hrn(epoch, cfg, model, data_loader, optimizer,
                            writer, device, styleAugmentor=None, scaler=None):
    time_meter = AverageMeter('ms')
    loss_meter = AverageMeter('-')

    model.train()

    for pg in optimizer.param_groups:
        lr = pg['lr']

    for idx, (images, target) in enumerate(data_loader):
        start  = time.time()
        B      = images.shape[0]

        images = images.to(device)
        target = target.to(device)

        if styleAugmentor is not None and random.random() < cfg.texture_ratio:
            images = styleAugmentor(images)

        if scaler is not None and cfg.use_cuda:
            with autocast():
                loss, summary = model(images, target)
        else:
            loss, summary = model(images, target)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        time_meter.update((time.time() - start) * 1000, B)
        loss_meter.update(summary['loss_hm'], B)

        report_progress(epoch=epoch, lr=lr, epoch_iter=idx + 1,
                        epoch_size=len(data_loader), time=time_meter,
                        is_train=True, loss_hm=loss_meter)

    if writer is not None:
        writer.add_scalar('train/loss_hm', loss_meter.avg, epoch)
