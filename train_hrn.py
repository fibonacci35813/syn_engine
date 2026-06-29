'''
Self-contained training script for HigherHRNet on SPEED+.
Does NOT modify any existing files (train.py, config.py, build.py, etc.).

Usage:
    python train_hrn.py [options]

Example (synthetic full, matching KRN setup):
    python train_hrn.py \
        --projroot /home/satyam/speedplusbaseline \
        --dataroot /home/satyam/speedplusv2 \
        --savedir  checkpoints/hrn/synthetic_full \
        --logdir   log/hrn/synthetic_full \
        --max_epochs 50 --batch_size 8 --use_fp16
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as osp
import json
import logging
from scipy.io import loadmat

import torch
from torch.utils.tensorboard import SummaryWriter

from src.nets.higherhrnet       import HigherHRNet
from src.core.trainer_hrn       import train_single_epoch_hrn
from src.core.inference_hrn     import valid_hrn
from src.datasets.build         import make_dataloader
from src.utils.utils            import (setup_logger, set_all_seeds,
                                        save_checkpoint, load_checkpoint,
                                        load_tango_3d_keypoints,
                                        load_camera_intrinsics,
                                        num_total_parameters,
                                        num_trainable_parameters)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_cfg():
    parser = argparse.ArgumentParser('HigherHRNet training on SPEED+')

    # Directories
    parser.add_argument('--seed',       type=int, default=2021)
    parser.add_argument('--projroot',   type=str, default='/home/satyam/speedplusbaseline')
    parser.add_argument('--dataroot',   type=str, default='/home/satyam/speedplusv2')
    parser.add_argument('--dataname',   type=str, default='speedplus')
    parser.add_argument('--savedir',    type=str, default='checkpoints/hrn/synthetic_full')
    parser.add_argument('--logdir',     type=str, default='log/hrn/synthetic_full')
    parser.add_argument('--pretrained', type=str, default='')

    # Model
    parser.add_argument('--num_keypoints',   type=int,   default=11)
    parser.add_argument('--heatmap_sigma',   type=float, default=2.0)
    parser.add_argument('--input_shape',     nargs='+',  type=int, default=(224, 224))
    parser.add_argument('--keypts_3d_model', type=str,
                        default='src/utils/tangoPoints.mat')
    parser.add_argument('--attitude_class',  type=str,
                        default='src/utils/attitudeClasses.mat')

    # Training
    parser.add_argument('--start_over',    dest='auto_resume', action='store_false', default=True)
    parser.add_argument('--use_fp16',      dest='fp16',        action='store_true',  default=False)
    parser.add_argument('--batch_size',    type=int,   default=8)
    parser.add_argument('--max_epochs',    type=int,   default=50)
    parser.add_argument('--num_workers',   type=int,   default=8)
    parser.add_argument('--test_epoch',    type=int,   default=-1)
    parser.add_argument('--optimizer',     type=str,   default='adamw')
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--momentum',      type=float, default=0.9)
    parser.add_argument('--weight_decay',  type=float, default=0.01)
    parser.add_argument('--lr_decay_alpha',type=float, default=0.95)
    parser.add_argument('--lr_decay_step', type=int,   default=1)

    # Dataset — NOTE: model_name is set to 'krn' so we reuse the same splits_krn/ CSVs
    # HRNet uses identical data format; only the backbone/head differ.
    parser.add_argument('--train_domain', type=str, default='synthetic')
    parser.add_argument('--test_domain',  type=str, default='synthetic')
    parser.add_argument('--train_csv',    type=str, default='train.csv')
    parser.add_argument('--test_csv',     type=str, default='validation.csv')

    # Misc
    parser.add_argument('--gpu_id',  type=int, default=0)
    parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', default=True)

    cfg = parser.parse_args()

    # HRN uses the same KRN dataset/transforms (same CSV structure, same crop)
    # We set model_name='krn' only for the dataloader path resolution.
    cfg.model_name   = 'krn'
    cfg.num_classes  = 5000   # unused by HRN; needed by some utility imports
    cfg.num_neighbors= 5
    cfg.dann         = False
    cfg.randomize_texture = False
    cfg.texture_alpha= 0.5
    cfg.texture_ratio= 0.5

    return cfg


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def get_optimizer(cfg, model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.optimizer == 'sgd':
        opt = torch.optim.SGD(params, lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam':
        opt = torch.optim.Adam(params, lr=cfg.lr,
                               betas=(cfg.momentum, 0.999), weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamw':
        opt = torch.optim.AdamW(params, lr=cfg.lr,
                                betas=(cfg.momentum, 0.999), weight_decay=cfg.weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}'.format(cfg.optimizer))
    logger.info('Optimizer: {}'.format(cfg.optimizer))
    return opt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg    = get_cfg()
    device = (torch.device('cuda:{}'.format(cfg.gpu_id))
              if torch.cuda.is_available() and cfg.use_cuda
              else torch.device('cpu'))

    setup_logger('train_hrn')
    logger.info('Random seed: {}'.format(cfg.seed))
    set_all_seeds(cfg.seed, cfg, cfg.use_cuda)

    os.makedirs(cfg.savedir, exist_ok=True)
    os.makedirs(cfg.logdir,  exist_ok=True)
    logger.info('Checkpoints → {}'.format(cfg.savedir))
    logger.info('Tensorboard → {}'.format(cfg.logdir))

    # Save config snapshot
    cfg_dict = {k: v for k, v in vars(cfg).items()}
    cfg_dict['model_name'] = 'hrn'   # override for the saved config record
    with open(osp.join(cfg.savedir, 'config.txt'), 'w') as f:
        json.dump(cfg_dict, f, indent=2)

    writer = SummaryWriter(cfg.logdir)

    # Model
    model = HigherHRNet(num_keypoints=cfg.num_keypoints,
                        heatmap_sigma=cfg.heatmap_sigma)
    logger.info('HigherHRNet created')
    logger.info('   - Total parameters:     {:,}'.format(num_total_parameters(model)))
    logger.info('   - Trainable parameters: {:,}'.format(num_trainable_parameters(model)))

    optimizer = get_optimizer(cfg, model)

    # Resume
    checkpoint_file = osp.join(cfg.savedir, 'checkpoint.pth.tar')
    if cfg.auto_resume and osp.exists(checkpoint_file):
        begin_epoch, best_perf = load_checkpoint(checkpoint_file, model, optimizer, device)
    else:
        begin_epoch = 0
        best_perf   = 0

    model = model.to(device)

    # Mixed precision
    scaler = None
    if cfg.fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info('Mixed-precision (fp16) enabled')

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha)

    # Dataloaders (reuse KRN dataset — same CSV / transforms)
    train_loader = make_dataloader(cfg, is_train=True,  is_source=True)
    test_loader  = make_dataloader(cfg, is_train=False, is_source=False)

    # Pose utilities
    corners3D = load_tango_3d_keypoints(cfg.keypts_3d_model)
    cameraMatrix, distCoeffs = load_camera_intrinsics(
        osp.join(cfg.dataroot, cfg.dataname, 'camera.json'))

    # Training loop
    for epoch in range(begin_epoch, cfg.max_epochs):
        train_single_epoch_hrn(
            epoch + 1, cfg, model, train_loader, optimizer,
            writer, device, styleAugmentor=None, scaler=scaler)

        lr_scheduler.step()

        # Periodic validation (set --test_epoch N to validate every N epochs)
        if cfg.test_epoch > 0 and (epoch + 1) % cfg.test_epoch == 0:
            valid_hrn(epoch + 1, cfg, model, test_loader,
                      cameraMatrix, distCoeffs, corners3D, writer, device)

        # Save checkpoint every epoch (always mark as best to mirror KRN behaviour)
        perf    = epoch + 1
        is_best = perf > best_perf
        if is_best:
            best_perf = perf

        save_checkpoint({
            'epoch':      epoch + 1,
            'model':      'hrn',
            'state_dict': model.state_dict(),
            'best_score': best_perf,
            'optimizer':  optimizer.state_dict(),
        }, is_best, cfg.savedir)

    # Final validation after all epochs
    logger.info('Running final validation ...')
    valid_hrn(cfg.max_epochs, cfg, model, test_loader,
              cameraMatrix, distCoeffs, corners3D, writer, device)

    writer.close()
    logger.info('Training complete. Results in {}'.format(cfg.logdir))


if __name__ == '__main__':
    main()
