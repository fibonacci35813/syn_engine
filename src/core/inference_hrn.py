'''
Validation loop for HigherHRNet (HRN).
Interface is identical to valid_krn: the model returns (xc, yc) normalised
coordinates which are then fed to solvePnP for 6-DoF pose recovery.
'''
from __future__ import absolute_import, division, print_function

import logging
import time
import os.path as osp

import torch

from src.utils.utils    import pnp, AverageMeter, report_progress
from src.utils.metrics  import error_orientation, error_translation, speed_score

import numpy as np

logger = logging.getLogger('Testing')


def valid_hrn(epoch, cfg, model, data_loader,
              cameraMatrix, distCoeffs, corners3D,
              writer, device, qClass=None):
    '''Validate HigherHRNet model — mirrors valid_krn from inference.py.'''

    time_meter     = AverageMeter('ms')
    err_q_meter    = AverageMeter('deg')
    err_t_meter    = AverageMeter('m')
    speed_meter    = AverageMeter('-')
    speed_meter_th = AverageMeter('-')
    acc_meter      = AverageMeter('%')

    err_q_all    = []
    err_t_all    = []
    speed_raw_all = []
    speed_mod_all = []

    model.eval()

    for idx, (images, bbox, q_gt, t_gt) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        images = images.to(device)
        with torch.no_grad():
            x_pr, y_pr = model(images)   # [B, K] each, normalised [0,1]

        time_meter.update((time.time() - start) * 1000, B)

        for b in range(B):
            try:
                q_pr, t_pr = _keypts_to_pose(
                    x_pr[b], y_pr[b], bbox[b], corners3D, cameraMatrix, distCoeffs)

                # Skip this sample if PnP returned non-finite values
                if not (np.isfinite(t_pr).all() and np.isfinite(q_pr).all()):
                    raise ValueError('PnP returned non-finite pose')

            except Exception:
                # Degenerate keypoints (collapsed / collinear) → treat as max-error
                q_pr = np.array([1.0, 0.0, 0.0, 0.0])
                t_pr = np.array([0.0, 0.0, 10.0])

            q_gt_i = q_gt[b].numpy()
            t_gt_i = t_gt[b].numpy()

            err_q     = error_orientation(q_pr, q_gt_i)
            err_t     = error_translation(t_pr, t_gt_i)
            speed_raw, acc = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=False)
            speed_mod, _   = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=True,
                                         rotThresh=0.169, posThresh=0.002173)

            # Update per-sample (bug fix: was outside loop, scoring only last sample)
            err_q_meter.update(err_q, 1)
            err_t_meter.update(err_t, 1)
            speed_meter.update(speed_raw, 1)
            speed_meter_th.update(speed_mod, 1)
            acc_meter.update(acc * 100, 1)

            err_q_all.append(err_q)
            err_t_all.append(err_t)
            speed_raw_all.append(speed_raw)
            speed_mod_all.append(speed_mod)

        report_progress(epoch=epoch, lr=np.nan, epoch_iter=idx + 1,
                        epoch_size=len(data_loader), time=time_meter,
                        is_train=False, eT=err_t_meter, eR=err_q_meter,
                        speed=speed_meter, acc=acc_meter)

    if writer is not None:
        writer.add_scalar('Valid/err_q [deg]',     err_q_meter.avg,    epoch)
        writer.add_scalar('Valid/err_t [m]',        err_t_meter.avg,    epoch)
        writer.add_scalar('Valid/speed (raw) [-]',  speed_meter.avg,    epoch)
        writer.add_scalar('Valid/speed (thr) [-]',  speed_meter_th.avg, epoch)

    performances = {
        'eR':           err_q_meter,
        'eT':           err_t_meter,
        'speed (raw)':  speed_meter,
        'speed (thr)':  speed_meter_th,
    }

    with open(osp.join(cfg.logdir, 'err_q.txt'), 'w') as f:
        for eq in err_q_all:
            f.write('{:.5f}\n'.format(eq))

    with open(osp.join(cfg.logdir, 'err_t.txt'), 'w') as f:
        for et in err_t_all:
            f.write('{:.5f}\n'.format(et))

    with open(osp.join(cfg.logdir, 'speed_raw.txt'), 'w') as f:
        for spd in speed_raw_all:
            f.write('{:.5f}\n'.format(spd))

    with open(osp.join(cfg.logdir, 'speed_mod.txt'), 'w') as f:
        for spd in speed_mod_all:
            f.write('{:.5f}\n'.format(spd))

    # Also write final summary (mirrors results.txt format used for KRN)
    with open(osp.join(cfg.logdir, 'results.txt'), 'w') as f:
        f.write('eR: {:.5f} [deg]\n'.format(err_q_meter.avg))
        f.write('eT: {:.5f} [m]\n'.format(err_t_meter.avg))
        f.write('speed (raw): {:.5f} [-]\n'.format(speed_meter.avg))
        f.write('speed (thr): {:.5f} [-]\n'.format(speed_meter_th.avg))

    return performances


def _keypts_to_pose(x_pr, y_pr, bbox, corners3D,
                    cameraMatrix, distCoeffs=np.zeros((1, 5))):
    '''Identical helper to inference.py:_keypts_to_pose.'''
    import torch
    corners2D_pr = torch.cat((x_pr.unsqueeze(0), y_pr.unsqueeze(0)), dim=0)  # [2, K]
    corners2D_pr = corners2D_pr.cpu().t().numpy()  # [K, 2]

    xmin, xmax, ymin, ymax = bbox.numpy()
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * (xmax - xmin) + xmin
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * (ymax - ymin) + ymin

    q_pr, t_pr = pnp(corners3D, corners2D_pr, cameraMatrix, distCoeffs)
    return q_pr, t_pr
