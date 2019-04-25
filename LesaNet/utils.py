# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains some utility codes.
# --------------------------------------------------------

import torch
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from openpyxl import load_workbook, Workbook
import cv2

from config import config, default

# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def checkpoint_name(epoch, last_bkup):
    return os.path.join(default.model_path,
                        '%s%s_epoch_%02d.pth.tar' % ('bkup_' if last_bkup else '', default.exp_name, epoch))


def save_checkpoint(epoch, model, optimizer, last_bkup=False):
    if not os.path.exists(default.model_path):
        os.mkdir(default.model_path)

    filename = checkpoint_name(epoch+1, last_bkup)
    state = {
            'epoch': epoch + 1,
            'default_arg': default,
            'config': config,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
    torch.save(state, filename)
    logger.info('saved %s', filename)

    prev_bkup = checkpoint_name(epoch, True)
    if os.path.exists(prev_bkup):  # remove previous backup
        os.remove(prev_bkup)
    return filename


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, avg_num=default.show_avg_loss):
        self.reset()
        self.avg_num = avg_num

    def reset(self):
        self.vals = []
        self.avg = 0
        self.val = 0

    def update(self, val):
        self.val = val
        self.vals = [val] + self.vals
        if len(self.vals) > self.avg_num:
            self.vals.pop()
        self.avg = np.mean(self.vals)


def var_size_collate(batch):
    # data tensors, nontensor info, target tensors
    def collate_tensors(batch):
        transposed = zip(*batch)
        collated = []
        for data in transposed:
            max_shape = np.array([d.shape for d in data]).max(axis=0)
            data_new = []
            for d in data:
                dshape = np.array(d.shape)
                if np.any(dshape != max_shape):
                    d = torch.nn.functional.pad(d, (0, max_shape[-1] - dshape[-1], 0, max_shape[-2] - dshape[-2]),
                                                'constant', 0)
                data_new.append(d)
            collated.append(torch.stack(data_new, 0))
        return collated

    batch_data = [b[0] for b in batch]
    collated_data = collate_tensors(batch_data)
    batch_target = [b[1] for b in batch]
    collated_target = collate_tensors(batch_target)
    collated_info = [b[2] for b in batch]

    return collated_data, collated_target, collated_info


def unique(l):
    lu = []
    for l1 in l:
        if l1 not in lu:
            lu.append(l1)
    return lu


from load_ct_img import windowing_rev, windowing
def debug(prob, target, infos, inputs):
    # save the most error im
    filenames = [info[0] for info in infos]
    indices = [info[1] for info in infos]
    err = np.max(np.abs(prob - target), axis=1)
    max_smp = np.argmax(err)
    err_term = np.argmax(np.abs(prob[max_smp] - target[max_smp]))

    fn_in = filenames[max_smp].replace('/', '_')
    index = indices[max_smp]
    fn_out = '%s_%d_%.2f_%05d_%s' % (default.term_list[err_term], target[max_smp, err_term],
                                prob[max_smp, err_term], index, fn_in)
    im = inputs[0][max_smp].detach().cpu().numpy()[0]
    bbox = inputs[2][max_smp].detach().cpu().numpy().astype(int)
    im = windowing(windowing_rev(im * 255 + config.PIXEL_MEANS, config.WINDOWING), [-175, 275]).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,255,0), thickness=2)

    fd = 'debug/'
    cv2.imwrite(fd+fn_out, im)


def print_accs(accs):
    items = ['m_AUC', 'pc_Pr', 'pc_Re', 'pc_F1',
             'wm_AUC', 'ov_Pr', 'ov_Re', 'ov_F1']
    res = {}
    res['m_AUC'] = accs['mean_auc']
    res['pc_F1'] = accs['mean_perclass_f1']
    res['pc_Pr'] = accs['mean_perclass_precision']
    res['pc_Re'] = accs['mean_perclass_recall']
    res['wm_AUC'] = accs['wt_mean_auc']
    res['ov_F1'] = accs['overall_f1']
    res['ov_Pr'] = accs['overall_precision']
    res['ov_Re'] = accs['overall_recall']

    print
    for key in items:
        print key, '\t',
    print
    for key in items:
        print '%.4f' % res[key], '\t',
    print

    for crit in ['auc_perclass', 'perclass_f1s', 'perclass_precisions', 'perclass_recalls']:
        msg = '\t'+crit+':\n'
        for cls in ['bodypart', 'type', 'attribute']:
            mask = np.array(default.term_class) == cls
            acc1 = accs[crit][mask]
            msg += '%s (%d): %.4f;\t' % (cls, np.count_nonzero(~np.isnan(acc1)), np.nanmean(acc1))

        msg += '\n'
        names = ['frequent', 'medium', 'rare']
        ranges = [[1001,100000], [101,1000], [0,100]]
        for idx in range(len(names)):
            mask = (ranges[idx][0] <= default.cls_sz_train) & (default.cls_sz_train <= ranges[idx][1])
            acc1 = accs[crit][mask]
            msg += '%s (%d): %.4f;\t' % (names[idx], np.count_nonzero(~np.isnan(acc1)), np.nanmean(acc1))
        print msg
    print


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    # print totalnorm
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)