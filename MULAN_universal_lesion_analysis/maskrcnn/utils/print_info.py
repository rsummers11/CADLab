# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Utilities to write or save results"""
import os
import numpy as np
from openpyxl import load_workbook, Workbook
import torch

from maskrcnn.config import cfg
from .miscellaneous import unique


def save_acc_to_file(acc_all, prefix=''):
    """Save tagging and tag-wise detection accuracies to a spreadsheet"""
    rownames = cfg.runtime_info.tag_list[:]
    colnames = ['train_positive', 'test_positive', 'auc', 'precision', 'recall', 'f1', 'threshold']
    if cfg.TEST.COMPUTE_DET_ACC_PER_TAG:
        colnames.extend(['det@FP%.1f' % fp for fp in cfg.TEST.VAL_FROC_FP])
    if prefix == 'test_handlabeled':
        test_cls_sz = cfg.runtime_info.manual_test_set_cls_sz
    elif prefix == 'val_mined':
        test_cls_sz = cfg.runtime_info.val_cls_sz
    tag_sel_val = cfg.runtime_info.tag_sel_val
    if isinstance(tag_sel_val, torch.Tensor):
        tag_sel_val = tag_sel_val.numpy()
    else:
        tag_sel_val = test_cls_sz * 0 + cfg.TEST.TAG.SELECTION_VAL
    matdata = np.vstack((cfg.runtime_info.train_cls_sz, test_cls_sz, acc_all['auc_perclass'],
                         acc_all['perclass_precisions'], acc_all['perclass_recalls'], acc_all['perclass_f1s'],
                         tag_sel_val))
    matdata = matdata.transpose()
    if cfg.TEST.COMPUTE_DET_ACC_PER_TAG:
        matdata = np.hstack((matdata, cfg.runtime_info.det_acc_per_tag))
    avg = [np.nan, np.nan, acc_all['mean_auc'], acc_all['mean_perclass_precision'],
                                    acc_all['mean_perclass_recall'], acc_all['mean_perclass_f1'], np.nan]
    if cfg.TEST.COMPUTE_DET_ACC_PER_TAG:
        avg += [np.nan] * len(cfg.TEST.VAL_FROC_FP)
    matdata = np.vstack((matdata, avg))
    rownames.append('Average')
    filename = os.path.join(cfg.RESULTS_DIR, '%s_%s.xlsx' % (prefix, cfg.EXP_NAME))
    if not os.path.exists(cfg.RESULTS_DIR):
        os.mkdir(cfg.RESULTS_DIR)
    save_to_xlsx_2d(filename, colnames, rownames, matdata)


def save_to_xlsx_2d(fn, col_names, row_names, mat):
    """Write data to a spreadsheet"""
    cellname = lambda row, col: '%s%d' % (chr(ord('A') + col - 1), row)
    wb = Workbook()
    sheet = wb.active
    for i, key in enumerate(row_names):
        sheet[cellname(i + 2, 1)] = key
    for i, key in enumerate(col_names):
        sheet[cellname(1, i + 2)] = key

    for row in range(len(row_names)):
        for col in range(len(col_names)):
            sheet[cellname(row + 2, col+2)] = mat[row, col]
    res = wb.save(fn)
    return res


def gen_tag_pred_str(pred_labels, scores):
    """Generate a string of tagging results.
    Only show tags in leaf nodes (not a parent tag)"""
    assert scores.ndim == 1
    tag_list = cfg.runtime_info.tag_list
    parent_list = cfg.runtime_info.parent_list

    if not np.any(pred_labels):
        # pred_labels = score2label(scores[None, :], K=5)
        pred_labels = np.argsort(scores)[::-1][:5]
    else:
        pred_labels = np.where(pred_labels)[0]

    # only show tags in leaf nodes (not a parent tag)
    all_parents = unique([p for l in pred_labels for p in parent_list[l]])
    pred_labels = [l for l in pred_labels if l not in all_parents]
    ord = np.argsort(scores[pred_labels])[::-1]
    msg = ''
    for idx in np.array(pred_labels)[ord]:
        msg += '%s: %.3f, ' % (tag_list[idx], scores[idx])
    # msg += '>parents: '
    # ord = np.argsort(scores[all_parents])[::-1]
    # for idx in np.array(all_parents)[ord]:
    #     msg += '%s: %.3f, ' % (tag_list[idx], scores[idx])

    return msg


def get_debug_info():
    """Format debug information"""
    info = ''
    for k in cfg.debug_info.keys():
        v = cfg.debug_info[k]
        if isinstance(v, (int, float)):
            s = '%g' % v
        elif isinstance(v, str):
            s = v
        else:
            s = str(v)

        info += '\t%s=%s' % (k, s)
    return info
