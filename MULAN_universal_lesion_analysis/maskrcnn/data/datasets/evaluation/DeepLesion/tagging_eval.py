# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Evaluation code for tags in DeepLesion"""
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch

from maskrcnn.data.datasets.DeepLesion_utils import unique
from maskrcnn.config import cfg


def score2label_np(pred_scores, K=5):
    """Convert tag scores to binary predictions with numpy"""
    # K: int or nparray

    pred_lbs = np.zeros_like(pred_scores, dtype=bool)
    parent_list = cfg.runtime_info.parent_list
    for smp, score in enumerate(pred_scores):  # iterate on each sample
        if isinstance(K, int) and K > 1:
            # top-K based
            K = min(K, len(score))
            rank = np.argsort(score)[::-1]
            expanded_labels = []
            cur_rank = -1
            for k in range(K):
                cur_rank += 1
                # while True:  # if rank[p] is in expanded_labels (is a parent of previous labels), ignore it
                #     cur_rank += 1
                #     if rank[cur_rank] not in expanded_labels:
                #         break
                expanded_labels = unique(expanded_labels + [rank[cur_rank]] + parent_list[rank[cur_rank]])
                # expanded_labels = unique(expanded_labels + [rank[cur_rank]])
        else:
            # threshold based, single threshold or array threshold
            labels = np.where(score >= K)[0].tolist()
            expanded_labels = unique(labels + [p for label1 in labels for p in parent_list[label1]])

        if cfg.TEST.TAG.FILTER_EXCLUSIVE_LABELS:
            # non-max suppression using exclusive terms
            ord = np.argsort(score[expanded_labels])[::-1]
            filtered_labels = []
            for ord1 in ord:
                lb1 = expanded_labels[ord1]
                if len(np.intersect1d(filtered_labels, cfg.runtime_info.exclusive_list[lb1])) == 0:
                    filtered_labels.append(lb1)
            expanded_labels = filtered_labels
        pred_lbs[smp, expanded_labels] = True
    return pred_lbs


def score2label(pred_scores, K=5):
    """Convert tag scores to binary predictions with Tensor"""
    # tensor version. K: int or tensor

    pred_lbs = torch.zeros_like(pred_scores, dtype=torch.uint8)
    if isinstance(K, int) and K > 1:
        # top-K based
        K = min(K, pred_lbs.shape[1])
        for smp, score in enumerate(pred_scores):  # iterate on each sample
            rank = torch.argsort(score, descending=True).tolist()
            labels = rank[:K]
            pred_lbs[smp, labels] = 1
    else:
        # threshold based, single threshold or array threshold
        pred_lbs = pred_scores >= K

    return pred_lbs


def compute_all_acc_wt(lb_te_all, pred_te_all, score_te_all, pred_wt_all):
    """Compute various tagging accuracy metrics"""
    # if pred_wt==0, ignore the label of the sample

    assert lb_te_all.shape == pred_te_all.shape
    num_smp, num_cls = lb_te_all.shape
    accs = {}
    precf = lambda lb1, pred1: (np.count_nonzero(lb1 & pred1) / float(np.count_nonzero(pred1))) if np.count_nonzero(pred1) else 1.
    recf = lambda lb1, pred1: (np.count_nonzero(lb1 & pred1) / float(np.count_nonzero(lb1))) if np.count_nonzero(lb1) else 1.
    f1f = lambda prec, rec: (2*prec*rec/(prec+rec)) if (prec+rec) else 0.

    # non-weighted class-level
    mps = np.zeros((num_cls, ))
    mrs = np.zeros((num_cls, ))
    mfs = np.zeros((num_cls, ))
    for cls in range(num_cls):
        wt1 = pred_wt_all[:, cls]
        lb1 = lb_te_all[wt1, cls]
        pred1 = pred_te_all[wt1, cls]
        if np.any(wt1) and np.any(lb1) and np.any(~lb1):
            mps[cls] = precf(lb1, pred1)
            mrs[cls] = recf(lb1, pred1)
            mfs[cls] = f1f(mps[cls], mrs[cls])
        else:
            mps[cls] = np.nan
            mrs[cls] = np.nan
            mfs[cls] = np.nan
    accs['perclass_precisions'] = mps
    accs['perclass_recalls'] = mrs
    accs['perclass_f1s'] = mfs
    accs['mean_perclass_precision'] = np.nanmean(mps)
    accs['mean_perclass_recall'] = np.nanmean(mrs)
    accs['mean_perclass_f1'] = np.nanmean(mfs)

    # weighted overall
    accs['overall_precision'] = precf(lb_te_all[pred_wt_all], pred_te_all[pred_wt_all])
    accs['overall_recall'] = recf(lb_te_all[pred_wt_all], pred_te_all[pred_wt_all])
    accs['overall_f1'] = f1f(accs['overall_precision'], accs['overall_recall'])

    # per class auc
    aucs = []
    for cls in range(num_cls):
        wt1 = pred_wt_all[:, cls]
        lb1 = lb_te_all[wt1, cls]
        pred1 = pred_te_all[wt1, cls]
        if np.count_nonzero(wt1) > 1 and np.any(lb1) and np.any(~lb1):
            fpr, tpr, threshold = roc_curve(lb1, score_te_all[wt1, cls])
            aucs.append(auc(fpr, tpr))
        else:
            aucs.append(np.nan)
    accs['auc_perclass'] = np.array(aucs)
    accs['mean_auc'] = np.nanmean(aucs)
    wt = (lb_te_all & pred_wt_all).sum(axis=0).astype(float) / (lb_te_all & pred_wt_all).sum()
    accs['wt_mean_auc'] = np.nansum([a*w for a,w in zip(aucs, wt)])

    return accs


def compute_thresholds(lb_te_all, score_te_all, pred_wt_all):
    """Find the best threshold for each tag from np.arange(.2, 1, .05)"""
    assert lb_te_all.shape == score_te_all.shape
    num_smp, num_cls = lb_te_all.shape
    precf = lambda lb1, pred1: (np.count_nonzero(lb1 & pred1) / float(np.count_nonzero(pred1))) if np.count_nonzero(pred1) else 1.
    recf = lambda lb1, pred1: (np.count_nonzero(lb1 & pred1) / float(np.count_nonzero(lb1))) if np.count_nonzero(lb1) else 1.
    f1f = lambda prec, rec: (2*prec*rec/(prec+rec)) if (prec+rec) else 0.
    accf = lambda lb1, pred1: np.count_nonzero(lb1 == pred1) / float(len(lb1))
    specf = lambda lb1, pred1: (np.count_nonzero((lb1==0) & (pred1==0)) / float(np.count_nonzero(lb1==0))) if np.count_nonzero(lb1==0) else 1.

    best_cls_ths = np.ones((num_cls,), dtype=float) #+ cfg.TEST.TAG.SELECTION_VAL
    criterion = 'F1'
    ths = np.arange(.2, 1, .05)

    for cls in range(num_cls):
        wt1 = pred_wt_all[:, cls]
        lb1 = lb_te_all[wt1, cls]
        prob1 = score_te_all[wt1, cls]
        best_th = 1#cfg.TEST.TAG.SELECTION_VAL
        best_res = 0
        if not (np.any(wt1) and np.any(lb1) and np.any(~lb1)):
            continue
        for th1 in ths:
            pred1 = prob1 >= th1
            if criterion == 'F1':
                mp = precf(lb1, pred1)
                mr = recf(lb1, pred1)
                m = f1f(mp, mr)
            elif criterion == 'acc':
                m = accf(lb1, pred1)
            elif criterion == 'Youden':
                msens = recf(lb1, pred1)
                mspec = specf(lb1, pred1)
                m = msens + mspec - 1
            if criterion == 'Youden-F1':
                mp = precf(lb1, pred1)
                mr = recf(lb1, pred1)
                m = mp + mr - 1
            if m > best_res:
                best_res = m
                best_th = th1
        best_cls_ths[cls] = best_th

    return best_cls_ths


def print_accs(accs, logger):
    """Print tag accuracy metrics"""
    items = ['m_AUC', 'pc_F1', 'pc_Pr', 'pc_Re',
             'wm_AUC', 'ov_F1', 'ov_Pr', 'ov_Re']
    res = {}
    res['m_AUC'] = accs['mean_auc']
    res['pc_F1'] = accs['mean_perclass_f1']
    res['pc_Pr'] = accs['mean_perclass_precision']
    res['pc_Re'] = accs['mean_perclass_recall']
    res['wm_AUC'] = accs['wt_mean_auc']
    res['ov_F1'] = accs['overall_f1']
    res['ov_Pr'] = accs['overall_precision']
    res['ov_Re'] = accs['overall_recall']

    msg = ""
    for key in items:
        msg += key + '\t'
    msg += '\n'
    for key in items:
        msg += '%.4f' % res[key] + '\t'
    msg += '\n'

    # for crit in ['auc_perclass', 'perclass_f1s', 'perclass_precisions', 'perclass_recalls']:
    #     msg = '\t'+crit+':\n'
    #     for cls in ['bodypart', 'finding', 'finding-attribute']:
    #         mask = np.array(config.term_class) == cls
    #         acc1 = accs[crit][mask]
    #         msg += '%s (%d): %.4f;\t' % (cls, np.count_nonzero(~np.isnan(acc1)), np.nanmean(acc1))
    #
    #     msg += '\n'
    #     names = ['frequent', 'medium', 'rare']
    #     ranges = [[1001,100000], [101,1000], [0,100]]
    #     for idx in range(len(names)):
    #         mask = (ranges[idx][0] <= default.cls_sz_train) & (default.cls_sz_train <= ranges[idx][1])
    #         acc1 = accs[crit][mask]
    #         msg += '%s (%d): %.4f;\t' % (names[idx], np.count_nonzero(~np.isnan(acc1)), np.nanmean(acc1))
    #     print(msg)
    # print

    # if len(default.term_list) < 10:
    #     for t, a in zip(default.term_list, accs['auc_perclass']):
    #         logger.info('%s: %.4f', t, a)
    # else:
    #     msg = ''
    #     for t, a in zip(default.term_list, accs['auc_perclass']):
    #         msg += '%s: %.3f | ' % (t, a)
    #     logger.info(msg)

    logger.info(msg)
