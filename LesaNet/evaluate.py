# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to evaluate the multi-label lesion
# classification accuracy.
# --------------------------------------------------------

import numpy as np
from sklearn.metrics import roc_curve, auc

from utils import unique
from config import default, config


def score2label(pred_scores, K0):
    pred_lbs = np.zeros_like(pred_scores, dtype=bool)
    for smp, score in enumerate(pred_scores):  # iterate on each sample
        if config.TEST.SCORE_PARAM > 1:
            # top-K based
            K = min(K0, len(score))
            rank = np.argsort(score)[::-1]
            labels = rank[:K].tolist()

            # label expansion, no significant change
            # expanded_labels = []
            # cur_rank = -1
            # for k in range(K):
            #     cur_rank += 1
            #     expanded_labels = unique(expanded_labels + [rank[cur_rank]] + default.parent_list[rank[cur_rank]])
        else:
            # threshold based
            th = K0
            labels = np.where(score >= th)[0].tolist()
            # expanded_labels = unique(labels + [p for label1 in labels for p in default.parent_list[label1]])

        if config.TEST.FILTER_EXCLUSIVE_LABELS:
            # non-max suppression using exclusive terms
            ord = np.argsort(score[labels])[::-1]
            filtered_labels = []
            for ord1 in ord:
                lb1 = labels[ord1]
                if len(np.intersect1d(filtered_labels, default.exclusive_list[lb1])) == 0:
                    filtered_labels.append(lb1)
            labels = filtered_labels
        pred_lbs[smp, labels] = True
    return pred_lbs


def compute_all_acc_wt(lb_te_all, pred_te_all, score_te_all, pred_wt_all):
    # if pred_wt==0, don't consider the label of the sample, because it is uncertain

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


def compute_all_acc_wt_th(lb_te_all, score_te_all, pred_wt_all, use_val_th):
    # if use_val_th is False, compute the best threshold for each label
    # if it is True, use the thresholds computed in the validation set (stored in default.best_cls_ths)

    assert lb_te_all.shape == score_te_all.shape
    num_smp, num_cls = lb_te_all.shape
    accs = {}
    precf = lambda lb1, pred1: (np.count_nonzero(lb1 & pred1) / float(np.count_nonzero(pred1))) if np.count_nonzero(pred1) else 1.
    recf = lambda lb1, pred1: (np.count_nonzero(lb1 & pred1) / float(np.count_nonzero(lb1))) if np.count_nonzero(lb1) else 1.
    f1f = lambda prec, rec: (2*prec*rec/(prec+rec)) if (prec+rec) else 0.

    def search_th(prob, lb):
        ths = np.arange(.3, 1, .1)
        best_th = 1
        best_f1 = 0
        for th1 in ths:
            pred1 = prob >= th1
            mp = precf(lb, pred1)
            mr = recf(lb, pred1)
            mf = f1f(mp, mr)
            if mf > best_f1:
                best_f1 = mf
                best_th = th1
        return best_th

    # non-weighted class-level
    mps = np.zeros((num_cls, ))
    mrs = np.zeros((num_cls, ))
    mfs = np.zeros((num_cls, ))
    pred_te_all = lb_te_all > 100
    for cls in range(num_cls):
        wt1 = pred_wt_all[:, cls]
        lb1 = lb_te_all[wt1, cls]
        prob1 = score_te_all[wt1, cls]
        if not (np.any(wt1) and np.any(lb1) and np.any(~lb1)):
            mps[cls] = np.nan
            mrs[cls] = np.nan
            mfs[cls] = np.nan
            continue

        if not use_val_th:
            th = search_th(prob1, lb1)
            default.best_cls_ths[cls] = th
        else:
            th = default.best_cls_ths[cls]
        pred1 = prob1 >= th
        pred_te_all[wt1, cls] = pred1

    for cls in range(num_cls):  # label expansion
        if len(default.parent_list[cls]) > 0:
            pred_te_all[:,default.parent_list[cls]] = pred_te_all[:,default.parent_list[cls]] | pred_te_all[:,[cls]]

    if config.TEST.FILTER_EXCLUSIVE_LABELS:
        pred_te_all1 = pred_te_all*0
        for smp in range(num_smp):
            score = score_te_all[smp]
            expanded_labels = np.where(pred_te_all[smp])[0]
            ord = np.argsort(score[expanded_labels])[::-1]
            filtered_labels = []
            for ord1 in ord:
                lb1 = expanded_labels[ord1]
                if len(np.intersect1d(filtered_labels, default.exclusive_list[lb1])) == 0:
                    filtered_labels.append(lb1)
            pred_te_all1[smp, filtered_labels] = True
        pred_te_all = pred_te_all1

    for cls in range(num_cls):
        wt1 = pred_wt_all[:, cls]
        lb1 = lb_te_all[wt1, cls]
        pred1 = pred_te_all[wt1, cls]
        if not (np.any(wt1) and np.any(lb1) and np.any(~lb1)):
            mps[cls] = np.nan
            mrs[cls] = np.nan
            mfs[cls] = np.nan
            continue
        mps[cls] = precf(lb1, pred1)
        mrs[cls] = recf(lb1, pred1)
        mfs[cls] = f1f(mps[cls], mrs[cls])

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
        if np.count_nonzero(wt1) > 1 and np.any(lb1) and np.any(~lb1):
            fpr, tpr, threshold = roc_curve(lb1, score_te_all[wt1, cls])
            aucs.append(auc(fpr, tpr))
        else:
            aucs.append(np.nan)
    accs['auc_perclass'] = np.array(aucs)
    accs['mean_auc'] = np.nanmean(aucs)
    wt = (lb_te_all & pred_wt_all).sum(axis=0).astype(float) / (lb_te_all & pred_wt_all).sum()
    accs['wt_mean_auc'] = np.nansum([a*w for a,w in zip(aucs, wt)])

    return accs, pred_te_all
