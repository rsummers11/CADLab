# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to train and validate LesaNet.
# --------------------------------------------------------

import time
import torch
import numpy as np

from utils import AverageMeter, logger, debug, print_accs, clip_gradient
from load_save_utils import save_test_scores_to_file, save_ft_to_file, save_acc_to_file
from evaluate import score2label, compute_all_acc_wt, compute_all_acc_wt_th
from config import config, default
from my_algorithm import select_triplets_multilabel


def train(train_loader, model, criterions, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets, infos) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # construct targets
        target_clsf, target_unc, target_ex = targets
        target_conf = (target_clsf + target_ex) > 0
        rhem_wt = torch.zeros_like(target_clsf).cuda()
        rhem_wt[target_conf] = 1.
        target_clsf = target_clsf.cuda()
        target_clsf_wt = 1-target_unc.cuda()

        # run model
        inputs = [input.cuda() for input in inputs]
        out = model(inputs)

        # compute losses
        emb = out['emb']
        A, P, N = select_triplets_multilabel(emb, target_clsf)
        loss_metric = criterions['metric'](A, P, N)

        prob1 = out['class_prob1']
        loss_ce1 = criterions['wce'](prob1, target_clsf, infos, wt=target_clsf_wt)
        loss_rhem = criterions['rhem'](prob1, target_clsf, infos, wt=rhem_wt)
        if config.SCORE_PROPAGATION:
            prob2 = out['class_prob2']
            loss_ce2 = criterions['wce'](prob2, target_clsf, infos, wt=target_clsf_wt)

            sub_losses = [loss_ce1, loss_rhem, loss_metric, loss_ce2]
            wts_names = ['CE_LOSS_WT_1', 'RHEM_LOSS_WT', 'TRIPLET_LOSS_WT', 'CE_LOSS_WT_2']
        else:
            sub_losses = [loss_ce1, loss_rhem, loss_metric]
            wts_names = ['CE_LOSS_WT_1', 'RHEM_LOSS_WT', 'TRIPLET_LOSS_WT']

        loss = 0
        wts = [eval('config.TRAIN.' + name1) for name1 in wts_names]
        for wt1, loss1 in zip(wts, sub_losses):
            loss += wt1 * loss1

        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(model, default.clip_gradient)
        optimizer.step()

        # measure accuracy
        if config.SCORE_PROPAGATION:
            prob_np = prob2.detach().cpu().numpy()
        else:
            prob_np = prob1.detach().cpu().numpy()

        pred_labels = score2label(prob_np, config.TEST.SCORE_PARAM)
        targets_np = target_clsf.detach().cpu().numpy()
        target_unc = target_unc.numpy()
        acc = compute_all_acc_wt(targets_np > 0, pred_labels, prob_np, target_unc == 0)[config.TEST.CRITERION]

        accs.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % default.frequent == 0:
            crit = 'mean_pcF1' if config.TEST.CRITERION == 'mean_perclass_f1' else config.TEST.CRITERION
            msg = 'Epoch: [{0}][{1}/{2}] Time {batch_time.val:.1f} ' \
                  '({batch_time.avg:.1f}, {data_time.val:.1f})\t' \
                  .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time)
            msg += 'Loss {loss.val:.3f} ({loss.avg:.3f}){{'.format(loss=losses)
            for wt1, loss1 in zip(wts, sub_losses):
                msg += '%.3f*%.1f, ' % (loss1, wt1)
            msg += '}}\t{crit} {accs.val:.3f} ({accs.avg:.3f})'.format(
                crit=crit, accs=accs, ms=prob_np.max())
            logger.info(msg)


def validate(val_loader, model, use_val_th=False):
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets, infos) in enumerate(val_loader):
            if default.generate_features_all:
                logger.info('generating features, batch %d', i)
            filenames = [info[0] for info in infos]
            lesion_idxs = [info[1] for info in infos]
            inputs = [input.cuda() for input in inputs]
            unc_targets = targets[1]
            targets = targets[0]

            # compute output
            out = model(inputs)
            if config.SCORE_PROPAGATION:
                prob_np = out['class_prob2'].detach().cpu().numpy()
                scores_np = out['class_score2'].detach().cpu().numpy()
            else:
                prob_np = out['class_prob1'].detach().cpu().numpy()
                scores_np = out['class_score1'].detach().cpu().numpy()

            target1 = targets.numpy() > 0
            pred_wt = unc_targets.numpy() == 0
            if i == 0:
                target_all = target1
                prob_all = prob_np
                score_all = scores_np
                lesion_idx_all = lesion_idxs
                pred_wt_all = pred_wt
                if default.generate_features_all:
                    ft_all = out['emb']
            else:
                target_all = np.vstack((target_all, target1))
                prob_all = np.vstack((prob_all, prob_np))
                score_all = np.vstack((score_all, scores_np))
                pred_wt_all = np.vstack((pred_wt_all, pred_wt))
                lesion_idx_all.extend(lesion_idxs)
                if default.generate_features_all:
                    ft_all = np.vstack((ft_all, out['emb']))

        if default.generate_features_all:
            save_ft_to_file(ft_all)
            assert 0, 'all features have been generated and saved.'

        if config.TEST.USE_CALIBRATED_TH:
            accs, pred_label_all = compute_all_acc_wt_th(target_all, prob_all, pred_wt_all, use_val_th)
        else:
            pred_label_all = score2label(prob_all, config.TEST.SCORE_PARAM)
            accs = compute_all_acc_wt(target_all, pred_label_all, prob_all, pred_wt_all)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % default.frequent == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        '{crit} {accs:.3f}'
                        .format(
                   i, len(val_loader), batch_time=batch_time, crit=config.TEST.CRITERION,
                   accs=accs[config.TEST.CRITERION]
            ))

        print_accs(accs)
        accs['ex_neg'] = np.sum((target_all == 0) & pred_wt_all, axis=0)

        if use_val_th:  # only save for test set not val set
            save_acc_to_file(accs, val_loader, 'all_terms')
        if default.mode == 'infer' and use_val_th:
            save_test_scores_to_file(score_all, pred_label_all, target_all, accs, lesion_idx_all)

    return accs


def adjust_learning_rate(optimizer, epoch):
    idx = np.where(epoch >= np.array([0]+default.lr_epoch))[0][-1]
    lr_factor = default.lr_factor ** idx
    for param_group in optimizer.param_groups:
        if 'ori_lr' not in param_group.keys():  # first iteration
            param_group['ori_lr'] = param_group['lr']
        param_group['lr'] = param_group['ori_lr'] * lr_factor
    logger.info('learning rate factor %g' % lr_factor)
