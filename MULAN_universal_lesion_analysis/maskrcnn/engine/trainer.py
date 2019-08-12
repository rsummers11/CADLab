# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Training codes, organized in epochs"""
import datetime
import logging
import time
import os

import torch
import torch.distributed as dist

from maskrcnn.utils.comm import get_world_size
from maskrcnn.utils.metric_logger import MetricLogger
from maskrcnn.config import cfg
from maskrcnn.utils.miscellaneous import clip_gradient
from maskrcnn.utils.print_info import get_debug_info
from maskrcnn.solver.build import adjust_learning_rate


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    checkpointer,
    device,
    eval_fun,
    arguments,
):
    logger = logging.getLogger("maskrcnn.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_epoch = arguments["start_epoch"]
    max_epoch = arguments["max_epoch"]

    best_acc = -1.
    accs = {}
    best_model_path = ''
    old_model_path = ''

    if cfg.EVAL_AT_BEGIN:
        acc = eval_fun(model, is_validation=True)
        accs[start_epoch] = acc
        best_acc = acc
        logger.info('epoch %d: %.4f' % (start_epoch, acc))

    model.train()
    start_training_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        adjust_learning_rate(optimizer, epoch)
        train_one_epoch(data_loader, device, model, meters, optimizer, logger, epoch)
        filename = checkpointer.get_save_name(epoch+1, prefix='')
        checkpointer.save(filename, **arguments)

        acc = eval_fun(model, is_validation=True)
        model.train()
        accs[epoch+1] = acc
        for key in sorted(accs.keys()):
            logger.info('%d: %.4f' % (key, accs[key]))

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if cfg.KEEP_BEST_MODEL:
            if is_best and len(best_model_path) > 0 and os.path.exists(best_model_path):
                os.remove(best_model_path)
            if len(old_model_path) > 0 and os.path.exists(old_model_path):
                os.remove(old_model_path)
            if not is_best:
                old_model_path = filename
        if is_best:
            cfg.runtime_info.best_model_path = filename
            best_model_path = filename

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch-start_epoch)
        )
    )


def train_one_epoch(data_loader, device, model, meters, optimizer, logger, epoch):
    end = time.time()
    max_iter_per_epoch = len(data_loader)
    for iteration, (images, targets, infos) in enumerate(data_loader):
        data_time = time.time() - end

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets, infos)

        keys = ['loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg']
        wts = [cfg.MODEL.RPN.CLSF_LOSS_WEIGHT, cfg.MODEL.RPN.REG_LOSS_WEIGHT,
               cfg.MODEL.ROI_BOX_HEAD.CLSF_LOSS_WEIGHT, cfg.MODEL.ROI_BOX_HEAD.REG_LOSS_WEIGHT,
               ]
        if cfg.MODEL.MASK_ON:
            keys += ['loss_mask']
            wts += [cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_WEIGHT]
        if cfg.MODEL.TAG_ON:
            keys += ['loss_tag', 'loss_tag_ohem']
            wts += [cfg.MODEL.ROI_TAG_HEAD.TAG_LOSS_WEIGHT, cfg.MODEL.ROI_TAG_HEAD.OHEM_LOSS_WEIGHT]
        if cfg.MODEL.REFINE_ON:
            keys += ['loss_clsf2', 'loss_tag2']
            wts += [1, 1]

        losses = 0
        for i, key in enumerate(keys):
            losses += loss_dict[key] * wts[i]

        # reduce losses over all GPUs for logging purposes (yk: ?)
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # meters.update(loss=losses_reduced, **loss_dict_reduced)
        meters.update(loss=losses, **loss_dict)

        optimizer.zero_grad()
        losses.backward()
        clip_gradient(model, cfg.SOLVER.CLIP_GRADIENT)

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time / cfg.SOLVER.IMS_PER_BATCH, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter_per_epoch - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % cfg.SOLVER.SHOW_LOSS_ITER == 0 or iteration == max_iter_per_epoch:
            logger.info(
                meters.delimiter.join(
                    [
                        # "eta: {eta}",
                        "epoch {epoch} iter {iter}/{total_iter}",
                        "{meters}",
                        "lr: {lr:g}",
                        # "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    iter=iteration,
                    total_iter=len(data_loader),
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            # logger.info(get_debug_info())
