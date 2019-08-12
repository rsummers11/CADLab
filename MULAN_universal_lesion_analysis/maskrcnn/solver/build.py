# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

from maskrcnn.config import cfg


def make_optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def adjust_learning_rate(optimizer, epoch):
    """The 'step' strategy in SGD"""
    steps = cfg.SOLVER.STEPS
    lr_factor = cfg.SOLVER.GAMMA
    idx = np.where(epoch >= np.array([0]+steps))[0][-1]
    lr_factor = lr_factor ** idx
    for param_group in optimizer.param_groups:
        if 'ori_lr' not in param_group.keys():  # first iteration
            param_group['ori_lr'] = param_group['lr']
        param_group['lr'] = param_group['ori_lr'] * lr_factor
    # logger.info('learning rate factor %g' % lr_factor)
