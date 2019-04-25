# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes of the losses of LesaNet.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time

from config import config, default
from utils import unique


class WeightedCeLoss(nn.Module):
    def __init__(self, pos_weight, neg_weight):
        super(WeightedCeLoss, self).__init__()
        self.pos_wt = torch.from_numpy(pos_weight).cuda()
        self.neg_wt = torch.from_numpy(neg_weight).cuda()

    def forward(self, prob, targets, infos, wt=None):
        prob = prob.clamp(min=1e-7, max=1-1e-7)
        if wt is None:
            wt1 = torch.ones_like(prob)
        if config.TRAIN.CE_LOSS_WEIGHTED and self.pos_wt is not None:
            wt1 = wt * (targets.detach() * self.pos_wt + (1-targets.detach()) * self.neg_wt)

        loss = -torch.mean(wt1 * (torch.log(prob) * targets + torch.log(1-prob) * (1-targets)))

        return loss


class CeLossRhem(nn.Module):
    def __init__(self):
        super(CeLossRhem, self).__init__()

    def forward(self, prob, targets, infos, wt=None):
        if wt is None:
            wt = torch.ones_like(prob)
        prob = prob.clamp(min=1e-7, max=1-1e-7)
        with torch.no_grad():
            prob_diff_wt = torch.abs((prob - targets) * wt) ** config.TRAIN.RHEM_POWER
            idx = torch.multinomial(prob_diff_wt.view(-1), config.TRAIN.RHEM_BATCH_SIZE, replacement=True)
            # hist = np.histogram(idx.cpu().numpy(), np.arange(torch.numel(prob)+1))[0]
            # hist = np.reshape(hist, prob.shape)
            # pos = np.where(hist == np.max(hist))
            # row = pos[0][0]
            # col = pos[1][0]
            # print np.max(hist), prob[row, col].item(), targets[row, col].item(), \
            #     default.term_list[col], int(self.pos_wt[col].item()), infos[row][0]#, prob_diff_wt.mean(0)[col].item()

        targets = targets.view(-1)[idx]
        prob = prob.view(-1)[idx]
        loss_per_smp = - (torch.log(prob) * targets + torch.log(1-prob) * (1-targets))
        loss = loss_per_smp.mean()

        return loss
