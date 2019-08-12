# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Loss functions for the tag head"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time

from maskrcnn.config import cfg


class WeightedCeLoss(nn.Module):
    def __init__(self, pos_weight, neg_weight):
        super(WeightedCeLoss, self).__init__()
        self.pos_wt = torch.from_numpy(pos_weight).cuda()
        self.neg_wt = torch.from_numpy(neg_weight).cuda()

    def forward(self, prob, targets, wt=None, infos=None):
        # prob = F.sigmoid(scores)
        if wt is None:
            wt = torch.ones_like(prob)
        prob = prob.clamp(min=1e-7, max=1-1e-7)
        wt[targets == -1] = 0
        if wt.sum() == 0:
            return 0.
        targets = targets.to(torch.float)
        if cfg.MODEL.ROI_TAG_HEAD.CE_LOSS_POS_WT and self.pos_wt is not None:
            wt = wt * (targets.detach() * self.pos_wt + (1-targets.detach()) * self.neg_wt)

        loss = -torch.sum(wt * (torch.log(prob) * targets + torch.log(1-prob) * (1-targets))) \
               / torch.sum(wt > 0).to(torch.float)

        return loss


class WeightedCeLossBatchOhem(nn.Module):
    """relational hard example mining"""
    def __init__(self, pos_weight, neg_weight):
        super(WeightedCeLossBatchOhem, self).__init__()
        self.pos_wt = torch.from_numpy(pos_weight).cuda()
        self.neg_wt = torch.from_numpy(neg_weight).cuda()

    def forward(self, prob, targets, wt=None, infos=None):
        # prob = F.sigmoid(scores)
        if wt is None:
            wt = torch.ones_like(prob)
        prob = prob.clamp(min=1e-7, max=1-1e-7)
        wt[targets == -1] = 0
        targets = targets.to(torch.float)
        wt = wt.to(torch.float)
        if wt.sum() == 0:
            return torch.tensor(0.)

        with torch.no_grad():
            prob_diff_wt = torch.abs((prob - targets) * wt) ** cfg.MODEL.ROI_TAG_HEAD.OHEM_POWER
            idx = torch.multinomial(prob_diff_wt.view(-1), cfg.MODEL.ROI_TAG_HEAD.OHEM_SEL_NUM, replacement=True)
            # hist = np.histogram(idx.cpu().numpy(), np.arange(torch.numel(prob)+1))[0]
            # hist = np.reshape(hist, prob.shape)
            # pos = np.where(hist == np.max(hist))
            # row = pos[0][0]
            # col = pos[1][0]
            # print(np.max(hist), prob[row, col].item(), targets[row, col].item(), \
            #     default.term_list[col], int(self.pos_wt[col].item()), infos[row][0]#, prob_diff_wt.mean(0)[col].item()

        targets = targets.view(-1)[idx]
        prob = prob.view(-1)[idx]
        loss_per_smp = - (torch.log(prob) * targets + torch.log(1-prob) * (1-targets))
        loss = loss_per_smp.mean()

        return loss
