# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from maskrcnn.config import cfg
import numpy as np


class FPNPredictor(nn.Module):
    def __init__(self):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)
        nn.init.constant_(self.cls_score.bias, -np.log(99/1))

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


_ROI_BOX_PREDICTOR = {
    "FPNPredictor": FPNPredictor,
}


def make_roi_box_predictor():
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func()
