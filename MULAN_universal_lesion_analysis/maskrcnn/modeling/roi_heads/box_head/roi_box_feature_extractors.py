# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn.modeling import registry
# from maskrcnn.modeling.backbone import resnet
from maskrcnn.modeling.poolers import Pooler
from maskrcnn.config import cfg


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    """
    Heads for Faster RCNN for classification
    """

    def __init__(self):
        super(MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.do_dropout = cfg.MODEL.ROI_BOX_HEAD.DROP_OUT
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        # self.fc8 = nn.Linear(representation_size, representation_size)
        self.dropout6 = nn.Dropout()
        self.dropout7 = nn.Dropout()
        for l in [self.fc6, self.fc7]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

            # in 3dce paper
            # nn.init.normal_(l.weight, mean=0, std=0.001)
            # nn.init.constant_(l.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x6 = F.relu(self.fc6(x))
        if self.do_dropout:
            x6 = self.dropout6(x6)
        x7 = F.relu(self.fc7(x6))
        if self.do_dropout:
            x7 = self.dropout7(x7)
        # x8 = F.relu(self.fc7(x7))

        return [x6, x7]


def make_roi_box_feature_extractor():
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func()
