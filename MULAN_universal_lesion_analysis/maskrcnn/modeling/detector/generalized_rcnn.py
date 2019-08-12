# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""
Implements the Generalized R-CNN framework
Ke Yan: added 3DCE
"""

import torch
from torch import nn

from maskrcnn.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from maskrcnn.config import cfg
from ..backbone.feature_fusion_3dce import FeatureFusion3dce


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone()
        self.rpn = build_rpn()
        self.roi_heads = build_roi_heads()
        if cfg.MODEL.USE_3D_FUSION:
            # the original 3DCE strategy, fuse last level feature maps
            self.feature_fuse = FeatureFusion3dce()

    def forward(self, images, targets=None, infos=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        if cfg.MODEL.USE_3D_FUSION:
            # the original 3DCE strategy, fuse last level feature maps
            features, images = self.feature_fuse(features, images)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, infos)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
