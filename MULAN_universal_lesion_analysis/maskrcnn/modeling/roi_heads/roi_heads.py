# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .tag_head.tag_head import build_roi_tag_head
from .refine_head.refine_head import build_roi_refine_head
from maskrcnn.config import cfg


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    Ke Yan: added tag head and score refinement layer
    """

    def __init__(self, heads):
        super(CombinedROIHeads, self).__init__(heads)
        # self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, infos=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        class_logits, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if cfg.MODEL.MASK_ON:
            mask_features = features
            mask_logits, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if cfg.MODEL.TAG_ON:
            tag_features = features
            tag_logits, detections, loss_tag = self.tag(tag_features, detections, targets)
            losses.update(loss_tag)

        if cfg.MODEL.REFINE_ON:
            if not self.training:
                class_logits = torch.cat([d.get_field('class_logits') for d in detections])
            refine_features = torch.cat((class_logits, tag_logits), dim=1)
            refine_logits, detections, loss_refine = self.refine(refine_features, detections, targets, infos)
            losses.update(loss_refine)

        return class_logits, detections, losses


def build_roi_heads():
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head()))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head()))
    if cfg.MODEL.TAG_ON:
        roi_heads.append(("tag", build_roi_tag_head()))
    if cfg.MODEL.REFINE_ON:
        roi_heads.append(("refine", build_roi_refine_head()))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(roi_heads)

    return roi_heads
