# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Only keep the truncated Densenet-121 with FPN and 3DCE"""
from collections import OrderedDict

from torch import nn

from maskrcnn.modeling import registry

from . import fpn as fpn_module
from . import densenet_custom_trunc
from maskrcnn.config import cfg


@registry.BACKBONES.register("DenseTrunc-121")
def build_densenet_trunc_backbone():
    model = densenet_custom_trunc.DenseNetCustomTrunc()
    return model


def build_backbone():
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY]()
