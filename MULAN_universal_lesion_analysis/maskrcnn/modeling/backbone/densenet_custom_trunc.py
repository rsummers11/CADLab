# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""The truncated Densenet-121 with FPN and 3DCE"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from torchvision.models.densenet import _DenseBlock, _Transition, model_urls
import math
import torch.utils.model_zoo as model_zoo
import re
from collections import OrderedDict

from maskrcnn.config import cfg


class DenseNetCustomTrunc(nn.Module):
    """The truncated Densenet-121 with FPN and 3DCE"""
    # truncated since transition layer 3 since we find it works better in DeepLesion
    # We only keep the finest-level feature map after FPN
    def __init__(self):
        super(DenseNetCustomTrunc, self).__init__()
        name = cfg.MODEL.BACKBONE.CONV_BODY
        self.depth = int(name.split('-')[1])
        self.feature_upsample = cfg.MODEL.BACKBONE.FEATURE_UPSAMPLE

        assert self.depth in [121]
        if self.depth == 121:
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 24)
            self.in_dim = [64, 256, 512, 1024]
        bn_size = 4
        drop_rate = 0

        # First convolution
        self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        cfg.runtime_info.backbone_ft_dim = self.in_dim[-1]

        # Final batch norm
        # self.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.feature_upsample:
            self.out_dim = cfg.MODEL.BACKBONE.OUT_CHANNELS
            self.fpn_finest_layer = cfg.MODEL.BACKBONE.FEATURE_UPSAMPLE_LEVEL-1
            for p in range(4, self.fpn_finest_layer - 1, -1):
                layer = nn.Conv2d(self.in_dim[p - 1], self.out_dim, 1)
                name = 'lateral%d' % p
                self.add_module(name, layer)

                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)
            cfg.runtime_info.backbone_ft_dim = self.out_dim

        self.indim_ilf = [64, 128, 256]
        self.num_image = cfg.INPUT.NUM_IMAGES_3DCE
        self.feature_fusion_level_list = cfg.MODEL.BACKBONE.FEATURE_FUSION_LEVELS
        for p in range(len(self.feature_fusion_level_list)):
            if self.feature_fusion_level_list[p]:
                layer = nn.Conv2d(self.num_image * self.indim_ilf[p], self.indim_ilf[p], 1)
                self.add_module('conv_ilf%d'%p, layer)

    def inter_layer_fuse(self, ft, level):
        """Improved 3DCE, 3D feature fusion"""
        if self.feature_fusion_level_list[level]:
            ft_ilf = ft.reshape(-1, self.num_image * self.indim_ilf[level], ft.shape[2], ft.shape[3])
            ft_ilf = getattr(self, 'conv_ilf%d'%level)(ft_ilf)
            ft_out = ft.clone()
            ft_out[int(self.num_image/2)::self.num_image] = ft_ilf
            return ft_out
        else:
            return ft

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        relu0 = self.relu0(x)
        pool0 = self.pool0(relu0)
        pool0_ilf = self.inter_layer_fuse(pool0, 0)

        db1 = self.denseblock1(pool0_ilf)
        ts1 = self.transition1(db1)
        ts1_ilf = self.inter_layer_fuse(ts1, 1)

        db2 = self.denseblock2(ts1_ilf)
        ts2 = self.transition2(db2)
        ts2_ilf = self.inter_layer_fuse(ts2, 2)

        db3 = self.denseblock3(ts2_ilf)

        # truncated since here since we find it works better in DeepLesion
        # ts3 = self.transition3(db3)
        # db4 = self.denseblock4(ts3)

        if self.feature_upsample:
            ftmaps = [relu0, db1, db2, db3]
            x = self.lateral4(ftmaps[-1])
            for p in range(3, self.fpn_finest_layer - 1, -1):
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                y = ftmaps[p-1]
                lateral = getattr(self, 'lateral%d' % p)(y)
                x += lateral
            return [x]
        else:
            return [db3]

    def load_pretrained_weights(self):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        state_dict1 = {}
        for key in list(state_dict.keys()):
            new_key = key.replace('features.', '')
            state_dict1[new_key] = state_dict[key]

        self.load_state_dict(state_dict1, strict=False)

    def freeze(self):
        for name, param in self.named_parameters():
            print('freezing', name)
            param.requires_grad = False