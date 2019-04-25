# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes of the network structure of LesaNet.
# --------------------------------------------------------

import torch.nn as nn
import torch
from roi_pooling.modules.roi_pool import _RoIPooling
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter

from utils import logger
from config import config


class VGG16bn(nn.Module):
    """VGG-16 multi-scale feature with batch normalization"""

    def __init__(self, num_cls, pretrained_weights=True):
        super(VGG16bn, self).__init__()
        self.num_cls = num_cls
        self._make_conv_layers(batch_norm=True)
        self._make_linear_layers(num_cls, roipool=5, fc=256, emb=config.EMBEDDING_DIM, norm=True)
        if pretrained_weights:
            self._load_pretrained_weights()
        self._initialize_weights()

    def _load_pretrained_weights(self):
        pretr = models.vgg16_bn(pretrained=True)
        pretrained_dict = pretr.state_dict().items()
        pretrained_dict = {k: v for k, v in pretrained_dict}
        model_dict = self.state_dict()
        pretr_keys = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        load_dict = {}
        for i in range(len(pretr_keys)):
            load_dict[self.conv_layer_names[i]+'.weight'] = pretrained_dict['features.%d.weight' % pretr_keys[i]]
            load_dict[self.conv_layer_names[i]+'.bias'] = pretrained_dict['features.%d.bias' % pretr_keys[i]]
            load_dict[self.bn_layer_names[i]+'.weight'] = pretrained_dict['features.%d.weight' % (pretr_keys[i]+1)]
            load_dict[self.bn_layer_names[i]+'.bias'] = pretrained_dict['features.%d.bias' % (pretr_keys[i]+1)]
            load_dict[self.bn_layer_names[i]+'.running_mean'] = pretrained_dict['features.%d.running_mean' % (pretr_keys[i]+1)]
            load_dict[self.bn_layer_names[i]+'.running_var'] = pretrained_dict['features.%d.running_var' % (pretr_keys[i]+1)]

        model_dict.update(load_dict)

    def _make_conv_layers(self, batch_norm=False):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        block_idx = 1
        layer_idx = 1
        in_channels = config.NUM_SLICES
        self.conv_layer_names = []
        self.bn_layer_names = []
        for v in cfg:
            if v == 'M':
                layer = nn.MaxPool2d(kernel_size=2, stride=2)
                name = 'pool%d' % block_idx
                exec('self.%s = layer' % name)
                block_idx += 1
                layer_idx = 1
            else:
                layer = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                name = 'conv%d_%d' % (block_idx, layer_idx)
                self.conv_layer_names.append(name)
                exec('self.%s = layer' % name)
                if batch_norm:
                    layer = nn.BatchNorm2d(v)
                    name = 'bn%d_%d' % (block_idx, layer_idx)
                    exec('self.%s = layer' % name)
                    self.bn_layer_names.append(name)
                layer = nn.ReLU(inplace=True)
                name = 'relu%d_%d' % (block_idx, layer_idx)
                exec('self.%s = layer' % name)

                in_channels = v
                layer_idx += 1

    def _make_linear_layers(self, num_cls, roipool=5, fc=512, emb=1024, norm=True):
        lin_in = roipool * roipool
        self.roipool1 = _RoIPooling(roipool, roipool, 1)
        self.fc_pool1 = nn.Linear(lin_in * 64, fc)

        self.roipool2 = _RoIPooling(roipool, roipool, .5)
        self.fc_pool2 = nn.Linear(lin_in * 128, fc)

        self.roipool3 = _RoIPooling(roipool, roipool, .25)
        self.fc_pool3 = nn.Linear(lin_in * 256, fc)

        self.roipool4 = _RoIPooling(roipool, roipool, .125)
        self.fc_pool4 = nn.Linear(lin_in * 512, fc)

        self.roipool5 = _RoIPooling(roipool, roipool, .0625)
        self.fc_pool5 = nn.Linear(lin_in * 512, fc)

        self.fc_emb = nn.Linear(fc*5, emb)
        self.class_scores1 = nn.Linear(fc * 5, num_cls)
        self.class_scores2 = nn.Linear(num_cls, num_cls)

    def forward(self, inputs):
        x, out_box, center_box = inputs
        bs = x.size(0)
        idxs = torch.arange(bs).unsqueeze(1).to(torch.float).cuda()
        out_box = torch.cat((idxs, out_box), 1)
        center_box = torch.cat((idxs, center_box), 1)

        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        pool1 = self.pool1(x)
        roipool1 = self.roipool1(x, center_box)
        roipool1 = roipool1.view(bs, -1)
        fc_pool1 = self.fc_pool1(roipool1)

        x = self.relu2_1(self.bn2_1(self.conv2_1(pool1)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        pool2 = self.pool2(x)
        roipool2 = self.roipool2(x, center_box)
        roipool2 = roipool2.view(bs, -1)
        fc_pool2 = self.fc_pool2(roipool2)

        x = self.relu3_1(self.bn3_1(self.conv3_1(pool2)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        pool3 = self.pool3(x)
        roipool3 = self.roipool3(x, center_box)
        roipool3 = roipool3.view(bs, -1)
        fc_pool3 = self.fc_pool3(roipool3)

        x = self.relu4_1(self.bn4_1(self.conv4_1(pool3)))
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        pool4 = self.pool4(x)
        roipool4 = self.roipool4(x, out_box)
        roipool4 = roipool4.view(bs, -1)
        fc_pool4 = self.fc_pool4(roipool4)

        x = self.relu5_1(self.bn5_1(self.conv5_1(pool4)))
        x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.relu5_3(self.bn5_3(self.conv5_3(x)))
        roipool5 = self.roipool5(x, out_box)
        roipool5 = roipool5.view(bs, -1)
        fc_pool5 = self.fc_pool5(roipool5)

        fc_cat = torch.cat((fc_pool1, fc_pool2, fc_pool3, fc_pool4, fc_pool5), dim=1)
        emb = self.fc_emb(fc_cat)
        emb = F.normalize(emb, p=2, dim=1)

        out = {}
        out['emb'] = emb

        class_score1 = self.class_scores1(fc_cat)
        class_prob1 = torch.sigmoid(class_score1)
        out['class_score1'] = class_score1
        out['class_prob1'] = class_prob1
        if config.SCORE_PROPAGATION:
            class_score2 = self.class_scores2(class_score1)
            class_prob2 = torch.sigmoid(class_score2)
            out['class_score2'] = class_score2
            out['class_prob2'] = class_prob2

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.shape == (self.num_cls, self.num_cls):
                m.weight.data[:] = torch.eye(self.num_cls)
                m.bias.data.zero_()
