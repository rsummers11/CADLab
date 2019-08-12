# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
from collections import defaultdict
from collections import deque, OrderedDict

import torch
from maskrcnn.config import cfg


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=cfg.SOLVER.COMPUTE_MEDIAN_LOSS):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        self_keys = self.meters.keys()
        abbreviations = [
            ('loss', 'loss'),
            ('loss_objectness', 'rpn_clsf'),
            ('loss_rpn_box_reg', 'rpn_reg'),
            ('loss_classifier', 'frcn_clsf'),
            ('loss_box_reg', 'frcn_reg'),
            ('loss_mask', 'mask'),
            ('loss_tag', 'tag'),
            ('loss_clsf2', 'clsf2'),
            ('loss_tag2', 'tag2'),
            ('loss_tag_ohem', 'ohem'),
            ('time', 'time'),
            # ('data', 'data'),
            ]

        abbrv_dicts = OrderedDict(abbreviations)
        for key in abbrv_dicts.keys():
            if key in self_keys:
                meter = self.meters[key]
                loss_str.append(
                    # "{}: {:.4f} ({:.4f})".format(to_show[key], meter.median, meter.global_avg)
                    "{}: {:.3f}".format(abbrv_dicts[key], meter.median)
                )
                # if len(loss_str)
        # for key in self_keys:
        #     if key not in abbrv_dicts.keys():
        #         meter = self.meters[key]
        #         loss_str.append(
        #             # "{}: {:.4f} ({:.4f})".format(to_show[key], meter.median, meter.global_avg)
        #             "{}: {:.4f}".format(key, meter.median)
        #         )
        return self.delimiter.join(loss_str)
