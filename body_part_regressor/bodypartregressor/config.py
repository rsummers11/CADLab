# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Ke Yan
# --------------------------------------------------------

""" config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import time


__C = edict()
# Consumers can get config by:
cfg = __C

#
# Training options
#

__C.TRAIN = edict()


# num of volumes to use per minibatch
__C.TRAIN.GROUPS_PER_BATCH = 1

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

__C.TRAIN.USE_PREFETCH = False

# yk: if DO_VALIDATION = True, then do validation on VALIDATION_ITERATION iterations
__C.TRAIN.DO_VALIDATION = True
__C.TRAIN.VALIDATION_ITERATION = ''

# if file size is less than this kilobytes, it will not be used for training since very small size indicates the image
# contains very little contents
__C.TRAIN.MIN_IM_SIZE_KB = 20

# make val2-val1 = val3-val2 = ... = val[SLICE_NUM]-val[SLICE_NUM-1]
__C.TRAIN.SLICE_NUM = 3

__C.TRAIN.CROP_RANDOM_PATCH = True

# end

#
# Testing options
#

__C.TEST = edict()

__C.TEST.USE_PREFETCH = False

#
# Overall options
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.PIXEL_MEANS = np.array([50])  # for CT images

# For reproducibility
__C.RNG_SEED = 31

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
__C.INFO_DIR = ''

# Default GPU device id
__C.GPU_ID = 0

__C.SCALE = 128
__C.MAX_SIZE = 128
__C.IMG_IS_16bit = False
__C.IMG_SUFFIX = '.png'


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    # with open(filename, 'r') as f: # not valid gramma in Python 2.5
    f = open(filename, 'r')
    yaml_cfg = edict(yaml.load(f))
    f.close()

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
