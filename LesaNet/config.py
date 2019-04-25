# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains some default configuration values,
# which will be overwritten by values in config.yml and default.yml
# --------------------------------------------------------

import numpy as np
from easydict import EasyDict as edict
import yaml


config = edict()

# algorithm related params
config.PIXEL_MEANS = np.array([50])

config.MAX_IM_SIZE = 512
config.SCALE = 512
config.NORM_SPACING = -1
config.SLICE_INTV = 2
config.WINDOWING = [-1024, 3071]
config.IMG_DO_CLIP = False  # clip the black borders of ct images

config.TRAIN = edict()
config.SAMPLES_PER_BATCH = 256
config.TRAIN.USE_PRETRAINED_MODEL = True

config.TEST = edict()

# default settings
default = edict()

# default network
default.network = 'vgg'
default.base_lr = 0.001

default.dataset = 'DeepLesion'
default.image_set = 'train'

# default training
default.frequent = 20
default.model_path = 'checkpoints/'
default.res_path = 'results/'
default.epoch = 10
default.lr = default.base_lr
default.lr_step = '7'
default.prefetch_thread_num = 4  # 0: no prefetch

default.world_size = 1  # number of distributed processes
default.dist_url = 'tcp://224.66.41.62:23456'  # url used to set up distributed training
default.dist_backend = 'gloo'  # distributed backend
default.seed = None  # seed for initializing training

default.gpus = '0'
default.val_gpu = default.gpus
default.val_image_set = 'val'
default.val_vis = False
default.val_shuffle = False
default.val_max_box = 5
default.val_thresh = 0
default.weight_decay = .0005
default.groundtruth_file = 'DL_info.csv'
default.image_path = ''
default.validate_at_begin = True
default.testing = False

default.flip = False
default.shuffle = True
default.begin_epoch = 0
default.show_avg_loss = 100  # 1: show exact loss of each batch. >1: smooth the shown loss


def merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # recursively merge dicts
        if type(v) is edict:
            merge_a_into_b(a[k], b[k])
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:  # not valid grammar in Python 2.5
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg
