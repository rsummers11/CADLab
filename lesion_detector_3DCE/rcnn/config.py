import numpy as np
from easydict import EasyDict as edict
import yaml


config = edict()

# network related params
config.PIXEL_MEANS = np.array([50])
config.IMAGE_STRIDE = 0
config.RPN_FEAT_STRIDE = 8
config.RCNN_FEAT_STRIDE = 8
config.FIXED_PARAMS = []
config.FIXED_PARAMS_SHARED = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

# dataset related params
config.NUM_CLASSES = 2
# config.SCALES = [(512, 512)]  # first is scale (the shorter side); second is max size. replaced by SCALE and MAX_SIZE by Ke
config.ANCHOR_SCALES = (8, 16, 32)
config.ANCHOR_RATIOS = (0.5, 1, 2)
config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)

config.TRAIN = edict()

# R-CNN and RPN
config.TRAIN.SAMPLES_PER_BATCH = 1
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = True
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = False

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

# ===============Ke added====================
config.GT_MARGIN = 0  # add a margin around the ground-truth box. generally not used
config.MAX_SIZE = 512
config.SCALE = 512
config.NORM_SPACING = -1
config.SLICE_INTV = 2
config.WINDOWING = [-1024, 3071]
config.TRAIN.RCNN_POS_UPSAMPLE = False

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.SAMPLES_PER_BATCH = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

# default settings
default = edict()

# default network
default.network = 'vgg'
default.pretrained = '/home/yk/ct/data/imagenet_models/MXNet/vgg16'
default.pretrained_epoch = 0
default.base_lr = 0.001

default.dataset = 'DeepLesion'
default.image_set = 'train'
# default.root_path = '/home/yk/ct/data/'
default.dataset_path = ''

# default training
default.frequent = 20
default.kvstore = 'device'
# default e2e
default.e2e_prefix = 'model/e2e'
default.e2e_epoch = 10
default.e2e_lr = default.base_lr
default.e2e_lr_step = '7'
# # default rpn
# default.rpn_prefix = 'model/rpn'
# default.rpn_epoch = 8
# default.rpn_lr = default.base_lr
# default.rpn_lr_step = '6'
# # default rcnn
# default.rcnn_prefix = 'model/rcnn'
# default.rcnn_epoch = 8
# default.rcnn_lr = default.base_lr
# default.rcnn_lr_step = '6'

# ===============Ke added====================
default.gpus = '0'
default.val_gpu = default.gpus
default.val_image_set = 'val'
default.val_vis = False
default.val_shuffle = False
default.val_has_rpn = True
default.proposal = 'rpn'
default.val_max_box = 5
default.val_iou_th = .5
default.val_thresh = 0
default.weight_decay = .0005
default.groundtruth_file = 'DL_info.csv'
default.image_path = ''
default.validate_at_begin = True
default.testing = False

default.flip = False
default.shuffle = True
default.work_load_list = None
default.resume = False  # resume from previous epoch
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
