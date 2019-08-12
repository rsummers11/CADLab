# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic run script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pprint
import random
import numpy as np
import logging
import time

import torch
import torch.backends.cudnn as cudnn

from maskrcnn.config import cfg, merge_a_into_b, cfg_from_file
from maskrcnn.utils.comm import synchronize, get_rank
from maskrcnn.utils.logger import setup_logger
from maskrcnn.config import cfg
from maskrcnn.engine.processor import train_model, test_model
from maskrcnn.engine.demo_process import exec_model
from maskrcnn.engine.batch_process import batch_exec_model


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
if len(os.path.dirname(__file__)) > 0:
    os.chdir(os.path.join(os.path.dirname(__file__)))  # go to root dir of this project


def main():
    config_file = 'config.yml'
    cfg_new = cfg_from_file(config_file)
    merge_a_into_b(cfg_new, cfg)

    log_dir = cfg.LOGFILE_DIR
    logger = setup_logger("maskrcnn", log_dir, cfg.EXP_NAME, get_rank())

    if cfg.MODE in ('demo', 'batch'):
        cfg_test = merge_test_config()
        logger.info(pprint.pformat(cfg_test))
    else:
        logger.info("Loaded configuration file {}".format(config_file))
        logger.info(pprint.pformat(cfg_new))
    check_configs()

    cfg.runtime_info.local_rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.runtime_info.distributed = num_gpus > 1
    if cfg.runtime_info.distributed:
        torch.cuda.set_device(cfg.runtime_info.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    logger.info("Using {} GPUs".format(num_gpus))

    if cfg.MODE in ('train',) and cfg.SEED is not None:
        random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        cudnn.deterministic = True
        np.random.seed(cfg.SEED)
        logger.info('Manual random seed %s', cfg.SEED)

    model, checkpointer = train_model()

    if cfg.MODE in ('train',):  # load best model and thresholds from previous training
        extra_checkpoint_data = checkpointer.load(cfg.runtime_info.best_model_path)

    if cfg.MODE in ('demo',):
        exec_model(model)
    if cfg.MODE in ('batch',):
        batch_exec_model(model)
    else:
        test_model(model, is_validation=True)
        test_model(model, is_validation=False)
    logger.info('Completed at %s: %s', time.strftime('%m-%d_%H-%M-%S'), cfg.EXP_NAME)


def check_configs():
    if cfg.MODE in ('train',):
        cfg.TEST.USE_SAVED_PRED_RES = 'none'
    elif cfg.MODE in ('vis',):
        cfg.TEST.EVAL_SEG_TAG_ON_GT = False
        cfg.LOG_IN_FILE = False
    elif cfg.MODE in ('demo', 'batch'):
        cfg.TEST.USE_SAVED_PRED_RES = 'none'
        cfg.TEST.EVAL_SEG_TAG_ON_GT = False

    scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
    assert scales == cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
    if cfg.MODEL.BACKBONE.FEATURE_UPSAMPLE:
        assert len(scales) == 1 and scales[0] == 1. / 2**(cfg.MODEL.BACKBONE.FEATURE_UPSAMPLE_LEVEL-1)
    anchor = cfg.MODEL.RPN.ANCHOR_STRIDE
    assert len(anchor) == 1 and anchor[0] == 1. / scales[0]

    if not cfg.MODEL.USE_3D_FUSION:
        assert cfg.INPUT.NUM_IMAGES_3DCE == 1
        assert cfg.MODEL.BACKBONE.FEATURE_FUSION_LEVELS == [False] * 3

    if cfg.GPU == '':
        import GPUtil
        deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.2)
        if len(deviceIDs) == 0:
            deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.9, maxLoad=1)
        cfg.GPU = str(deviceIDs[0])


def merge_test_config():
    config_file = 'demo_batch_config.yml'
    cfg_new = cfg_from_file(config_file)
    # cfg.GPU = cfg_new.GPU
    cfg.TEST.TEST_SLICE_INTV_MM = cfg_new.TEST_SLICE_INTV_MM
    cfg.TEST.VISUALIZE.SCORE_THRESH = cfg_new.DETECTION_SCORE_THRESH
    cfg.TEST.VISUALIZE.DETECTIONS_PER_IMG = cfg_new.MAX_DETECTIONS_PER_IMG
    cfg.TEST.MIN_LYMPH_NODE_DIAM = cfg_new.MIN_LYMPH_NODE_DIAM_TO_SHOW
    cfg.TEST.MASK.THRESHOLD = cfg_new.MASK_THRESHOLD
    cfg.TEST.VISUALIZE.NMS = cfg_new.BBOX_NMS_OVERLAP
    cfg.INPUT.IMG_DO_CLIP = cfg_new.IMG_DO_CLIP
    cfg.TEST.TAGS_TO_KEEP = cfg_new.TAGS_TO_KEEP
    cfg.TEST.RESULT_FIELDS = cfg_new.RESULT_FIELDS

    return cfg_new


if __name__ == "__main__":
    main()
