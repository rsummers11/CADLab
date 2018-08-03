import numpy as np
from scipy.io import loadmat

from ..logger import logger
from ..config import config, default
from ..dataset import *


def load_gt_roidb(dataset_name, image_set_name, dataset_path,
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, dataset_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb):
    """ remove noisy bboxes, then remove roidb entries without usable rois """
    filtered_roidb = []
    num_boxes = 0
    num_boxes_new = 0
    for r in roidb:
        num_boxes += r['boxes'].shape[0]
        r['boxes'] = r['boxes'][r['noisy']==0]
        num_boxes_new += r['boxes'].shape[0]
        if r['boxes'].shape[0] > 0:
            filtered_roidb.append(r)
    logger.info('noisy boxes filtered, images: %d -> %d, bboxes: %d -> %d' %
                (len(roidb), len(filtered_roidb), num_boxes, num_boxes_new))

    return filtered_roidb
