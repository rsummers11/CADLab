import _init_paths
import mxnet as mx
import numpy as np
import sys
import os
from scipy.io import savemat

from rcnn.logger import logger
from rcnn.config import config, default, cfg_from_file, merge_a_into_b
from rcnn.symbol import *
from rcnn.core.loader import TestLoader
from rcnn.core.tester import Predictor, pred_eval
from rcnn.utils.load_model import load_param
from rcnn.dataset import *
from rcnn.utils.load_data import filter_roidb


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)) # go to root dir of this project


def test_rcnn(network, dataset, image_set,
              dataset_path,
              ctx, prefix, epoch,
              vis, shuffle, has_rpn, proposal, max_box, thresh):
    # set config
    assert has_rpn, "only end-to-end case was checked in this project."
    config.TEST.HAS_RPN = True

    # load symbol and testing data
    sym = eval('get_' + network)(is_train=False, num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    imdb = eval(dataset)(image_set, dataset_path)
    roidb = imdb.gt_roidb()
    roidb = filter_roidb(roidb)
    imdb.num_images = len(roidb)

    # get test data iter
    test_data = TestLoader(roidb, batch_size=1, shuffle=shuffle, has_rpn=has_rpn, nThreads=default.prefetch_thread_num)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data)
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

    # check parameters
    for k in sym.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data]
    label_names = None  # [k[0] for k in test_data.provide_label]
    max_data_shape = [('data', (config.NUM_IMAGES_3DCE, config.NUM_SLICES, config.MAX_SIZE, config.MAX_SIZE))]
    if not has_rpn:
        max_data_shape.append(('rois', (1, config.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, #provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    acc = pred_eval(predictor, test_data, imdb, vis=vis, max_box=max_box, thresh=thresh)

    return acc


def test_net(prefix, iter_no):
    logger.info('Testing ...')
    default.testing = True
    # ctx = mx.gpu(int(default.val_gpu))
    ctx = mx.gpu(int(default.gpus.split(',')[0]))
    acc = test_rcnn(default.network, default.dataset, default.test_image_set,
                      default.dataset_path,
                      ctx, prefix, iter_no,
                      default.val_vis, default.val_shuffle,
                      default.val_has_rpn, default.proposal,
                      default.val_max_box, default.val_thresh)

    prop_file = 'proposals_%s_%s.mat' % (default.test_image_set, default.exp_name)
    savemat(prop_file, default.res_dict)
    default.testing = False

if __name__ == '__main__':
    config_file = cfg_from_file('config.yml')
    merge_a_into_b(config_file, config)
    config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)

    default_file = cfg_from_file('default.yml')
    merge_a_into_b(default_file, default)
    default.e2e_prefix = 'model/' + default.exp_name

    if default.gpus == '':  # auto select
        import GPUtil
        deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxLoad=0.5, maxMemory=0.5)
        GPUs = GPUtil.getGPUs()
        default.gpus = str(len(GPUs)-1-deviceIDs[0])
        logger.info('using gpu '+default.gpus)
    default.val_gpu = default.gpus
    default.prefetch_thread_num = min(default.prefetch_thread_num, config.TRAIN.SAMPLES_PER_BATCH)

    print config
    print default

    test_net(default.e2e_prefix, default.begin_epoch)
