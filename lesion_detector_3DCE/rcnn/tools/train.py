# main file to train the detector. This file may be called by train.sh and generate log files.
# Or, it can be debugged by itself (e.g. in PyCharm), since all parameters are set in config.yml and default.yml,
# and no log file will be generated
import _init_paths

import pprint
import mxnet as mx
import numpy as np
import os
import random

from rcnn.logger import logger
from rcnn.config import config, default, cfg_from_file, merge_a_into_b
from rcnn.symbol import *
from rcnn.core import callback, metric
from rcnn.core.loader import AnchorLoader
from rcnn.core.module import MutableModule
from rcnn.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from rcnn.utils.load_model import load_param

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)) # go to root dir of this project


def get_optimizer(args, arg_names, num_iter_per_epoch, iter_size):
    # decide learning rate
    base_lr = args.e2e_lr
    lr_factor = args.lr_factor
    lr_epoch = [float(ep) for ep in args.e2e_lr_step.split(',')]
    lr_epoch_diff = [ep - args.begin_epoch for ep in lr_epoch if ep > args.begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(ep * num_iter_per_epoch / iter_size) for ep in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)

    # optimizer
    lr_dict = dict()
    wd_dict = dict()
    param_idx2name = {}
    for i, arg_name in enumerate(arg_names):
        param_idx2name[i] = arg_name
        lr_dict[arg_name] = 1.
        if arg_name.endswith('_bias'):  # for biases, set the learning rate to 2, weight_decay to 0
            wd_dict[arg_name] = 0
            lr_dict[arg_name] = 2

    optimizer_params = {'momentum': 0.9,
                        'wd': args.weight_decay,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        # 'rescale_grad': (1.0 / batch_size),  # rescale_grad is done in loss functions
                        'param_idx2name': param_idx2name,
                        'clip_gradient': 5}

    opt = mx.optimizer.SGD(**optimizer_params)
    opt.set_wd_mult(wd_dict)
    opt.set_lr_mult(lr_dict)

    return opt


def init_params(args, sym, train_data):
    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    logger.info('output shape %s' % pprint.pformat(out_shape_dict))

    if args.resume:  # load params from previously trained model
        arg_params, aux_params = load_param(args.e2e_prefix, args.begin_epoch, convert=True)
    else:  # initialize weights from pretrained model and random numbers
        arg_params, aux_params = load_param(args.pretrained, args.pretrained_epoch, convert=True)

        # deal with multiple input CT slices, see 3DCE paper.
        # if NUM_SLICES = 3, pretrained weights won't be changed
        # if NUM_SLICES > 3, extra input channels in conv1_1 will be initialized to 0
        nCh = config.NUM_SLICES
        w1 = arg_params['conv1_1_weight'].asnumpy()
        w1_new = np.zeros((64, nCh, 3, 3), dtype=float)
        w1_new[:, (nCh - 3) / 2:(nCh - 3) / 2 + 3, :, :] = w1

        arg_params['conv1_1_new_weight'] = mx.nd.array(w1_new)
        arg_params['conv1_1_new_bias'] = arg_params['conv1_1_bias']
        del arg_params['conv1_1_weight']

        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])

        if config.FRAMEWORK == '3DCE':
            arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv_new_1_weight'])
            arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv_new_1_bias'])
            arg_params['fc6_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['fc6_weight'])
            arg_params['fc6_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc6_bias'])

            arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
            arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
            arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
            arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

        elif config.FRAMEWORK == 'RFCN':
            arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv_new_1_weight'])
            arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv_new_1_bias'])
            arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rfcn_cls_weight'])
            arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=arg_shape_dict['rfcn_cls_bias'])
            arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rfcn_bbox_weight'])
            arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=arg_shape_dict['rfcn_bbox_bias'])

        elif config.FRAMEWORK == 'Faster':
            arg_params['fc6_small_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['fc6_small_weight'])
            arg_params['fc6_small_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc6_small_bias'])
            arg_params['fc7_small_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['fc7_small_weight'])
            arg_params['fc7_small_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc7_small_bias'])

            arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
            arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
            arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
            arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    logger.info('load param done')
    return arg_params, aux_params


def train_net(args):

    if args.rand_seed > 0:
        np.random.seed(args.rand_seed)
        mx.random.seed(args.rand_seed)
        random.seed(args.rand_seed)

    # print config
    logger.info(pprint.pformat(config))
    logger.info(pprint.pformat(args))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in args.image_set.split('+')]
    roidbs = [load_gt_roidb(args.dataset, image_set, args.dataset_path,
                            flip=args.flip)
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb)
    logger.info('%d training slices' % (len(roidb)))

    # load symbol
    sym = eval('get_' + args.network)(is_train=True, num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.SAMPLES_PER_BATCH * batch_size

    # load training data
    train_data = AnchorLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=args.shuffle,
                              ctx=ctx, work_load_list=args.work_load_list,
                              feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
                              anchor_ratios=config.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
                              nThreads=default.prefetch_thread_num)

    # infer max shape
    max_data_shape = [('data', (input_batch_size*config.NUM_IMAGES_3DCE, config.NUM_SLICES, config.MAX_SIZE, config.MAX_SIZE))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (input_batch_size*config.NUM_IMAGES_3DCE, 5, 5)))
    logger.info('providing maximum shape %s %s' % (max_data_shape, max_label_shape))

    # load and initialize and check params
    arg_params, aux_params = init_params(args, sym, train_data)

    # create solver
    fixed_param_prefix = config.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=args.work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    # rpn_eval_metric = metric.RPNAccMetric()
    rpn_cls_metric = metric.RPNLogLossMetric()
    rpn_bbox_metric = metric.RPNL1LossMetric()
    # eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_cls_metric, rpn_bbox_metric,  cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), config.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), config.NUM_CLASSES)
    epoch_end_callback = (callback.do_checkpoint(args.e2e_prefix, means, stds),
                          callback.do_validate(args.e2e_prefix))

    arg_names = [x for x in sym.list_arguments() if x not in data_names+label_names]
    opt = get_optimizer(args, arg_names, len(roidb) / input_batch_size, args.iter_size)

    # train
    default.testing = False
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=args.kvstore,
            optimizer=opt, iter_size=args.iter_size,
            arg_params=arg_params, aux_params=aux_params,
            begin_epoch=args.begin_epoch, num_epoch=args.e2e_epoch)


if __name__ == '__main__':
    config_file = cfg_from_file('config.yml')
    merge_a_into_b(config_file, config)
    config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)

    if config.FRAMEWORK != '3DCE':
        assert config.NUM_IMAGES_3DCE == 1, "Combining multiple images is only possible in 3DCE"

    default_file = cfg_from_file('default.yml')
    merge_a_into_b(default_file, default)
    default.e2e_prefix = 'model/' + default.exp_name
    if default.begin_epoch != 0:
        default.resume = True
    default.accs = dict()

    if default.gpus == '':  # auto select GPU
        import GPUtil
        deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.2)
        if len(deviceIDs) == 0:
            deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.9, maxLoad=1)

        GPUs = GPUtil.getGPUs()
        default.gpus = str(len(GPUs)-1-deviceIDs[0])
        logger.info('using gpu '+default.gpus)
    default.val_gpu = default.gpus[0]
    # default.prefetch_thread_num = min(default.prefetch_thread_num, config.TRAIN.SAMPLES_PER_BATCH)

    train_net(default)

    # test the best model on the test set
    from test import test_net
    test_net(default.e2e_prefix, default.best_epoch)