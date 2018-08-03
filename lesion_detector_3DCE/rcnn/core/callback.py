import time
import numpy as np
import os
import logging
import mxnet as mx
from rcnn.tools.validate import validate
from ..config import default, config


class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.begin_validated = False

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        # if self.last_count > count:
        #     self.init = False
        # self.last_count = count

        if not self.init:
            self.init = True
            self.tic = time.time()

        if count % self.frequent == 0:
            speed = self.frequent * self.batch_size / (time.time() - self.tic)
            if param.eval_metric is not None:
                name, value = param.eval_metric.get()
                s = "Epoch %d Batch %d\t%.1f smp/sec\t" % (param.epoch, count, speed)
                for n, v in zip(name, value):
                    s += "%s=%.3g,\t" % (n, v)
                logging.info(s)
            else:
                logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                             param.epoch, count, speed)
            self.tic = time.time()

            # wt = param.locals['self'].get_params()[0]['conv1_1_weight'].asnumpy()
            # logging.info('conv1 mean=%f, std=%f\n', wt.mean(), wt.std())
            # if np.isnan(wt).any():
            #     logging.info('detected nan')

        if default.validate_at_begin and not self.begin_validated:
            # means = np.tile(np.array(config.TRAIN.BBOX_MEANS), config.NUM_CLASSES)
            # stds = np.tile(np.array(config.TRAIN.BBOX_STDS), config.NUM_CLASSES)
            mod = param.locals['self']
            arg_params, aux_params = mod.get_params()
            if not default.resume:
                for callback1 in param.locals['epoch_end_callback']:
                    callback1(-1, mod.symbol, arg_params, aux_params)
            else:
                validate(default.e2e_prefix, default.begin_epoch-1)
            self.begin_validated = True




def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        if config.FRAMEWORK == 'RFCN':
            bbox_param = 'rfcn_bbox'  # for RFCN, because its final layer is pooling
        else:
            bbox_param = 'bbox_pred'  # for faster rcnn
        if config.TRAIN.BBOX_NORMALIZE_TARGETS:
            arg[bbox_param+'_weight_test'] = (arg[bbox_param+'_weight'].T * mx.nd.array(stds)).T
            arg[bbox_param+'_bias_test'] = arg[bbox_param+'_bias'] * mx.nd.array(stds) + mx.nd.array(means)

        path = os.path.dirname(prefix)
        if not os.path.exists(path):
            os.mkdir(path)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)

        if config.TRAIN.BBOX_NORMALIZE_TARGETS:
            arg.pop(bbox_param+'_weight_test')
            arg.pop(bbox_param+'_bias_test')

    return _callback


def do_validate(prefix):
    def _callback(iter_no, sym, arg, aux):
        validate(prefix, iter_no)

    return _callback
