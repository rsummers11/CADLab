import mxnet as mx
import numpy as np
import sys

from rcnn.config import config, default


def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names():
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if config.TRAIN.END2END:
        pred.append('rcnn_label')
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


class BufferedEvalMetric(mx.metric.EvalMetric):
    """
    Keep a buffer of values instead of accumulating all values. Won't be affected by reset().
    """
    def __init__(self, name='metric', buffer_len=default.show_avg_loss):
        super(BufferedEvalMetric, self).__init__(name)
        self.val_buffer = []
        self.num_buffer = []
        self.buffer_len = buffer_len

    def addval(self, val, num=1):
        self.val_buffer = [val] + self.val_buffer
        self.num_buffer = [num] + self.num_buffer
        if len(self.val_buffer) > self.buffer_len:
            self.val_buffer.pop()
            self.num_buffer.pop()
        self.sum_metric = np.sum(self.val_buffer)
        self.num_inst = np.sum(self.num_buffer)


class RPNLogLossMetric(BufferedEvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.addval(cls_loss, label.shape[0])


class RCNNLogLossMetric(BufferedEvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')
        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.addval(cls_loss, label.shape[0])


class RPNL1LossMetric(BufferedEvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        # num_inst = np.sum(bbox_weight)
        num_inst = 1
        # print np.sum(bbox_loss), num_inst
        # sys.stdout.flush()

        self.addval(np.sum(bbox_loss) * config.TRAIN.RPN_REG_LOSS_WEIGHT, num_inst)


class RCNNL1LossMetric(BufferedEvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst
        keep_inds = np.where(label != 0)[0]
        # num_inst = len(keep_inds) * 4 * (config.NUM_CLASSES-1)
        num_inst = 1
        # print np.sum(bbox_loss), num_inst
        # sys.stdout.flush()

        self.addval(np.sum(bbox_loss) * config.TRAIN.RCNN_REG_LOSS_WEIGHT, num_inst)
