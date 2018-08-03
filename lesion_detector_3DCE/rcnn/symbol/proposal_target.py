"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import logging
import mxnet as mx
import numpy as np
from distutils.util import strtobool

from ..logger import logger
from rcnn.fio.rcnn import sample_rois
from rcnn.config import config


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._fg_fraction = fg_fraction

        if logger.level == logging.DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        # assert self._batch_rois % self._batch_images == 0, \
        #     'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)
        rois_per_image = self._batch_rois  # / self._batch_images
        fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()

        rois = np.empty((0, 5))
        labels = np.empty((0,))
        bbox_targets = np.empty((0, self._num_classes * 4))
        bbox_weights = np.empty((0, self._num_classes * 4))
        num_image = config.NUM_IMAGES_3DCE
        key_idx = (num_image-1)/2
        for i0 in range(self._batch_images):
            i = i0*num_image+key_idx
            # Include ground-truth boxes in the set of candidate rois
            gt_boxes1 = gt_boxes[i, :, :]
            gt_boxes1 = gt_boxes1[gt_boxes1[:, -1] > 0, :]
            batch_ind = np.zeros((gt_boxes1.shape[0], 1), dtype=gt_boxes.dtype) + i0
            all_rois1 = all_rois[all_rois[:, 0] == i0, :]
            all_rois1[:, 0] = i0
            all_rois1 = np.vstack((all_rois1, np.hstack((batch_ind, gt_boxes1[:, :-1]))))
            # Sanity check: single batch only
            # assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

            rois1, labels1, bbox_targets1, bbox_weights1 = \
                sample_rois(all_rois1, fg_rois_per_image, rois_per_image, self._num_classes, gt_boxes=gt_boxes1)

            rois = np.vstack((rois, rois1))
            labels = np.hstack((labels, labels1))
            bbox_targets = np.vstack((bbox_targets, bbox_targets1))
            bbox_weights = np.vstack((bbox_weights, bbox_weights1))

            if logger.level == logging.DEBUG:
                logger.debug("labels: %s" % labels1)
                logger.debug('num fg: {}'.format((labels1 > 0).sum()))
                logger.debug('num bg: {}'.format((labels1 == 0).sum()))
                self._count += 1
                self._fg_num += (labels1 > 0).sum()
                self._bg_num += (labels1 == 0).sum()
                logger.debug("self._count: %d" % self._count)
                logger.debug('num fg avg: %d' % (self._fg_num / self._count))
                logger.debug('num bg avg: %d' % (self._bg_num / self._count))
                logger.debug('ratio: %.3f' % (float(self._fg_num) / float(self._bg_num)))

        for ind, val in enumerate([rois, labels, bbox_targets, bbox_weights]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction='0.25'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        output_rois_shape = (self._batch_rois * self._batch_images, 5)
        label_shape = (self._batch_rois * self._batch_images, )
        bbox_target_shape = (self._batch_rois * self._batch_images, self._num_classes * 4)
        bbox_weight_shape = (self._batch_rois * self._batch_images, self._num_classes * 4)

        return [rpn_rois_shape, gt_boxes_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
