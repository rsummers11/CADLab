import mxnet as mx
import numpy as np
from mxnet.executor_manager import _split_input_slice
from rcnn.utils.timer import Timer
import threading
import sys
import logging
from Queue import Queue

from rcnn.config import config
from rcnn.fio.image import tensor_vstack
from rcnn.fio.rpn import get_rpn_testbatch, get_rpn_batch, assign_anchor
from rcnn.fio.rcnn import get_rcnn_testbatch, get_rcnn_batch


class AnchorLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, nThreads=0):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.prefetch_indices = None

        # get first batch to fill in provide_data and provide_label
        self.nThreads = 0
        self.get_batch()

        self.nThreads = nThreads
        if nThreads > 0:
            self.prefetched_data = Queue(maxsize=max(nThreads, self.batch_size))

            def prefetch_func(self):
                """Thread entry"""
                while True:
                    if not self.started:
                        return
                    ri = self.prefetch_indices.get()
                    assert ri is not None
                    data = get_rpn_batch(self.roidb[ri:ri+1])
                    self.prefetched_data.put(data)

            self.prefetch_threads = []
            self.prefetch_func = prefetch_func
            self.started = False

        self.reset()

    def __del__(self):
        self.started = False
        for i in range(self.nThreads):
            self.prefetch_threads[i].join()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0

        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

        if self.nThreads > 0:
            if self.started:
                for th in self.prefetch_threads:
                    th.join(timeout=1)
                    del th
                self.prefetch_threads = []
                del self.prefetch_indices

            self.prefetch_indices = Queue()
            for i in self.index:
                self.prefetch_indices.put(i)
            for i in self.index:  # put one more time to avoid prefetch_indices is empty. may have better solutions
                self.prefetch_indices.put(i)
            self.prefetched_data.empty()

            self.started = True
            for i in range(self.nThreads):
                self.prefetch_threads += [threading.Thread(target=self.prefetch_func, args=(self,))]
                self.prefetch_threads[i].setDaemon(True)
                self.prefetch_threads[i].start()
            logging.info('Loading images with %d threads.', self.nThreads)


    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch(self):
        data_list = []
        label_list = []
        if self.nThreads == 0:
            # slice roidb
            cur_from = self.cur
            cur_to = min(cur_from + self.batch_size, self.size)
            roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

            # decide multi device slice
            work_load_list = self.work_load_list
            ctx = self.ctx
            if work_load_list is None:
                work_load_list = [1] * len(ctx)
            assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
                "Invalid settings for work load. "
            slices = _split_input_slice(self.batch_size, work_load_list)

            # get testing data for multigpu
            for islice in slices:
                iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
                data, label = get_rpn_batch(iroidb)
                data_list.append(data)
                label_list.append(label)

        else:
            for p in range(len(self.ctx)):
                data, label = self.multithread_get_rpn_batch()
                data_list.append(data)
                label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])  # to unify the shape of data
        k = 0
        for data in data_list:
            bs = data['data'].shape[0]
            data['data'] = data_tensor[k:k+bs, :]
            k += bs

        # re-organize images and labels for 3DCE
        new_label_list = []
        num_smp = config.TRAIN.SAMPLES_PER_BATCH
        num_image = config.NUM_IMAGES_3DCE
        num_slices = config.NUM_SLICES
        for data, label1 in zip(data_list, label_list):
            data0 = data['data']
            im_info0 = data['im_info']
            gt0 = label1['gt_boxes']
            data_new = np.empty((num_smp*num_image, num_slices, data0.shape[2], data0.shape[3]))
            iminfo_new = np.empty((num_smp*num_image, 3))
            gt_new = np.zeros((num_smp*num_image, gt0.shape[1], 5))
            for p in range(num_smp):
                for q in range(num_image):
                    data_new[p*num_image+q, :,:,:] = data0[p, q*num_slices:(q+1)*num_slices, :,:]
                    iminfo_new[p*num_image+q, :] = im_info0[p, :]
                    if q == (num_image-1)/2:
                        gt_new[p*num_image+q, :,:] = gt0[p, :, :]

            data['data'] = data_new
            data['im_info'] = iminfo_new
            label1['gt_boxes'] = gt_new

            # infer label shape
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
            feat_shape = [int(i) for i in feat_shape[0]]

            # add gt_boxes to data for e2e
            data['gt_boxes'] = label1['gt_boxes']#[np.newaxis, :, :]

            # assign anchor for label
            for i in range(len(label1['gt_boxes'])):
                label = assign_anchor(feat_shape, label1['gt_boxes'][i], data['im_info'][i:i+1, :],
                                      self.feat_stride, self.anchor_scales,
                                      self.anchor_ratios, self.allowed_border)
                if i % num_image != (num_image-1)/2:
                    label['label'][:] = -1
                    label['bbox_weight'][:] = 0

                new_label_list.append(label)

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]

    def multithread_get_rpn_batch(self):
        d = []
        l = []
        for i in range(self.batch_size):
            d1, l1 = self.prefetched_data.get()
            d.append(d1)
            l.append(l1)

        data = {}
        for key in d[0].keys():
            data[key] = tensor_vstack([batch[key] for batch in d])

        label = {}
        for key in l[0].keys():
            label[key] = tensor_vstack([batch[key] for batch in l])

        return data, label


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size=1, shuffle=False,
                 has_rpn=False, nThreads=0):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn
        self.t = Timer()

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if has_rpn:
            self.data_name = ['data', 'im_info']
        else:
            self.data_name = ['data', 'rois']
        self.label_name = ['gt_boxes']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = None
        self.im_info = None
        self.imname = None
        self.prefetch_indices = None

        # get first batch to fill in provide_data and provide_label
        self.nThreads = 0
        self.get_batch()

        if nThreads > 0:
            self.prefetched_data = Queue(maxsize=max(nThreads, self.batch_size))

        self.nThreads = nThreads
        self.reset()

        if nThreads > 0:
            self.started = True

            def prefetch_func(self):
                """Thread entry"""
                while True:
                    if not self.started:
                        return
                    ri = self.prefetch_indices.get()
                    if ri is None:
                        return
                    data = get_rpn_testbatch(self.roidb[ri:ri + 1])
                    self.prefetched_data.put(data)

            self.prefetch_threads = []
            for i in range(nThreads):
                self.prefetch_threads += [threading.Thread(target=prefetch_func, args=(self,))]
                self.prefetch_threads[i].setDaemon(True)
                self.prefetch_threads[i].start()
            logging.info('Loading images with %d threads.', nThreads)

    def __del__(self):
        self.started = False
        for i in range(self.nThreads):
            self.prefetch_threads[i].join()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None
        # return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

        if self.nThreads > 0:
            self.prefetch_indices = Queue()
            for i in self.index:
                self.prefetch_indices.put(i)
            self.prefetched_data.empty()

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.im_info, self.imname, self.crop, \
                   mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        if self.nThreads == 0:
            cur_from = self.cur
            cur_to = min(cur_from + self.batch_size, self.size)
            roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
            assert self.has_rpn
            data, label, im_info, imname, crop = get_rpn_testbatch(roidb)
        else:
            assert self.has_rpn
            data, label, im_info, imname, crop = self.prefetched_data.get()

        num_smp = config.TEST.SAMPLES_PER_BATCH
        num_image = config.NUM_IMAGES_3DCE
        num_slices = config.NUM_SLICES

        data0 = data['data']
        im_info0 = data['im_info']
        gt0 = label['gt_boxes']
        data_new = np.empty((num_smp * num_image, num_slices, data0.shape[2], data0.shape[3]))
        iminfo_new = np.empty((num_smp * num_image, 3))
        gt_new = np.zeros((num_smp * num_image, gt0.shape[0], 4))
        for p in range(num_smp):
            for q in range(num_image):  # adjust data and label for 3DCE
                data_new[p * num_image + q, :, :, :] = data0[p, q * num_slices:(q + 1) * num_slices, :, :]
                iminfo_new[p * num_image + q, :] = im_info0[p, :]
                if q == (num_image - 1) / 2:
                    gt_new[p * num_image + q, :, :] = gt0

        data['data'] = data_new
        data['im_info'] = iminfo_new
        label['gt_boxes'] = gt_new

        self.data = [mx.nd.array(data[name]) for name in self.data_name]
        self.im_info = im_info
        self.label = [mx.nd.array(label[name]) for name in self.label_name]
        self.imname = imname
        self.crop = crop
