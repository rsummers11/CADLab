import cPickle
import os
import time
import mxnet as mx
import numpy as np
import sys
from scipy.io import savemat

from module import MutableModule

from rcnn.logger import logger
from rcnn.config import config, default
from rcnn.fio import image
from rcnn.fio.load_ct_img import map_box_back
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

from rcnn.utils.timer import Timer

from rcnn.utils.evaluation import recall_all


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))


def im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)

    # drop the batch index
    boxes = output['rois_output'].asnumpy()[:, 1:]
    scores = output['rois_score'].asnumpy()

    # transform to original scale
    boxes = boxes / scale

    return scores, boxes, data_dict


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, boxes, data_dict = im_proposal(predictor, data_batch, data_names, scale)
        t2 = time.time() - t
        t = time.time()

        # assemble proposals
        dets = np.hstack((boxes, scores))
        original_boxes.append(dets)

        # filter proposals
        keep = np.where(dets[:, 4:] > thresh)[0]
        dets = dets[keep, :]
        imdb_boxes.append(dets)

        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale)

        logger.info('generating %d/%d ' % (i + 1, imdb.num_images) +
                    'proposal %d ' % (dets.shape[0]) +
                    'data %.4fs net %.4fs' % (t1, t2))
        i += 1

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    logger.info('wrote rpn proposals to %s' % rpn_file)
    return imdb_boxes


def im_detect(predictor, data_batch, data_names, scale):
    output = predictor.predict(data_batch)

    data_dict = dict(zip(data_names, data_batch.data))
    if config.TEST.HAS_RPN:
        rois = output['rois_output'].asnumpy()[:, 1:]
    else:
        rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
    im_shape = data_dict['data'].shape

    # save output
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
    # print 'im_det', scores[:,1]

    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    # we used scaled image & roi to train, so it is necessary to transform them back
    pred_boxes = pred_boxes / scale

    return scores, pred_boxes, data_dict


from rcnn.utils.evaluation import sens_at_FP
def my_evaluate_detections(all_boxes, all_gts):

    print 'Sensitivity @', default.val_avg_fp, 'average FPs per image:',
    res = sens_at_FP(all_boxes[1], all_gts[1], default.val_avg_fp, default.val_iou_th)  # cls 0 is background
    print res
    return res[3]  # sens@4FP


def pred_eval(predictor, test_data, imdb, vis=False, max_box=-1, thresh=1e-3):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param max_box: maximum number of boxes detected in each image
    :param thresh: valid detection threshold
    :return:
    """
    # assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = max_box

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    kept_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_gts = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_iminfos = []
    all_imnames = []
    all_crops = []

    i = 0
    _t = {'data': Timer(), 'im_detect' : Timer(), 'misc' : Timer()}
    _t['data'].tic()
    num_image = config.NUM_IMAGES_3DCE
    key_idx = (num_image - 1) / 2  # adjust image for 3DCE
    for im_info, imname, crop, data_batch in test_data:
        _t['data'].toc()
        _t['im_detect'].tic()

        all_iminfos.append(im_info)
        all_imnames.append(imname)
        all_crops.append(crop)

        # scale = im_info[0, 2]
        scale = 1.  # we have scaled the label in get_image(), so no need to scale the pred_box
        gt_boxes = data_batch.label[0].asnumpy()[key_idx, :, :]
        data_batch.label = None
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scale)
        _t['im_detect'].toc()
        _t['misc'].tic()

        for j in range(1, imdb.num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4]
            cls_boxes = map_box_back(cls_boxes, crop[2], crop[0], im_info[0,2])
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            all_boxes[j][i] = cls_dets[keep, :]
            all_gts[j][i] = map_box_back(gt_boxes, crop[2], crop[0], im_info[0,2])

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    kept_boxes[j][i] = all_boxes[j][i][keep, :]

        if vis:
            boxes_this_image = [[]] + [kept_boxes[j][i] for j in range(1, imdb.num_classes)]
            vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scale)

        _t['misc'].toc()

        if i % 200 == 0:
            if i <= 400:
                logger.info('im_detect: {:d}/{:d} data {:.3f}s im_detect {:.3f}s misc {:.3f}s'
                            .format(i, imdb.num_images, _t['data'].average_time, _t['im_detect'].average_time,
                                    _t['misc'].average_time))
            else:
                print i,
                sys.stdout.flush()
        # logger.info('testing %d/%d data %.4fs net %.4fs post %.4fs' % (i, imdb.num_images, t1, t2, t3))
        i += 1
        _t['data'].tic()

    print
    sys.stdout.flush()
    det_file = os.path.join(imdb.cache_path, imdb.name + '_detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(kept_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    default.res_dict = {'imname': all_imnames, 'boxes': all_boxes[1], 'gts': all_gts[1]}
    # default.res_dict = {'imname': all_imnames, 'im_info': all_iminfos, 'crops': all_crops, 'boxes': all_boxes[1], 'gts': all_gts[1]}

    acc = my_evaluate_detections(all_boxes, all_gts)
    sys.stdout.flush()
    return acc


def vis_all_boxes(im_array, boxes):
    """
    visualize all boxes in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    from ..fio.load_ct_img import windowing_rev, windowing

    im = windowing_rev(im_array+config.PIXEL_MEANS, config.WINDOWING)
    im = windowing(im, [-175,275]).astype(np.uint8)  # soft tissue window
    plt.imshow(im)
    color = (0.,1.,0.)
    for bbox in boxes:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)
        if boxes.shape[1] == 5:
            score = bbox[-1]
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.cv.CV_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im
