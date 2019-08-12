# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Visualization utilities"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import torch

from maskrcnn.config import cfg
from maskrcnn.data.datasets.load_ct_img import windowing, windowing_rev
from .print_info import gen_tag_pred_str


def vis_all_detection(im, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    im = windowing_rev(im + cfg.INPUT.PIXEL_MEAN, cfg.INPUT.WINDOWING)
    im = windowing(im, [-175, 275]).astype('uint8')
    plt.imshow(im, cmap='gray')
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        elif name == 'gt':
            color = (0, 1, 0)
        else:
            color = (1, 1, 0)
        # color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)
            if det.shape[0] > 4:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1] - 2,
                               '{:s} {:.3f}'.format(name, score),
                               # bbox=dict(facecolor=color, alpha=0.5),
                               fontsize=12, color=color)
    plt.show()


def visualize(im, target, result, info, masker):
    """Visualize gt and predicted box, tag, mask, and RECIST"""
    print('\n lesion_idxs', info['lesion_idxs'], ':', info['image_fn'])
    if 'tags' in info.keys():
        print('gt tags:')
        for tags in target.get_field('tags'):
            idx = torch.nonzero(tags > 0)
            msg = ', '.join([cfg.runtime_info.tag_list[idx1] for idx1 in idx])
            print(msg)
    if 'diameters' in info.keys():
        print('gt diameters:')
        print(info['diameters'])

    im = windowing_rev(im + cfg.INPUT.PIXEL_MEAN, cfg.INPUT.WINDOWING)
    im = windowing(im, info['window']).astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    gt = target.bbox.cpu().numpy()
    gt_recists = info['recists']

    scale = cfg.TEST.VISUALIZE.SHOW_SCALE
    im = cv2.resize(im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gt *= scale
    gt_recists *= scale
    overlay, msgs = draw_results(im, gt, recists=gt_recists, colors=(0, 255, 0), thickness=(2,1,1))

    pred = result.bbox.cpu().numpy()
    labels = result.get_field('labels').cpu().numpy()
    scores = result.get_field('scores').cpu().numpy()
    tag_scores = result.get_field('tag_scores').cpu().numpy()
    tag_predictions = result.get_field('tag_predictions').cpu().numpy()

    # if cfg.MODEL.MASK_ON and len(result.get_field('contour_mm')) > 0:
    mm2pix = info['im_scale'] / info['spacing'] * scale
    contours = result.get_field('contour_mm').cpu().numpy() * mm2pix
    contours = [c[c[:,0]>0, :] for c in contours]
    contours = [c+1*scale for c in contours]  # there seems to be a small offset in the mask?
    recists = result.get_field('recist_mm').cpu().numpy() * mm2pix
    recists += 1*scale   # there seems to be a small offset in the mask?
    diameters = result.get_field('diameter_mm').cpu().numpy()

    pred *= scale
    overlay, msgs = draw_results(overlay, pred, labels, scores, tag_predictions=tag_predictions, tag_scores=tag_scores,
                           contours=contours, recists=recists, diameters=diameters)
    plt.figure(1)
    plt.imshow(overlay)
    for msg in msgs:
        print(msg)

    if cfg.TEST.VISUALIZE.SHOW_MASK_HEATMAPS and 'mask' in result.extra_fields.keys():
        plt.figure(2)
        mask = result.get_field('mask')
        mask = masker([mask], [result])[0]
        mask = mask.numpy()
        mask0 = cv2.resize(mask[0, 0], None, None, fx=scale, fy=scale)
        mask1 = np.empty((mask.shape[0], mask.shape[1], mask0.shape[0], mask0.shape[1]), dtype=mask.dtype)
        for i1 in range(mask.shape[0]):
            for i2 in range(mask.shape[1]):
                mask1[i1, i2] = cv2.resize(mask[i1, i2], None, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_NEAREST)

        pred *= scale
        heatmap = np.sum(mask1, (0, 1)).astype('uint8')
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        plt.imshow(heatmap, cmap='gray')
    plt.show()


def draw_results(im, boxes, labels=None, scores=None, class_names=None, tag_predictions=None, tag_scores=None,
                 contours=None, recists=None, diameters=None, colors=None, thickness=(1,1,1)):
    """
    draw boxes, masks on image; generate message of predictions
    """
    num_box = boxes.shape[0]
    if labels is None:
        labels = np.ones((num_box, ), dtype=int)
    if colors is None:
        colors = np.random.rand(num_box, 3) * 255
        colors[range(num_box), colors.argmax(axis=1)] = 255  # increase saturation for display on gray image
    else:
        if isinstance(colors, (tuple, list)):
            colors = np.tile(colors, (num_box, 1))
        else:
            assert colors.shape == (num_box, 3)

    msgs = []
    for i, box in enumerate(boxes.astype(int)):
        msg = ''
        color = colors[i]
        im = cv2.rectangle(
            im, tuple(box[:2]), tuple(box[2:4]), color.tolist(), thickness=thickness[0]
        )

        if scores is not None:
            assert len(scores) == num_box
            msg += 'lesion %s, score: %.3f'%(chr(65+i), scores[i])

        if contours is not None:
            assert len(contours) == num_box
            contour = [contours[i][:, None, :].round().astype(int)]
            im = cv2.drawContours(im, contour, -1, [c*2/3 for c in color], thickness=thickness[1])

        if recists is not None:
            assert len(recists) == num_box
            recist = recists[i].astype(int)
            im = cv2.line(im, tuple(recist[:2]), tuple(recist[2:4]),
                          [c*2/3 for c in color], thickness=thickness[2])
            im = cv2.line(im, tuple(recist[4:6]), tuple(recist[6:]),
                          [c*2/3 for c in color], thickness=thickness[2])

        if diameters is not None:
            assert len(diameters) == num_box
            msg += ' | %d x %d mm' % (diameters[i,0], diameters[i,1])

        if tag_scores is not None:
            assert len(tag_scores) == num_box
            msg += ' | '+gen_tag_pred_str(tag_predictions[i], tag_scores[i])

        txt = ""
        if class_names is not None:
            txt += class_names[labels[i]]
        if scores is not None:
            txt += '%s: %.3f'%(chr(65+i), scores[i])
            # txt += '%.3f'%(scores[i])
            cv2.putText(im, txt, (box[0], box[1] - 2),
                        cv2.FONT_HERSHEY_DUPLEX , fontScale=.5,
                        color=color.tolist(), thickness=1)

        msgs.append(msg)
    return im, msgs
