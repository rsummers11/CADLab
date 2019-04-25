# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to run a demo of LesaNet on user-provided data.
# --------------------------------------------------------

import os
import numpy as np
import logging
from time import time
import torch
import nibabel as nib
import cv2

from config import config, default
from load_ct_img import load_prep_img, get_patch, windowing, windowing_rev
from evaluate import score2label
from utils import logger


def demo_fun(model):
    # test model on user-provided data, instead of the preset DeepLesion dataset
    model.eval()

    while True:
        while True:
            info = "Please input the path of a nifti CT volume >> "
            logger.info(info)
            im_path = raw_input('')
            if not os.path.exists(im_path):
                logger.info('file does not exist!')
                continue
            try:
                logger.info('reading image ...')
                nifti_data = nib.load(im_path)
                break
            except:
                logger.info('load nifti file error!')

        while True:
            info = 'Please input the path of a coordinate file, in which each line contains ' \
                   'the coordinates of a 2D b-box of a lesion (slice, x1, y1, x2, y2. Top left is 0) >> '
            logger.info(info)
            path = raw_input('')
            if not os.path.exists(path):
                logger.info('file does not exist!')
                continue
            with open(path) as f:
                coords = [np.array([float(num) for num in line.split()]) for line in f.readlines()]
            break

        infos = []
        infos.append('-----------------------------------------')
        for box in coords:
            infos.append('lesion coordinates: %s' % str(box.astype(int)))
            input = get_input(nifti_data, box, os.path.basename(im_path), infos)
            result = model(input)
            if 'class_prob2' in result:
                class_prob = result['class_prob2']
            else:
                class_prob = result['class_prob1']
            class_prob = class_prob.detach().cpu().numpy()

            if config.TEST.USE_CALIBRATED_TH:
                infos.append('Predictions (using a separate threshold for each label on the val set):')
                class_pred = class_prob[0] > default.best_cls_ths
            elif config.TEST.SCORE_PARAM < 1:
                infos.append('Predictions (uniform threshold %f):' % config.TEST.SCORE_PARAM)
                class_pred = score2label(class_prob, config.TEST.SCORE_PARAM)[0]
            else:
                infos.append('Predictions (top %d):' % config.TEST.SCORE_PARAM)
                class_pred = score2label(class_prob, config.TEST.SCORE_PARAM)[0]
            infos = output_scores(class_pred, class_prob[0], infos)

            if config.TEST.USE_CALIBRATED_TH or config.TEST.SCORE_PARAM < 1:
                K = 5
                infos.append('\nPredictions (top %d):' % K)
                class_pred = score2label(class_prob, K)[0]
                infos = output_scores(class_pred, class_prob[0], infos)

            infos.append('============================================')
        logger.info('\n'.join(infos))


def output_scores(pred, prob, infos):
    pred_labels = np.where(pred)[0]
    for cls in pred_labels[np.argsort(prob[pred_labels])[::-1]]:
        infos.append('%.4f %s' % (prob[cls], default.term_list[cls]))
    return infos


def get_input(nifti_data, box0, imname, infos):
    slice_idx = box0[0]
    box = box0[1:].copy()
    vol = nifti_data.get_data()
    aff = nifti_data.get_affine()[:3, :3]
    spacing = np.abs(aff[:2, :2]).max()
    slice_intv = np.abs(aff[2, 2])

    # Ad-hoc code for normalizing the orientation of the volume.
    # The aim is to make vol[:,:,i] an supine right-left slice
    # It works for the authors' data, but maybe not suitable for some kinds of nifti files
    if np.abs(aff[0, 0]) > np.abs(aff[0, 1]):
        vol = np.transpose(vol, (1,0,2))
        aff = aff[[1,0,2], :]
    if np.max(aff[0, :2]) > 0:
        vol = vol[::-1, :, :]
    if np.max(aff[1, :2]) > 0:
        vol = vol[:, ::-1, :]

    im, im_scale, c = load_prep_img(vol, int(slice_idx), spacing=spacing,
                                    slice_intv=slice_intv, do_clip=False, num_slice=config.NUM_SLICES)
    box *= im_scale
    patch, center_box, patch_scale = get_patch(im, box)

    im_show = windowing(windowing_rev(patch[:,:,1] + config.PIXEL_MEANS, config.WINDOWING), [-175, 275]).astype(np.uint8)
    im_show = cv2.cvtColor(im_show, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(im_show, (int(center_box[0]), int(center_box[1])),
                  (int(center_box[2]), int(center_box[3])), color=(0, 255, 0), thickness=1)
    coord_str = '_'.join([str(x) for x in box0.astype(int).tolist()])
    fn = os.path.join(default.res_path, '%s_%s.png' % (imname, coord_str))
    cv2.imwrite(fn, im_show)
    infos.append('Image patch saved to %s.' % fn)

    patch = patch.astype(float) / 255
    patch = torch.from_numpy(patch.transpose((2, 0, 1)))

    out_box = torch.tensor([[0, 0, patch.shape[2], patch.shape[1]]]).to(dtype=torch.float).cuda()
    center_box = torch.tensor(center_box[None,:]).to(dtype=torch.float).cuda()
    patch = patch[None, :,:,:].to(dtype=torch.float).cuda()
    return patch, out_box, center_box
