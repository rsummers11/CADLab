# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Post-process for the prediction results on DeepLesion"""
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from maskrcnn.config import cfg
from maskrcnn.data.datasets.evaluation.DeepLesion.tagging_eval import score2label


def post_process_results(result, info):
    """Convert tag scores to predictions, compute mask and RECIST, and more post-process"""
    if cfg.MODEL.TAG_ON:
        tag_predictions = score2label(result.get_field('tag_scores'), cfg.runtime_info.tag_sel_val)
        result.add_field('tag_predictions', tag_predictions)
    if cfg.MODEL.MASK_ON:
        contours, recists, diameters = predict_mask(result, info)
        # result.extra_fields['mask'] = mask
        del result.extra_fields['mask']  # delete it to save space
        result.add_field('contour_mm', contours)
        result.add_field('recist_mm', recists)
        result.add_field('diameter_mm', diameters)

    if cfg.TEST.POSTPROCESS_ON:
        do_post_process(result)


def do_post_process(result):
    """more post-process according to the predicted tags"""
    # lymph nodes with short diameter less than 10mm should be removed according to RECIST guidelines
    LN_idx = cfg.runtime_info.tag_list.index('lymph node')
    threshold = cfg.TEST.MIN_LYMPH_NODE_DIAM
    is_LN = result.get_field('tag_predictions')[:, LN_idx]
    is_small = result.get_field('diameter_mm')[:, 1] < threshold
    rmv = is_LN & is_small
    if 'is_gt' in result.extra_fields.keys():
        is_gt = result.get_field('is_gt')
        rmv = rmv & (~is_gt)

    # in demo and batch mode, you can choose to keep only certain tags
    if 'TAGS_TO_KEEP' in cfg.TEST.keys():
        # remove boxes with scores of certain tags less than predefined thresholds
        for tag in cfg.TEST.TAGS_TO_KEEP.keys():
            tag_idx = cfg.runtime_info.tag_list.index(tag)
            rmv = rmv | (result.get_field('tag_scores')[:, tag_idx] < cfg.TEST.TAGS_TO_KEEP[tag])

    if cfg.MODE in ('vis', 'demo', 'batch'):
        score_thresh = cfg.TEST.VISUALIZE.SCORE_THRESH
    else:
        score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    rmv = rmv | (result.get_field('scores') < score_thresh)
    
    result.bbox = result.bbox[~rmv]
    for k, v in result.extra_fields.items():
        result.add_field(k, v[~rmv])


def predict_mask(result, info):
    """Binarize mask, compute contour, RECIST, and RECIST diameters"""
    spacing = info['spacing']
    im_scale = info['im_scale']
    boxes_mm = result.bbox.numpy()/im_scale*spacing
    masks = result.get_field('mask').numpy()
    masks_new = []
    recists = []
    diameters = []
    contours = []
    if cfg.MODE in ("train", "eval"):
        is_gt = result.get_field('is_gt').numpy()>0

    for i in range(len(boxes_mm)):

        if cfg.MODE in ("train", "eval") and cfg.TEST.EVAL_SEG_TAG_ON_GT and not is_gt[i]:
            # don't compute to save time
            recists.append(-np.ones((8, ), dtype=np.float32))
            diameters.append([-1, -1])
            contours.append(-np.ones((1,2)))
            continue

        # coordinate offset caused by IMG_DO_CLIP is not considered
        box = boxes_mm[i]
        mask = (masks[i, 0] >= cfg.TEST.MASK.THRESHOLD).astype('uint8')
        if mask.sum() == 0:
            # if no mask is predicted, we treat the whole box as the mask
            mask = mask+1
        # masks_new.append(mask)

        tmp = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(tmp) == 3:  # yk: why?
            contours1 = tmp[1]
        else:
            contours1 = tmp[0]
        contour = contours1[np.argmax([len(contour1) for contour1 in contours1])]
        contour = contour.squeeze().astype('float32')
        if contour.ndim == 1:
            contour = contour[None, :]
        reshape_ratio = (box[2:]-box[:2]) / np.array(mask.shape[::-1])
        contour *= reshape_ratio
        contour += box[:2]

        long_diameter, short_diameter, long_diameter_len, short_diameter_len = find_recist_in_contour(contour)

        recist = np.vstack((contour[long_diameter], contour[short_diameter])).reshape((-1))
        recists.append(recist)
        diameters.append([long_diameter_len, short_diameter_len])
        contours.append(contour)

    # masks_new = torch.from_numpy(np.dstack(masks_new))
    if len(contours) > 0:
        contours_cat = -np.ones((len(contours),  max([len(c) for c in contours]), 2), dtype='float32')
        for i, c in enumerate(contours):
            contours_cat[i, :len(c), :] = c
        contours_cat = torch.from_numpy(contours_cat)
        recists = torch.from_numpy(np.vstack(recists))
        diameters = torch.from_numpy(np.vstack(diameters)).to(torch.float)
    else:
        contours_cat = torch.zeros(0,0,2, dtype=torch.float)
        recists = torch.zeros(0,8, dtype=torch.float)
        diameters = torch.zeros(0,2, dtype=torch.float)
    return contours_cat, recists, diameters


def find_recist_in_contour(contour):
    if len(contour) == 1:
        return np.array([0,0]), np.array([0,0]), 1, 1
    D = squareform(pdist(contour)).astype('float32')
    long_diameter_len = D.max()
    endpt_idx = np.where(D==long_diameter_len)
    endpt_idx = np.array([endpt_idx[0][0], endpt_idx[1][0]])
    long_diameter_vec = (contour[endpt_idx[0]] - contour[endpt_idx[1]])[:, None]

    side1idxs = np.arange(endpt_idx.min(), endpt_idx.max()+1)
    side2idxs = np.hstack((np.arange(endpt_idx.min()+1), np.arange(endpt_idx.max(), len(contour))))
    perp_diameter_lens = np.empty((len(side1idxs), ), dtype=float)
    perp_diameter_idx2s = np.empty((len(side1idxs), ), dtype=int)
    for i, idx1 in enumerate(side1idxs):
        short_diameter_vecs = contour[side2idxs] - contour[idx1]
        dot_prods_abs = np.abs(np.matmul(short_diameter_vecs, long_diameter_vec))
        idx2 = np.where(dot_prods_abs == dot_prods_abs.min())[0]
        if len(idx2) > 1:
            idx2 = idx2[np.sum(short_diameter_vecs[idx2]**2, axis=1).argmax()]
        idx2 = side2idxs[idx2]  # find the one that is perpendicular with long axis
        perp_diameter_idx2s[i] = idx2
        perp_diameter_lens[i] = D[idx1, idx2]
    short_diameter_len = perp_diameter_lens.max()
    short_diameter_idx1 = side1idxs[perp_diameter_lens.argmax()]
    short_diameter_idx2 = perp_diameter_idx2s[perp_diameter_lens.argmax()]
    short_diameter = np.array([short_diameter_idx1, short_diameter_idx2])
    long_diameter = endpt_idx

    return long_diameter, short_diameter, long_diameter_len, short_diameter_len