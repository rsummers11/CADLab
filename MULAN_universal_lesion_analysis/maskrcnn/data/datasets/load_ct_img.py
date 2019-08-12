# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Load and pre-process CT images in DeepLesion"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_dilation
import nibabel as nib

from maskrcnn.config import cfg


def load_prep_img(data_dir, imname, spacing, slice_intv, do_clip=False, num_slice=3, is_train=False):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    im, mask = load_multislice_img_16bit_png(data_dir, imname, slice_intv, do_clip, num_slice)

    im = windowing(im, cfg.INPUT.WINDOWING)

    if do_clip:  # clip black border
        c = get_range(mask, margin=0)
        if cfg.INPUT.DATA_AUG_POSITION is not False and is_train:
            offset_aug = np.random.randint(16)-8
            c[0] += offset_aug
            offset_aug = np.random.randint(16)-8
            c[2] += offset_aug
            c = [max(0, val) for val in c]
        im = im[c[0]:c[1] + 1, c[2]:c[3] + 1, :]
        # mask = mask[c[0]:c[1] + 1, c[2]:c[3] + 1]
        # print(im.shape)
    else:
        c = [0, im.shape[0]-1, 0, im.shape[1]-1]

    im_shape = im.shape[0:2]
    if spacing is not None and cfg.INPUT.NORM_SPACING > 0:  # spacing adjust, will overwrite simple scaling
        im_scale = float(spacing) / cfg.INPUT.NORM_SPACING
    else:
        im_scale = float(cfg.INPUT.MAX_IM_SIZE) / float(np.max(im_shape))  # simple scaling

    if cfg.INPUT.DATA_AUG_SCALE is not False and is_train:
        aug_scale = cfg.INPUT.DATA_AUG_SCALE
        im_scale *= np.random.rand(1)*(aug_scale[1]-aug_scale[0])+aug_scale[0]
    max_shape = np.max(im_shape)*im_scale
    if max_shape > cfg.INPUT.MAX_IM_SIZE:
        im_scale1 = float(cfg.INPUT.MAX_IM_SIZE) / max_shape
        im_scale *= im_scale1

    if im_scale != 1:
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return im, im_scale, c


def load_multislice_img_16bit_png(data_dir, imname, slice_intv, do_clip, num_slice):
    data_cache = {}
    def _load_data_from_png(imname, delta=0):
        imname1 = get_slice_name(data_dir, imname, delta)
        if imname1 not in data_cache.keys():
            data_cache[imname1] = cv2.imread(os.path.join(data_dir, imname1), -1)
            assert data_cache[imname1] is not None, 'file reading error: ' + imname1
            # if data_cache[imname1] is None:
            #     print('file reading error:', imname1)
        return data_cache[imname1]

    def _load_data_from_nifti(imname, delta=0):
        # in this case, data_dir is the numpy volume and imname is the slice index
        vol = data_dir
        idx = min(vol.shape[2]-1, max(int(imname+delta), 0))
        return vol[:,:,idx]

    if isinstance(data_dir, str) and isinstance(imname, str):
        _load_data = _load_data_from_png
    elif isinstance(data_dir, np.ndarray) and isinstance(imname, int):
        _load_data = _load_data_from_nifti

    im_cur = _load_data(imname)

    mask = get_mask(im_cur) if do_clip else None

    if cfg.INPUT.SLICE_INTV == 0 or np.isnan(slice_intv) or slice_intv < 0:
        ims = [im_cur] * num_slice  # only use the central slice

    else:
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = float(cfg.INPUT.SLICE_INTV) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
            for p in range(int((num_slice-1)/2)):
                im_prev = _load_data(imname, - rel_pos * (p + 1))
                im_next = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
        else:
            for p in range(int((num_slice-1)/2)):
                intv1 = rel_pos*(p+1)
                slice1 = _load_data(imname, - np.ceil(intv1))
                slice2 = _load_data(imname, - np.floor(intv1))
                im_prev = a * slice1 + b * slice2  # linear interpolation

                slice1 = _load_data(imname, np.ceil(intv1))
                slice2 = _load_data(imname, np.floor(intv1))
                im_next = a * slice1 + b * slice2

                ims = [im_prev] + ims + [im_next]

    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    im = im.astype(np.float32,
                       copy=False) - 32768  # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit

    return im, mask


def get_slice_name(data_dir, imname, delta=0):
    """Infer slice name with an offset"""
    if delta == 0:
        return imname
    delta = int(delta)
    dirname, slicename = imname.split(os.sep)
    slice_idx = int(slicename[:-4])
    imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)

    # if the slice is not in the dataset, use its neighboring slice
    while not os.path.exists(os.path.join(data_dir, imname1)):
        # print('file not found:', imname1)
        delta -= np.sign(delta)
        imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
        if delta == 0:
            break

    return imname1


def windowing(im, win):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


def windowing_rev(im, win):
    """backward windowing"""
    im1 = im.astype(float)/255
    im1 *= win[1] - win[0]
    im1 += win[0]
    return im1


def get_mask(im):
    """use a intensity threshold to roughly find the mask of the body"""
    th = 32000  # an approximate background intensity value
    mask = im > th
    mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
    # mask = binary_dilation(mask)
    # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

    if mask.sum() == 0:  # maybe atypical intensity
        mask = im * 0 + 1
    return mask.astype(dtype=np.int32)


def get_range(mask, margin=0):
    """Get up, down, left, right extreme coordinates of a binary mask"""
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return [u, d, l, r]


def map_box_back(boxes, cx=0, cy=0, im_scale=1.):
    """Reverse the scaling and offset of boxes"""
    boxes /= im_scale
    boxes[:, [0,2]] += cx
    boxes[:, [1,3]] += cy
    return boxes
