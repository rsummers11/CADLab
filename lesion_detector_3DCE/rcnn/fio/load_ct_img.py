import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_dilation
import mxnet as mx
import nibabel as nib

from ..config import config, default
from ..utils.timer import Timer


t = Timer()


def load_prep_img(imname, slice_idx, spacing, slice_intv, do_clip=False, num_slice=3):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    if imname.endswith('.nii.gz') or imname.endswith('.nii'):
        im, mask = load_multislice_img_nifti(imname, slice_idx, slice_intv, do_clip, num_slice)
    else:
        im, mask = load_multislice_img_16bit_png(imname, slice_idx, slice_intv, do_clip, num_slice)

    im = windowing(im, config.WINDOWING)

    if do_clip:  # clip black border
        c = get_range(mask, margin=0)
        im = im[c[0]:c[1] + 1, c[2]:c[3] + 1, :]
        # mask = mask[c[0]:c[1] + 1, c[2]:c[3] + 1]
        # print im.shape
    else:
        c = [0, im.shape[0]-1, 0, im.shape[1]-1]

    im_shape = im.shape[0:2]
    if spacing is not None and config.NORM_SPACING > 0:  # spacing adjust, will overwrite simple scaling
        im_scale = float(spacing) / config.NORM_SPACING
    else:
        im_scale = float(config.SCALE) / float(np.min(im_shape))  # simple scaling

    max_shape = np.max(im_shape)*im_scale
    if max_shape > config.MAX_SIZE:
        im_scale1 = float(config.MAX_SIZE) / max_shape
        im_scale *= im_scale1

    if im_scale != 1:
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return im, im_scale, c


def load_multislice_img_16bit_png(imname, slice_idx, slice_intv, do_clip, num_slice):
    data_cache = {}
    def _load_data(imname, delta=0):
        imname1 = get_slice_name(imname, delta)
        if imname1 not in data_cache.keys():
            data_cache[imname1] = cv2.imread(fullpath(imname1), -1)
            if data_cache[imname1] is None:
                print 'file reading error:', imname1
        return data_cache[imname1]

    im_cur = _load_data(imname)

    mask = get_mask(im_cur) if do_clip else None

    if config.SLICE_INTV == 0 or np.isnan(slice_intv) or slice_intv < 0:
        ims = [im_cur] * num_slice  # only use the central slice

    else:
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = float(config.SLICE_INTV) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
            for p in range((num_slice-1)/2):
                im_prev = _load_data(imname, - rel_pos * (p + 1))
                im_next = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
        else:
            for p in range((num_slice-1)/2):
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
    im = im.astype(np.float32, copy=False)-32768  # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit

    return im, mask


def get_slice_name(imname, delta=0):
    if delta == 0: return imname
    delta = int(delta)
    dirname, slicename = imname.split(os.sep)
    slice_idx = int(slicename[:-4])
    imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)

    while not os.path.exists(fullpath(imname1)):  # if the slice is not in the dataset, use its neighboring slice
        # print 'file not found:', imname1
        delta -= np.sign(delta)
        imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
        if delta == 0: break

    return imname1


def fullpath(imname):
    imname_full = os.path.join(default.image_path, imname)
    return imname_full


def load_multislice_img_nifti(imname, slice_idx, slice_intv, do_clip, num_slice):
    # nifti files store 3D volumes. If you do not need all slices in the volume, it may be slower than loading 2D png files
    # t.tic()
    vol = nib.load(default.image_path + imname).get_data()
    # tt = t.toc()
    # print '%s, %.2f'% (default.image_path + imname, tt)

    if do_clip:
        mask = get_mask(vol[:,:,slice_idx])
    else:
        mask = None

    im_cur = vol[:, :, slice_idx]

    if config.SLICE_INTV == 0 or np.isnan(slice_intv) or slice_intv < 0:
        ims = [im_cur] * num_slice

    else:
        max_slice = vol.shape[2] - 1
        ims = [im_cur]
        # linear interpolate
        rel_pos = float(config.SLICE_INTV) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:
            for p in range((num_slice-1)/2):
                im_prev = vol[:,:, int(slice_idx - rel_pos * (p + 1))]
                im_next = vol[:,:, int(slice_idx + rel_pos * (p + 1))]
                ims = [im_prev] + ims + [im_next]
        else:
            for p in range((num_slice-1)/2):
                intv1 = rel_pos*(p+1)
                slice1 = int(max(slice_idx - np.ceil(intv1), 0))
                slice2 = int(max(slice_idx - np.floor(intv1), 0))
                im_prev = a * vol[:,:,slice1] + b * vol[:,:,slice2]

                slice1 = int(min(slice_idx + np.ceil(intv1), max_slice))
                slice2 = int(min(slice_idx + np.floor(intv1), max_slice))
                im_next = a * vol[:, :, slice1] + b * vol[:, :, slice2]

                ims = [im_prev] + ims + [im_next]

    im = cv2.merge(ims)
    return im, mask


def windowing(im, win):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


# backward windowing
def windowing_rev(im, win):
    im1 = im.astype(float)/255
    im1 *= win[1] - win[0]
    im1 += win[0]
    return im1


def get_mask(im):
    # use a intensity threshold to roughly find the mask of the body
    th = 32000  # an approximate background intensity value
    mask = im > th
    mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
    # mask = binary_dilation(mask)
    # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

    if mask.sum() == 0:  # maybe atypical intensity
        mask = im * 0 + 1
    return mask.astype(dtype=np.int32)


def get_range(mask, margin=0):
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return u, d, l, r


def im_list_to_blob(ims, use_max_size=False):
    """Convert a list of images into a network input.
    """
    # max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # min_shape = np.array([im.shape for im in ims]).min(axis=0)
    # print max_shape, min_shape
    if use_max_size:
        max_shape = np.array([config.MAX_SIZE, config.MAX_SIZE])
    else:
        max_shape = np.array([im.shape for im in ims]).max(axis=0)

    num_images = len(ims)
    num_channel = ims[0].shape[2] if ims[0].ndim == 3 else 3
    blob = np.zeros((num_images, num_channel, max_shape[0], max_shape[1]),
                    dtype=np.float32)
    rois = np.zeros((num_images, 4))
    for i in xrange(num_images):
        im = ims[i]

        # # put images in the center
        # m = (max_shape - im.shape) / 2
        # rois[i, :] = np.array([m[1], m[0], m[1] + im.shape[1], m[0] + im.shape[0]])
        # if im.ndim == 2:
        # 	for chn in range(3):
        # 		blob[i, chn, m[0]:m[0] + im.shape[0], m[1]:m[1] + im.shape[1]] = im
        # elif im.ndim == 3:
        # 	blob[i, :, m[0]:m[0] + im.shape[0], m[1]:m[1] + im.shape[1]] = im.transpose((2, 0, 1))

        # put images on the corner
        rois[i, :] = np.array([0, 0, im.shape[1], im.shape[0]])
        if im.ndim == 2:
            for chn in range(num_channel):
                blob[i, chn, :im.shape[0], :im.shape[1]] = im
        elif im.ndim == 3:
            blob[i, :, :im.shape[0], :im.shape[1]] = im.transpose((2, 0, 1))

    return blob, rois


def map_box_back(boxes, cx=0, cy=0, im_scale=1.):
    boxes /= im_scale
    boxes[:, [0,2]] += cx
    boxes[:, [1,3]] += cy
    return boxes


def get_patch(im, box):
    # box = box0.copy()  # shouldn't change box0!
    # if spacing is not None and config.NORM_SPACING > 0:  # spacing adjust, will overwrite simple scaling
    #     im_scale = float(spacing) / config.NORM_SPACING
    #     box *= im_scale

    mg = config.GT_MARGIN
    if config.ROI_METHOD == 'FIXED_MARGIN' or config.ROI_METHOD == 'VAR_SIZE_FIXED_MARGIN':
        # method 1: crop real lesion size + margin. will pad zero for diff size patches
        box1 = np.round(box).astype(int)
        box1[0] = np.maximum(0, box1[0] - mg)
        box1[1] = np.maximum(0, box1[1] - mg)
        box1[2] = np.minimum(im.shape[1] - 1, box1[2] + mg)
        box1[3] = np.minimum(im.shape[0] - 1, box1[3] + mg)
        patch = im[box1[1]:box1[3] + 1, box1[0]:box1[2] + 1]

        offset_x = np.maximum(box[0] - mg, 0)
        offset_y = np.maximum(box[1] - mg, 0)
        box_new = box - np.array([offset_x, offset_y] * 2)

        if config.ROI_METHOD == 'FIXED_MARGIN':
            # handle diff size
            if box1[0] - mg < 0:
                patch = cv2.copyMakeBorder(patch, 0, 0, mg - box1[0], 0, cv2.BORDER_REPLICATE)
            elif box1[2] + mg > im.shape[1] - 1:
                patch = cv2.copyMakeBorder(patch, 0, 0, 0, box1[2] + mg - im.shape[1] + 1, cv2.BORDER_REPLICATE)
            elif box1[1] - mg < 0:
                patch = cv2.copyMakeBorder(patch, mg - box1[1], 0, 0, 0, cv2.BORDER_REPLICATE)
            elif box1[3] + mg > im.shape[0] - 1:
                patch = cv2.copyMakeBorder(patch, 0, box1[3] + mg - im.shape[0] + 1, 0, 0, cv2.BORDER_REPLICATE)

            im_scale = float(config.FM_PATCH_SCALE) / np.array(patch.shape[0:2])
            patch = cv2.resize(patch, None, None, fx=im_scale[1], fy=im_scale[0], interpolation=cv2.INTER_LINEAR)

    elif config.ROI_METHOD == 'FIXED_CONTEXT':
        # method 2: crop fixed size context, so no need to pad zeros
        center = np.round((box[:2] + box[2:]) / 2)
        box1 = np.zeros((4,), dtype=int)
        box1[0] = np.maximum(0, center[0] - mg)
        box1[1] = np.maximum(0, center[1] - mg)
        box1[2] = np.minimum(im.shape[1] - 1, center[0] + mg - 1)
        box1[3] = np.minimum(im.shape[0] - 1, center[1] + mg - 1)

        patch = im[box1[1]:box1[3] + 1, box1[0]:box1[2] + 1]

        # handle diff size
        xdiff = mg * 2 - patch.shape[1]
        ydiff = mg * 2 - patch.shape[0]
        if xdiff > 0:
            if center[0] - mg < 0:
                patch = cv2.copyMakeBorder(patch, 0, 0, xdiff, 0, cv2.BORDER_REPLICATE)
            else:
                patch = cv2.copyMakeBorder(patch, 0, 0, 0, xdiff, cv2.BORDER_REPLICATE)
        if ydiff > 0:
            if center[1] - mg < 0:
                patch = cv2.copyMakeBorder(patch, ydiff, 0, 0, 0, cv2.BORDER_REPLICATE)
            else:
                patch = cv2.copyMakeBorder(patch, 0, ydiff, 0, 0, cv2.BORDER_REPLICATE)

    return patch.copy(), box_new

