# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Procedure in the batch mode"""
import os
import numpy as np
from time import time
import torch
import nibabel as nib
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from maskrcnn.config import cfg
from maskrcnn.structures.image_list import to_image_list
from maskrcnn.data.datasets.evaluation.DeepLesion.post_process import post_process_results
from maskrcnn.data.datasets.load_ct_img import map_box_back, windowing
from .demo_process import import_tag_data, load_preprocess_nifti, get_ims
from maskrcnn.utils.draw import draw_results


def batch_exec_model(model):
    """test model on user-provided folder of data, instead of the preset DeepLesion dataset"""
    import_tag_data()
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)

    info = "Please input the path which contains all nifti CT volumes to predict in batch >> "
    while True:
        path = input(info)
        if not os.path.exists(path):
            print('folder does not exist!')
            continue
        else:
            break

    nifti_paths = []
    for dirName, subdirList, fileList in os.walk(path):
        print('found directory: %s' % dirName)
        for fname in fileList:
            if fname.endswith('.nii.gz') or fname.endswith('.nii'):
                nifti_paths.append(os.path.join(path, dirName, fname))
    print('%d nifti files found' % len(nifti_paths))

    total_time = 0
    results = {}
    total_slices = 0
    for file_idx, nifti_path in enumerate(nifti_paths):
        print('(%d/%d) %s' % (file_idx+1, len(nifti_paths), nifti_path))
        print('reading image ...')
        try:
            nifti_data = nib.load(nifti_path)
        except:
            print('load nifti file error!')
            continue

        vol, spacing, slice_intv = load_preprocess_nifti(nifti_data)
        slice_num_per_run = max(1, int(float(cfg.TEST.TEST_SLICE_INTV_MM)/slice_intv+.5))
        num_total_slice = vol.shape[2]
        results[nifti_path] = {}

        slices_to_process = range(int(slice_num_per_run/2), num_total_slice, slice_num_per_run)
        total_slices += len(slices_to_process)
        for slice_idx in tqdm(slices_to_process):
            ims, im_np, im_scale, crop = get_ims(slice_idx, vol, spacing, slice_intv)
            im_list = to_image_list(ims, cfg.DATALOADER.SIZE_DIVISIBILITY).to(device)
            start_time = time()
            with torch.no_grad():
                result = model(im_list)
            result = [o.to("cpu") for o in result]

            info = {'spacing': spacing, 'im_scale': im_scale, 'crop': crop}
            post_process_results(result[0], info)
            result = sort_results_for_batch(result[0], info)
            results[nifti_path][slice_idx] = result
            total_time += time() - start_time

            # # sanity check
            # im = vol[:, :, slice_idx].astype(float) - 32768
            # im = windowing(im, [-175, 275]).astype('uint8')
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            # overlay, msgs = draw_results(im, np.array(result['boxes']), scores=np.array(result['scores']),
            #                              tag_scores=np.array(result['tag_scores']), tag_predictions=np.array(result['tag_scores'])>.5,
            #                              contours=np.array(result['contour_mm']))
            # plt.imshow(overlay)
            # print(msgs)
            # plt.show()

    output_dir = os.path.join(cfg.RESULTS_DIR)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_fn = os.path.join(output_dir, '%s.pth' % os.path.basename(path))
    torch.save(results, output_fn)
    print('result images and text saved to', output_fn)
    print('processing time: %d ms per slice' % int(1000.*total_time/total_slices))


def sort_results_for_batch(result, info):
    """Organize prediction results to a dict"""
    # all units in pixel of the original image
    boxes = map_box_back(result.bbox, info['crop'][2], info['crop'][0], info['im_scale'])
    res_new = {'boxes': boxes.tolist()}
    for key in cfg.TEST.RESULT_FIELDS:
        val = result.get_field(key)
        if key == 'contour_mm':
            contours = []
            for p in range(val.shape[0]):
                contour = val[p, val[p,:,0]>0, :] / info['spacing'] + torch.Tensor([info['crop'][2], info['crop'][0]])
                contours.append(contour.tolist())
            val = contours
            res_new[key] = val
            continue
        elif key == 'recist_mm':
            val = val / info['spacing'] + torch.Tensor([info['crop'][2], info['crop'][0]] * 4)  # in pixel
        elif key == 'diameter_mm':
            val = val / info['spacing']  # in pixel
        res_new[key] = val.tolist()
    return res_new

