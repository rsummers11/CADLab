# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize
from maskrcnn.config import cfg
from maskrcnn.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn.data.datasets.evaluation.DeepLesion.post_process import post_process_results
from maskrcnn.data.datasets.evaluation.DeepLesion.DL_eval import do_evaluation
from maskrcnn.utils.draw import visualize


def compute_on_dataset(model, data_loader, device, logger):
    """Inference and save predictions to results_dict (or visualize)"""
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    total_time = 0.

    if cfg.MODE in ('vis',):
        mask_threshold = -1 if cfg.TEST.VISUALIZE.SHOW_MASK_HEATMAPS else cfg.TEST.MASK.THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)

    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, infos = batch
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.no_grad():
            start_time = time.time()
            outputs = model(images, targets, infos)
            total_time += time.time() - start_time
            outputs = [o.to(cpu_device) for o in outputs]
            if cfg.MODEL.USE_3D_FUSION:
                num_image = cfg.INPUT.NUM_IMAGES_3DCE
                images = images[int(num_image / 2)::num_image]  # only keep central ones

            for p in range(len(infos)):
                d = {'target': targets[p], 'result': outputs[p], 'info': infos[p]}
                results_dict.update({infos[p]['image_fn']: d})
                # visualization
                if cfg.MODE in ('vis',):
                    post_process_results(outputs[p], infos[p])
                    sz = images.image_sizes[p]
                    im = images.tensors[p].cpu().numpy()[1, : sz[0], : sz[1]]
                    visualize(im, targets[p], outputs[p], infos[p], masker)

    assert cfg.MODE not in ('vis',), 'all test images have been visualized!'
    logger.info('Total forwarding time per image: %.4f s', total_time / len(data_loader.dataset))
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if isinstance(image_ids[-1], int) and len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    # predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        is_validation,
        device="cuda",
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn.inference")
    dataset = data_loader.dataset

    stage = 'val' if is_validation else 'test'
    predictions = None
    if cfg.TEST.USE_SAVED_PRED_RES == 'raw':  # load saved results without postprocess to save time
        res_save_name = os.path.join(output_folder, cfg.EXP_NAME + '_' + stage + "_raw_predictions.pth")
        if os.path.exists(res_save_name):
            predictions = torch.load(res_save_name)
            logger.info('loaded saved predictions in from %s', res_save_name)
        logger.info('postprocessing: generating tag predictions, mask contours, RECIST measurements.')
        for d in tqdm(predictions.values()):
            post_process_results(d['result'], d['info'])

    elif cfg.TEST.USE_SAVED_PRED_RES == 'proc':  # load saved results with postprocess to save time
        res_save_name = os.path.join(output_folder, cfg.EXP_NAME + '_' + stage + "_processed_predictions.pth")
        if os.path.exists(res_save_name):
            predictions = torch.load(res_save_name)
            logger.info('loaded saved predictions in from %s', res_save_name)

    # if no previous results are loaded, do inference
    if predictions is None:
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
        start_time = time.time()
        predictions = compute_on_dataset(model, data_loader, device, logger)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        if not is_main_process():
            return

        # res_save_name = os.path.join(output_folder, cfg.EXP_NAME + '_' + stage + "_raw_predictions.pth")
        # if output_folder:
        #     torch.save(predictions, res_save_name)

        # for DeepLesion
        logger.info('postprocessing: generating tag predictions, mask contours, RECIST measurements.')
        for d in tqdm(predictions.values()):
            post_process_results(d['result'], d['info'])

        res_save_name = os.path.join(output_folder, cfg.EXP_NAME + '_' + stage + "_processed_predictions.pth")
        if output_folder:
            torch.save(predictions, res_save_name)

    return do_evaluation(dataset, predictions, is_validation)