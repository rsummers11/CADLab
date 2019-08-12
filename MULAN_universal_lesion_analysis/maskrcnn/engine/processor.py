# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Procedures for training and testing"""
import os
import numpy as np
import logging
from time import time
import torch

from maskrcnn.data import make_data_loader, make_datasets
from maskrcnn.solver import make_optimizer
from maskrcnn.engine.inference import inference
from maskrcnn.engine.trainer import do_train
from maskrcnn.modeling.detector import build_detection_model
from maskrcnn.utils.checkpoint import DetectronCheckpointer
from maskrcnn.utils.comm import synchronize, get_rank
from maskrcnn.utils.miscellaneous import mkdir
from maskrcnn.config import cfg


def train_model():
    logger = logging.getLogger('maskrcnn.train')
    datasets = make_datasets('train')
    logger.info('building model ...')
    model = build_detection_model()  # some model parameters rely on initialization of the dataset
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if cfg.runtime_info.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.runtime_info.local_rank], output_device=cfg.runtime_info.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    if cfg.MODE in ('train',):
        optimizer = make_optimizer(model)
        arguments = {}
        arguments['start_epoch'] = cfg.BEGIN_EPOCH
        arguments['max_epoch'] = cfg.SOLVER.MAX_EPOCH
    else:
        optimizer = None

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(model, optimizer, scheduler=None, save_dir=cfg.CHECKPT_DIR,
                                         prefix=cfg.EXP_NAME, save_to_disk=save_to_disk)

    if cfg.BEGIN_EPOCH == 0:
        if not cfg.MODEL.INIT_FROM_PRETRAIN:
            logger.info('No pretrained weights')
        else:
            # extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
            model.backbone.load_pretrained_weights()
    else:
        name = checkpointer.get_save_name(cfg.BEGIN_EPOCH, prefix=cfg.FINETUNE_FROM)
        extra_checkpoint_data = checkpointer.load(name)
        # if cfg.MODE in ('train',):
        #     arguments.update(extra_checkpoint_data)
            # scheduler.milestones = ori_milestones  # in case step is adjusted and diff from in the checkpt

    if not cfg.MODE in ('train',):
        return model, checkpointer

    if cfg.MODEL.BACKBONE.FREEZE:
        model.backbone.freeze()

    data_loader = make_data_loader(
        datasets,
        is_train=True,
        is_distributed=cfg.runtime_info.distributed,
    )

    do_train(
        model,
        data_loader,
        optimizer,
        checkpointer,
        device,
        test_model,
        arguments,
    )

    return model, checkpointer


def test_model(model, is_validation):
    if cfg.runtime_info.distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps

    if is_validation:
        old_mode = cfg.MODE
        old_eval_gt = cfg.TEST.EVAL_SEG_TAG_ON_GT
        cfg.MODE = "eval"  # use val set to compute tag thresholds, so no visualize
        dataset_names = cfg.DATASETS.VAL
        stage = 'val'
        if cfg.MODEL.TAG_ON:
            cfg.runtime_info.tag_sel_val = cfg.TEST.TAG.SELECTION_VAL
            cfg.TEST.EVAL_SEG_TAG_ON_GT = True  # use val set to compute tag thresholds, so should compute on gt
    else:
        # if cfg.TEST.TAG.CALIBRATE_TH, cfg.runtime_info.tag_sel_val should be computed in DL_eval
        if cfg.MODE in ('vis',):
            cfg.TEST.USE_SAVED_PRED_RES = 'none'
        dataset_names = cfg.DATASETS.TEST
        stage = 'test'

    output_folders = [None] * len(dataset_names)
    if cfg.RESULTS_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.RESULTS_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    datasets = make_datasets(stage)
    data_loaders = make_data_loader(datasets, is_train=False, is_distributed=cfg.runtime_info.distributed)
    logger = logging.getLogger('maskrcnn.test')
    t = time()
    for output_folder, dataset_name, data_loader in zip(output_folders, dataset_names, data_loaders):
        acc = inference(
            model,
            data_loader,
            dataset_name=dataset_name,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
            is_validation=is_validation
        )
        synchronize()
    logger.info('total test time: %4f\n', time()-t)

    if is_validation:
        cfg.MODE = old_mode
        cfg.TEST.EVAL_SEG_TAG_ON_GT = old_eval_gt

    return acc
