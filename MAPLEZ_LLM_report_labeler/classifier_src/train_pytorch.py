# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Use `python <python_file> --help` to check settings and hyperparameters
# This script needs the correct location of several dataset files, as following:
# report_labels_mimic_0.0_llama2_all_tasks_generic_1_30_fixed.csv
# report_labels_mimic_0.0_llama2_all_tasks_1_30_fixed.csv
# vqa_dataset_converted.csv
# reflacx_dataset_converted.csv
# train_df.csv
# train_df_all.csv
# val_df.csv
# val_df_all.csv
# test_df.csv
# test_df_all.csv
# MIMIC-CXR images in ./MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/files/
# ./datasets/chexpert_test/groundtruth.csv
# ./datasets/chexpert_test/CheXpert/val_labels.csv
# CheXpert images in ./datasets/chexpert_test/val/ or ./datasets/chexpert_test/test/
# ./datasets/nih_chestxray14/Data_Entry_2017_v2020.csv
# pneumothorax_relabeled_dataset_converted.csv
# pneumonia_relabeled_dataset_converted.csv
# train_nih_labeled.csv.csv
# val_nih_labeled.csv.csv
# test_nih_labeled.csv.csv
# image_1_image_2_nih_conversion.csv
# NIH ChestXray14 dataset images in ./datasets/nih_chestxray14/images/
# Description:
# Script used to train and evaluate the classification models
# File substantially modified from https://github.com/pytorch/vision/blob/main/references/classification/train.py

import datetime
import os
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix
import socket
import output as outputs
import numpy as np
from random import randint
import model_mine as models
from list_labels import str_labels_mimic

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, output, args, metric_logger, model_ema=None, scaler=None, log_suffix=""):
    model.train()

    header = f"Epoch: [{epoch}]"
    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        (image, target, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, vqa_new_gt, \
             vqa_severities, vqa_location_labels, vqa_location_vector_index, vqa_probabilities, reflacx_new_gt, reflacx_probabilities, reflacx_present) = batch
        if not args.ignore_comparison_uncertainty:
            unchanged_uncertainties = unchanged_uncertainties*0
        if args.invert_image:
            image = 1- image
        if i==0:
            output.save_image(image, 'image_train', epoch)
        start_time = time.time()
        image, target, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, \
            vqa_new_gt, vqa_severities, vqa_location_labels, vqa_location_vector_index, vqa_probabilities = \
            image.to(device), target.to(device), new_gt.to(device), \
            severities.to(device), location_labels.to(device), \
            location_vector_index.to(device), probabilities.to(device), \
            unchanged_uncertainties.to(device), vqa_new_gt.to(device), \
            vqa_severities.to(device), vqa_location_labels.to(device), \
            vqa_location_vector_index.to(device), vqa_probabilities.to(device)
        if args.labeler == 'llm':
            target_to_use = new_gt
            location_label_to_use = location_labels
            location_vector_to_use = location_vector_index
            severity_to_use = severities
            probability_to_use = probabilities
        elif args.labeler == 'vqa':
            target_to_use = vqa_new_gt
            location_label_to_use = vqa_location_labels
            location_vector_to_use = vqa_location_vector_index
            severity_to_use = vqa_severities
            probability_to_use = vqa_probabilities
        elif args.labeler == 'chexpert':
            target_to_use = target
            location_label_to_use = None
            location_vector_to_use = None
            severity_to_use = None
            probability_to_use = None

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            model_output = model(image)
            loss = criterion(model_output, target_to_use, severity_to_use, location_label_to_use, location_vector_to_use, probability_to_use, unchanged_uncertainties)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        batch_size = image.shape[0]
        
        for index_class in range(target.shape[1]):
            allowed_indices = (1-unchanged_uncertainties[:,index_class][:,None]).bool()[:,0]
            metric_logger.meters[f"output_{log_suffix}_{index_class}"].update(model_output[index_class][0][allowed_indices].detach().cpu().numpy())
            this_target = target[:, index_class]
            this_target[this_target==-2] = 0
            this_target[this_target==-1] = 1
            metric_logger.meters[f"target_{log_suffix}_{index_class}"].update(this_target[allowed_indices].detach().cpu().numpy())
        metric_logger.meters[f"loss_{log_suffix}"].update(loss.item())
        metric_logger.meters[f"lr"].update(optimizer.param_groups[0]["lr"])
        metric_logger.meters[f"img/s"].update(batch_size / (time.time() - start_time))

def evaluate(model, criterion, data_loader, data_loader_test_chexpert_images, data_loader_test_nih_images, device, epoch, output , metric_logger, print_freq=100, log_suffix=""):
    model.eval()
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for i, batch in enumerate(data_loader_test_chexpert_images):
            image, target = batch
            if args.invert_image:
                image = 1- image
            if i==0:
                output.save_image(image, 'image_val_chexpertdataset', epoch)
            image, target = \
                image.to(device, non_blocking=True), target.to(device, non_blocking=True), 
            model_output = model(image)
            for index_class in range(target.shape[1]):
                metric_logger.meters[f"output_chexpertdataset_{log_suffix}_{index_class}"].update(model_output[index_class][0].detach().cpu().numpy())    
                this_target = target[:, index_class]
                this_target[this_target==-2] = 0
                this_target[this_target==-1] = args.uncertainty_label
                metric_logger.meters[f"target_chexpertdataset_{log_suffix}_{index_class}"].update(this_target.detach().cpu().numpy())

        for i, batch in enumerate(data_loader_test_nih_images):
            image,pneumothorax_labels, pneumothorax_label_present, pneumonia_labels, pneumonia_label_present = batch
            if args.invert_image:
                image = 1- image
            if i==0:
                output.save_image(image, 'image_val_nihdataset', epoch)
            image, pneumothorax_labels, pneumothorax_label_present, pneumonia_labels, pneumonia_label_present = \
                image.to(device, non_blocking=True), pneumothorax_labels.to(device, non_blocking=True), \
                pneumothorax_label_present.to(device, non_blocking=True), pneumonia_labels.to(device, non_blocking=True), \
                pneumonia_label_present.to(device, non_blocking=True)
            model_output = model(image)
            for index_class in range(target.shape[1]):
                if index_class==str_labels_mimic.index('Pneumothorax'):
                    if (pneumothorax_label_present).sum()>0:
                        metric_logger.meters[f"output_nihpneumothorax_{log_suffix}"].update(model_output[index_class][0][(pneumothorax_label_present).bool()].detach().cpu().numpy())
                        metric_logger.meters[f"target_nihpneumothorax_{log_suffix}"].update(pneumothorax_labels[(pneumothorax_label_present).bool()].detach().cpu().numpy())                
                if index_class==str_labels_mimic.index('Consolidation'):
                    if (pneumonia_label_present).sum()>0:
                        metric_logger.meters[f"output_nihpneumonia_{log_suffix}"].update(model_output[index_class][0][(pneumonia_label_present).bool()].detach().cpu().numpy())
                        metric_logger.meters[f"target_nihpneumonia_{log_suffix}"].update(pneumonia_labels[(pneumonia_label_present).bool()].detach().cpu().numpy())       

        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            (image, target, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, vqa_new_gt,\
             vqa_severities, vqa_location_labels, vqa_location_vector_index, vqa_probabilities, reflacx_new_gt, reflacx_probabilities, reflacx_present) = batch
            if not args.ignore_comparison_uncertainty:
                unchanged_uncertainties = unchanged_uncertainties*0
            if args.invert_image:
                image = 1- image
            if i==0:
                output.save_image(image, 'image_val', epoch)
            image, target, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, \
                vqa_new_gt, vqa_severities, vqa_location_labels, vqa_location_vector_index, vqa_probabilities, reflacx_new_gt, \
                reflacx_probabilities, reflacx_present = \
                image.to(device, non_blocking=True), target.to(device, non_blocking=True), new_gt.to(device, non_blocking=True), \
                severities.to(device, non_blocking=True), location_labels.to(device, non_blocking=True), \
                location_vector_index.to(device, non_blocking=True), probabilities.to(device, non_blocking=True), \
                unchanged_uncertainties.to(device, non_blocking=True), vqa_new_gt.to(device, non_blocking=True), \
                vqa_severities.to(device, non_blocking=True), vqa_location_labels.to(device, non_blocking=True), \
                vqa_location_vector_index.to(device, non_blocking=True), vqa_probabilities.to(device, non_blocking=True), \
                reflacx_new_gt.to(device, non_blocking=True), reflacx_probabilities.to(device, non_blocking=True), \
                reflacx_present.to(device, non_blocking=True)
            if args.labeler == 'llm':
                target_to_use = new_gt
                location_label_to_use = location_labels
                location_vector_to_use = location_vector_index
                severity_to_use = severities
                probability_to_use = probabilities
            elif args.labeler == 'vqa':
                target_to_use = vqa_new_gt
                location_label_to_use = vqa_location_labels
                location_vector_to_use = vqa_location_vector_index
                severity_to_use = vqa_severities
                probability_to_use = vqa_probabilities
            elif args.labeler == 'chexpert':
                target_to_use = target
                location_label_to_use = None
                location_vector_to_use = None
                severity_to_use = None
                probability_to_use = None
        
            model_output = model(image)
            loss = criterion(model_output, target_to_use, severity_to_use, location_label_to_use, location_vector_to_use, probability_to_use, unchanged_uncertainties)

            batch_size = image.shape[0]
            metric_logger.meters[f"loss_{log_suffix}"].update(loss.item())
            for index_class in range(target.shape[1]):
                allowed_indices = (1-unchanged_uncertainties[:,index_class][:,None]).bool()[:,0]
                metric_logger.meters[f"output_{log_suffix}_{index_class}"].update(model_output[index_class][0][allowed_indices].detach().cpu().numpy())
                
                if (reflacx_present).sum()>0:
                    metric_logger.meters[f"output_{log_suffix}_reflacx_{index_class}"].update(model_output[index_class][0][(reflacx_present).bool()].detach().cpu().numpy())
                for val_labeler in args.val_labeler:
                    if val_labeler=='reflacx':
                        this_target = reflacx_new_gt[(reflacx_present).bool(), :]
                        if len(this_target)==0:
                            continue
                    elif val_labeler == 'llm':
                        this_target = new_gt[allowed_indices, :]
                    elif val_labeler == 'vqa':
                        this_target = vqa_new_gt[allowed_indices, :]
                    elif val_labeler == 'chexpert':
                        this_target = target[allowed_indices, :]
        
                    this_target = this_target[:, index_class]
                    this_target[this_target==-3] = 0
                    this_target[this_target==-2] = 0
                    this_target[this_target==-1] = args.uncertainty_label
                    metric_logger.meters[f"target_{log_suffix}_{val_labeler}_{index_class}"].update(this_target.detach().cpu().numpy())
            num_processed_samples += batch_size
        
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

from sklearn.metrics import roc_auc_score

def bootstrap(score_fn, gt_vector, pred_vector):
    bootstrapped_scores = []
    for i in range(200):
        rng = np.random.default_rng(seed=i)
        sampled_gt = rng.choice(gt_vector, gt_vector.shape[0])
        rng = np.random.default_rng(seed=i)
        sampled_pred = rng.choice(pred_vector, pred_vector.shape[0])
        score = score_fn(sampled_gt, sampled_pred)
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    # confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    # confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    average = sorted_scores[int(0.5 * len(sorted_scores))]
    return average #f'{average} [{confidence_lower},{confidence_upper}]'

def auc_metric(metric_logger, target_column, output_column):
    if len(metric_logger.meters[target_column])==0:
        return None
    if metric_logger.meters[target_column].deque.std()==0:
        return None
    return roc_auc_score((metric_logger.meters[target_column].deque>0.)*1., metric_logger.meters[output_column].deque, average=None)
    
    # return bootstrap(lambda x,y: roc_auc_score(x,y, average=None),(metric_logger.meters[target_column].deque>0.)*1., metric_logger.meters[output_column].deque)

def suffix_metric_average(metric_logger, auc_column, n_metrics):
    metric_in_all_indices = []
    for index_metric in range(n_metrics):
        if metric_logger.meters[f'{auc_column}_{index_metric}'] is None:
            return None
        metric_in_all_indices.append(metric_logger.meters[f'{auc_column}_{index_metric}'])
    # print(metric_logger.meters[auc_column])
    return sum(metric_in_all_indices)/len(metric_in_all_indices)

def main(args):
    start_time = time.time()
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    output = outputs.Outputs(args, args.output_dir)
    output.save_run_state(os.path.dirname(__file__))

    last_best_validation_metric = args.initialization_comparison

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    
    from mimic_dataset import get_dataset, get_chexpert_by_split, get_nih_by_split
    if args.skip_train:
        data_loader, train_sampler = None, None
    else:
        data_loader, train_sampler = get_dataset(split='train', h5_filename = f'joint_dataset_mimic_noseg_{"paerect" if not args.include_ap else "all"}_new_labels_included_8_labels{"_generic" if args.do_generic else ""}', args = args)
    data_loader_test, _ = get_dataset(split=args.split_validation, h5_filename = f'joint_dataset_mimic_noseg_{"paerect" if not args.include_ap else "all"}_new_labels_included_8_labels{"_generic" if args.do_generic else ""}', args = args)
    data_loader_test_chexpert_images, _ = get_dataset(split=args.split_validation, args = args, h5_filename = 'joint_dataset_chexpert_images', fn_create_dataset = get_chexpert_by_split)
    data_loader_test_nih_images, _ = get_dataset(split=args.split_validation, args = args, h5_filename = 'joint_dataset_nih_images_test', fn_create_dataset = get_nih_by_split)

    print("Creating model")
    model = models.get_model(args)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = models.loss_ce(args)
    # nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.start_epoch >= args.epochs_only_last_layer:
            parameters_to_add = []
            for name, param in model.named_parameters():
                param.requires_grad = True
                if f'module.{args.name_last_parameters}' not in name:
                    # parameters_to_add.append(param)
                    optimizer.param_groups[0]['params'].append(param)
            # optimizer.add_param_group({'params': parameters_to_add})
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
    
    metric_logger = utils.MetricLogger(delimiter="  ", start_time_script = start_time)
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    for index_class in range(args.num_classes):
        metric_logger.add_meter(f"target_train_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"target_chexpertdataset_val_{index_class}", utils.MeterValue(window_size=None))
        for val_labeler in args.val_labeler:
            metric_logger.add_meter(f"target_val_{val_labeler}_{index_class}", utils.MeterValue(window_size=None))
            metric_logger.add_meter(f"target_ema_{val_labeler}_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"output_train_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"output_val_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"output_chexpertdataset_val_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"output_ema_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"output_train_reflacx_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"output_val_reflacx_{index_class}", utils.MeterValue(window_size=None))
        metric_logger.add_meter(f"output_ema_reflacx_{index_class}", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"target_nihpneumothorax_val", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"target_nihpneumonia_val", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"output_nihpneumothorax_ema", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"output_nihpneumonia_ema", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"target_nihpneumothorax_ema", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"target_nihpneumonia_ema", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"output_nihpneumothorax_val", utils.MeterValue(window_size=None))
    metric_logger.add_meter(f"output_nihpneumonia_val", utils.MeterValue(window_size=None))
    metric_logger.add_meter("loss_train", utils.SmoothedValue(window_size=None, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("loss_val", utils.SmoothedValue(window_size=None, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("loss_ema", utils.SmoothedValue(window_size=None, fmt="{global_avg:.4f}"))

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, data_loader_test_chexpert_images, data_loader_test_nih_images, device=device, epoch = -1, output = output, metric_logger = metric_logger, log_suffix="ema", print_freq = args.print_freq)
        else:
            evaluate(model, criterion, data_loader_test, data_loader_test_chexpert_images, data_loader_test_nih_images , device=device, epoch = -1, output = output, metric_logger = metric_logger, print_freq = args.print_freq, log_suffix="val")
        metric_logger.synchronize_between_processes()
        for index_class in range(args.num_classes):
            output.save_model_outputs(metric_logger.meters[f"target_chexpertdataset_val_{index_class}"].deque, metric_logger.meters[ f"output_chexpertdataset_val_{index_class}"].deque, f'val_chexpertdataset_{index_class}')
            metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_chexpertdataset_val_{index_class}", f"output_chexpertdataset_val_{index_class}"), f'auc_chexpertdataset_val_{index_class}')
            metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_chexpertdataset_ema_{index_class}", f"output_chexpertdataset_ema_{index_class}"), f'auc_chexpertdataset_ema_{index_class}')
            for val_labeler in args.val_labeler:
                output.save_model_outputs(metric_logger.meters[f'target_val_{val_labeler}_{index_class}'].deque, metric_logger.meters[f"output_val{'' if val_labeler!='reflacx' else '_reflacx'}_{index_class}"].deque, f'val_{val_labeler}_{index_class}')
                metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f'target_val_{val_labeler}_{index_class}', f"output_val{'' if val_labeler!='reflacx' else '_reflacx'}_{index_class}"), f'auc_val_{val_labeler}_{index_class}')
                metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f'target_ema_{val_labeler}_{index_class}', f"output_ema{'' if val_labeler!='reflacx' else '_reflacx'}_{index_class}"), f'auc_ema_{val_labeler}_{index_class}')
        output.save_model_outputs(metric_logger.meters[f"target_nihpneumothorax_val"].deque, metric_logger.meters[ f"output_nihpneumothorax_val"].deque, 'nihpneumothorax_val')
        output.save_model_outputs(metric_logger.meters[ f"target_nihpneumonia_val"].deque, metric_logger.meters[ f"output_nihpneumonia_val"].deque, 'nihpneumonia_val')
        metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumothorax_val", f"output_nihpneumothorax_val"), f'auc_nihpneumothorax_val')
        metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumothorax_ema", f"output_nihpneumothorax_ema"), f'auc_nihpneumothorax_ema')
        metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumonia_val", f"output_nihpneumonia_val"), f'auc_nihpneumonia_val')
        metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumonia_ema", f"output_nihpneumonia_ema"), f'auc_nihpneumonia_ema')
        metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_chexpertdataset_val', args.num_classes), f'auc_chexpertdataset_val_average')
        metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_chexpertdataset_ema', args.num_classes), f'auc_chexpertdataset_ema_average')
        for val_labeler in args.val_labeler:
            metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_val_{val_labeler}', args.num_classes), f'auc_val_{val_labeler}_average')
            metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_ema_{val_labeler}', args.num_classes), f'auc_ema_{val_labeler}_average')
        
        output.log_added_values_pytorch(0, metric_logger)
        return
    else:
        print("Start training")
        
        for epoch in range(args.start_epoch, args.epochs):
            print('Epoch:', epoch)
            if epoch == args.epochs_only_last_layer:
                parameters_to_add = []
                for name, param in model.named_parameters():
                    param.requires_grad = True
                    if f'module.{args.name_last_parameters}' not in name:
                        # parameters_to_add.append(param)
                        optimizer.param_groups[0]['params'].append(param)
                # optimizer.add_param_group({'params': parameters_to_add})
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, output, args, metric_logger, model_ema, scaler, log_suffix="train")
            lr_scheduler.step()
            evaluate(model, criterion, data_loader_test, data_loader_test_chexpert_images, data_loader_test_nih_images, device=device, epoch = epoch, output = output, metric_logger=metric_logger, print_freq = args.print_freq, log_suffix="val")
            if model_ema:
                evaluate(model_ema, criterion, data_loader_test, data_loader_test_chexpert_images, data_loader_test_nih_images, device=device, epoch = epoch, output = output, metric_logger=metric_logger, print_freq = args.print_freq, log_suffix="ema")
            metric_logger.synchronize_between_processes()
            for index_class in range(args.num_classes):
                metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_chexpertdataset_val_{index_class}", f"output_chexpertdataset_val_{index_class}"), f'auc_chexpertdataset_val_{index_class}')
                metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_chexpertdataset_ema_{index_class}", f"output_chexpertdataset_ema_{index_class}"), f'auc_chexpertdataset_ema_{index_class}')
                
                for val_labeler in args.val_labeler:
                    metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f'target_val_{val_labeler}_{index_class}', f"output_val{'' if val_labeler!='reflacx' else '_reflacx'}_{index_class}"), f'auc_val_{val_labeler}_{index_class}')
                    metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f'target_ema_{val_labeler}_{index_class}', f"output_ema{'' if val_labeler!='reflacx' else '_reflacx'}_{index_class}"), f'auc_ema_{val_labeler}_{index_class}')
            metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumothorax_val", f"output_nihpneumothorax_val"), f'auc_nihpneumothorax_val')
            metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumothorax_ema", f"output_nihpneumothorax_ema"), f'auc_nihpneumothorax_ema')
            metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumonia_val", f"output_nihpneumonia_val"), f'auc_nihpneumonia_val')
            metric_logger.apply_function_to_metrics(lambda x: auc_metric(x, f"target_nihpneumonia_ema", f"output_nihpneumonia_ema"), f'auc_nihpneumonia_ema')
            metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_chexpertdataset_val', args.num_classes), f'auc_chexpertdataset_val_average')
            metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_chexpertdataset_ema', args.num_classes), f'auc_chexpertdataset_ema_average')
            for val_labeler in args.val_labeler:
                metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_val_{val_labeler}', args.num_classes), f'auc_val_{val_labeler}_average')
                metric_logger.apply_function_to_metrics(lambda x: suffix_metric_average(x, f'auc_ema_{val_labeler}', args.num_classes), f'auc_ema_{val_labeler}_average')
            
            output.log_added_values_pytorch(epoch, metric_logger)
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                }
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                this_validation_metric = metric_logger.meters[args.metric_to_validate]
                if args.function_to_compare_validation_metric(this_validation_metric,last_best_validation_metric):
                    last_best_validation_metric = this_validation_metric
                    utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_best_epoch.pth"))
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint_last_epoch.pth"))

            metric_logger_2 = utils.MetricLogger(delimiter="  ", start_time_script = None)
            metric_logger_2.meters[args.metric_to_validate + '_best'] = last_best_validation_metric
            output.log_added_values_pytorch(epoch, metric_logger_2)
            metric_logger.reset_meters()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--model", default="shufflenet", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=600, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.5, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.00002,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=0.0,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.1, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./runs/", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm", type=str2bool, default="false"
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model", type=str2bool, default="false"
    )
    
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.1, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", type=str2bool, default="false", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", type=str2bool, default="false", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", type=str2bool, default="false", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", type=str2bool, default="false", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=4, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", type=str2bool, default="false", help="Use V2 transforms")

    parser.add_argument("--load_to_memory", type=str2bool, default="false", help="")
    parser.add_argument("--use_data_aug", type=str2bool, default="true", help="")
    parser.add_argument("--use_old_aug", type=str2bool, default="false", help="")
    parser.add_argument("--split_validation", default='val', type=str, help="")
    parser.add_argument("--adjust_lr", type=str2bool, default="true", help="")
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--epochs_only_last_layer", default=4, type=int, help="")
    parser.add_argument("--pad", type=str2bool, default="true", help="")
    parser.add_argument("--invert_image", type=str2bool, default="false", help="")

    parser.add_argument("--use_hard_labels", type=str2bool, default="true", help="")
    parser.add_argument("--presence_loss_type", type=str, default="ce", help="")
    parser.add_argument("--severity_loss_type", type=str, default="ce", help="")
    parser.add_argument("--severity_loss_multiplier", type=float, default=0, help="")
    parser.add_argument("--location_loss_multiplier", type=float, default=0, help="")
    parser.add_argument("--n_hidden_neurons_in_heads", type=int, default=256, help="")
    parser.add_argument("--labeler", type=str, default='llm', choices = ['llm', 'vqa', 'chexpert'], help="")
    parser.add_argument("--ignore_comparison_uncertainty", type=str2bool, default='true', help="")
    parser.add_argument("--uncertainty_label", default=1, type=int, help="")
    parser.add_argument("--label_smoothing", default=0, type=float, help="")
    parser.add_argument("--severity_smoothing", default=0, type=float, help="")
    parser.add_argument("--location_smoothing", default=0, type=float, help="")
    parser.add_argument("--include_ap", type=str2bool, default="false", help="")
    parser.add_argument("--share_first_classifier_layer", type=str2bool, default="false", help="")
    parser.add_argument("--do_generic", type=str2bool, default="false", help="")
    parser.add_argument("--stable_probability", default=50, type=float, help="")
    
    # lr-warmup-epochs 2
    args = parser.parse_args()
    import torch
    args.pytorch_version = torch.__version__ 
    import torchvision
    args.torchvision_version = torchvision.__version__

    args.num_classes = len(str_labels_mimic)
    if args.split_validation=='val':
        args.val_labeler = ['llm', 'vqa', 'chexpert']
    else:
        args.val_labeler = ['llm', 'vqa', 'chexpert', 'reflacx']

    args.skip_train = args.test_only
    if args.adjust_lr:
        args.lr = (args.batch_size*1)/1024*args.lr
    
    if args.model=='v2_s':
        args.weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        args.resolution = 384
        args.name_last_parameters = 'classifier'
    if args.model=='v2_m':
        args.weights = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        args.resolution = 480
        args.name_last_parameters = 'classifier'
    if args.model=='v2_l':
        args.weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        args.resolution = 480
        args.name_last_parameters = 'classifier'
    if args.model=='mobilenet':
        args.weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        args.resolution = 224
        args.name_last_parameters = 'classifier'
    if args.model=='shufflenet':
        args.weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        args.resolution = 224
        args.name_last_parameters = 'fc'
    if args.model=='b0':
        args.weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        args.resolution = 224
        args.name_last_parameters = 'classifier'
    if args.model=='b3':
        args.weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        args.resolution = 300
        args.name_last_parameters = 'classifier'
    if args.model=='b6':
        args.weights = torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1
        args.resolution = 528
        args.name_last_parameters = 'classifier'
    if args.model=='b7':
        args.weights = torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
        args.resolution = 600
        args.name_last_parameters = 'classifier'
    if args.model=='swin':
        args.weights = torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1
        args.resolution = 256
        args.name_last_parameters = 'head'
    if args.model=='chexzero':
        from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
        from types import SimpleNamespace
                # load data
        transformations = [
            # means computed from sample in `cxr_stats` notebook
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ]
        # resize to input resolution of pretrained clip model
        args.resolution = 224
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC, antialias = True))
        transform = Compose(transformations)
        args.weights = SimpleNamespace(model_path = './chexzero_weights/best_64_0.0001_original_35000_0.864.pt', transforms = transform)
        args.name_last_parameters = 'fc'
    if args.model=='xrv':
        from types import SimpleNamespace
        args.resolution = 512
        args.name_last_parameters = 'model.fc'
        import torchxrayvision as xrv
        class normalize(object):
            def __init__(self):
                pass
            def __call__(self, img):
                img = (2 * (img / 255.) - 1.) * 1024
                img = img.mean(0)[None, ...] # Make single color channel
                return img

        transform = torchvision.transforms.Compose([normalize(), xrv.datasets.XRayCenterCrop(), torchvision.transforms.Resize(512, antialias=True)])
        args.weights = SimpleNamespace(transforms = transform)
    args.metric_to_validate = f'auc_val_{args.labeler}_average' # if valiadtion is performed, use the validation AUC for deciding the best epoch
    args.function_to_compare_validation_metric = lambda x,y:x>=y #the larger the metric the better
    args.initialization_comparison = float('-inf')

    #gets the current time of the run, and adds a four digit number for getting
    #different folder name for experiments run at the exact same time.
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + str(randint(10000,99999))
    args.timestamp = timestamp
    
    #register a few values that might be important for reproducibility
    args.screen_name = os.getenv('STY')
    args.hostname = socket.gethostname()
    args.slurm_job_id = os.getenv('SLURM_JOB_ID')

    import platform
    args.python_version = platform.python_version()
    args.numpy_version = np.__version__
    args.output_dir = args.output_dir + '/' + args.experiment + '_' + args.timestamp + '/'
    args.num_workers = args.workers
    return args 


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
