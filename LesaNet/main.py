# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to initialize the training and
# inference of LesaNet.
# --------------------------------------------------------

import os
import random
import warnings
import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from config import config, default, cfg_from_file, merge_a_into_b
from utils import save_checkpoint, checkpoint_name, var_size_collate, logger
from load_save_utils import load_demo_labels
from dataset_DeepLesion import DeepLesion
from dataset_DeepLesion_handlabeled import DeepLesion_handlabeled
from my_process import train, validate, adjust_learning_rate
from my_loss import WeightedCeLoss, CeLossRhem
from network_vgg import VGG16bn
from demo_function import demo_fun


os.chdir(os.path.join(os.path.dirname(__file__))) # go to root dir of this project

best_acc1 = 0.
accs = {}


def init_platform():
    config_file = cfg_from_file('config.yml')
    default_file = cfg_from_file('default.yml')
    logger.info(pprint.pformat(default_file))
    logger.info(pprint.pformat(config_file))

    merge_a_into_b(config_file, config)
    merge_a_into_b(default_file, default)
    default.best_model_path = ''

    if default.gpu == '':
        default.gpu = None
    if default.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = default.gpu

    default.distributed = default.world_size > 1
    if default.distributed:
        dist.init_process_group(backend=default.dist_backend, init_method=default.dist_url,
                                world_size=default.world_size)

    default.lr_epoch = [int(ep) for ep in default.lr_step.split(',')]

    if default.seed is not None:
        seed = default.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True


def init_model():
    model = eval(config.ARCH)(num_cls=default.num_cls,
                              pretrained_weights=config.TRAIN.USE_PRETRAINED_MODEL)

    if default.gpu is not None:
        model = model.cuda()
    elif default.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterions = {}
    if default.mode == 'train':
        criterions['wce'] = WeightedCeLoss(pos_weight=default.cls_pos_wts, neg_weight=default.cls_neg_wts).cuda()
        criterions['rhem'] = CeLossRhem().cuda()
        criterions['metric'] = nn.TripletMarginLoss(margin=config.TRAIN.TRIPLET_LOSS_MARGIN)

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': default.lr*2,
                            'weight_decay': 0}]
            else:
                params += [{'params': [value]}]
    optimizer = torch.optim.SGD(params,
                                default.lr,
                                momentum=default.momentum,
                                weight_decay=default.weight_decay)

    # optionally resume from a checkpoint
    if default.begin_epoch > 0:
        filename = checkpoint_name(default.begin_epoch, False)
        assert os.path.isfile(filename), "no checkpoint found at '{}'".format(filename)
        checkpoint = torch.load(filename)
        if 'acc' in checkpoint:
            accs[default.begin_epoch] = checkpoint['acc']
            best_acc1 = checkpoint['acc']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))

    cudnn.benchmark = True
    return model, criterions, optimizer


def init_data():
    # Data loading code
    train_dataset = DeepLesion(split='train')
    default.num_cls = train_dataset.num_labels
    default.term_list = train_dataset.term_list
    default.cls_sz_train = train_dataset.cls_sz
    default.term_class = train_dataset.term_class
    default.best_cls_ths = np.zeros((default.num_cls,), dtype=float)

    if default.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.SAMPLES_PER_BATCH, shuffle=(train_sampler is None), drop_last=default.drop_last_batch,
        num_workers=default.prefetch_thread_num, pin_memory=True, sampler=train_sampler, collate_fn=var_size_collate)

    val_dataset = DeepLesion(split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=config.SAMPLES_PER_BATCH, shuffle=False,
        num_workers=default.prefetch_thread_num, pin_memory=True, collate_fn=var_size_collate)

    test_dataset = DeepLesion(split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=config.SAMPLES_PER_BATCH, shuffle=False,
        num_workers=default.prefetch_thread_num, pin_memory=True, collate_fn=var_size_collate)

    test_dataset2 = DeepLesion_handlabeled()
    test_loader2 = torch.utils.data.DataLoader(test_dataset2,
        batch_size=config.SAMPLES_PER_BATCH, shuffle=False,
        num_workers=default.prefetch_thread_num, pin_memory=True, collate_fn=var_size_collate)

    return train_loader, [val_loader, test_loader, test_loader2], train_sampler


def init_demo():
    default.term_list, best_cls_ths = load_demo_labels(default.demo_labels_file)
    default.best_cls_ths = np.array(best_cls_ths, dtype=float)
    default.num_cls = len(default.term_list)


def evaluate(val_loaders, model):
    print('----------- Text-mined Validation -----------')
    acc_all = validate(val_loaders[0], model, use_val_th=False)
    print('----------- Text-mined Test -----------')
    acc_all1 = validate(val_loaders[1], model, use_val_th=True)
    print('----------- Hand-labeled Test -----------')
    acc_all2 = validate(val_loaders[2], model, use_val_th=True)
    return [acc_all, acc_all1, acc_all2]


def main():
    init_platform()
    logger.info('Experiment name: %s', default.exp_name)
    logger.info('Initializing ...')

    if default.mode in ('train', 'infer'):
        train_loader, val_loaders, train_sampler = init_data()
    elif default.mode == 'demo':
        init_demo()

    model, criterions, optimizer = init_model()

    if default.mode == 'infer':
        evaluate(val_loaders, model)
        return
    elif default.mode == 'demo':
        demo_fun(model)
        return

    # mode == 'train'
    global accs, best_acc1
    if default.validate_at_begin:
        acc_all = evaluate(val_loaders, model)
        accs[default.begin_epoch] = acc_all[0][config.TEST.CRITERION]
        best_acc1 = acc_all[0][config.TEST.CRITERION]
        logger.info('iter %d: %.4f' % (default.begin_epoch, accs[default.begin_epoch]))

    for epoch in range(default.begin_epoch, default.epochs):
        if default.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterions, optimizer, epoch)
        fn = save_checkpoint(epoch, model, optimizer, last_bkup=True)

        # evaluate on validation set
        acc_all = evaluate(val_loaders, model)
        accs[epoch+1] = acc_all[0][config.TEST.CRITERION]
        for key in sorted(accs.keys()):
            print 'iter %d: %.4f' % (key, accs[key])

        # remember best acc and save checkpoint
        is_best = accs[epoch+1] > best_acc1
        best_acc1 = max(accs[epoch+1], best_acc1)
        if is_best or not default.keep_best_model:
            new_fn = checkpoint_name(epoch+1, False)
            os.rename(fn, new_fn)
            if default.keep_best_model:
                if os.path.exists(default.best_model_path):
                    os.remove(default.best_model_path)
                default.best_model_path = new_fn


if __name__ == '__main__':
    main()
