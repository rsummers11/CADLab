#!/usr/bin/env python3
##(Code for training the CNN)
import sys, argparse, os, time
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import nibabel as nib
import torchvision
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter  # Changed by D.E. Elton. Need to install TensorFlow but better to use this to avoid library conflicts that arise when tensorboardX and tensorflow are both installed
import collections
import multiprocessing
from datetime import datetime
from importlib import import_module
from model.utils import *
from model.radam import RAdam
from model.resnet_3d import *
from model.lung_nodule_13_layer_3DCNN import Net
#from prettytable import PrettyTable
torch.backends.cudnn.benchmark = True #autotune - slower start but seems to help slightly.
from model.BoxDataLoader import BoxDataLoader

def main():
    #------------------------------- argument parsing --------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file_name',  type=str,      help='folder to segfiles')
    parser.add_argument('--model_file_name', type=str, default="", help='(if not included then assumes name is same as config_file_name)')
    parser.add_argument('-notrain', action='store_true')
    args = parser.parse_args()
    config_file_name = args.config_file_name
    model_file_name = args.model_file_name

    if (model_file_name == ""):
        model_file_name = config_file_name

    num_iter = 25000
    mixup_alpha = 0.35

    #--------------------- read in data and config file ------------------------
    config_file_name = os.path.splitext(config_file_name)[0]

    config = getattr(import_module('configs.' + config_file_name), 'config')

    data_root = config.get("data_root", None)
    if (data_root == None):
        print("please set data_root in config file")
        exit()

    os.makedirs('./log/', exist_ok=True)

    pprint(config)

    n_classes = config['n_classes']
    testsplits = config.get('testsplits', 5)
    nonrigid = config.get('nonrigid', True)
    num_train = config.get('num_train', 215)
    box_size = config.get('box_size', 24)
    cliplow = config.get('cliplow', -200)
    cliphigh = config.get('cliphigh', 1000)
    batch_size = config.get('batch_size', 10)
    n_augmented_versions = config.get('n_augmented_versions', 1)
    num_base_filters = config.get('num_base_filters', 64)
    start_epoch = 0

    writer = SummaryWriter('./runs/' + str(datetime.now()) +model_file_name  )

    model_file_name = os.path.join('./log/' + model_file_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_F1_ring = collections.deque(maxlen=3)

    #-------------------------------- make model instance ----------------------
    if 'resnet18' in config['model']:
        model = resnet18(sample_input_W=box_size, sample_input_H=box_size, sample_input_D=box_size,
                                num_seg_classes=n_classes, num_base_filters=num_base_filters)
    elif 'resnet34' in config['model']:
        model = resnet34(sample_input_W=box_size, sample_input_H=box_size, sample_input_D=box_size,
                                num_seg_classes=n_classes, num_base_filters=num_base_filters)
    elif '13layerCNN' in config['model']:
        model = Net()
    else:
        print("Unknown model")
        exit()

    if (device != "cpu"):
        use_cuda = True
        model = nn.DataParallel(model)
    else:
        use_cuda = False

    model = model.to(device)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if not (key[:10] == 'module.fin'):
            params += [{'params': [value], 'lr': config['initial_LR']}]
        else:
            params += [{'params': [value], 'lr':  config['initial_LR']/ config['factor_LR'] }]  # was 0.00005

    optimizer = RAdam(params)

    if os.path.exists(model_file_name) and not model_file_name == 'none' :
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, filename=model_file_name)
        model = model.to(device)
        model.train(True)

    print("number parameters = ", get_n_params(model))

    model.train(True)

    aug3d = config.get('augment3d','true')
    if (aug3d=='true'):
        augment3d = True
    else:
        augment3d = False

    #-------------------------- make parallel dataloaders ------------------------
    train_dataset = BoxDataLoader(data_root, cliplow = cliplow, cliphigh = cliphigh,
                n_augmented_versions = n_augmented_versions, n_classes = n_classes, mode = 'CVtrain')

    val_dataset = BoxDataLoader(data_root, cliplow = cliplow, cliphigh = cliphigh,
                n_augmented_versions = n_augmented_versions, n_classes = n_classes, mode = 'CVval')

    num_workers = multiprocessing.cpu_count()

    if num_workers > 32:
        num_workers = 1  #for some reason paralell loader is slower than non-parlallel!! best to go with 1 here most likely!

    train_sampler = WeightedRandomSampler(train_dataset.weights(), num_samples=len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)#useCPUs)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=1)

    # Schedulers
    best_val_F1 = 0
    end_training = False
    stopearly = EarlyStopping(patience=config['patience']*2, min_delta = 0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['step_factor_LR'],
                        patience=config['patience'], verbose=True)

    #criterion = FocalLoss(logits=True, gamma=1)
    criterion = nn.BCEWithLogitsLoss()
    final_layer = nn.Sigmoid()

    def mixup_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    if not args.notrain:
        print('starting training with len(train_loader) = ', len(train_loader))
        iter = start_epoch * len(train_loader)
        epoch = start_epoch
        while not (end_training):
            for i, data in enumerate(train_loader):
                time_iter_start = time.time()
                inputs = data['data']
                labels = data['label']

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, use_cuda)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

                outputs = model(inputs)

                outputs = outputs.view(-1).float().to(device)

                #print(inputs.shape, targets_a.shape, outputs.shape)

                targets_a, targets_b = targets_a.float().to(device), targets_b.float().to(device)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                #loss = criterion(outputs, labels.float())

                #loss = F.binary_cross_entropy_with_logits(outputs.float().to(device), labels.float().to(device))

                loss.backward()
                optimizer.step()

                # Tensorboard output
                writer.add_scalar('data/trainingloss', loss, iter)
                writer.add_scalar('data/timeperIteration', time.time() - time_iter_start, iter)

                iter = iter + 1

                if iter > num_iter:
                    end_training = True
                    print('numiter reached')

                if iter % config['validation_every_n'] == 0:
                    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),  'optimizer': optimizer.state_dict(), },  model_file_name)

                    val_F1 = validate(model, validation_loader)

                    print('validation loss:', val_F1)

                    writer.add_scalar('data/validation_loss', val_F1, iter)

                    val_F1_ring.append(val_F1)
                    smooth_val_F1 = np.mean(val_F1_ring)
                    scheduler.step(smooth_val_F1)
                    lrs = [group['lr'] for group in optimizer.param_groups]
                    writer.add_scalar('data/learning_rate', lrs[0])

                    #if stopearly.step(smooth_val_F1):
                    #    end_training = True

                    if val_F1 > best_val_F1:
                        best_val_F1 = val_F1
                        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },  model_file_name +'_best')

                    #if (iter % 1000 == 0) and (iter > 32000):
                    #    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),  'optimizer': optimizer.state_dict(), },  model_file_name+str(iter))


                if (iter % 10 == 0):
                    print("epoch = ", epoch, "iter = ", iter, "loss =", float(loss.detach().cpu().numpy()), "time =", np.round(time.time() - time_iter_start, 2),  flush=True)

                if end_training:
                    break
            epoch += 1

    print('Finished Training\n')


#-----------------------------------------------------------------------------
def get_n_params(model):
    #table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
    #print(table)
    #print(f"Total Trainable Params: {total_params}")
    return total_params

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
