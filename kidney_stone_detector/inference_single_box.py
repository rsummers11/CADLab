#!/usr/bin/env python3
import sys, argparse, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import nibabel as nib
import torchvision
import numpy as np
import collections
from importlib import import_module
from model.utils import *
from model.resnet_3d import *
from model.lung_nodule_13_layer_3DCNN import Net

#-----------------------------------------------------------------------------

class InferenceSingleBox:
    def __init__(self):
        #config_file_name = "kidney_stone_classifier_34"
        #model_file_name = "kidney_stone_classifier_resnet34_best"
        config_file_name = "kidney_stone_classifier_13_layer_CNN"
        #model_file_name = "kidney_stone_classifier_13_layer_CNN_mixup"
        #model_file_name = "kidney_stone_classifier_13_layer_CNN_larger_dataset"
        model_file_name = "kidney_stone_classifier_13_layer_CNN_larger_again_BCE_mixup_best" #config_file_name
        #model_file_name = "kidney_stone_classifier_13_layer_CNN_larger_mixup_3_best"
        config = getattr(import_module('configs.' + config_file_name), 'config')

        ## read config file
        n_classes = config['n_classes']
        box_size = config.get('box_size', 24)
        cliplow = config.get('cliplow', -200)
        cliphigh = config.get('cliphigh', 1000)
        num_base_filters = config.get('num_base_filters', 64)

        model_file_name = os.path.join('./log/' + model_file_name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        if (self.device != "cpu"):
            model = nn.DataParallel(model)

        model = model.to(self.device)

        if os.path.exists(model_file_name):
            if (torch.cuda.is_available() == True):
                checkpoint = torch.load(model_file_name)
            else:
                print("WARNING: torch.cuda.is_available() =", torch.cuda.is_available(), " running on CPU - code will be SLOW..")
                checkpoint = torch.load(model_file_name, map_location='cpu')

            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(self.device)
            model.train(True)

        model.train(False)
        model.eval()
        self.model = model
        self.cliplow = cliplow
        self.cliphigh = cliphigh
        self.box_size = box_size
        self.final_layer = nn.Sigmoid()

    #---------------------------------------------------------------------------
    def inference_single_box(self, ct):

        try:
            assert(ct.shape == (self.box_size, self.box_size, self.box_size))
        except:
            print("invalid shape for subbox : ", ct.shape)

        # normalize CT
        ct = np.clip(ct, self.cliplow , self.cliphigh)
        ct = ct - np.mean(ct)
        std =  np.std(ct)
        if (std > 0):
            ct = ct / std

        ct = torch.from_numpy(ct)
        ct = torch.unsqueeze(ct, 0)
        ct = torch.unsqueeze(ct, 0)
        ct = torch.round(ct*1000)/1000
        inputs  = ct.to(self.device)
        outputs = self.model(inputs)
        outputs = self.final_layer(outputs)
        outputs = outputs.view(-1)

        return float(outputs.detach().cpu().numpy())

#-----------------------------------------------------------------------------
if __name__ == '__main__':
     test_file = "/home/dan/data/kits19/case_00000/ct_vol.nii.gz" #sd1_199_10s.nii.gz0__1__.nii.gz"

     ctnib = nib.load(test_file)
     ct = ctnib.get_fdata().astype(np.float32) #+ HUOffset
     ct = ct[0:24,0:24,0:24]

     bo = InferenceSingleBox()
     print(bo.inference_single_box(ct))
