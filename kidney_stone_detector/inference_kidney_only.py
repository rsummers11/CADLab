#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torchvision
import numpy as np
import nibabel as nib
from importlib import import_module
from skimage.measure import label as label_connected_components

class InferenceKidney:
    def __init__(self):

        config_file_name =   'KidneyNC' #'dan_kidney_192'
        model_file_name =  'KidneyNC' #'dan_kidney_192.py_CV0'  'dan_kidney_192_CV0_best'  #config_file_name

        config = getattr(import_module('configs.' + config_file_name), 'config')
        self.config = config
        dynamic = import_module(config['model'])
        MultiDataModel = getattr(dynamic, "MultiDataModel")

        self.n_classes = config.get('n_classes', 1)
        self.cliplow = config.get('cliplow', -500)
        self.cliphigh = config.get('cliphigh', 2000)
        self.XY = config.get('originalXY', 256)
        self.Z = config.get('originalZ', 192)
        self.task = config.get('taskstouse', (13,))[0]

        # define and load model
        model_file_name = os.path.join('./log/' + model_file_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MultiDataModel(1, self.n_classes, n_filter_per_level=config['n_filter_per_level'])
        model = model.to(self.device)
        model = nn.DataParallel(model)
        if os.path.exists(model_file_name):
            if str(self.device) == 'cpu':
                checkpoint = torch.load(model_file_name, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(model_file_name)

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}')".format(model_file_name))
        else:
            print("ERROR => no liverNC checkpoint found at '{}'".format(model_file_name))
            exit()
        model.train(False)
        #model.half()
        model.eval()

        self.model = model

    #--------------------------------------------------------------------------------
    def isolate_two_largest_components(self, seg):
        labeled_components, num_components = label_connected_components(seg, return_num=True)

        ## this could be improved using argmax and numpy label bin functions
        sizes = {}
        for i in range(0, num_components+1):
            this_component = np.where(labeled_components == i, 1, 0)
            size = np.sum(this_component)
            sizes[i] = size

        sizes_list = sorted(sizes, key=sizes.get, reverse=True)

        if num_components > 2:
            largest_component = sizes_list[1]
            next_largest_component = sizes_list[2]
        else:
            return seg

        largest_component = np.where(labeled_components == largest_component, 1, 0)
        next_largest_component = np.where(labeled_components == next_largest_component, 1, 0)

        seg = largest_component + next_largest_component

        return seg



    #--------------------------------------------------------------------------------
    def inference_kidney_only(self, ct, out_filepath, affine=np.eye(4),
                need_flipx=False, need_flipy=False, need_flipz=False):

        model = self.model

        with torch.no_grad():
            #model = model.eval()

            orig_shape = ct.shape

            ct = np.clip(ct, self.cliplow, self.cliphigh)
            ct = ct - np.mean(ct)
            std = np.std(ct)
            if std > 0:
                ct = ct / np.std(ct)

            ct = torch.from_numpy(ct)
            ct = torch.unsqueeze(ct, 0)
            ct = torch.unsqueeze(ct, 0)
            ct = torch.nn.functional.interpolate(ct, size =[self.XY, self.XY, self.Z], mode ='trilinear', align_corners = False)

            inputs = ct

            task = torch.LongTensor([self.task])

            inputs, task = inputs.to(self.device), task.to(self.device)

            result = model(inputs, task, self.config['taskstouse'])

            #complete = result['complete']
            complete = result['final_layerA']

            print(complete.shape)

            ##### FREE UP MEMORY
            del result
            del inputs

            complete = torch.nn.functional.interpolate(complete, size = orig_shape, mode ='trilinear', align_corners=False)

            complete = complete.cpu().numpy()

            ## REMOVE ORIGINAL CT
            #complete = complete[0,0:self.n_classes+1,:,:,:]

            #numpy_seg = complete[1,:,:,:]*0
            first_chanel = complete[0,:,:,:]
            numpy_seg = np.where(first_chanel > 0.5, 1, 0)

            #for i in range(1, self.n_classes+1):
            #    class_i = complete[i,:,:,:]
            #    #numpy_seg = class_i# np.where(class_i > 0.1, class_i, 0)
            #    numpy_seg[class_i > 0.5] = i
            # quantitize:
            #numpy_seg = np.round(numpy_seg)

            del complete
            del first_chanel
            #print(numpy_seg.shape)

            numpy_seg = self.isolate_two_largest_components(numpy_seg[0,:,:,:])

            ## save segmentation to file
            if len(out_filepath) > 0:
                numpy_seg_out = numpy_seg.copy()
                if (need_flipx): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 0))
                if (need_flipy): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 1))
                if (need_flipz): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 2))
                nib.Nifti1Image(numpy_seg_out.astype(np.int16), affine).to_filename(out_filepath)

        return numpy_seg.astype(np.uint16)


if __name__ == '__main__':
    _ = inference_liver("10000_20131004_2.nii.gz")
