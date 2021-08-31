#!/usr/bin/env python3

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import nibabel as nib
import numpy as np
import pandas as pd
import collections
import multiprocessing
from importlib import import_module
from model.utils import *
from model.DatasetTestList import DecathlonData
import argparse
import pprint
from scipy.ndimage.morphology import binary_erosion
from metrics import dc

#--------------------------------------------------------------------------------
# read config and save
def read_filelist(filename):
    filelist = list()
    with open(filename) as f:
      for line in f:
        arg1 = line.rstrip('\n')
        arg1 = arg1.strip()
        if len(arg1)<2:
            continue
        if arg1[0]=='#':
            continue
        if arg1=="-1":
            break
        filelist.append(arg1)

    return filelist

#--------------------------------------------------------------------------------
def main():
    SAVE_CSV = False

    pp = pprint.PrettyPrinter(indent=4)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list',   help='list of test images')
    parser.add_argument('--single_file', action='store', help='name of a single file')
    parser.add_argument('--result_root', help='folder to store result')
    parser.add_argument('--model',       help='Name of the model file (and the config file)')
    parser.add_argument('--test',      action='store_true', help= '') #if flag isn't there default is False
    parser.add_argument('--HUoffset',  type=float,    action='store', help='HUoffset ') #if flag isn't there default is False

    parser.set_defaults(runBMD=False)
    args = parser.parse_args()

    config_file_name =  os.path.splitext(args.model)[0]
    model_file_name = os.path.splitext(args.model)[0]

    # read filelist
    if (args.test_list):
        filelist =  read_filelist(args.test_list)
    else:
        filelist = [args.single_file.rstrip('\n').strip()]

    print('processing these files:')
    pp.pprint(filelist)

    config = getattr(import_module('configs.' + config_file_name), 'config')
    dynamic = import_module(config['model'])
    MultiDataModel = getattr(dynamic, "MultiDataModel")
    print('Configuration')
    pp.pprint(config)

    n_classes = config.get('n_classes', 1)

    cliplow = config.get('cliplow', -500)
    cliphigh = config.get('cliphigh', 2000)

    # define and load model
    model_file_name = os.path.join('./configs/' + model_file_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiDataModel(1, n_classes, n_filter_per_level=config['n_filter_per_level'])
    model = model.to(device)
    model = nn.DataParallel(model)
    if os.path.exists(model_file_name):
        checkpoint = torch.load(model_file_name)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}')".format(model_file_name))
    else:
        print("=> no checkpoint found at '{}'".format(model_file_name))
    model.train(False)
    model.eval()

    if (args.HUoffset==None):
        HUoffset = config.get('HUoffset', 0)
    else:
        HUoffset = args.HUoffset

    # make datasets
    test_dataset = DecathlonData(filelist, config['originalXY'], config['originalZ'], cliplow, cliphigh,
                                n_augmented_versions=0, mode= 'all', config = config, HUoffset=HUoffset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,  num_workers=1)

    results =[]

    with torch.no_grad():
        results = list()

        for i, data in enumerate(test_loader):

            model = model.eval()

            inputs = data['data']
            task   = data['task']
            file   = data['file']
            affine = data['affine'].cpu().numpy().squeeze()
            zooms = data['zooms']
            zooms = np.asarray(zooms)
            print(zooms)

            originalCT = data['originalCT']
            print('Processing:', file)

            inputs,  task = inputs.to(device),  task.to(device)

            result = model(inputs, task, config['taskstouse'])

            complete = result['complete']

            ##### FREE UP MEMORY
            data = None
            result = None
            ###########

            complete = torch.nn.functional.interpolate(complete,size =[originalCT.shape[1], originalCT.shape[2],originalCT.shape[3]], mode ='trilinear', align_corners=False)

            complete[0,0,:,:,:] = originalCT #add in orginal CT

            complete = complete.cpu().numpy()

            # figure out filename
            resultname = os.path.basename(str(file[0]))
            resultname = resultname.rsplit('.', maxsplit=2)[0]
            resultnameNii = resultname + '_' +  config_file_name +'.nii.gz'
            #print(resultnameNii)

            #construct labelmap ------------------------------------

            # REMOVE ORIGINAL CT
            complete = complete[0,0:n_classes+1,:,:,:]

            numpy_segs = complete[1,:,:,:]*0

            for i in range(1, n_classes+1):
                class_i = complete[i,:,:,:]
                #numpy_segs = class_i# np.where(class_i > 0.1, class_i, 0)
                numpy_segs[class_i > 0.1] = i

            # quantitize:
            numpy_segs = np.round(numpy_segs)

            if (args.test):
                labelfile = file[0].replace('imagesTr', 'labelsTr')
                labels= nib.load(labelfile).get_data().astype(np.float32)

                dice = dc(numpy_segs, labels)
                print(i, file, "dice = ", dice)
                results += [dice]

                with open(os.path.join(args.result_root,"dice_scores.csv"), 'a')  as out_file:
                    out_file.write("%s, %d \n" % (resultnameNii, dice))

            ##save NII
            nib.Nifti1Image(numpy_segs.astype(np.float),affine).to_filename(os.path.join(args.result_root,resultnameNii))


            # evaluate CT
            #print(zooms)
            voxelsize = zooms[0]*zooms[1]*zooms[2]
            volume = (np.sum(numpy_segs) * voxelsize)/1000 # cubic mm to ml
            print("Total volume = ", volume, " ml" )


            #CT = originalCT.cpu().numpy()
            #originalCT = None
            #labels = complete[0,:,:,:]
            #evaluationeroded = evaluate_CT(CT[0,:,:,:], labels , voxelsize, zooms, resultname, config_file_name, erosion=True)
            #evaluation =       evaluate_CT(CT[0,:,:,:], labels , voxelsize, zooms, resultname, config_file_name, erosion=False)
            #evaluation = evaluation.append(evaluationeroded, ignore_index=True)

            #print(evaluation)
            ##save CSV
            #resultnameCSV = resultname + '_' +  config_file_name +'_perSlice.csv'
            #evaluation.to_csv(os.path.join(args.result_root, resultnameCSV), index=False)


    if (args.test):
        mean_dice = np.mean(results)
        std_dice = np.std(results)
        print(mean_dice, std_dice)
        with open(os.path.join(args.result_root,"average_dice.csv"), 'a') as out_file:
            out_file.write("%f, %f, %f\n" % (HUoffset, mean_dice, std_dice))





def evaluate_CT(CT, binarylabel, voxelsize, zooms, filename, config, erosion):
    binarylabel = (binarylabel >0.5)
    if erosion:
        zerosion = int(20 / zooms[2])  # make a 20 mm z erosion (1 cm in each direction)
        if zerosion <= 0:
            zerosion = 1
        binarylabel = binary_erosion(binarylabel, structure = np.ones((1,1,zerosion)) , iterations =1)

    CTfull = np.where(binarylabel == False, np.nan, CT)
    binarylabelfull = binarylabel
    evaluationAll = {'file': 'NONE', 'config': 'NONE' , 'erosion_performed':'NONE',  'Z':'NONE',  'mean': 'NONE', 'median':'NONE', 'stdev':'NONE', 'volume': 'NONE' }
    evaluationAll = pd.DataFrame(data = evaluationAll,  index=[0])
    for z in range(CTfull.shape[2]):

        CT = CTfull[:,:,z]
        binarylabel = binarylabelfull[:,:,z]

        std = np.nanstd(CT)
        median = np.nanmedian(CT)
        mean = np.nanmean(CT)
        volume = (np.sum(binarylabel) * voxelsize)/1000 # cubic mm to ml
        evaluation = {'file':filename, 'config': config , 'erosion_performed':erosion,  'Z':z,  'mean': mean, 'median':median, 'stdev':std, 'volume': volume }
        evaluation = pd.DataFrame(data = evaluation,  index=[0])
        evaluationAll = pd.concat([evaluationAll, evaluation])#, ignore_index=True)
        #print(evaluationAll)
    return evaluationAll


if __name__ == '__main__':
    main()
