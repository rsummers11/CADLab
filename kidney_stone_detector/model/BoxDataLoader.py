import os, random, glob
import torch
from torch.utils.data import Dataset
#from multiprocessing import Pool
import nibabel as nib
import numpy as np
#from joblib import Memory
import random
#from  scipy.ndimage import rotate
import glob
from random import Random

#-------------------------------------------------------------------------------
class BoxDataLoader(Dataset):
    def __init__(self, data_path, cliplow=-200, cliphigh=1000, n_augmented_versions=2, n_classes=1,
                mode = 'CVtrain', augment3d=False, cropz=.1, deformfactor = 10, nonrigid=True ):

        nonrigid=False

        self.data_path = data_path
        self.cliplow = cliplow
        self.cliphigh = cliphigh
        self.n_augmented_versions = n_augmented_versions
        self.augment3d = augment3d
        self.nonrigid = nonrigid
        self.deformfactor = deformfactor
        self.n_classes=n_classes
        self.cropz = cropz

        allfiles = sorted(glob.iglob(os.path.join(data_path, '*.nii.gz')))

        print("len allfiles=", len(allfiles))

        if mode == 'CVtrain':
            self.files = allfiles

        if mode == 'CVval':
            validation_path = os.path.join(data_path, "validation/")
            allfiles_val = sorted(glob.iglob(os.path.join(validation_path, '*.nii.gz')))
            self.files = allfiles_val

        if mode == 'all':
            self.files = allfiles
            if not self.exclude_in_test == None:
                self.files = [f for f in self.files if not self.exclude_in_test  in f]

#-------------------------------------------------------------------------------
    def __getitem__(self, index):
        file =  self.files[index]
        transformversion = random.randint(0, self.n_augmented_versions)

        sample = load_and_resize(file , transformversion, cliplow = self.cliplow, cliphigh = self.cliphigh, flip=True,
                    augment3d = self.augment3d, deformfactor=self.deformfactor, n_classes=self.n_classes,
                    nonrigid = self.nonrigid, cropz=self.cropz)

        sample['file'] = file

        return sample

    def __len__(self):
        return len(self.files)

#-----------------------------------
    def weights(self):

        wts = list()

        num_pos = 0
        num_neg = 0
        for f in self.files:
            if ("__1__" in os.path.basename(f)):
                num_pos += 1
            else:
                num_neg += 1

        ratio = float(num_neg)/float(num_pos) #set up 50-50 weighting

        for f in self.files:
            if ("__1__" in f):
                wts += [ratio]
            else:
                wts += [1]

        return wts

#-------------------------------------------------------------------------------
#@memory.cache
def load_and_resize(ctfile, transformversion,  cliplow=-200, cliphigh=1000, cropz = 0.0, flip = True, augment3d = True,
                    deformfactor=3, nonrigid=True,  n_classes=1 ):

    #random.seed(transformversion)

    ctnib = nib.load(ctfile)
    filename = os.path.basename(ctfile)
    ct = ctnib.get_fdata().astype(np.float32)
    assert (ct.ndim == 3),  print(ctfile, flush=True) #check for multiple channels

    if (ct.shape[0] == 32):
        ct = ct[4:-4, 4:-4, 4:-4]

    if not (ct.shape == (24,24,24)):
        print(ct.shape, ctfile, flush=True)
        os.system("rm "+ctfile)

    sample = {}

    #get voxelspacing
    header = ctnib.header
    voxelsize = header.get_zooms()
    originalsize = ctnib.shape

    if transformversion>0 :
        if random.random()>0.5 and flip==True:
            ct = np.flip(ct, axis = 0).copy()
        if random.random()>0.5 and flip==True:
            ct = np.flip(ct, axis = 1).copy()
        if random.random()>0.5 and flip==True:
            ct = np.flip(ct, axis = 2).copy()
        if random.random()>0.5 and flip==True:
            ct = ct.swapaxes(0,1).copy()
        #if augment3d == True:
        #    ct, labels = augment_3d(ct, labels,  nonrigid = nonrigid  )

    # normalize CT
    ct = np.clip(ct, cliplow , cliphigh)
    ct = ct - np.mean(ct)
    ct = ct / np.std(ct)

    ct = torch.from_numpy(ct)
    ct = torch.unsqueeze(ct, 0)
    #ct = torch.unsqueeze(ct, 0)
    ct = torch.round(ct*1000)/1000

    if "__1__" in filename:
        label = 1
    else:
        label = 0

    sample['data'] = ct
    sample['label'] = label

    return sample
