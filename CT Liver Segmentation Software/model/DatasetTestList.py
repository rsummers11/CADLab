import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import random
import SimpleITK as sitk

class DecathlonData(Dataset):
    def __init__(self, data_list, XY, Z, cliplow=-500, cliphigh=2000, n_augmented_versions = 2,
                mode = 'train', CVfold=0, flip=True,augment3d= False,
                nonrigid = False, number_train = 10, config= None, HUoffset=0):

        self.config = config
        self.data_list = data_list
        self.XY = XY
        self.Z = Z
        self.cliplow = cliplow
        self.cliphigh = cliphigh
        self.n_augmented_versions = n_augmented_versions
        self.CVfold = CVfold
        self.flip = flip
        self.augment3d = augment3d
        self.HUoffset = HUoffset

        allfiles = data_list
        #print(data_path +'/*.nii.gz')
        self.files = allfiles


    def __getitem__(self, index):
        file =  self.files[index]
        transformversion = random.randint(0, self.n_augmented_versions)
        sample = load_and_resize(file , self.XY, self.XY, self.Z, self.cliphigh, self.cliplow, transformversion,
                                    flip = False, augment3d = False, HUoffset = self.HUoffset)
        sample['file'] = file
        sample['task'] = 5
        return sample

    def __len__(self):
        return len(self.files)


def load_and_resize(ctfile, X, Y, Z, cliphigh, cliplow, transformversion, cropz = 0.2, flip = True, augment3d = False, HUoffset = 0, ):
    try:
        ctNIB= nib.load(ctfile)
        ct = ctNIB.get_data().astype(np.float32)
        affine = ctNIB.affine
        zooms = ctNIB.header.get_zooms()
        if len(zooms) > 3:
            zooms = zooms[0:3]
    except:
        sitk_t1 = sitk.ReadImage(ctfile)
        ct = sitk.GetArrayFromImage(sitk_t1).astype(np.float32)
        affine = np.eye(4)
        zooms = [1,1,1]

    ct = ct + HUoffset
    sample = {}
    if len(ct.shape) == 6:
        ct = ct[:,:,:,0,0,0]

    #ct = np.flip(ct, axis=1)
    #ct = np.flip(ct, axis=0)

    #ct = np.ascontiguousarray(ct)

    #ct = ct[:,:,40:-70] #crop

    print(ct.shape)
    # resize and squeeze
    ct = torch.from_numpy(ct)
    ct = torch.unsqueeze(ct, 0)
    ct = torch.unsqueeze(ct, 0)
    sample['originalCT'] = ct[0,0,:,:,:].numpy()
    ct = torch.nn.functional.interpolate(ct,size =[X,Y,Z], mode ='trilinear', align_corners= False )
    ct = ct[0,0,:,:,:]

    ct = ct.numpy()
    ct_kidneys = ct.copy()

    ct = np.clip(ct, cliplow , cliphigh)
    ct = ct - np.mean(ct)
    ct = ct / np.std(ct)
    ct = torch.from_numpy(ct)

    # reformat img
    ct = torch.unsqueeze(ct, 0)

    ct_kidneys = np.clip(ct_kidneys, -500, 500)
    ct_kidneys = ct_kidneys - np.mean(ct_kidneys)
    ct_kidneys = ct_kidneys / np.std(ct_kidneys)
    ct_kidneys = torch.from_numpy(ct_kidneys)

    # reformat img
    ct_kidneys = torch.unsqueeze(ct_kidneys, 0)
    sample['data_kidneys'] = ct_kidneys

    sample['data'] = ct
    sample['file'] =ctfile
    sample['affine']= affine
    sample['zooms']= zooms

    return sample
