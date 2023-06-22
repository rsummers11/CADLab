"""
Author : Seung Yeon Shin
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
"""

""" Custom dataset class and data-augmentation functions
including 'ComputeILP' for computing the corresponding ILP volume from each input volume """


import os
import numpy as np
import scipy.ndimage
import nibabel as nib
import numpy.random as npr
import medpy.filter.smoothing as medpy_smooth
from skimage.measure import label
from scipy import stats
from scipy import optimize

import torch
from torch.utils.data import Dataset

from data_augmentation import rotate, elastic_transform


class CustomDataset(Dataset):
    """Custom dataset"""

    def __init__(self, set_txt, root_dir, data_type_list=['img'],
                 subset_ratio=1., transform=None):
        """
        Args:
            set_txt (string) : Path to the txt file with a list of cases
            root_dir (string) : Directory with all the cases
            data_type_list (list of string) : List of data types
            subset_ratio (float) : Subset ratio
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(set_txt) as f:
            self.case_list = [x.strip() for x in f.readlines()]
        self.root_dir = root_dir
        self.data_type_list = data_type_list
        self.subset_ratio = subset_ratio
        self.transform = transform
        
        if self.subset_ratio < 1:
            self.case_list = list(map(lambda x: self.case_list[x], range(0,len(self.case_list),int(1/self.subset_ratio))))
        elif self.subset_ratio > 1:
            self.case_list = self.case_list*int(self.subset_ratio)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        case_path = os.path.join(self.root_dir,
                                  self.case_list[idx])
        
        sample = {}
        for cur_key in self.data_type_list:
            
            cur_data = nib.load(os.path.join(case_path, cur_key + '_crop.nii.gz'))
            cur_data = cur_data.get_fdata()
            
            sample[cur_key] = cur_data

        if self.transform:
            sample = self.transform(sample)

        return sample


# augmentation
class Scale(object):
    """Scale the images in a sample

    Args:
        scale_range
    """

    def __init__(self, scale_range=(0.9,1.1)):
        assert isinstance(scale_range, (tuple, list))
        self.scale_range = scale_range

    def __call__(self, sample):
        
        sample_deformed = {}
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        for cur_key in list(sample.keys()):
            cur_data = sample[cur_key]
            cur_data_deformed = scipy.ndimage.zoom(cur_data, scale, order=3 if cur_key=='img' else 0)
            sample_deformed[cur_key] = cur_data_deformed

        return sample_deformed
    

# augmentation
class Rotation(object):
    """Rotate the images in a sample
    This simulates mildly rotated volumes.
    This is a callable class which wraps the function 'rotate'
    Please refer to the arguments of the function 'rotate'

    Args:
        rot_range
        rot_plane
    """

    def __init__(self, rot_range=(-10,10), rot_plane=(0,1)):
        assert isinstance(rot_range, (tuple, list))
        assert isinstance(rot_plane, (int, tuple))
        self.rot_range = rot_range
        self.rot_plane = rot_plane

    def __call__(self, sample):

        img_list = []
        order_list = []
        for cur_key in list(sample.keys()):
            cur_data = sample[cur_key]
            img_list.append(cur_data)
            cur_order = 3 if cur_key=='img' else 0
            order_list.append(cur_order)
            
        img_list_deformed = \
        rotate(img_list, order_list, \
               rot_angle=np.random.uniform(self.rot_range[0], self.rot_range[1]), \
               rot_plane=self.rot_plane)
        """if np.random.uniform()>=0.5:
            img_list_deformed = \
            rotate(img_list, order_list, \
                   rot_angle=180, \
                   rot_plane=self.rot_plane)
        else:
            img_list_deformed = img_list"""
  
        sample_deformed = {}
        idx = 0
        for cur_key in list(sample.keys()):
            sample_deformed[cur_key] = img_list_deformed[idx]
            idx = idx + 1

        return sample_deformed
    

# augmentation
class ElasticTransform(object):
    """Elastically transform the images in a sample.
    This is a callable class which wraps the function 'elastic_transform'
    Please refer to the arguments of the function 'elastic_transform'

    Args:
        alpha_range
        sigma_range
    """

    def __init__(self, alpha_range=(20,100), sigma_range=(4,8), use_segm=True):
        assert isinstance(alpha_range, (tuple, list))
        assert isinstance(sigma_range, (tuple, list))
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.use_segm = use_segm

    def __call__(self, sample):
        
        img_list = []
        order_list = []
        for cur_key in list(sample.keys()):
            cur_data = sample[cur_key]
            img_list.append(cur_data)
            cur_order = 3 if cur_key=='img' else 0
            order_list.append(cur_order)
            
        if self.use_segm and ('segm' in list(sample.keys())):
            ctrl_pts = np.stack(np.where(sample['segm'].astype(np.bool)), axis=1)
        else: # must check
            shape = (np.array(sample['img'].shape)/2).astype(int)
            x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            ctrl_pts = np.stack([x.reshape(-1),y.reshape(-1),z.reshape(-1)], axis=1)*2
         
        img_list_deformed = \
        elastic_transform(img_list, ctrl_pts, order_list, \
                          alpha=np.random.uniform(self.alpha_range[0], self.alpha_range[1]), \
                          sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1]))
  
        sample_deformed = {}
        idx = 0
        for cur_key in list(sample.keys()):
            sample_deformed[cur_key] = img_list_deformed[idx]
            idx = idx + 1

        return sample_deformed


# preprocessing
class Normalization(object):
    """Normalize the intensity values of the image in a sample

    Args:
        int_bound : intensity bound, (min, max)
    """

    def __init__(self, int_bound=[-135,215]):
        assert isinstance(int_bound, (tuple, list))
        self.int_bound = int_bound

    def __call__(self, sample):
        sample_deformed = {}
        for cur_key in list(sample.keys()):
            cur_data = sample[cur_key]
            if cur_key == 'img':
                cur_data_deformed = (cur_data - self.int_bound[0]) / (self.int_bound[1] - self.int_bound[0])
                cur_data_deformed[cur_data_deformed>1] = 1.
                cur_data_deformed[cur_data_deformed<0] = 0.
            else:
                cur_data_deformed = cur_data.copy()
            sample_deformed[cur_key] = cur_data_deformed

        return sample_deformed
    

# preprocessing    
class ZeroCentering(object):
    """Zero center the intensity values of the image in a sample

    Args:
        int_bound
        set_name
    """

    def __init__(self, int_bound=[-135,215], set_name='carcinoid_det_pos'):
        
        self.int_bound = int_bound
        self.set_name = set_name
        
        if int_bound==[-135,215]:
        
            if set_name=='carcinoid_det_pos':
                self.pixel_mean = 0.2182
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def __call__(self, sample):
        sample_deformed = {}
        for cur_key in list(sample.keys()):
            cur_data = sample[cur_key]
            if cur_key == 'img':
                cur_data_deformed = cur_data - self.pixel_mean    
            else:
                cur_data_deformed = cur_data.copy()
            sample_deformed[cur_key] = cur_data_deformed

        return sample_deformed
    
    
# preprocessing
class PatchSampling(object):
    """Extract a random patch given a whole scan

    Args:
        size
    """

    def __init__(self, size, lesion_segm_name, tumor_margin=10):
        if type(size)==list:
            self.size = size
        else:
            self.size = [size]*3
            
        self.lesion_segm_name = lesion_segm_name
        self.tumor_margin = tumor_margin

    def __call__(self, sample):
        sample_deformed = {}
        len_x, len_y, len_z = sample['img'].shape
        
        cc_map, cc_num = label(sample[self.lesion_segm_name].astype(int), return_num=True)
        
        sel_cc_idx = npr.randint(1,cc_num+1)
        xs, ys, zs = np.where(cc_map==sel_cc_idx)
        x_max = np.minimum(np.amax(xs) + self.tumor_margin, len_x-1)
        y_max = np.minimum(np.amax(ys) + self.tumor_margin, len_y-1)
        z_max = np.minimum(np.amax(zs) + self.tumor_margin, len_z-1)
        x_min = np.maximum(np.amin(xs) - self.tumor_margin, 0)
        y_min = np.maximum(np.amin(ys) - self.tumor_margin, 0)
        z_min = np.maximum(np.amin(zs) - self.tumor_margin, 0)

        if len_x>=self.size[0]:
            x_st = npr.randint(np.maximum(0, x_max-self.size[0]+1), np.minimum(x_min, len_x-self.size[0])+1)
            cur_len_x = self.size[0]
        else:
            x_st = 0
            cur_len_x = len_x
            
        if len_y>=self.size[1]:
            y_st = npr.randint(np.maximum(0, y_max-self.size[1]+1), np.minimum(y_min, len_y-self.size[1])+1)
            cur_len_y = self.size[1]
        else:
            y_st = 0
            cur_len_y = len_y
            
        if len_z>=self.size[2]:
            z_st = npr.randint(np.maximum(0, z_max-self.size[2]+1), np.minimum(z_min, len_z-self.size[2])+1)
            cur_len_z = self.size[2]
        else:
            z_st = 0
            cur_len_z = len_z

        for cur_key in list(sample.keys()):
            cur_data = sample[cur_key]
            sample_deformed[cur_key] = cur_data[x_st:x_st+cur_len_x, \
                                                y_st:y_st+cur_len_y, \
                                                z_st:z_st+cur_len_z]
                    
        return sample_deformed
    
    
# new label
class ComputeILP(object):
    """Compute intensity-based lesion probability (ILP) for each voxel

    Args:
        target_int_file
    """

    def __init__(self, hu_range=[60,140], shift=0, \
                 target_int_file='lesion_stat/tumor_int_hist_list_sbct.npy'):

        assert isinstance(hu_range, (tuple, list))
        self.hu_range = np.array(hu_range) # only HU values within this range will be considered, otherwise 0
        self.shift = shift
        if self.shift!=0:
            self.hu_range = self.hu_range + shift  
        
        tumor_int_hist_list = np.load(target_int_file)  
    
        ### histogram ###
        if 'sbct' in target_int_file:
            hist_bin_min = 0
            hist_bin_max = 200
            hist_bin_size = 10
            self.denois_niter = 5
            self.resize_factor = 1
        else:
            raise NotImplementedError
            
        bins = list(range(hist_bin_min,hist_bin_max+1,hist_bin_size))
        
        cum_hist = np.zeros(tumor_int_hist_list.shape[1])
        for idx in range(tumor_int_hist_list.shape[0]):
            norm_hist = tumor_int_hist_list[idx] / np.sum(tumor_int_hist_list[idx])
            cum_hist = cum_hist + norm_hist
        cum_hist = cum_hist/tumor_int_hist_list.shape[0]
        
        ### KDE ###
        N = 1000
        data = []
        for idx in range(len(cum_hist)):
            cur_ub = bins[idx+1]
            cur_data = [cur_ub-hist_bin_size*0.5]*int(np.round(N*cum_hist[idx]))
            data += cur_data
    
        kde = stats.gaussian_kde(data)
        
        # check out the maximum
        opt = optimize.minimize_scalar(lambda x: -kde(x))
        #print('max : %.5f @ %.5f'%(-opt.fun[0],opt.x[0]))
        
        self.ilp_func = kde
        self.ilp_scale = -opt.fun[0]        
        
    def __call__(self, sample):
        
        sample_deformed = {}
        for cur_key in list(sample.keys()):
            cur_data = sample[cur_key]
            sample_deformed[cur_key] = cur_data.copy()
            if cur_key == 'img':
                
                b_resized = False
                if (self.resize_factor != 1) and (not np.any(np.array([cur_data.shape[0]%2, cur_data.shape[1]%2, cur_data.shape[2]%2]))):
                    b_resized = True
                    cur_data = scipy.ndimage.zoom(cur_data, self.resize_factor, order=1)
                
                orig_shape = cur_data.shape
                
                cur_data = medpy_smooth.anisotropic_diffusion(cur_data, \
                           niter=self.denois_niter, kappa=50, gamma=0.1, voxelspacing=None, option=3)
                    
                mask = (cur_data>=self.hu_range[0]) & (cur_data<=self.hu_range[1])
                in_val = cur_data[mask]
                if self.shift==0:
                    out_val = self.ilp_func(in_val)
                else:
                    out_val = self.ilp_func(in_val-self.shift)    
                out_val = out_val/self.ilp_scale
                
                ilp = np.zeros(orig_shape)
                ilp[mask] = out_val
                
                if b_resized:
                    ilp = scipy.ndimage.zoom(ilp, 1/self.resize_factor, order=1)
                
                sample_deformed['ilp'] = ilp
                    
        return sample_deformed


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        
        sample_deformed = {}
        for cur_key in list(sample.keys()):    
            temp = np.expand_dims(sample[cur_key], axis=0)
            sample_deformed[cur_key] = torch.from_numpy(temp)
        
        return sample_deformed