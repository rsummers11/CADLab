"""
Author : Seung Yeon Shin
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
"""

""" To compute an intensity histogram and an intensity-based lesion probability (ILP) function of target lesions """

import numpy as np
import nibabel as nib
import argparse
import os
import matplotlib.pyplot as plt
import medpy.filter.smoothing as medpy_smooth
from scipy import stats
from scipy import optimize


fontsize = 25


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='dataset_stat')
    
    ### data related ###
    parser.add_argument('--data_root_path', \
                        default='../../dataset', \
                        help='Data path', type=str)

    parser.add_argument('--set_txt_path', \
                        default='../../dataset/carcinoid/carcinoid_det_pos.txt', \
                        help='Set txt file path', type=str)    
        
    parser.add_argument('--hist_bin_min', \
                        default=0, \
                        help='Histogram bin min', type=int)
    parser.add_argument('--hist_bin_max', \
                        default=200, \
                        help='Histogram bin max', type=int)
    parser.add_argument('--hist_bin_size', \
                        default=10, \
                        help='Histogram bin size', type=int)  
    
    args = parser.parse_args()
    
    return args


# examine the GT lesion segm of the SBCT dataset to construct an intensity-based lesion probability (ILP) function
def gt_lesion_segm_stat_sbct(args):
    
    stat_save_path = 'lesion_stat'
    if not os.path.exists(stat_save_path):
        os.makedirs(stat_save_path)
    
    
    ### load set txt ###
    with open(args.set_txt_path) as f:
        case_list = [x.strip() for x in f.readlines()]

    
    ### compute stats ###
    bins = list(range(args.hist_bin_min,args.hist_bin_max+1,args.hist_bin_size))
    tumor_int_hist_list = []
    for cur_case_path in case_list:

        img_path = os.path.join(args.data_root_path, cur_case_path, 'img_crop.nii.gz')
        segm_path = os.path.join(args.data_root_path, cur_case_path, 'tumor_segm_crop.nii.gz')
        
        img = nib.load(img_path)
        img = img.get_fdata()
        img = medpy_smooth.anisotropic_diffusion(img, \
              niter=5, kappa=50, gamma=0.1, voxelspacing=None, option=3)
        
        segm = nib.load(segm_path)
        segm = segm.get_fdata()
        segm = segm.astype(np.uint8)
        
        for tumor_idx in list(np.unique(segm))[1:]:
            
            cur_tumor_segm = (segm==tumor_idx)
            
            cur_tumor_int_list = list(img[cur_tumor_segm].reshape(-1))
            cur_tumor_int_list = sorted(cur_tumor_int_list)
            
            hist, bin_edges = np.histogram(cur_tumor_int_list, bins)
            tumor_int_hist_list.append(hist)


    tumor_int_hist_list = np.array(tumor_int_hist_list)
    np.save(os.path.join(stat_save_path, 'tumor_int_hist_list_sbct.npy'), tumor_int_hist_list)
 

    #tumor_int_hist_list = np.load(os.path.join(stat_save_path, 'tumor_int_hist_list_sbct.npy'))
       
    
    ### draw hist & ILP  ###    
    cum_hist = np.zeros(tumor_int_hist_list.shape[1])
    for idx in range(tumor_int_hist_list.shape[0]):
        norm_hist = tumor_int_hist_list[idx] / np.sum(tumor_int_hist_list[idx])
        cum_hist = cum_hist + norm_hist
    cum_hist = cum_hist/tumor_int_hist_list.shape[0]
    
    #fig, ax = plt.subplots()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.set_size_inches(18, 9)
    
    # Plot the histogram heights against integers on the x axis
    ax1.bar(bins[:-1], cum_hist, \
            width=8, color='gray', alpha=1, \
            label="intensity histogram")
    
    # Set the ticks to the middle of the bars
    ax1.set_xticks([i*args.hist_bin_size-0.5*args.hist_bin_size for i in range(len(cum_hist)+1)])
    #ax1.set_xticklabels(bins)
    temp = []
    for idx, elem in enumerate(bins):
        if idx%2==0:
            temp.append(str(elem))
        else:
            temp.append("")
    ax1.set_xticklabels(temp)
    
    ax1.set_yticks([0, 0.04, 0.08, 0.12, 0.16, 0.20])
    ax1.set_yticklabels([0, 0.04, 0.08, 0.12, 0.16, 0.20])
    
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.set_xlabel("Hounsfield unit", fontsize=fontsize)
    ax1.set_ylabel("proportion", fontsize=fontsize, color='gray')
    ax1.legend(loc=2, fontsize=fontsize)
        
    # KDE
    N = 1000
    data = []
    for idx in range(len(cum_hist)):
        cur_ub = bins[idx+1]
        cur_data = [cur_ub-args.hist_bin_size*0.5]*int(np.round(N*cum_hist[idx]))
        data += cur_data

    kde = stats.gaussian_kde(data)

    # check out the maximum
    opt = optimize.minimize_scalar(lambda x: -kde(x))
    max_val = -opt.fun[0] 
    
    xx = np.linspace(args.hist_bin_min, args.hist_bin_max, args.hist_bin_max-args.hist_bin_min)
    ax2.plot(xx, kde(xx)/max_val, color='black', label="ILP function")
    
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    ax2.set_ylabel('probability', fontsize=fontsize, color='black')
    ax2.legend(loc=1, fontsize=fontsize)
    
    #plt.legend(loc=2, fontsize=fontsize)
    plt.savefig(os.path.join(stat_save_path, 'tumor_int_hist_func_sbct.eps'))
    

if __name__ == '__main__':
    
    args = parse_args()
    
    gt_lesion_segm_stat_sbct(args)