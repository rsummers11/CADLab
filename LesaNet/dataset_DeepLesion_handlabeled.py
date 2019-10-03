# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to load data and labels of the
# hand-labeled test set of the DeepLesion dataset.
# --------------------------------------------------------

import torch.utils.data as data
import csv
import cPickle
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import json

from config import config, default
from load_ct_img import load_prep_img, get_patch
from utils import unique
from load_save_utils import load_DL_info


class DeepLesion_handlabeled(data.Dataset):
    """`DeepLesion <https://nihcc.box.com/v/DeepLesion>`_ Dataset.

    """

    def __init__(self):
        self.loadinfo(default.groundtruth_file)
        with open(default.hand_split_file, 'r') as f:
            data = json.load(f)
        self.term_list = default.term_list
        self.num_cls = len(self.term_list)

        self.smp_idxs = [d['lesion_idx'] for d in data]
        self.labels = [[self.term_list.index(t) for t in d['expanded_terms'] if t in self.term_list] for d in data]
        self.uncertain_labels = [[] for d in data]
        self.smp_idxs = self.smp_idxs
        self.labels = self.labels
        self.num_smp = len(self.smp_idxs)

        self.labels = [unique(l) for l in self.labels]
        self.uncertain_labels = [unique(l) for l in self.uncertain_labels]

        print '>>>', len(self.smp_idxs), 'hand-labeled samples,',
        keep = [i for i in range(len(self.smp_idxs)) if (not self.noisy[self.smp_idxs[i]])
                and len(self.labels[i]) > 0]
        self.smp_idxs = [self.smp_idxs[i] for i in keep]
        self.labels = [self.labels[i] for i in keep]
        self.uncertain_labels = [self.uncertain_labels[i] for i in keep]
        print 'num of positive labels:', np.hstack(self.labels).shape[0]
        print 'num of uncertain labels:', np.hstack(self.uncertain_labels).shape[0]

        if default.generate_features_all:
            self.smp_idxs = range(len(self.filenames))
            self.labels = [[0] for _ in self.smp_idxs]
            self.uncertain_labels = [[0] for _ in self.smp_idxs]
            print 'Fake evaluation, generating features for all 32735 lesions'

        self.num_smp = len(self.smp_idxs)
        print self.num_smp, 'after removing noisy and empty ones:',

        all_labels = [lb for lbs in self.labels for lb in lbs]
        print self.num_cls, 'classes'
        self.cls_sz = np.array([all_labels.count(cls) for cls in range(self.num_cls)], dtype=np.float32)
        if self.num_cls < 10:
            print 'number of positive samples:'
            for cls in range(self.num_cls):
                print self.term_list[cls], int(self.cls_sz[cls])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_idx, labels, unc_labels = self.smp_idxs[index], self.labels[index], self.uncertain_labels[index]
        patch, center_box = self.load_process_img(img_idx)
        patch = patch.astype(float) / 255
        patch = torch.from_numpy(patch.transpose((2, 0, 1))).to(dtype=torch.float)

        center_box = torch.tensor(center_box).to(dtype=torch.float)
        out_box = torch.tensor([0, 0, patch.shape[2], patch.shape[1]]).to(dtype=torch.float)

        target = torch.zeros(self.num_cls).to(torch.float32)
        target[labels] = 1

        unc_target = torch.zeros(self.num_cls).to(torch.float32)

        imname = self.filenames[img_idx]

        ex_target = torch.zeros(self.num_cls).to(torch.float32)
        ex_target[labels] = 0

        return [patch, out_box, center_box], [target, unc_target, ex_target], [imname, img_idx]
        # data tensors, target tensors, nontensor info

    def load_process_img(self, img_idx):
        imname = str(self.filenames[img_idx])
        slice_idx = self.slice_idx[img_idx]
        spacing = self.spacing[img_idx]
        slice_intv = self.slice_intv[img_idx]
        box = self.boxes[img_idx].copy()

        im, im_scale, crop = load_prep_img(imname, slice_idx, spacing, slice_intv,
                                            do_clip=False, num_slice=config.NUM_SLICES)
        box *= im_scale
        patch, new_box, patch_scale = get_patch(im, box)
        return patch, new_box

    def __len__(self):
        return len(self.smp_idxs)

    def loadinfo(self, path):
        # load annotations and meta-info from DL_info.csv
        if not hasattr(default, 'DL_info'):
            default.DL_info = load_DL_info(path)
        for key in default.DL_info.keys():
            setattr(self, key, default.DL_info[key])
