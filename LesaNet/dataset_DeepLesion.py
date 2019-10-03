# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to load data and labels of the DeepLesion
# dataset and the lesion ontology.
# --------------------------------------------------------

import torch.utils.data as data
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
from load_save_utils import load_ontology_from_xlsfile, load_DL_info


class DeepLesion(data.Dataset):
    """`DeepLesion <https://nihcc.box.com/v/DeepLesion>`_ Dataset.

    """

    def __init__(self, split):
        self.train = split == 'train'
        self.loadinfo(default.groundtruth_file)

        fn = default.split_file
        with open(fn, 'r') as f:
            data = json.load(f)
            print 'loaded', fn
        self.term_list = data['term_list']
        self.num_labels = len(self.term_list)

        prefix = split
        self.smp_idxs, self.labels, self.uncertain_labels = \
            data['%s_lesion_idxs'%prefix], data['%s_relevant_labels'%prefix], \
            data['%s_uncertain_labels'%prefix]
        if self.train:
            self.irrelevant_labels = data['train_irrelevant_labels']
        self.num_smp = len(self.smp_idxs)

        if not hasattr(default, 'ontology'):
            default.ontology = load_ontology_from_xlsfile(default.ontology_file)
        self.ontology = default.ontology
        self.gen_parents_list()
        self.gen_children_list()
        self.gen_exclusive_list()

        if config.TRAIN.TEXT_MINED_LABEL == 'RUI' and self.train:
            self.labels = [r+u+i for r,u,i in zip(self.labels, self.uncertain_labels, self.irrelevant_labels)]
        elif config.TRAIN.TEXT_MINED_LABEL == 'RU' and self.train:
            self.labels = [r+u for r,u in zip(self.labels, self.uncertain_labels)]
        self.labels = [unique(l) for l in self.labels]
        self.uncertain_labels = [unique(u) for u in self.uncertain_labels]

        terms_all = [d['term'] for d in self.ontology]
        self.term_class = [self.ontology[terms_all.index(t)]['class'] for t in self.term_list]

        print '>>>', len(self.smp_idxs), prefix, 'samples,',
        keep = [i for i in range(len(self.smp_idxs))
                if (not self.noisy[self.smp_idxs[i]])
                   and len(self.labels[i]) > 0]
        self.smp_idxs = [self.smp_idxs[i] for i in keep]
        self.num_smp = len(self.smp_idxs)
        print self.num_smp, 'after removing noisy and empty ones:'

        print self.num_labels, 'labels,',
        self.labels = [self.labels[i] for i in keep]
        self.uncertain_labels = [self.uncertain_labels[i] for i in keep]
        print '%d relevant cases,' % np.hstack(self.labels).shape[0],
        print '%d uncertain cases.' % np.hstack(self.uncertain_labels).shape[0]

        if default.generate_features_all:
            self.smp_idxs = range(len(self.filenames))
            self.labels = [[0] for _ in self.smp_idxs]
            self.uncertain_labels = [[0] for _ in self.smp_idxs]
            print 'Fake evaluation, generating features for all 32735 lesions'

        all_labels = [lb for lbs in self.labels for lb in lbs]
        self.cls_sz = np.array([all_labels.count(cls) for cls in range(self.num_labels)], dtype=np.float32)
        self.gen_class_weights()
        print

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

        target = torch.zeros(self.num_labels).to(torch.float32)
        target[labels] = 1

        unc_labels = unc_labels + [u1 for u in unc_labels for u1 in self.parent_list[u]]
        unc_labels = [u for u in unc_labels if u not in labels and all([eu not in labels for eu in self.exclusive_list[u]]) ]
        unc_target = torch.zeros(self.num_labels).to(torch.float32)
        unc_target[unc_labels] = 1

        imname = self.filenames[img_idx]

        ex_target = torch.zeros(self.num_labels).to(torch.float32)
        for l in labels:
            ex_target[self.exclusive_list[l]] = 1
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

    def gen_parents_list(self):
        if hasattr(default, 'parent_list'):
            self.parent_list = default.parent_list
            return

        parents_map = {t['term']: t['parents'] for t in self.ontology}
        self.parent_list = []
        for t in self.term_list:
            ps = parents_map[t]
            self.parent_list.append([self.term_list.index(p) for p in ps if p in self.term_list])
        print '%d parent-child pairs extracted' % len([p1 for p in self.parent_list for p1 in p])
        default.parent_list = self.parent_list

    def gen_children_list(self):
        if hasattr(default, 'all_children_list'):
            self.all_children_list = default.all_children_list
            return

        self.all_children_list = [[] for _ in self.term_list]
        for i, parent in enumerate(self.parent_list):
            for p1 in parent:
                self.all_children_list[p1].append(i)
        default.all_children_list = self.all_children_list

    def gen_exclusive_list(self):
        if hasattr(default, 'exclusive_list'):
            self.exclusive_list = default.exclusive_list
            return

        self.exclusive_list = []
        all_d_terms = [t['term'] for t in self.ontology]
        for p in range(self.num_labels):
            idx = all_d_terms.index(self.term_list[p])
            self.exclusive_list.append([self.term_list.index(ex) for ex in
                                        self.ontology[idx]['exclusive'] if ex in self.term_list])

        # if labels A and B are exclusive, any child of A and any child of B should also be exclusive
        while True:
            flag = False
            for p in range(self.num_labels):
                cur_ex = self.exclusive_list[p]
                next_ex = cur_ex[:]
                for ex in cur_ex:
                    next_ex += self.all_children_list[ex]
                for parent in self.parent_list[p]:
                    next_ex += self.exclusive_list[parent]
                next_ex = unique(next_ex)
                flag = flag or (set(next_ex) != set(cur_ex))
                self.exclusive_list[p] = next_ex
            if not flag:
                break

        print '%d mutually exclusive pairs extracted' % (len([p1 for p in self.exclusive_list for p1 in p]) / 2)
        default.exclusive_list = self.exclusive_list

    def gen_class_weights(self):
        if hasattr(default, 'cls_pos_wts'):
            self.cls_pos_wts = default.cls_pos_wts
            self.cls_neg_wts = default.cls_neg_wts
            return

        self.cls_pos_wts = self.num_smp / self.cls_sz / 2
        self.cls_neg_wts = self.num_smp / (self.num_smp - self.cls_sz) / 2
        self.cls_pos_wts = np.minimum(config.TRAIN.CE_POS_WT_CLAMP, self.cls_pos_wts)  # clamp positive weight
        default.cls_pos_wts = self.cls_pos_wts
        default.cls_neg_wts = self.cls_neg_wts
