"""
DeepLesion database
"""

import cPickle
import cv2
import os
import numpy as np
# from scipy.io import loadmat
import csv
import sys

from ..logger import logger
from imdb import IMDB
from rcnn.config import config, default

DEBUG = False


class DeepLesion(IMDB):
    def __init__(self, image_set, devkit_path):
        """
        fill basic information to initialize imdb
        """
        # year, image_set = image_set.split('_')
        super(DeepLesion, self).__init__('DeepLesion', image_set, devkit_path, devkit_path)  # set self.name
        # self.year = year
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path)

        self.classes = ['__background__',  # always index 0
                        'lesion']
        self.num_classes = len(self.classes)
        self.loadinfo(os.path.join(self.devkit_path, default.groundtruth_file))
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('%s num_images %d' % (self.name, self.num_images))

    def loadinfo(self, path):
        # load annotations and meta-info from DL_info.csv
        info = []
        with open(path, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                filename = row[0]  # replace the last _ in filename with / or \
                idx = filename.rindex('_')
                row[0] = filename[:idx] + os.sep + filename[idx+1:]
                info.append(row)
        info = info[1:]

        # the information not used in this project are commented
        self.filenames = np.array([row[0] for row in info])
        # self.patient_idx = np.array([int(row[1]) for row in info])
        # self.study_idx = np.array([int(row[2]) for row in info])
        # self.series_idx = np.array([int(row[3]) for row in info])
        self.slice_idx = np.array([int(row[4]) for row in info])
        # self.d_coordinate = np.array([[float(x) for x in row[5].split(',')] for row in info])
        # self.d_coordinate -= 1
        self.boxes = np.array([[float(x) for x in row[6].split(',')] for row in info])
        self.boxes -= 1  # coordinates in info file start from 1
        # self.diameter = np.array([[float(x) for x in row[7].split(',')] for row in info])
        # self.norm_location = np.array([[float(x) for x in row[8].split(',')] for row in info])
        # self.type = np.array([int(row[9]) for row in info])
        self.noisy = np.array([int(row[10]) > 0 for row in info])
        # self.slice_range = np.array([[int(x) for x in row[11].split(',')] for row in info])
        self.spacing3D = np.array([[float(x) for x in row[12].split(',')] for row in info])
        self.spacing = self.spacing3D[:, 0]
        self.slice_intv = self.spacing3D[:, 2]  # slice intervals
        # self.image_size = np.array([[int(x) for x in row[13].split(',')] for row in info])
        # self.DICOM_window = np.array([[float(x) for x in row[14].split(',')] for row in info])
        # self.gender = np.array([row[15] for row in info])
        # self.age = np.array([float(row[16]) for row in info])  # may be NaN
        self.train_val_test = np.array([int(row[17]) for row in info])


    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        # image_set_index_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '.txt')
        # assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        # with open(image_set_index_file) as f:
        #     image_set_index = [x.strip() for x in f.readlines()]

        set_list = ['train', 'val', 'test']
        index = set_list.index(self.image_set)
        image_set_index = self.filenames[self.train_val_test == index + 1]
        image_set_index = np.unique(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'Images_16bit', index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        use_cache = default.use_roidb_cache
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('%s gt roidb loaded from %s' % (self.name, cache_file))
        else:
            logger.info('loading gt roidb from %s ...', os.path.join(self.devkit_path, default.groundtruth_file))
            roidb = [self._load_annotation(filename) for filename in self.image_set_index]
            with open(cache_file, 'wb') as fid:
                cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
            logger.info('%s wrote gt roidb to %s' % (self.name, cache_file))

        return roidb

    def _load_annotation(self, filename):
        """
        Load annotations from .mat file.
        """
        idx = np.where(self.filenames == filename)[0]  # there may be multiple boxes (lesions) in a image
        assert idx.shape[0] >= 1, "The groundtruth file contains no entry of %s!" % (filename)
        boxes = self.boxes[idx, :]
        i = idx[0]
        slice_no = self.slice_idx[i]
        slice_intv = self.slice_intv[i]
        spacing = self.spacing[i]
        noisy = self.noisy[idx]

        num_boxes = boxes.shape[0]
        gt_classes = np.ones((num_boxes,), dtype=np.int32)  # we only have one class: lesion

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'image': filename,
                'slice_no': slice_no,
                'spacing': spacing,
                'slice_intv': slice_intv,
                'noisy': noisy,
                'flipped': False}
