# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""The DeepLesion dataset loader, include box, tag, and masks"""
import torch
import torchvision
import numpy as np
import os
import csv
import logging
import json

from maskrcnn.data.datasets.load_ct_img import load_prep_img
from maskrcnn.structures.bounding_box import BoxList
from maskrcnn.structures.segmentation_mask import SegmentationMask
from maskrcnn.config import cfg
from maskrcnn.data.datasets.DeepLesion_utils import load_tag_dict_from_xlsfile, gen_mask_polygon_from_recist, load_lesion_tags
from maskrcnn.data.datasets.DeepLesion_utils import gen_parent_list, gen_exclusive_list, gen_children_list


class DeepLesionDataset(object):

    def __init__(
        self, split, data_dir, ann_file, transforms=None
    ):
        self.transforms = transforms
        self.split = split
        self.data_path = data_dir
        self.classes = ['__background__',  # always index 0
                        'lesion']
        self.num_classes = len(self.classes)
        self.loadinfo(ann_file)
        self.image_fn_list, self.lesion_idx_grouped = self.load_split_index()
        self.num_images = len(self.image_fn_list)
        self.logger = logging.getLogger(__name__)

        # for classification
        if cfg.MODEL.TAG_ON:
            self._process_tags()
            if split == 'test':
                self.logger.info('loading 500 hand-labeled test tags')
                self._process_manual_annot_test_tags()

        self.logger.info('DeepLesion %s num_images: %d' % (split, self.num_images))

    def _process_manual_annot_test_tags(self):
        fn = os.path.join(cfg.PROGDAT_DIR, cfg.DATASETS.TAG.MANUAL_ANNOT_TEST_FILE)
        with open(fn, 'r') as f:
            data = json.load(f)
        self.manual_annot_test_tags = {}
        for d in data:
            lb = [self.tag_list.index(t) for t in d['expanded_terms'] if t in self.tag_list]
            self.manual_annot_test_tags.update({d['lesion_idx']: lb})
        assert np.all(self.train_val_test[list(self.manual_annot_test_tags.keys())] == 3)
        all_tags = [t for tags in self.manual_annot_test_tags.values() for t in tags]
        cfg.runtime_info.manual_test_set_cls_sz = np.array([all_tags.count(cls) for cls in range(self.num_tags)])

    def _process_tags(self):
        cache_fn = os.path.join(cfg.PROGDAT_DIR, cfg.DATASETS.TAG.USE_CACHE_FILE)
        if os.path.exists(cache_fn):
            with open(cache_fn, 'r') as f:
                data = json.load(f)
            self.tag_dict_list, lesion_tags = data['tag_dict_list'], data['lesion_tags']
            self.lesion_tags = {k: v for k, v in lesion_tags}
            self.logger.info('lesion tags loaded from %s' % cache_fn)
        else:
            self.logger.info('Generating lesion tags')
            tag_dict_file = os.path.join(cfg.PROGDAT_DIR, cfg.DATASETS.TAG.TAG_DICT_FILE)
            split_file = os.path.join(cfg.PROGDAT_DIR, cfg.DATASETS.TAG.SPLIT_FILE)
            self.tag_dicts = load_tag_dict_from_xlsfile(tag_dict_file)
            self.tag_dict_list, self.lesion_tags = \
                load_lesion_tags(split_file, self.tag_dicts)
            lesion_tags_save = [(k, v) for k, v in self.lesion_tags.items()]
            with open(cache_fn, 'w') as f:
                json.dump({'tag_dict_list': self.tag_dict_list, 'lesion_tags': lesion_tags_save}, f, indent=2)

        pos_tag_num = sum([len(tt) for tt in self.lesion_tags.values()])
        self.tag_list = [d['tag'] for d in self.tag_dict_list]
        self.num_tags = len(self.tag_list)
        assert [d['ID'] for d in self.tag_dict_list] == list(range(self.num_tags))
        self.logger.info('%d tags, %d lesions with tags, %d positive tags altogether' %
                         (self.num_tags, len(self.lesion_tags.keys()), pos_tag_num))

        cfg.runtime_info.num_tags = self.num_tags
        all_lesion_idxs = [li for lis in self.lesion_idx_grouped for li in lis]
        all_taged_lesion_idxs = np.intersect1d(all_lesion_idxs, list(self.lesion_tags.keys()))
        self.num_taged_lesion = len(all_taged_lesion_idxs)
        all_tags = [lb for li in all_taged_lesion_idxs for lb in self.lesion_tags[li]]
        self.cls_sz = np.array([all_tags.count(cls) for cls in range(self.num_tags)], dtype=np.float32)

        self.tag_classes = [d['class'] for d in self.tag_dict_list]
        cfg.runtime_info.tag_list = self.tag_list
        if 'parent_list' not in cfg.runtime_info:
            tag_dict_file = os.path.join(cfg.PROGDAT_DIR, cfg.DATASETS.TAG.TAG_DICT_FILE)
            self.tag_dicts = load_tag_dict_from_xlsfile(tag_dict_file)
            parent_list = gen_parent_list(self.tag_dicts, self.tag_list)
            cfg.runtime_info.parent_list = parent_list

            all_children_list, direct_children_list = gen_children_list(parent_list, self.tag_list)
            cfg.runtime_info.exclusive_list = gen_exclusive_list(self.tag_dicts, self.tag_list, parent_list,
                                                     all_children_list)
            self.logger.info('%d parent-children relation pairs; %d exclusive relation pairs',
                             sum([len(p) for p in parent_list]), sum([len(p) for p in cfg.runtime_info.exclusive_list])/2)
        self.exclusive_list = cfg.runtime_info.exclusive_list

        if self.split == 'train':
            cfg.runtime_info.train_cls_sz = self.cls_sz
            cfg.runtime_info.cls_pos_wts = self.num_taged_lesion / self.cls_sz / 2
            cfg.runtime_info.cls_neg_wts = self.num_taged_lesion / (self.num_taged_lesion - self.cls_sz) / 2
            cfg.runtime_info.cls_pos_wts = np.minimum(cfg.MODEL.ROI_TAG_HEAD.CE_POS_WT_CLAMP,
                                                      cfg.runtime_info.cls_pos_wts)  # clamp positive weight
        elif self.split == 'val':
            cfg.runtime_info.val_cls_sz = self.cls_sz
        elif self.split == 'test':
            cfg.runtime_info.test_cls_sz = self.cls_sz

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, info).
        """
        image_fn = self.image_fn_list[index]
        lesion_idx_grouped = self.lesion_idx_grouped[index]
        boxes0 = self.boxes[lesion_idx_grouped]
        # slice_no = self.slice_idx[lesion_idx_grouped][0]
        slice_intv = self.slice_intv[lesion_idx_grouped][0]
        spacing = self.spacing[lesion_idx_grouped][0]
        recists = self.d_coordinate[lesion_idx_grouped]
        diameters = self.diameter[lesion_idx_grouped]
        window = self.DICOM_window[lesion_idx_grouped][0]
        gender = float(self.gender[lesion_idx_grouped][0] == 'M')
        age = self.age[lesion_idx_grouped][0]/100
        if np.isnan(age) or age == 0:
            age = .5
        z_coord = self.norm_location[lesion_idx_grouped[0], 2]

        num_slice = cfg.INPUT.NUM_SLICES * cfg.INPUT.NUM_IMAGES_3DCE
        is_train = self.split=='train'
        if is_train and cfg.INPUT.DATA_AUG_3D is not False:
            slice_radius = diameters.min() / 2 * spacing / slice_intv * abs(cfg.INPUT.DATA_AUG_3D)  # lesion should not be too small
            slice_radius = int(slice_radius)
            if slice_radius > 0:
                if cfg.INPUT.DATA_AUG_3D > 0:
                    delta = np.random.randint(0, slice_radius+1)
                else:  # give central slice higher prob
                    ar = np.arange(slice_radius+1)
                    p = slice_radius-ar.astype(float)
                    delta = np.random.choice(ar, p=p/p.sum())
                if np.random.rand(1) > .5:
                    delta = -delta

                dirname, slicename = image_fn.split(os.sep)
                slice_idx = int(slicename[:-4])
                image_fn1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
                if os.path.exists(os.path.join(self.data_path, image_fn1)):
                    image_fn = image_fn1
        im, im_scale, crop = load_prep_img(self.data_path, image_fn, spacing, slice_intv,
                                           cfg.INPUT.IMG_DO_CLIP, num_slice=num_slice, is_train=is_train)

        im -= cfg.INPUT.PIXEL_MEAN
        im = torch.from_numpy(im.transpose((2, 0, 1))).to(dtype=torch.float)

        boxes_new = boxes0.copy()
        if cfg.INPUT.IMG_DO_CLIP:
            offset = [crop[2], crop[0]]
            boxes_new -= offset*2
        boxes_new *= im_scale
        boxes = torch.as_tensor(boxes_new).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (im.shape[2], im.shape[1]), mode="xyxy")

        num_boxes = boxes.shape[0]
        classes = torch.ones(num_boxes, dtype=torch.int)  # lesion/nonlesion
        target.add_field("labels", classes)
        if cfg.MODEL.TAG_ON:
            tags = torch.zeros(num_boxes, self.num_tags, dtype=torch.int)
            reliable_neg_tags = torch.zeros(num_boxes, self.num_tags, dtype=torch.int)
            for p in range(num_boxes):
                if lesion_idx_grouped[p] in self.lesion_tags.keys():
                    pos_tags = self.lesion_tags[lesion_idx_grouped[p]]
                    tags[p, pos_tags] = 1
                    ex_tags = [e for l in pos_tags for e in self.exclusive_list[l] if e not in pos_tags]
                    reliable_neg_tags[p, ex_tags] = 1
                else:
                    tags[p] = -1  # no tag exist for this lesion, the loss weights for this lesion should be zero
                    reliable_neg_tags[p] = -1  # no tag exist for this lesion, the loss weights for this lesion should be zero
            target.add_field("tags", tags)
            target.add_field("reliable_neg_tags", reliable_neg_tags)

            if self.split == 'test':
                tags = torch.zeros(num_boxes, self.num_tags, dtype=torch.int)
                for p in range(num_boxes):
                    if lesion_idx_grouped[p] in self.manual_annot_test_tags.keys():
                        tags[p, self.manual_annot_test_tags[lesion_idx_grouped[p]]] = 1
                    else:
                        tags[p] = -1  # no tag exist for this lesion, the loss weights for this lesion should be zero
                target.add_field("manual_annot_test_tags", tags)

        if cfg.INPUT.IMG_DO_CLIP:
            recists -= offset * 4
        recists *= im_scale
        if cfg.MODEL.MASK_ON:
            masks = []
            for recist in recists:
                mask = gen_mask_polygon_from_recist(recist)
                masks.append([mask])
            masks = SegmentationMask(masks, (im.shape[-1], im.shape[-2]))
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            im, target = self.transforms(im, target)

        infos = {'im_index': index, 'lesion_idxs': lesion_idx_grouped, 'image_fn': image_fn, 'diameters': diameters*spacing,
                 'crop': crop, 'recists': recists, 'window': window, 'spacing': spacing, 'im_scale': im_scale,
                 'gender': gender, 'age': age, 'z_coord': z_coord}
        return im, target, infos

    def __len__(self):
        return len(self.image_fn_list)

    def load_split_index(self):
        """
        need to group lesion indices to image indices, since one image can have multiple lesions
        :return:
        """

        split_list = ['train', 'val', 'test', 'small']
        index = split_list.index(self.split)
        if self.split != 'small':
            lesion_idx_list = np.where((self.train_val_test == index + 1) & ~self.noisy)[0]
        else:
            lesion_idx_list = np.arange(30)
        fn_list = self.filenames[lesion_idx_list]
        fn_list_unique, inv_ind = np.unique(fn_list, return_inverse=True)
        lesion_idx_grouped = [lesion_idx_list[inv_ind==i] for i in range(len(fn_list_unique))]
        return fn_list_unique, lesion_idx_grouped

    def loadinfo(self, path):
        """load annotations and meta-info from DL_info.csv"""
        info = []
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                filename = row[0]  # replace the last _ in filename with / or \
                idx = filename.rindex('_')
                row[0] = filename[:idx] + os.sep + filename[idx + 1:]
                info.append(row)
        info = info[1:]

        # the information not used in this project are commented
        self.filenames = np.array([row[0] for row in info])
        # self.patient_idx = np.array([int(row[1]) for row in info])
        # self.study_idx = np.array([int(row[2]) for row in info])
        # self.series_idx = np.array([int(row[3]) for row in info])
        self.slice_idx = np.array([int(row[4]) for row in info])
        self.d_coordinate = np.array([[float(x) for x in row[5].split(',')] for row in info])
        self.d_coordinate -= 1
        self.boxes = np.array([[float(x) for x in row[6].split(',')] for row in info])
        self.boxes -= 1  # coordinates in info file start from 1
        self.diameter = np.array([[float(x) for x in row[7].split(',')] for row in info])
        self.norm_location = np.array([[float(x) for x in row[8].split(',')] for row in info])
        # self.type = np.array([int(row[9]) for row in info])
        self.noisy = np.array([int(row[10]) > 0 for row in info])
        # self.slice_range = np.array([[int(x) for x in row[11].split(',')] for row in info])
        self.spacing3D = np.array([[float(x) for x in row[12].split(',')] for row in info])
        self.spacing = self.spacing3D[:, 0]
        self.slice_intv = self.spacing3D[:, 2]  # slice intervals
        # self.image_size = np.array([[int(x) for x in row[13].split(',')] for row in info])
        self.DICOM_window = np.array([[float(x) for x in row[14].split(',')] for row in info])
        self.gender = np.array([row[15] for row in info])
        self.age = np.array([float(row[16]) for row in info])  # may be NaN
        self.train_val_test = np.array([int(row[17]) for row in info])
