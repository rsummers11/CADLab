# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Added Batch3dceCollator"""

from maskrcnn.structures.image_list import to_image_list
from maskrcnn.config import cfg


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        infos = transposed_batch[2]
        return images, targets, infos


class Batch3dceCollator(object):
    """
    Batch collator for 3DCE. One multi-slice image is split to NUM_IMAGES_3DCE inputs
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible
        self.num_slice = cfg.INPUT.NUM_SLICES
        self.num_image = cfg.INPUT.NUM_IMAGES_3DCE

    def __call__(self, batch):
        # split multiple slices to multiple images
        images = ()
        targets = []
        infos = []
        for im, target, info in batch:
            images += im.split(int(im.shape[0]/self.num_image))
            targets += [target]#*self.num_image  # only the targets and info for central image of each batch is useful
            infos += [info]#*self.num_image
        images = to_image_list(images, self.size_divisible)
        return images, tuple(targets), tuple(infos)
