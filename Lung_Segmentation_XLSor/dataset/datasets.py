'''
Youbao Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
April 2019

This file contains two classes. One (i.e. XRAYDataSet) is for loading training data
and the other one (i.e. XRAYDataTestSet) is for loading testing data.
'''

import os.path as osp
import glob
import numpy as np
import random
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt

def rotate_bound(image, angle, flag):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flag)

class XRAYDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 512), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=0):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            if name[:12]=='Augmentation':
                label_file = osp.join(self.root, name[:17] + '_mask.png')
            else:
                label_file = osp.join(self.root, name.replace('.png', '_mask.png'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.35 + random.random() * 0.9
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        angle = -15.0 + random.random() * 30.0
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = rotate_bound(image, angle, cv2.INTER_CUBIC)
        image = cv2.resize(image, (576, 576), interpolation=cv2.INTER_CUBIC)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = rotate_bound(label, angle, cv2.INTER_NEAREST)
        label = cv2.resize(label, (576, 576), interpolation=cv2.INTER_NEAREST)/255
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        top_p = random.randint(0, pad_h)
        left_p = random.randint(0, pad_w)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, top_p, pad_h - top_p, left_p,
                                         pad_w - left_p, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, top_p, pad_h - top_p, left_p,
                                           pad_w - left_p, cv2.BORDER_CONSTANT,
                                           value=(0,))
        else:
            img_pad, label_pad = image, label

        img_pad -= self.mean
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = np.expand_dims(label, axis=0)
        return image.copy(), label.copy(), np.array(size), name

class XRAYDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128), scale=False, mirror=False):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.files = []
        if osp.exists(list_path):
            self.img_ids = [i_id.strip() for i_id in open(list_path)]
            for name in self.img_ids:
                img_file = osp.join(self.root, name)
                self.files.append({
                    "img": img_file
                })
        else:
            self.img_ids = glob.glob(self.root + '/*' + list_path)
            for name in self.img_ids:
                self.files.append({
                    "img": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        size = image.shape
        name = datafiles["img"]
        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, np.array(size), name


if __name__ == '__main__':
    dst = XRAYDataSet("./data", "./data/train_list.txt")
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, _, _  = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
