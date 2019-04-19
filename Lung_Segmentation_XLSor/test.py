'''
Youbao Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
April 2019

For testing, you need to modify some arguments according to your own setting and run the command "python test.py".
'''



import argparse
import numpy as np

import torch
from torch.utils import data
from networks.xlsor import XLSor
from dataset.datasets import XRAYDataTestSet
import os
from PIL import Image as PILImage

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 1

DATA_DIRECTORY = './data/NIH/images'
DATA_LIST_PATH = 'png'
RESTORE_FROM = './models/XLSor.pth'

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    return parser.parse_args()


def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model = XLSor(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(XRAYDataTestSet(args.data_dir, args.data_list, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, size, name = batch
        with torch.no_grad():
            prediction = model(image.cuda(), args.recurrence)
            if isinstance(prediction, list):
                prediction = prediction[0]
            prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
        output_im = PILImage.fromarray((np.clip(prediction[:,:,0],0,1)* 255).astype(np.uint8))
        output_im.save('./outputs/' + os.path.basename(name[0]).replace('.png', '_xlsor.png'), 'png')


if __name__ == '__main__':
    main()
