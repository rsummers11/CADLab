"""
Yuxing Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
March 2020

THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function
import os
import argparse
import distutils.util
import numpy as np
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
import copy

from CXR_Trainer_bin import Trainer



parser = argparse.ArgumentParser(description='PyTorch NIH-CXR Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet121')
parser.add_argument('--pretrained', '-p', type=distutils.util.strtobool, 
	default='True')
parser.add_argument('--img_size', '-sz', default=256, type=int)
parser.add_argument('--crop_size', '-cs', default=224, type=int)
parser.add_argument('--epoch', '-ep', default=50, type=int)
parser.add_argument('--batch_size', '-bs', default=64, type=int)
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
parser.add_argument('--gpu_id', '-gpu', default=0, type=int)

def main():
	global args
	args = parser.parse_args()
	# print(args)

	if args.pretrained:
		print("=> using pre-trained model '{}'".format(args.arch))
		model = models.__dict__[args.arch](pretrained=True)
	else:
		print("=> creating model '{}'".format(args.arch))
		model = models.__dict__[args.arch](pretrained=False)

	torch.cuda.set_device(args.gpu_id)

	# number of classes
	numClass = 1 # 1 for binary classification

	# image folder location
	img_dir = './images-nih'

	# dataset split files
	split_file_dir = './dataset_split'
	splits = ['train', 'val']

	split_file_suffix = '.txt'

	split_files = {}
	for split in splits:
		split_files[split] = os.path.join(split_file_dir, 
			split+split_file_suffix)

	print(model)

	# modify the last FC layer to number of classes
	num_ftrs = model.classifier.in_features
	model.classifier = nn.Linear(num_ftrs, numClass)
	# model = nn.DataParallel(model).cuda()
	model = model.cuda()

	trainer_cxr = Trainer()
	trainer_cxr.train(img_dir, split_files['train'], split_files['val'], 
		model, args.batch_size, args.epoch, args.img_size, args.crop_size,
		args.learning_rate, args.arch, args.gpu_id)


if __name__ == '__main__':
	main()
