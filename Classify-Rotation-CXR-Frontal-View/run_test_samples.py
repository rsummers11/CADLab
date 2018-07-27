"""
Yuxing Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
July 2018

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
from __future__ import division
import os
import argparse
import distutils.util
import numpy as np
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
import time
import copy
from shutil import copyfile
from sklearn import metrics
from DataGenerator import DataGenerator

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def mk_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def run_test():

	model = models.__dict__['resnet18'](pretrained=False)

	# number of classes
	numClass = 1
	# image folder location
	img_dir = './images-sample'

	# dataset split files
	split_name = 'test'
	splits = [split_name]
	split_file_suffix = '_sample_list.txt'
	split_files = {}
	for split in splits:
		split_files[split] = os.path.join(split+split_file_suffix)

	# modify the last FC layer to number of classes
	num_ftrs = model.fc.in_features
	model.avgpool = nn.AvgPool2d(7)
	model.fc = nn.Linear(num_ftrs, numClass)
	model = model.cuda()

	model.load_state_dict(torch.load('./trained-models/resnet18_cxr_rotation_cls.pth')['state_dict'])

	mk_dir('./images-90/')
	mk_dir('./images-0/')
	
	test(img_dir, split_files[split_name], split_name, model, batch_size=32, img_size=256, crop_size=224)


def test(img_dir, split_test, split_name, model, batch_size, img_size, crop_size):
	since = time.time()

	# -------------------- SETTINGS: DATA TRANSFORMS
	normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
	data_transforms = {split_name: transforms.Compose([
		transforms.Resize(img_size),
		transforms.CenterCrop(crop_size),
		transforms.ToTensor(),
		transforms.Normalize(normalizer[0], normalizer[1])])}

	# -------------------- SETTINGS: DATASET BUILDERS
	datasetTest = DataGenerator(img_dir=img_dir, split_file=split_test,
								transform=data_transforms[split_name])
	dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size,
								shuffle=False, num_workers=32)

	dataloaders = {}
	dataloaders[split_name] = dataLoaderTest

	print('Number of testing CXR images: {}'.format(len(datasetTest)))
	dataset_sizes = {split_name: len(datasetTest)}

	# -------------------- TESTING
	model.train(False)
	running_corrects = 0
	output_list = []
	# label_list = []
	preds_list = []

	# Iterate over data.
	for data in dataloaders[split_name]:
		inputs, img_names = data
		
		# wrap them in Variable
		inputs = Variable(inputs.cuda(), volatile=True)

		# forward
		outputs = model(inputs)
		score = torch.sigmoid(outputs)
		score_np = score.data.cpu().numpy()
		preds = score>0.5
		preds_np = preds.data.cpu().numpy()
		preds = preds.type(torch.cuda.LongTensor)

		outputs = outputs.data.cpu().numpy()

		for j in range(len(img_names)):
			print(str(img_names[j]) + ': ' + str(score_np[j]))
			img_name = str(img_names[j]).rsplit('/',1)[1]
			if score_np[j] > 0.5:
				copyfile(str(img_names[j]), './images-90/'+img_name)

			if score_np[j] < 0.5:
				copyfile(str(img_names[j]), './images-0/'+img_name)

		for i in range(outputs.shape[0]):
			output_list.append(outputs[i].tolist())
			preds_list.append(preds_np[i].tolist())

# if __name__ == '__main__':
run_test()
