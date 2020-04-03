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

import numpy as np
import torch
import torch.nn as nn
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
import os
import copy
from sklearn.metrics import roc_auc_score
from CXR_Data_Generator import DataGenerator

class Trainer:

	def epoch_train(self, model, dataLoader, optimizer, criterion, gpu_id):
		model.train()
		loss_train = 0
		loss_train_norm = 0
		loss_tensor_mean_train = 0
		output_list = []
		label_list = []
		# current_iter = 0
		for data in dataLoader:
			# current_iter += 1
			inputs, labels, img_names = data
			inputs = inputs.cuda(gpu_id, non_blocking=True)
			labels = labels.cuda(gpu_id, non_blocking=True)
			labels = labels.view(labels.size()[0],-1) #add for BCE loss

			optimizer.zero_grad()			
			outputs = model(inputs)
			# _, preds = torch.max(outputs.data, 1)
			if isinstance(outputs, tuple):
				outputs = outputs[0]
				score = torch.sigmoid(outputs)
			else:
				score = torch.sigmoid(outputs)
			preds = score>0.5
			preds = preds.type(torch.cuda.LongTensor)
			
			labels = labels.type(torch.cuda.FloatTensor) #add for BCE loss
			loss = criterion(outputs, labels)
			loss_tensor_mean_train += loss

			labels = labels.data.cpu().numpy()
			outputs = outputs.data.cpu().numpy()

			for i in range(outputs.shape[0]):
				output_list.append(outputs[i].tolist())
				label_list.append(labels[i].tolist())

			loss_train_norm += 1
			loss.backward()
			optimizer.step()

		loss_tensor_mean_train = np.float(loss_tensor_mean_train) / loss_train_norm
		epoch_auc =  roc_auc_score(np.array(label_list), np.array(output_list))
		return loss_tensor_mean_train, epoch_auc

	def epoch_val(self, model, dataLoader, criterion, gpu_id):
		model.eval()
		loss_val = 0
		loss_val_norm = 0
		loss_tensor_mean_val = 0
		running_corrects = 0
		output_list = []
		label_list = []
		with torch.no_grad():
			for data in dataLoader:
				inputs, labels, img_names = data
				labels = labels.type(torch.FloatTensor) #add for BCE loss
				inputs = inputs.cuda(gpu_id, non_blocking=True)
				labels = labels.cuda(gpu_id, non_blocking=True)
				labels = labels.view(labels.size()[0],-1) #add for BCE loss

				# outputs = model(inputs)[0] #inception
				outputs = model(inputs)
				loss_tensor = criterion(outputs, labels)
				loss_tensor_mean_val += loss_tensor
				
				labels = labels.data.cpu().numpy()
				outputs = outputs.data.cpu().numpy()

				for i in range(outputs.shape[0]):
					output_list.append(outputs[i].tolist())
					label_list.append(labels[i].tolist())

				loss_val_norm += 1
		loss_tensor_mean_val = np.float(loss_tensor_mean_val) / loss_val_norm
		epoch_auc =  roc_auc_score(np.array(label_list), np.array(output_list))
		return loss_tensor_mean_val, epoch_auc
		
	def train(self, img_dir, split_train, split_val, model, batch_size, 
		num_epoch, img_size, crop_size, lr, net_arc, gpu_id):

		best_model_wts = copy.deepcopy(model.state_dict())
		
		#-------------------- SETTINGS: CXR DATA TRANSFORMS -------------------
		normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
		data_transforms = {
		'train': transforms.Compose([
			transforms.Resize(img_size),
			transforms.CenterCrop(crop_size),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.25, contrast=0.25),
			transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
			transforms.ToTensor(),
			transforms.Normalize(normalizer[0], normalizer[1])]),
		'val': transforms.Compose([
			transforms.Resize(img_size),
			transforms.CenterCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize(normalizer[0], normalizer[1])])}
			# tramsforms.Tencrop(crop_size)
			# transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
			# normalize = transforms.Normalize(normalizer[0], normalizer[1])
			# transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
			# ])}

		#-------------------- CXR DATASET BUILDERS -------------------
		datasetTrain = DataGenerator(img_dir = img_dir, split_file = split_train, 
			transform = data_transforms['train'])
		dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=batch_size, 
			shuffle=True, num_workers=32, pin_memory=True)

		datasetVal = DataGenerator(img_dir = img_dir, split_file = split_val,
			transform = data_transforms['val'])
		dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch_size, 
			shuffle=True, num_workers=32,  pin_memory=True)

		dataloaders = {}
		dataloaders['train'] =dataLoaderTrain
		dataloaders['val'] =dataLoaderVal

		print('Number of training CXR images: {}\nNumber of validation CXR images: {}\n'
			.format(len(datasetTrain), len(datasetVal)))
		dataset_sizes = {'train': len(datasetTrain), 'val': len(datasetVal)}

		#-------------------- SETTINGS: OPTIMIZER & SCHEDULER -------------------
		optimizer = optim.SGD(model.parameters(), lr, weight_decay=0.0001, momentum=0.9)
		scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience = 5)

		#-------------------- SETTINGS: LOSS FUNCTION -------------------
		criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()

		#-------------------- TRAINING -------------------
		loss_min = 9999999

		for epoch in range(num_epoch):
			# print('Epoch {}/{}'.format(epoch+1, num_epoch))

			trainer_cxr = Trainer()
			loss_train, auc_train = trainer_cxr.epoch_train(model, dataLoaderTrain, optimizer, criterion, gpu_id)

			loss_val, auc_val = trainer_cxr.epoch_val(model, dataLoaderVal, criterion, gpu_id)

			scheduler.step(loss_val)

			if loss_val < loss_min:
				loss_min = loss_val
				torch.save({'Epoch': epoch + 1, 'state_dict': model.state_dict(), 
					'best_loss': loss_min, 'optimizer' : optimizer.state_dict()}, \
					'./trained_models_nih/'+net_arc+'_'+str(img_size)+'_'+str(batch_size)+'_'+str(lr))
				print ('Epoch {0:02d}/{1} [save]'.format(epoch + 1, num_epoch))
				
			else:
				print ('Epoch {0:02d}/{1} [skip]'.format(epoch + 1, num_epoch))

			print ('*'*20)	
			print ('Train_AUC: {:.4f}     Train_loss: {:.4f}'\
				.format(auc_train, loss_train))
			print ('  Val_AUC: {:.4f}     Val_loss: {:.4f}'\
				.format(auc_val, loss_val))
			print ('\n')
