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

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DataGenerator(Dataset):

	def __init__(self, img_dir, split_file, transform):

		self.img_name_list = []
		self.img_label_list = []
		self.transform = transform
		# self.img_directory = img_dir

		with open(split_file, 'r') as split_name:
			img_and_label_list = split_name.readlines()

		for index in img_and_label_list:
			img_path = os.path.join(img_dir, index.split()[0])
			# img_label = [int(index.split()[1])]
			img_label = int(index.split()[1])
			self.img_name_list.append(img_path)
			self.img_label_list.append(img_label)

	def __getitem__(self, index):

		img_name = self.img_name_list[index]
		# img_path = os.path.join(img_dir, img_name)
		image_data = Image.open(img_name).convert('RGB')
		image_data = self.transform(image_data)
		# image_label= torch.FloatTensor(self.img_label_list[index])
		image_label= self.img_label_list[index]

		return (image_data, image_label, img_name)

	def __len__(self):

		return len(self.img_name_list)
