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

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DataGenerator(Dataset):

	def __init__(self, img_dir, split_file, transform):

		self.img_name_list = []
		self.transform = transform

		with open(split_file, 'r') as split_name:
			img_and_label_list = split_name.readlines()

		for index in img_and_label_list:
			img_path = os.path.join(img_dir, index.split()[0])
			
			self.img_name_list.append(img_path)
			

	def __getitem__(self, index):

		img_name = self.img_name_list[index]
		image_data = Image.open(img_name).convert('RGB')
		image_data = self.transform(image_data)

		return (image_data, img_name)

	def __len__(self):

		return len(self.img_name_list)
