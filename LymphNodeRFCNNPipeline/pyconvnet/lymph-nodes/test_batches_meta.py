'make cuda-convnet batches from images in the input dir; start numbering batches from 7'

import os
import sys
import numpy as np
import cPickle as pickle
from natsort import natsorted
from PIL import Image
from PIL import ImageOps
from array import *

def unpickle(fname):
	#print 'loading file ' + fname
	fo = open(fname, 'rb')
	data = pickle.load(fo)
	fo.close()
	return data

fname='D:/HolgerRoth/DropConnect/MyDropConnect/data/cifar-10-py-colmajor/batches.meta'	
fname='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/Cropped_test_png_batches/batches.meta'
data = unpickle(fname)
data_mean = data['data_mean']
#print data_mean
#print data_mean.size
#print len(data_mean)
print data
