import os
import sys
import numpy as np
import cPickle as pickle
import math
from natsort import natsorted
from PIL import Image
from PIL import ImageOps
from array import *
import random
import fnmatch
import shutil

def recursive_glob(rootdir='.', str1=''):
	print " searching for '{}' in {} ...".format( str1, rootdir )
	return [os.path.join(rootdir, filename)
		for rootdir, dirnames, filenames in os.walk(rootdir)
		for filename in filenames if (str1 in filename)]

def main():
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	search_ending = sys.argv[3]

	names = recursive_glob(input_dir, search_ending)
	nr_names = names.__len__()
	print " Found {} files with search pattern '{}'.".format( nr_names, search_ending )

	print " ... Converting files with '{}' from RGB to Gray ...".format( search_ending )
	counter = 0
	conv_counter = 0
	already_gray_counter = 0
	for n in names:
		img = Image.open( n, 'r')
		img.load()
		# check if already RGB image
		if img.getbands().__len__() is 3:
			red, green, blue = img.split()
			#red.dtype = 'uint16'
			red.save( os.path.join(output_dir, os.path.basename(n) ) ) # assuming red is the same as the gray image (the way I implemented it)
			conv_counter += 1
		else:
			#print "image is already gray."
			already_gray_counter += 1
			img.save( os.path.join(output_dir, os.path.basename(n) ) ) # just resave already gray image

		counter += 1
		if counter % 1000 == 0:
			print " {}: {} of {} images processed ({} already grayscale)".format(n,counter, nr_names, already_gray_counter)

	print " Converted {} of a total of {} images.".format( conv_counter, counter )

if __name__ == '__main__':
	main()
