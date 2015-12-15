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

	print " ... Cleaning zero files with '{}' ...".format( search_ending )
	counter = 0
	clean_count = 0
	for n in names:
		img = Image.open( n, 'r')
		img.load()
		# check if already RGB image
		if img.getbands().__len__() is 3:
			print "WARNING: This assumes grayscale images!!! Skip..."
		else: 
			arr = np.asarray(img)
			tot = arr.sum(0).sum(0)
			#print tot
			if tot == 0:
				shutil.move(n, output_dir)
				clean_count += 1

		counter += 1
		if counter % 1000 == 0:
			print " {}: {} of {} images ({} cleaned)".format(n,counter, nr_names, clean_count)

	print " Cleaned {} of a total of {} images.".format( clean_count, nr_names )

if __name__ == '__main__':
	main()
