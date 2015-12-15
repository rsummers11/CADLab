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

def recursive_glob(rootdir='.', str1='', str2=''):
	print " searching for '{}' and '{}' in {} ...".format( str1, str2, rootdir )
	return [os.path.join(rootdir, filename)
		for rootdir, dirnames, filenames in os.walk(rootdir)
		for filename in filenames if (str1 in filename and str2 in filename)]
	
def main():	
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]

	search_str1 = sys.argv[3]
	search_str2 = sys.argv[4]

	names = recursive_glob(input_dir, search_str1, search_str2)
	print " Found {} files with search pattern '{}' and '{}'.".format( names.__len__(), search_str1, search_str2 )
	
	print " ... copying files to '{}' ...".format( output_dir )
	counter = 0
	for n in names:
		#path, filename = os.path.split( n )
		#print('Path is %s and file is %s' % (path, filename))
		#outname = os.path.join(output_dir,filename)
		#print('out name is %s ' % (outname))

		# copy file n to output_dir
		shutil.copy(n, output_dir)
		
		counter += 1
		if counter % 1000 == 0:
			print " copy image {} of {}:".format( counter, names.__len__() )
			print n

if __name__ == '__main__':
	main()
