import os
import sys
import numpy as np
import cPickle as pickle
import math
from PIL import Image
from PIL import ImageOps
from array import *
import random
import fnmatch
import shutil

def recursive_glob(rootdir='.', str=''):
	return [os.path.join(rootdir, filename)
		for rootdir, dirnames, filenames in os.walk(rootdir)
		for filename in filenames if str in filename]		
	
def main():	
	batch_dir = sys.argv[1]
	batch_prefix = sys.argv[2]
	batch_range = sys.argv[3]
	batch_combined_file = sys.argv[4]

	batch_start = int( batch_range[0:batch_range.find('-')] )
	batch_end = int( batch_range[batch_range.find('-')+1:batch_range.__len__()] )
	
	print " combining batches {} to {}:".format(batch_start, batch_end)
	# define filenames to be combined
	filenames = []
	for x in range(batch_start, batch_end+1):	
		fname = batch_dir + '/' + batch_prefix+str(x)
		#print " looking for " + fname
		filenames.append( fname )

	# combine them
	with open(batch_combined_file, 'w') as outfile:
	    for fname in filenames:
		print "adding file: " + fname
		with open(fname) as infile:
		    for line in infile:
		        outfile.write(line)

	# and save
	print " saved combined log to " + batch_combined_file

if __name__ == '__main__':
	main()

