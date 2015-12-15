'make cuda-convnet batches from images in the input dir; start numbering batches from 7'

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

def process( image ):
	image = np.array( image )           # 32 x 32 x 3
	image = np.rollaxis( image, 2 )     # 3 x 32 x 32
	image = image.reshape( -1 )         # 3072
	return image
	
def get_batch_path( output_dir, number ):
	filename = "data_batch_{}".format( number )
	return os.path.join( output_dir, filename )	

def get_batch_log_path( output_dir, number ):
	filename = "data_batch_{}.txt".format( number )
	return os.path.join( output_dir, filename )		
	
def get_empty_batch():	
	return np.zeros(( 3072, 0 ), dtype = np.uint8 ) # 32 x 32 x 3
	
def write_batch( path, batch, labels):
	print "writing {}...\n".format( path )
	d = { 'labels': labels, 'data': batch }
	pickle.dump( d, open( path, "wb" ))

def recursive_glob(rootdir='.', str=''):
	return [os.path.join(rootdir, filename)
		for rootdir, dirnames, filenames in os.walk(rootdir)
		for filename in filenames if str in filename]		
	
def main():	
	input_dir1 = sys.argv[1]
	input_dir2 = sys.argv[2]
	output_dir = sys.argv[3]

	batch_counter = int( sys.argv[4] )
	batch_number = int( sys.argv[5] )
	search_str = sys.argv[6]

	print "searching for: '{}' in '{}' and \n\t\t\t\t'{}' ...".format( search_str, input_dir1, input_dir2 )	
	names1 = recursive_glob(input_dir1, search_str)
	names2 = recursive_glob(input_dir2, search_str)
	
	names = names1 + names2
	
	print " Combining {} and {} files to a total of {} files.".format(names1.__len__(), names2.__len__(), names.__len__())
	
	print "reading file names..."
	batch_size = int( math.floor( names.__len__()/batch_number ) )
	print " Generating {} batch(es) from {} images with {} images per batch ...".format(batch_number, names.__len__(), batch_size)
	
	# Data Mean: To my understanding, this is the mean images of all training and test images (computed below)
	data_mean = np.zeros((3072,1), dtype=np.float32) 

	current_batch = get_empty_batch()
	current_labels = []
	counter = 0
	pos_counter = 0
	neg_counter = 0
	
	new_log = True
	
	print "Randomly shuffel names..."
	random.shuffle(names)
	for n in names:
		# read label
		#print " read label {} of {}: {}".format(counter+1, names.__len__(), n)
			
		if new_log is True:
			batch_log_path = get_batch_log_path( output_dir, batch_counter )
			fid = open( batch_log_path, 'w')
			new_log = False;
		
		# check if positive or negative example in filename (NOT WHOLE PATH)
		curr_path,curr_filename=os.path.split(n)
		if "pos" in curr_filename: 
			label = 1;
			pos_counter += 1
			#print " {} is POSITIVE: label = {}".format(curr_filename,label)
		elif "neg" in curr_filename: 
			label = 0;
			neg_counter += 1
			#print " {} is NEGATIVE: label = {}".format(curr_filename,label)
		else:
			print "Error!  This image is neither positive nor negative example ({}). Remove from folder or rename...".format(n)
			sys.exit( 1 )
			
		#print extension
		current_labels.append(label)
		
		# read image
		imagename = n
		image = Image.open( imagename )
		fid.write("{}, {}".format( imagename, label ) )
		fid.write( '\n' ) # new line in log
		try:
			image = process( image )
		except ValueError:
			print "problem with image {}".format( n )
			sys.exit( 1 )

		image = image.reshape( -1, 1 )
		current_batch = np.hstack(( current_batch, image ))
		data_mean = data_mean + image
		
		if current_batch.shape[1] == batch_size:
			new_log = True
			fid.close() # close last log
			batch_path = get_batch_path( output_dir, batch_counter )
			write_batch( batch_path, current_batch, current_labels )
			
			batch_counter += 1
			current_batch = get_empty_batch()
			current_labels = []
		
		counter += 1
		if counter % 1000 == 0:
			print " {}: {} of {} images".format(n,counter, names.__len__())
			print "    {} POSITIVE and {} NEGATIVE images so far.".format(pos_counter, neg_counter)	

	# if last batch is not full, delete log file
	if current_batch.shape[1] > 0 and current_batch.shape[1] < batch_size:
		print "Last batch not full -> label log as unused ..."
		fid.close() # close last log
		if os.path.isfile( batch_log_path ):
			os.rename( batch_log_path,  batch_log_path + '_unsused' )
		batch_path = get_batch_path( output_dir, batch_counter )
		write_batch( batch_path, current_batch, current_labels )
		if os.path.isfile( batch_path ):
			os.rename( batch_path,  batch_path + '_unsused' )		
			
	# compute total data mean and save as image
	data_mean = data_mean/names.__len__()
	mean_image_values = data_mean.reshape((3,32,32))
	mean_image_values = np.swapaxes(mean_image_values,0,2) # following conversions in convdata.py
	mean_image_values = np.swapaxes(mean_image_values,0,1)
	mean_image = Image.fromarray( np.uint8( mean_image_values ) )
	mean_image.save( os.path.join( output_dir, 'data_mean.png' ) )
	
	# Save lymph node dictionary (saved as batches.meta)
	dict = ({'num_cases_per_batch': batch_size,
			'label_names': ['false_lymphnode', 'true_lymphnode'],
			'num_vis': 3072,
			'data_mean': data_mean
	})
	with open( os.path.join( output_dir, 'batches.meta' ), 'wb') as handle:
		pickle.dump(dict, handle)				
		
	print " Batched a total of {} POSITIVE and {} NEGATIVE images.".format(pos_counter, neg_counter)	
			
if __name__ == '__main__':
	main()
