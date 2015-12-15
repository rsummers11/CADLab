'make cuda-convnet batches from images in the input dir; start numbering batches from 7'

import os
import sys
import numpy as np
import cPickle as pickle
from natsort import natsorted
from PIL import Image
from PIL import ImageOps
from array import *

def process( image ):
	image = np.array( image )           # 32 x 32 x 3
	image = np.rollaxis( image, 2 )     # 3 x 32 x 32
	image = image.reshape( -1 )         # 3072
	return image
	
def get_batch_path( output_dir, number ):
	filename = "data_batch_{}".format( number )
	return os.path.join( output_dir, filename )

def get_empty_batch():	
	return np.zeros(( 3072, 0 ), dtype = np.uint8 ) # 32 x 32 x 3
	
def write_batch( path, batch, labels):
	print "writing {} ...\n".format( path )
	#labels = [ 1 for x in range( batch.shape[1] ) ]
	print batch.shape[1]
	print labels
	d = { 'labels': labels, 'data': batch }
	pickle.dump( d, open( path, "wb" ))
	
def main():	
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	search_str = sys.argv[3]

	print "searching for: {} ...\n".format( search_str )
	names = [ d for d in os.listdir( input_dir ) if d.endswith(search_str) ]	#e.g. '.png'
	batch_size = names.__len__()
	print batch_size

	# lymph node dictionary (saved as batches.meta)
	##'label_names': ['false_lymphnode', 'true_lymphnode', 'notlabeled3', 'notlabeled4', 'notlabeled5', 'notlabeled6', 'notlabeled7', 'notlabeled8', 'notlabeled9', 'notlabeled10'],	
	##'label_names': ['false_lymphnode', 'true_lymphnode'],
	data_mean = np.zeros((3072,1), dtype=np.float32) # To my understanding, this is the mean images of all training and test images (zero for now)
	dict = ({'num_cases_per_batch': batch_size,
			 'label_names': ['false_lymphnode', 'true_lymphnode'],
			 'num_vis': 3072,
			 'data_mean': data_mean
	})
	with open( os.path.join( output_dir, 'batches.meta' ), 'wb') as handle:
		pickle.dump(dict, handle)
	
	# generate batches
	try:
		batch_counter = 1
	except IndexError:
		batch_counter = 7
	
	print "reading file names..."
	names = natsorted( names )
	
	if batch_counter > 7:
		omit_batches = batch_counter - 7
		omit_images = omit_batches * batch_size
		names = names[omit_images:]
		print "omiting {} images".format( omit_images )

	current_batch = get_empty_batch()
	current_labels = []
	counter = 0
	
	for n in names:
		# read label
		shortname = os.path.splitext( os.path.join( input_dir, n ))[0] # name without extension
		label = 2 # unknown category (as we are using these for testing)
		
		#print extension
		current_labels.append(label)
		
		# read image
		image = Image.open( os.path.join( input_dir, n ))
		try:
			image = process( image )
		except ValueError:
			print "problem with image {}".format( n )
			sys.exit( 1 )

		image = image.reshape( -1, 1 )
		current_batch = np.hstack(( current_batch, image ))
		
		if current_batch.shape[1] == batch_size:
			batch_path = get_batch_path( output_dir, batch_counter )
			write_batch( batch_path, current_batch, current_labels )
			
			batch_counter += 1
			current_batch = get_empty_batch()
			current_labels = []
		
		counter += 1
		if counter % 1000 == 0:
			print " read image {} of {}:".format(counter+1, names.__len__())
			print n

	

if __name__ == '__main__':
	main()
