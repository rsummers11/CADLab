'make cuda-convnet batches from images in the input dir; start numbering batches from 7'

import os
import sys
import numpy as np
import cPickle as pickle
from natsort import natsorted
from PIL import Image
from PIL import ImageOps

def process( image ):
	image = np.array( image )           # 32 x 32 x 3
	image = np.rollaxis( image, 2 )     # 3 x 32 x 32
	image = image.reshape( -1 )         # 3072
	return image
	
def get_batch_path( output_dir, number ):
	filename = "data_batch_{}".format( number )
	return os.path.join( output_dir, filename )

def get_empty_batch():	
	return np.zeros(( 3072, 0 ), dtype = np.uint8 )
	
def write_batch( path, batch ):
	print "writing {}...\n".format( path )
	labels = [ 0 for x in range( batch.shape[1] ) ]
	d = { 'labels': labels, 'data': batch }
	pickle.dump( d, open( path, "wb" ))
	
def main():
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]

	try:
		batch_counter = int( sys.argv[3] )
	except IndexError:
		batch_counter = 7
	
	batch_size = 10000
	
	print "reading file names..."
	names = [ d for d in os.listdir( input_dir ) if d.endswith( '.png') ]
	names = natsorted( names )
	
	if batch_counter > 7:
		omit_batches = batch_counter - 7
		omit_images = omit_batches * batch_size
		names = names[omit_images:]
		print "omiting {} images".format( omit_images )

	current_batch = get_empty_batch()
	counter = 0
	
	for n in names:
		
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
			write_batch( batch_path, current_batch )
			
			batch_counter += 1
			current_batch = get_empty_batch()
		
		counter += 1
		if counter % 1000 == 0:
			print n

	

if __name__ == '__main__':
	main()
