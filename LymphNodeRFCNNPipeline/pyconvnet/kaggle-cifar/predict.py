'produce a submission file from cuda-convnet batch predictions'

import csv
import sys
import os
import cPickle as pickle
import numpy as np
from natsort import natsorted

input_dir = sys.argv[1]
output_file = sys.argv[2]

#

label_names = [ 'airplane', 'automobile', 'bird',  'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
labels_dict = { i: x for i, x in enumerate( label_names ) }

batch_names = [ d for d in os.listdir( input_dir ) if d.startswith( 'data_batch_' ) ]
batch_names = natsorted( batch_names )

writer = csv.writer( open( output_file, 'wb' ))
writer.writerow( [ 'id', 'label' ] )
counter = 1

for n in batch_names:
	
	print n
	batch_path = os.path.join( input_dir, n )
	d = pickle.load( open( batch_path, 'rb' ))
	
	label_indexes = np.argmax( d['data'], axis = 1 )
	print label_indexes
	
	for i in label_indexes:
		label = labels_dict[i]
		writer.writerow( [ counter, label ] )
		counter += 1
		
assert( counter == 300001 )
