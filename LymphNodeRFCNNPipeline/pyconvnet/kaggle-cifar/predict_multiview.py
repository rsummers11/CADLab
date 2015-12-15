'produce a submission file from cuda-convnet multiview predictions'
'needs Python 2.7 for dictionary comprehension (labels_dict)'

import csv
import sys
import os
import cPickle as pickle
import numpy as np

input_file = sys.argv[1]
output_file = sys.argv[2]

#

label_names = [ 'airplane', 'automobile', 'bird',  'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
labels_dict = { i: x for i, x in enumerate( label_names ) }

writer = csv.writer( open( output_file, 'wb' ))
writer.writerow( [ 'id', 'label' ] )
counter = 1

d = pickle.load( open( input_file, 'rb' ))

label_indexes = np.argmax( d['data'], axis = 1 )
print label_indexes

for i in label_indexes:
	label = labels_dict[i]
	writer.writerow( [ counter, label ] )
	counter += 1
		
assert( counter == 300001 )
