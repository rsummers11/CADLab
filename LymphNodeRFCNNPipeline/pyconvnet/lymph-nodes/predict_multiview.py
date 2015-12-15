'produce a submission file from cuda-convnet multiview predictions'
'needs Python 2.7 for dictionary comprehension (labels_dict)'

import csv
import sys
import os
import cPickle as pickle
import numpy as np

input_file = sys.argv[1]
output_file = sys.argv[2]
use_multiview = int( sys.argv[3] )

#

#label_names = [ 'airplane', 'automobile', 'bird',  'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
#label_names = ['false_lymphnode', 'true_lymphnode', 'notlabeled3', 'notlabeled4', 'notlabeled5', 'notlabeled6', 'notlabeled7', 'notlabeled8', 'notlabeled9', 'notlabeled10']
label_names = ['false_lymphnode', 'true_lymphnode']
labels_dict = { i: x for i, x in enumerate( label_names ) }

print "Writing to " + output_file
writer = csv.writer( open( output_file, 'wb' ))
writer.writerow( [ 'id', 'label', 'pred' ] )
counter = 1

d = pickle.load( open( input_file, 'rb' ))

data = d['data']
print "data: {}".format(data)

label_indexes = np.argmax( d['data'], axis = 1 ) # get index of label prediction with highest probability
print "label_indexes: {}".format( label_indexes )

meta_data = d['metadata']
print "meta_data: {}".format( meta_data )

for i in label_indexes:
	label = labels_dict[i]
	pred = data[counter-1]
	if use_multiview: # multi-view: 10 observations per volume summed up!
		pred = pred/10
	writer.writerow( [ counter, label, pred ] )
	counter += 1
		
#assert( counter == 300001 )
