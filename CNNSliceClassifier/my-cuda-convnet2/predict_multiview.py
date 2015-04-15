'produce a submission file from cuda-convnet multiview predictions'
'needs Python 2.7 for dictionary comprehension (labels_dict)'

import csv
import sys
import os
import cPickle as pickle
import numpy as np
from make_data import make_general_data as data_functions

input_file = sys.argv[1]
output_file = sys.argv[2]
data_batch_text_file = sys.argv[3]
label_type = sys.argv[4]	
#use_multiview = int( sys.argv[3] )

label_names = data_functions.get_label_names( label_type )
print "There are {} label names:".format(label_names.__len__())
for label_name in label_names:
	print "    {}".format(label_name)
labels_dict = { i: x for i, x in enumerate( label_names ) }

print "read batch text file " + data_batch_text_file
reader = csv.reader( open( data_batch_text_file, 'rb') )
number_files = reader.next()

print "Writing to " + output_file
writer = csv.writer( open( output_file, 'wb' ))
writer.writerow( [ str(number_files), 'id', 'filename', 'label', 'prediction', 'probs', label_names ] )
counter = 1

d = pickle.load( open( input_file, 'rb' ))

labels = d['labels']
print "labels: {}".format( labels )

data = d['data']
print "data: {}".format(data)

label_indexes = np.argmax( d['data'], axis = 1 ) # get index of label prediction with highest probability
print "label_indexes: {}".format( label_indexes )

for i in label_indexes:
	label = labels_dict[i]
	pred = data[counter-1]
	#if use_multiview: # multi-view: 10 observations per volume summed up!
	#	pred = pred/10
	inputs = reader.next()
	writer.writerow( [ counter-1, ','.join(str(ip) for ip in inputs), label, ','.join(str(p) for p in pred) ] )
	counter += 1

print " wrote {} predictions.".format(counter)
		
#assert( counter == 300001 )

