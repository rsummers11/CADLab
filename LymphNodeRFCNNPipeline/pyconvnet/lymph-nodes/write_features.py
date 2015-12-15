'produce a submission file from cuda-convnet multiview predictions'
'needs Python 2.7 for dictionary comprehension (labels_dict)'

import csv
import sys
import os
import cPickle as pickle
import numpy as np
import re

def parseRange(value):
	m = re.match("^(\d+)\-(\d+)$", value)
	try:
		if m: return range(int(m.group(1)), int(m.group(2)) + 1)
		return [int(value)]
	except:
		raise OptionException("argument is neither an integer nor a range")
def get_batch_path( output_dir, number ):
	filename = "data_batch_{}".format( number )
	return os.path.join( output_dir, filename )	

def main():
	input_dir = sys.argv[1]
	output_dir = os.path.basename( input_dir )
	
	batch_range = parseRange( sys.argv[2] )
	if batch_range.__len__() is 1:
		batch_range.append( batch_range[0] )

	counter = 0
	print "converting batches from {} to {}...".format(batch_range[0], batch_range[batch_range.__len__()-1])
	for batch in range(batch_range[0], batch_range[batch_range.__len__()-1]+1):
		input_file = get_batch_path( input_dir, batch)
		output_file = input_file + ".txt"
		print "Converting {}".format(input_file)
		writer = csv.writer( open( output_file, 'wb' ))
		writer.writerow( [ 'data', 'labels' ] )
	
		d = pickle.load( open( input_file, 'rb' ))
	
		data = np.squeeze( d['data'] )
		#print "data: {}".format(data)
		print "data shape: {}".format(data.shape)	
		
		labels = np.squeeze( d['labels'] )
		#print "labels: {}".format(labels)	
		print "labels shape: {}".format(labels.shape)	
		
		# disable print summarization rather than full representation of matrix
		np.set_printoptions(threshold='nan')		
		
		print "converting {} feature entries....".format( labels.__len__() )
		for i in range(0,labels.__len__()):
			label = labels[i]
			feature = data[i]
			#print "  {}. feature: {}".format(i, feature)			
			#print "  {}. label: {}".format(i, label)	
			writer.writerow( [ feature, label ] )
			counter += 1
		
		print "wrote to {}".format(output_file)

if __name__ == '__main__':
	main()
	