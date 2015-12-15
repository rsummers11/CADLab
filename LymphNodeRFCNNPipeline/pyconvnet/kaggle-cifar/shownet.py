# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *

import numpy as np
import time

try:
	import pylab as pl
except:
	print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
	sys.exit(1)

class ShowNetError(Exception):
	pass

class ShowConvNet(ConvNet):
	def __init__(self, op, load_dic):
		ConvNet.__init__(self, op, load_dic)
	
	def get_gpus(self):
		self.need_gpu = self.op.get_value('show_preds') \
			or self.op.get_value('write_features') \
			or self.op.get_value('write_predictions')
		if self.need_gpu:
			ConvNet.get_gpus(self)
	
	def init_data_providers(self):
		class Dummy:
			def advance_batch(self):
				pass
		if self.need_gpu:
			ConvNet.init_data_providers(self)
		else:
			self.train_data_provider = self.test_data_provider = Dummy()
	
	def import_model(self):
		if self.need_gpu:
			ConvNet.import_model(self)
			
	def init_model_state(self):
		#ConvNet.init_model_state(self)
		if self.op.get_value('show_preds'):
			self.sotmax_idx = self.get_layer_idx(self.op.get_value('show_preds'), check_type='softmax')
		if self.op.get_value('write_features'):
			self.ftr_layer_idx = self.get_layer_idx(self.op.get_value('write_features'))
			
	def init_model_lib(self):
		if self.need_gpu:
			ConvNet.init_model_lib(self)

	def plot_cost(self):
		if self.show_cost not in self.train_outputs[0][0]:
			raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
		train_errors = [o[0][self.show_cost][self.cost_idx] for o in self.train_outputs]
		test_errors = [o[0][self.show_cost][self.cost_idx] for o in self.test_outputs]

		numbatches = len(self.train_batch_range)
		test_errors = numpy.row_stack(test_errors)
		test_errors = numpy.tile(test_errors, (1, self.testing_freq))
		test_errors = list(test_errors.flatten())
		test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
		test_errors = test_errors[:len(train_errors)]

		numepochs = len(train_errors) / float(numbatches)
		pl.figure(1)
		x = range(0, len(train_errors))
		pl.plot(x, train_errors, 'k-', label='Training set')
		pl.plot(x, test_errors, 'r-', label='Test set')
		pl.legend()
		ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
		epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
		epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
		ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

		pl.xticks(ticklocs, ticklabels)
		pl.xlabel('Epoch')
#		pl.ylabel(self.show_cost)
		pl.title(self.show_cost)
		
	def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
		FILTERS_PER_ROW = 16
		MAX_ROWS = 16
		MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
		num_colors = filters.shape[0]
		f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
		filter_end = min(filter_start+MAX_FILTERS, num_filters)
		filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
	
		filter_size = int(sqrt(filters.shape[1]))
		fig = pl.figure(fignum)
		fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
		num_filters = filter_end - filter_start
		if not combine_chans:
			bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
		else:
			bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)
	
		for m in xrange(filter_start,filter_end ):
			filter = filters[:,:,m]
			y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
			if not combine_chans:
				for c in xrange(num_colors):
					filter_pic = filter[c,:].reshape((filter_size,filter_size))
					bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
						   1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
			else:
				filter_pic = filter.reshape((3, filter_size,filter_size))
				bigpic[:,
					   1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
					   1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
				
		pl.xticks([])
		pl.yticks([])
		if not combine_chans:
			pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
		else:
			bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
			pl.imshow(bigpic, interpolation='nearest')		
		
	def plot_filters(self):
		filter_start = 0 # First filter to show
		layer_names = [l['name'] for l in self.layers]
		if self.show_filters not in layer_names:
			raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
		layer = self.layers[layer_names.index(self.show_filters)]
		filters = layer['weights'][self.input_idx]
		if layer['type'] == 'fc': # Fully-connected layer
			num_filters = layer['outputs']
			channels = self.channels
		elif layer['type'] in ('conv', 'local'): # Conv layer
			num_filters = layer['filters']
			channels = layer['filterChannels'][self.input_idx]
			if layer['type'] == 'local':
				filters = filters.reshape((layer['modules'], layer['filterPixels'][self.input_idx] * channels, num_filters))
				filter_start = r.randint(0, layer['modules']-1)*num_filters # pick out some random modules
				filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
				num_filters *= layer['modules']

		filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
		# Convert YUV filters to RGB
		if self.yuv_to_rgb and channels == 3:
			R = filters[0,:,:] + 1.28033 * filters[2,:,:]
			G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
			B = filters[0,:,:] + 2.12798 * filters[1,:,:]
			filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
		combine_chans = not self.no_rgb and channels == 3
		
		# Make sure you don't modify the backing array itself here -- so no -= or /=
		filters = filters - filters.min()
		filters = filters / filters.max()

		self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans)
	
	def plot_predictions(self):
		data = self.get_next_batch(train=False)[2] # get a test batch
		num_classes = self.test_data_provider.get_num_classes()
		NUM_ROWS = 2
		NUM_COLS = 4
		NUM_IMGS = NUM_ROWS * NUM_COLS
		NUM_TOP_CLASSES = min(num_classes, 4) # show this many top labels
		
		label_names = self.test_data_provider.batch_meta['label_names']
		if self.only_errors:
			preds = n.zeros((data[0].shape[1], num_classes), dtype=n.single)
		else:
			preds = n.zeros((NUM_IMGS, num_classes), dtype=n.single)
			rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
			data[0] = n.require(data[0][:,rand_idx], requirements='C')
			data[1] = n.require(data[1][:,rand_idx], requirements='C')
		data += [preds]

		# Run the model
		self.libmodel.startFeatureWriter(data, self.sotmax_idx)
		self.finish_batch()
		
		fig = pl.figure(3)
		fig.text(.4, .95, '%s test case predictions' % ('Mistaken' if self.only_errors else 'Random'))
		if self.only_errors:
			err_idx = nr.permutation(n.where(preds.argmax(axis=1) != data[1][0,:])[0])[:NUM_IMGS] # what the net got wrong
			data[0], data[1], preds = data[0][:,err_idx], data[1][:,err_idx], preds[err_idx,:]
			
		data[0] = self.test_data_provider.get_plottable_data(data[0])
		for r in xrange(NUM_ROWS):
			for c in xrange(NUM_COLS):
				img_idx = r * NUM_COLS + c
				if data[0].shape[0] <= img_idx:
					break
				pl.subplot(NUM_ROWS*2, NUM_COLS, r * 2 * NUM_COLS + c + 1)
				pl.xticks([])
				pl.yticks([])
				img = data[0][img_idx,:,:,:]
				pl.imshow(img, interpolation='nearest')
				true_label = int(data[1][0,img_idx])

				img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
				pl.subplot(NUM_ROWS*2, NUM_COLS, (r * 2 + 1) * NUM_COLS + c + 1, aspect='equal')

				ylocs = n.array(range(NUM_TOP_CLASSES)) + 0.5
				height = 0.5
				width = max(ylocs)
				pl.barh(ylocs, [l[0]*width for l in img_labels], height=height, \
						color=['r' if l[1] == label_names[true_label] else 'b' for l in img_labels])
				pl.title(label_names[true_label])
				pl.yticks(ylocs + height/2, [l[1] for l in img_labels])
				pl.xticks([width/2.0, width], ['50%', ''])
				pl.ylim(0, ylocs[-1] + height*2)
	
	def do_write_features(self):
		if not os.path.exists(self.feature_path):
			os.makedirs(self.feature_path)
		next_data = self.get_next_batch(train=False)
		b1 = next_data[1]
		num_ftrs = self.layers[self.ftr_layer_idx]['outputs']
		while True:
			batch = next_data[1]
			data = next_data[2]
			ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
			self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)
			
			# load the next batch while the current one is computing
			next_data = self.get_next_batch(train=False)
			self.finish_batch()
			path_out = os.path.join(self.feature_path, 'data_batch_%d' % batch)
			pickle(path_out, {'data': ftrs, 'labels': data[1]})
			print "Wrote feature file %s" % path_out
			if next_data[1] == b1:
				break
		pickle(os.path.join(self.feature_path, 'batches.meta'), {'source_model':self.load_file,
																 'num_vis':num_ftrs})

	def make_predictions(self, net, data, labels, num_classes):
		data = np.require(data, requirements='C')
		labels = np.require(labels, requirements='C')

		preds = np.zeros((data.shape[1], num_classes), dtype=np.single)
		softmax_idx = net.get_layer_idx('probs', check_type='softmax')

		t0 = time.time()
		net.libmodel.startFeatureWriter(
			[data, labels, preds], softmax_idx)
		net.finish_batch()
		print "Predicted %s cases in %.2f seconds." % (
			labels.shape[1], time.time() - t0)

		if net.multiview_test:
			# We have to deal with num_samples * num_views
			# predictions.
			num_views = net.test_data_provider.num_views
			num_samples = labels.shape[1] / num_views
			split_sections = range(
				num_samples, num_samples * num_views, num_samples)
			preds = np.split(preds, split_sections, axis=0)
			labels = np.split(labels, split_sections, axis=1)
			preds = reduce(np.add, preds)
			labels = labels[0]

		return preds, labels

	def get_predictions(self):
		num_classes = self.test_data_provider.get_num_classes()
		all_preds = np.zeros((0, num_classes), dtype=np.single)
		all_labels = np.zeros((0, 1), dtype=np.single)
		all_metadata = []
		num_batches = len(self.test_data_provider.batch_range)
		db = self.test_data_provider.batch_meta.get('metadata', {})

		for batch_index in range(num_batches):
			epoch, batchnum, (data, labels) = self.get_next_batch(train=False)
			if data.shape[1] != labels.shape[1]:
				data = data[:, :labels.shape[1]]
			preds, labels = self.make_predictions(self, data, labels, num_classes)
			all_preds = np.vstack([all_preds, preds])
			all_labels = np.vstack([all_labels, labels.T])
			if db:
				ids = self.test_data_provider.get_batch(batchnum).get('ids')
				all_metadata.extend([db[id] for id in ids])

		self._predictions = all_preds, all_labels, all_metadata
		return self._predictions
	
	def do_write_predictions(self):
		preds, labels, metadata = self.get_predictions()
		pickle(self.write_predictions, {'data': preds, 'labels': labels})
		print "Wrote predictions file %s" % self.write_predictions		

	def start(self):
		self.op.print_values()
		if self.show_cost:
			self.plot_cost()
		if self.show_filters:
			self.plot_filters()
		if self.show_preds:
			self.plot_predictions()
		if self.write_features:
			self.do_write_features()
		if self.write_predictions:
			self.do_write_predictions()
			
		pl.show()
		sys.exit(0)
			
	@classmethod
	def get_options_parser(cls):
		op = ConvNet.get_options_parser()
		for option in list(op.options):
			if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range'):
				op.delete_option(option)
		op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
		op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
		op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
		op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
		op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
		op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
		op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
		op.add_option("show-preds", "show_preds", StringOptionParser, "Show predictions made by given softmax on test set", default="")
		op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions (to be used with --show-preds)", default=False, requires=['show_preds'])
		op.add_option("write-features", "write_features", StringOptionParser, "Write test data features from given layer", default="", requires=['feature-path'])
		op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")
		
		op.add_option("write-predictions", "write_predictions", StringOptionParser, "Write predictions to a file", default="")		
		op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0)	# requires=['logreg_name']		
		
		op.options['load_file'].default = None
		return op
	
if __name__ == "__main__":
	try:
		op = ShowConvNet.get_options_parser()
		op, load_dic = IGPUModel.parse_options(op)
		model = ShowConvNet(op, load_dic)
		model.start()
	except (UnpickleError, ShowNetError, opt.GetoptError), e:
		print "----------------"
		print "Error:"
		print e 
