# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################


# This script makes batches suitable for training from raw ILSVRC 2012 tar files.

import tarfile
from StringIO import StringIO
from random import shuffle
import sys
from time import time
from pyext._MakeDataPyExt import resizeJPEG
import itertools
import os
import cPickle
import scipy.io
import math
import argparse as argp

# Set this to True to crop images to square. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels, and then the
# center OUTPUT_IMAGE_SIZE x OUTPUT_IMAGE_SIZE patch will be extracted.
#
# Set this to False to preserve image borders. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels. This was
# demonstrated to be superior by Andrew Howard in his very nice paper:
# http://arxiv.org/abs/1312.5402
CROP_TO_SQUARE          = True
OUTPUT_IMAGE_SIZE       = 64

# Number of threads to use for JPEG decompression and image resizing.
NUM_WORKER_THREADS      = 8

# Don't worry about these.
OUTPUT_BATCH_SIZE = 3072
OUTPUT_SUB_BATCH_SIZE = 1024

def get_label_names_liver():
	# can be adjusted for different classes but need to be searchable in image filename
	#label_names = ['leg', 'pelvis', 'liver', 'lung', 'neck', 'head']
	label_names = ['leg', 'pelvis', 'pancreas', 'liver', 'lung', 'neck'] # ISBI 2015
	return label_names

def get_label_names_pancreas():
	# can be adjusted for different classes but need to be searchable in image filename
	label_names = ['neg','pos']
	return label_names

def get_label_names_lymphnodes():
	# can be adjusted for different classes but need to be searchable in image filename
	label_names = ['neg','pos']
	return label_names	

def get_label_names_prostate():
	# can be adjusted for different classes but need to be searchable in image filename
	label_names = ['neg','pos']
	return label_names

def get_label_names_neg_pos():
	# can be adjusted for different classes but need to be searchable in image filename
	label_names = ['neg','pos']
	return label_names	
	
def get_label_names_vertebra():
	# can be adjusted for different classes but need to be searchable in image filename
	label_names = ['label100','label200','label300','label400']
	return label_names		
	
def get_label_names( label_type ):
	if label_type == "liver":
		label_names = get_label_names_liver()
	elif label_type == "pancreas":
		label_names = get_label_names_pancreas()
	elif label_type == "lymphnodes":
		label_names = get_label_names_lymphnodes()
	elif label_type == "prostate":
		label_names = get_label_names_prostate()
	elif label_type == "neg-pos":
		label_names = get_label_names_neg_pos();
	elif label_type == "polyps":
		label_names = get_label_names_polyps()
	elif label_type == "vertebra":
		label_names = get_label_names_vertebra()
	else:
		raise Exception( "No such label type: " + label_type + " !" )
	return label_names

def recursive_glob(rootdir='.', str=''):
	return [os.path.join(rootdir, filename)
		for rootdir, dirnames, filenames in os.walk(rootdir)
		for filename in filenames if str in filename]		
	
def pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

def unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

def partition_list(l, partition_size):
    divup = lambda a,b: (a + b - 1) / b
    return [l[i*partition_size:(i+1)*partition_size] for i in xrange(divup(len(l),partition_size))]

def open_tar(path, name):
    if not os.path.exists(path):
        print "ILSVRC 2012 %s not found at %s. Make sure to set ILSVRC_SRC_DIR correctly at the top of this file (%s)." % (name, path, sys.argv[0])
        sys.exit(1)
    return tarfile.open(path)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#def parse_devkit_meta(ILSVRC_DEVKIT_TAR):
#    tf = open_tar(ILSVRC_DEVKIT_TAR, 'devkit tar')
#    fmeta = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/meta.mat'))
#    meta_mat = scipy.io.loadmat(StringIO(fmeta.read()))
#    labels_dic = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
#    label_names_dic = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
#    label_names = [tup[1] for tup in sorted([(v,label_names_dic[k]) for k,v in labels_dic.items()], key=lambda x:x[0])]
#
#    fval_ground_truth = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
#    validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
#    tf.close()
#    return labels_dic, label_names, validation_ground_truth

def write_batches(target_dir, name, start_batch_num, labels, jpeg_files):
    jpeg_files = partition_list(jpeg_files, OUTPUT_BATCH_SIZE)
    labels = partition_list(labels, OUTPUT_BATCH_SIZE)
    makedir(target_dir)
    print "Writing %s batches..." % name
    for i,(labels_batch, jpeg_file_batch) in enumerate(zip(labels, jpeg_files)):
        t = time()
        jpeg_strings = list(itertools.chain.from_iterable(resizeJPEG([jpeg.read() for jpeg in jpeg_file_batch], OUTPUT_IMAGE_SIZE, NUM_WORKER_THREADS, CROP_TO_SQUARE)))
        batch_path = os.path.join(target_dir, 'data_batch_%d' % (start_batch_num + i))
        makedir(batch_path)
        for j in xrange(0, len(labels_batch), OUTPUT_SUB_BATCH_SIZE):
            pickle(os.path.join(batch_path, 'data_batch_%d.%d' % (start_batch_num + i, j/OUTPUT_SUB_BATCH_SIZE)), 
                   {'data': jpeg_strings[j:j+OUTPUT_SUB_BATCH_SIZE],
                    'labels': labels_batch[j:j+OUTPUT_SUB_BATCH_SIZE]})
        print "Wrote %s (%s batch %d of %d) (%.2f sec)" % (batch_path, name, i+1, len(jpeg_files), time() - t)
    return i + 1

if __name__ == "__main__":
    search_str = ".jpg"

    parser = argp.ArgumentParser()
    parser.add_argument('--src-dir', help='Directory containing JPG images.', required=True)
    parser.add_argument('--tgt-dir', help='Directory to output batches.', required=True)
    parser.add_argument('--label-type', help='Type of labels.', required=True)
    args = parser.parse_args()

    label_names = get_label_names( args.label_type )
    print "There are {} label names:".format(label_names.__len__())
    for label_name in label_names:
	print "    {}".format(label_name)
    label_counts = [0] * label_names.__len__()
	
    print "CROP_TO_SQUARE: %s" % CROP_TO_SQUARE
    print "OUTPUT_IMAGE_SIZE: %s" % OUTPUT_IMAGE_SIZE
    print "NUM_WORKER_THREADS: %s" % NUM_WORKER_THREADS

    ILSVRC_TRAIN_TAR = os.path.join(args.src_dir, 'ILSVRC2012_img_train.tar')

    assert OUTPUT_BATCH_SIZE % OUTPUT_SUB_BATCH_SIZE == 0
#    labels_dic, label_names, validation_labels = parse_devkit_meta(ILSVRC_DEVKIT_TAR)

    print "searching for: '{}' in '{}' ...".format( search_str, args.src_dir )	
    jpeg_files = recursive_glob(args.src_dir, search_str)
    print "   found {} files:".format(jpeg_files.__len__())

    # check if positive or negative example in filenames (NOT WHOLE PATH)
    counter = 0
    undefined_counter = 0
    labels = []
    for n in jpeg_files:
        curr_path,curr_filename=os.path.split(n)
        label_found = False
        for label_idx, label_name in enumerate(label_names):
            if label_name in curr_filename: 
	        label = label_idx
	        label_found = True
	        label_counts[label_idx] += 1
       	label_idx += 1
	
        if not label_found:
	    print "could not find label in {} -> skip".format(curr_filename)
	    undefined_counter += 1
	    continue

	labels.append(label)
    
    # Write training batches
    i = write_batches(args.tgt_dir, 'training', 0, labels, jpeg_files)
    
    # Write meta file
    meta = unpickle('input_meta')
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    meta.update({'batch_size': OUTPUT_BATCH_SIZE,
                 'num_vis': OUTPUT_IMAGE_SIZE**2 * 3,
                 'label_names': label_names})
    pickle(meta_file, meta)
    print "Wrote %s" % meta_file
    print "All done! Batches are in %s" % args.tgt_dir

