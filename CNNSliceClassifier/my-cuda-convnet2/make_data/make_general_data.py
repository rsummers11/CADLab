'make cuda-convnet batches from images in the input dir'
# This function supports the output of batches with 1, 2 and 3 channel images

SHOW_IMAGES = False

import os
import sys
import numpy as np
import cPickle as pickle
import math
from PIL import Image
from PIL import ImageOps
from array import *
import random
import fnmatch
import shutil
from matplotlib import pylab as pl
import scipy.io # For Matlab export, Usage: scipy.io.savemat('test.mat', dict(image=image))

def get_label_names_anatomy():
    # can be adjusted for different classes but need to be searchable in image filename
    label_names = ['leg', 'pelvis', 'pancreas', 'liver', 'lung', 'neck'] # ISBI 2015 + pancreas
    return label_names

def get_label_names_neg_pos():
    # can be adjusted for different classes but need to be searchable in image filename
    label_names = ['neg','pos']
    return label_names    
    
def get_label_names( label_type ):
    if label_type == "anatomy":
        label_names = get_label_names_anatomy()
    elif label_type == "neg-pos":
        label_names = get_label_names_neg_pos();
    else:
        raise Exception( "No such label type: " + label_type + " !" )
    return label_names

def process( image, requested_channels ):
    #image_channels = image.layers
    if "RGB" in image.mode:
	image_channels = 3
    else:
	image_channels = 1

    image = np.array( image )           # img_size x img_size x img_channels
    #check if image has more channels than requested
    if image_channels > requested_channels:
        image = image[:,:,0:requested_channels] # Note: range ends before "stop" value

    if image.shape.__len__() is 3:
        image = np.rollaxis( image, 2 )     # img_channels x img_size x img_size
        image = image.reshape( -1 )         # N_elements
    else:
        image = image.reshape( -1 )         # N_elements

    return image
    
def get_batch_path( output_dir, number ):
    filename = "data_batch_{}".format( number )
    return os.path.join( output_dir, filename )    

def get_batch_log_path( output_dir, number ):
    filename = "data_batch_{}.txt".format( number )
    return os.path.join( output_dir, filename )        
    
def get_empty_batch( N_elements ):    
    return np.zeros(( N_elements, 0 ), dtype = np.uint8 ) # img_size x img_size x img_channels
    
def write_batch( path, batch, labels):
    #labels = [ 1 for x in range( batch.shape[1] ) ]
    #print batch.shape[1]
    #print labels
    d = { 'labels': labels, 'data': batch }
    pickle.dump( d, open( path, "wb" ))

def recursive_glob(rootdir='.', str=''):
    return [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(rootdir)
        for filename in filenames if str in filename]        
    
def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    batch_counter = int( sys.argv[3] )
    batch_number = int( sys.argv[4] )
    search_str = sys.argv[5]
    img_size = int( sys.argv[6] )
    img_channels = int( sys.argv[7] )
    do_shuffle = int( sys.argv[8] )
    label_type = sys.argv[9]

    # BEGING DEBUG
    # input_dir = "C:/HR/Data/Pancreas/MICCAI/rois_unary/1003_test"
    # output_dir = "C:/HR/Data/Pancreas/MICCAI/rois_unary/train_64_dice50_onlypos_2scales_8def_Batchtest/1003"
    #
    # batch_counter = int( "1" )
    # batch_number = int( "10" )
    # search_str = ".jpg"
    # img_size = int( "64" )
    # img_channels = int( "2" )
    # do_shuffle = int( "1" )
    # label_type = "pancreas"
    # END DEBUG

    if len(sys.argv) is 11:
        COMPUTE_MEAN = False # use existing mean from file
	ISTRAINING = False
        data_mean_filename = sys.argv[10]
        data_mean = Image.open(data_mean_filename)
        data_mean = process( data_mean, img_channels ) # same processing as image patch
        data_mean = data_mean.reshape( -1, 1)
        print " Using existing mean image from (TESTING): {}".format( data_mean_filename )
    else:
        print " Computing mean image from data (TRAINING)..."
        COMPUTE_MEAN = True
	ISTRAINING = True

    label_names = get_label_names( label_type )

    print "There are {} label names:".format(label_names.__len__())
    for label_name in label_names:
        print "    {}".format(label_name)
    label_counts = [0] * label_names.__len__()
        
    N_elements = img_size * img_size * img_channels
    print "Batched image info: size={}x{}, channels={} -> {} elements.".format( img_size, img_size, img_channels, N_elements )    
    
    print "searching for: '{}' in '{}' ...".format( search_str, input_dir )    
    names = recursive_glob(input_dir, search_str)
    print "   found {} files:".format(names.__len__())
    
    print "reading file names..."
    batch_size = int( math.floor( names.__len__()/batch_number ) )
    print " Generating {} batch(es) from {} images with {} images per batch ...".format(batch_number, names.__len__(), batch_size)
    
    # Data Mean: To my understanding, this is the mean images of all training and test images (computed below)
    if COMPUTE_MEAN is True:
        data_mean = np.zeros((N_elements,1), dtype=np.float32) 

    current_batch = get_empty_batch( N_elements )
    current_labels = []
    counter = 0
    undefined_counter = 0
    
    new_log = True
    
    if SHOW_IMAGES:
        pl.figure()    
    
    if do_shuffle:
        print " Randomly shuffel names..."
        random.shuffle(names)
    else:
        print " No shuffling..."
        names.sort()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for n in names:
        # read label
        #print " read label {} of {}: {}".format(counter+1, names.__len__(), n)
            
        if new_log is True:
            batch_log_path = get_batch_log_path( output_dir, batch_counter )
            fid = open( batch_log_path, 'w')
            fid.write("#{}\n".format( batch_size ) )
            new_log = False;
        
        # check if positive or negative example in filename (NOT WHOLE PATH)
        curr_path,curr_filename=os.path.split(n)
        label_found = False
        for label_idx, label_name in enumerate(label_names):
            if label_name in curr_filename: 
                label = label_idx
                label_found = True
                label_counts[label_idx] += 1
            label_idx += 1
                
        if not label_found and ISTRAINING:
            print "could not find label in {} -> skip".format(curr_filename)
            undefined_counter += 1
            continue
        if not label_found and not ISTRAINING: # Testing case
	    label = 0
            #print "could not find label in {} -> batch as undefined".format(curr_filename)
            undefined_counter += 1
            
        #print extension
        current_labels.append(label)
        
        # read image
        imagename = os.path.join( input_dir, n )
        image = Image.open( imagename )
        fid.write("{}, {}\n".format( imagename, label ) )
        # show image
        if SHOW_IMAGES:
            pl.imshow(image, interpolation='lanczos')
            pl.show()

        try:
            image = process( image, img_channels ) # check if it has more channels than needed
        except ValueError:
            print "problem with image {}".format( n )
            sys.exit( 1 )

        image = image.reshape( -1, 1 )
        current_batch = np.hstack(( current_batch, image ))
        if COMPUTE_MEAN is True:
            data_mean = data_mean + image
        
        if current_batch.shape[1] == batch_size:
            new_log = True
            fid.close() # close last log
            batch_path = get_batch_path( output_dir, batch_counter )
            print "writing {} of {}...\n".format( batch_path, batch_number )
            write_batch( batch_path, current_batch, current_labels )
            
            batch_counter += 1
            current_batch = get_empty_batch( N_elements )
            current_labels = []
        
        counter += 1
        if counter % 1000 == 0:
            print "\r {}: {} of {} images".format(n,counter, names.__len__())
            print "    so far {} undefined.".format(undefined_counter)
            print "%d%% ..." % int(round(100.0 * float(counter) / len(names)))

    # if last batch is not full, delete log file
    if current_batch.shape[1] > 0 and current_batch.shape[1] < batch_size:
        print "Last batch not full -> label log as unused ..."
        print "writing {} of {}...\n".format( batch_path, batch_number )
        fid.close() # close last log
        if os.path.isfile( batch_log_path ):
            shutil.move( batch_log_path,  batch_log_path + '_unsused' )
        batch_path = get_batch_path( output_dir, batch_counter )
        write_batch( batch_path, current_batch, current_labels )
        if os.path.isfile( batch_path ):
            shutil.move( batch_path,  batch_path + '_unsused' )        

    # compute total data mean and save as image
    if COMPUTE_MEAN is True:
        data_mean = data_mean/names.__len__()
    mean_image_values = data_mean.reshape((img_channels,img_size,img_size))
    mean_image_values = np.swapaxes(mean_image_values,0,2) # following conversions in convdata.py
    mean_image_values = np.swapaxes(mean_image_values,0,1)
    if img_channels is 1:
        mean_image_values = np.squeeze( mean_image_values )
    if img_channels is 2: # duplicate last channel to make mean image
        a1 = mean_image_values
        a2 = mean_image_values[:,:,img_channels-1]
        a2 = a2[:,:,np.newaxis]
        mean_image_values = np.concatenate((a1,a2),axis=2)
    mean_image = Image.fromarray( np.uint8( mean_image_values ))
    mean_image.save( os.path.join( output_dir, 'data_mean.png' ) )
    
    # Save lymph node dictionary (saved as batches.meta)
    dict = ({'num_cases_per_batch': batch_size,
            'label_names': label_names,
            'num_vis': N_elements,
            'data_mean': data_mean,
            'img_size': img_size,
            'img_channels': img_channels,
                        'batch_size': batch_size
    })
    with open( os.path.join( output_dir, 'batches.meta' ), 'wb') as handle:
        pickle.dump(dict, handle)                

    print " Batched {} of {} images".format(batch_number*batch_size,names.__len__())
    for label_idx, label_name in enumerate(label_names):
        print " {}: {}".format(label_name,label_counts[label_idx])

    if undefined_counter > 0:
        print " Warning! There were {} undefined labels.".format(undefined_counter)

if __name__ == '__main__':
    main()
    
