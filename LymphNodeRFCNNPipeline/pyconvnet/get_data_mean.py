# compute the mean of data given data list

from PIL import Image
import numpy as n
from sys import stdout
from imagenetdata import *
from util import pickle
from util import unpickle
from util import plot_array_image

DATA_PATH = '/home/snwiz/data/imagenet12/code/data'
num_class = 1
MEAN_FILE_EXT = "_mean"
if num_class == 1000:
    INPUT_FILE = DATA_PATH + "/imagenet_data"
else:
    INPUT_FILE = DATA_PATH + "/imagenet_data_tiny" + str(int(num_class) ) 

OUTPUT_FILE = INPUT_FILE + MEAN_FILE_EXT
IMAGE_SIZE = 256
VERIFY_RESULT = True
#VERIFY_RESULT = False

out = unpickle( INPUT_FILE );

num = 0;
m = n.zeros( (IMAGE_SIZE, IMAGE_SIZE,3), n.float32 )
if VERIFY_RESULT:
    sum_m  = n.zeros( (IMAGE_SIZE, IMAGE_SIZE,3), n.float32 )

for cls_index in range( num_class ):
    num_cls_index = len(out['index_map_train'][cls_index])
    for index in range( num_cls_index ):
        i = out['index_map'][cls_index][index]
        image_path = out['image_path'][i]
        im = Image.open( image_path )
        #if cls_index == 6 and index == 0:
        #    im.show()
        #    import pdb; pdb.set_trace()
        assert( im.size == ( IMAGE_SIZE, IMAGE_SIZE ) )
        im_value = PIL2array( im ).astype(n.float32)
        m = m * (1.0 * num/(num+1))
        m = m + im_value/(num+1)
        num += 1
        if VERIFY_RESULT:
            sum_m += im_value
        #print "\r" + str(i) + "/" + str(len(provider.data_list))
        stdout.write( "%4d/%%4d " % (cls_index+1) % num_class ) 
        stdout.write( "%8d/%%8d\r" % (index+1) % len(out['index_map_train'][cls_index]) )
        stdout.flush()
print "\n"

if VERIFY_RESULT:
    sum_m /= num
    diff = sum_m - m
    plot_array_image( diff, (IMAGE_SIZE, IMAGE_SIZE, 3 ) )
    print n.mean( abs(diff) )
    plot_array_image( m, (IMAGE_SIZE,IMAGE_SIZE,3) );

mean_info = {}
mean_info['data'] = m
mean_info['file'] = INPUT_FILE
pickle( OUTPUT_FILE, mean_info )

