# image net data provider

from PIL import Image
from util import pickle,unpickle, plot_array_image, rbgImage2col, plot_col_image
import numpy as n
import sys
import signal
from numpy.random import random_integers
from time import time, asctime, localtime, strftime, sleep
from math import *
import threading
from Queue import Queue
from Queue import Empty
import random

MEAN_FILE_EXT = "_mean"
#PRINT_DEBUG_INFO = True
PRINT_DEBUG_INFO = False

#util functions
def clean_queue( q ):
   try:
       q.get( False )
   except Empty:
       return

def PIL2array(img):
   #if img.mode == 'L':
   #   r = n.array(img.getdata(), n.uint8).reshape(img.size[1], img.size[0] )
   #   result = n.zeros( (img.size[1], img.size[0],3 ), n.uint8 )
   #   result[:,:,0] = r
   #   result[:,:,1] = r
   #   result[:,:,2] = r
   #   return result
   #else:
   #   return n.array(img.getdata(), n.uint8).reshape(img.size[1], img.size[0], 3)
   if img.mode == 'L':
      I = n.asarray( img )
      result = n.zeros( (img.size[1], img.size[0],3 ), n.uint8 )
      result[:,:,0] = I
      result[:,:,1] = I
      result[:,:,2] = I
      return result
   else:
      return n.asarray( img )

def array2PIL(arr):
   return Image.fromarray( n.uint8(arr) )

# load image from file
class ReadImage( threading.Thread ):
    def __init__( self,
                  raw_image_queue,  # shared Queue to store raw image
                  data,             # data file contians image path information
                  mean_file,        # mean file
                  root_path,        # root path of images
                  data_mode,        # 'all','train','val'
                  batch_size = 128, # size of batch
                  batch_index = 0,  # start batch index
                  epoch_index = 1   # start epoch index
                  ):
        threading.Thread.__init__( self, name = "Load Image Thread" )
        self.stop = False
        self.sharedata = raw_image_queue
        self.data = data
        self.num_classes = len(self.data['val'])
        self.data_mode = data_mode
        self.root_path = root_path
        if data_mode == "val":
           self.images = self.data['val']
           self.total_samples = self.data['num_data_val']
           self.shuffle = False
           print 'Validation data is not randomized'
        elif data_mode == "train":
           self.images = self.data['train']
           self.total_samples = self.data['num_data_train']
           self.shuffle = False 
           #self.shuffle = True
           print 'Traing data shuffle: ', self.shuffle
        else:
           print "data_mode: " + str(data_mode) + " not valid"
           import pdb; pdb.set_trace()
           sys.exit(1)
        # iterator on classes
        self.iclass = -1
        # iterator for samples of each class
        self.isamples = self.num_classes * [-1]

        # class_iter = range(num_classes)
        # if shuffle: random.shuffle(class_iter)
        # classes_iter = []
        # for i in range(num_classes):
        #    classes_iter.append(range(len(images[i])))
        #    if shuffle: random.shuffle(classes_iter[i])

        # # get batch queue
        # self.batch_queue = []
        # has_add = True
        # while has_add:
        #    has_add = False
        #    for i in range( self.num_classes ):
        #       if len(index_map[i]) > 0:
        #          index = index_map[i].pop()
        #          self.batch_queue.append( index )
        #          has_add = True

        # self.num_images = len( self.batch_queue )

        #init current index and batch size
        self.total_processed = 0
        self.batch_size = batch_size
        self.batch_index = batch_index
        self.epoch_index = epoch_index
        # read data mean from file
        data_mean_file = unpickle(mean_file)
        self.data_mean = data_mean_file['data']
        self.data_mean = self.data_mean.astype(n.float32)
        # store it as uint8
        #self.data_mean = n.round( self.data_mean).astype(n.uint8)
        print data_mode + ': total_samples: ' + str(self.total_samples) \
            + ' batch_size: ' + str(batch_size) \
            + ' num_batches: ' + str(self.get_num_batches())

    def stopThread( self ):
        self.stop = True

    def run( self ):
        while not self.stop:
            data = self.produceData()
            self.sharedata.put( data )

    def produceData( self ):
       loading_time = time()
       batch_index = self.batch_index + 1
       epoch_index = self.epoch_index
       #print "(batch index, num batches): " + str( self.batch_index ) + \
       #        str( self.get_num_batches() )

       images = []
       labels = []
       diff_size = self.total_samples - self.batch_index * self.batch_size
       assert( diff_size > 0 )
       if diff_size > self.batch_size:
           diff_size = self.batch_size

       # fill current batch with diff_size
       for i in range( diff_size ):
           image_i,label_i = self.next()
           images.append( image_i )
           labels.append( label_i )
       self.batch_index += 1
       # reset when we complete each epoch
       if( self.batch_index > self.get_num_batches() - 1):
           self.epoch_index +=1
           self.batch_index = 0
       return (epoch_index, batch_index, images, labels,
               time() - loading_time)

    def next(self):
       if self.shuffle: # class and sample selection are random
          self.iclass = int(random.uniform(0, self.num_classes))
          self.isamples[self.iclass] = int(random.uniform(0, len(self.images[self.iclass])))
       else: # not random
          self.iclass += 1
          if self.iclass == self.num_classes: self.iclass = 0
          self.isamples[self.iclass] += 1
          if self.isamples[self.iclass] == len(self.images[self.iclass]):
             self.isamples[self.iclass] = 0
       self.total_processed += 1
       image_path = self.root_path + "/" + self.images[self.iclass][self.isamples[self.iclass]]
       # print 'loading class ' + str(self.iclass) \
       #     + ' / sample ' + str(self.isamples[self.iclass]) + ': ' + image_path
       im = Image.open(image_path)
       image_matrix = PIL2array( im )
       return image_matrix,self.iclass

    def get_num_batches( self ):
        # this is wrong:
        #return int(ceil(self.total_samples / self.batch_size))
        return int(ceil( 1.0 * self.total_samples / self.batch_size))
    # int(ceil( 1.0 * len(self.batch_queue) / self.batch_size ))

    def get_num_classes( self ):
        return self.num_classes

    def print_data_summary( self ):
        label_hist = [0] * self.get_num_classes()
        total = 0
        for i in range(len(self.images)):
           label_hist[i] += len(self.images[i])
           total += len(self.images[i])
        print "#samples: " + str(total)
        print "Class Label Hist: ", label_hist, len(label_hist)
        #print "Num Batches     : ", self.get_num_batches()


# read image from raw_image_queue and store to batch_data_queue
class ProcessImage( threading.Thread ):
    def __init__( self,
                  data,
                  raw_image_queue,        #[in]  queue to store ( epoch, batch_index, image )
                  batch_data_queue,       #[out] queue to store transformed batches
                  crop_width,             #[in]  crop width
                  crop_height,            #[in]  crop height
                  data_mean,              #[in]  data mean matrix
                  data_mode,
                  random_transform = True #[in]  whether apply transformation
                  ):
        threading.Thread.__init__( self, name = "Image Process Thread" )
        self.classes = data['classes']
        self.raw_image_queue = raw_image_queue
        self.batch_data_queue = batch_data_queue
        self.random_transform = random_transform
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.data_mean = data_mean
        self.data_mode = data_mode
        self.stop = False
        self.libmodel = __import__('_ConvNet')

    def stopThread( self ):
        self.stop = True

    def run( self ):
        while not self.stop:
            data = self.raw_image_queue.get()
            data = self.process_image( data )
            self.batch_data_queue.put( data )

    def process_image( self, data ):
        pp_time = time()
        epoch = data[0]
        batch_index = data[1]
        images = data[2]
        labels = data[3]
        loading_time = data[4]
        assert( len(images) == len(labels) )
        num_images = len( images )

        # result_data = n.zeros( ( self.crop_width * self.crop_height * 3,
        #     num_images ), n.float32, order='C' )
        # result_label = n.zeros( (1, num_images ), n.float32, order='C' )

        # for i in range( num_images ):
        #     image_matrix = images[i]
        #     image_matrix = image_matrix.astype(n.float32)
        #     image_matrix = image_matrix - self.data_mean
        #     #image_matrix = images[i].astype(n.float32)

        #     x = 0
        #     y = 0
        #     (w,h,a) = image_matrix.shape
        #     # compute image_matrix
        #     if self.random_transform:
        #         # random crop
        #         x += random_integers( 0, w - self.crop_width - 1)
        #         y += random_integers( 0, h - self.crop_height - 1)
        #     else:
        #         # fixed crop
        #         x += (w - self.crop_width)/2
        #         y += (h - self.crop_height)/2

        #     #crop image
        #     assert( x + self.crop_width < w )
        #     assert( y + self.crop_height < h )
        #     #im = im.crop( (x,y, x + self.crop_width, y + self.crop_height ) )
        #     image_matrix = image_matrix[ x:x+self.crop_width, y:y+self.crop_width, : ]

        #     if self.random_transform:
        #         # flip: roll a dice to whether flip image
        #         if random_integers( 0,1 ) > 0.5:
        #             #im = im.transpose( Image.FLIP_LEFT_RIGHT )
        #             image_matrix = image_matrix[:, -1::-1, :]

        #     # print self.data_mode + " set, image " + str(i) + ": " + str(image_matrix.shape) \
        #     #    + " min " + str(image_matrix.min()) + " max " + str(image_matrix.max()) \
        #     #    + ", label " + str(labels[i]) + " (" + self.classes[labels[i]][2] + ")"
        #     # if self.data_mode == 'train':
        #     #    plot_array_image( n.round(image_matrix).astype(n.uint8),(224,224,3) )

        #     #-------------------------------------------------------------------------------
        #     #bug code: (Dec11-2012)
        #     # we expect matrix after reshape is [r(1,1),r(1,2),...,r(1,n),...,r(m,n)] followed
        #     # by blue and green, but this code does not do this, what I observed is:
        #     # image_matrix[0:224*224*3] -> origional image (ok)
        #     # image_matrix[0:224*224]   -> should be r channel image, but the it is not
        #     # code:
        #     #image_matrix = image_matrix.reshape( (self.crop_width * self.crop_height * 3, ) )
        #     #-------------------------------------------------------------------------------
        #     #bug fix: (Dec11-2012)
        #     # this function and corresponding viewer is in util.py
        #     image_matrix = rbgImage2col( image_matrix )
        #     #-------------------------------------------------------------------------------



        #     #store to result_data
        #     result_data[:,i] = image_matrix;
        #     result_label[0,i] = labels[i]

        # #return process tuple
        # return epoch, batch_index, result_data, result_label

        # allocate numpy arrays
        data_array = n.empty((self.crop_width * self.crop_height * 3, num_images), n.float32, order='C')
        labels_array = n.empty((1, num_images), n.float32, order='C')
        # preprocess and pack images into a single array
        assert(self.libmodel.preprocess(data_array, self.data_mean, images,
                                        self.crop_height, self.crop_width))
        # pack labels into a single array
        for i in range(num_images): labels_array[0, i] = labels[i]
        # return data
        return epoch, batch_index, [data_array, labels_array], loading_time, (time() - pp_time)
#        return epoch, batch, data_array, labels_array


# main class

# Imagenet data provider
class ImagenetDataProvider:
   def __init__( self,
                 data_file,
                 mean_file,
                 root_path,
                 data_mode = "train",
                 batch_index = 0,
                 epoch_index = 1,
                 random_transform = False,
                 batch_size = 128,
                 crop_width = 224,
                 crop_height = 224,
                 buffer_size = 2
                 ):
       # init data Q
       self.raw_image_queue = Queue( buffer_size )
       self.batch_data_queue = Queue( buffer_size )
       self.stop = False
       print 'Loading data from ' + str(data_file)
       self.data = unpickle(data_file)
       # init read/tranfrom image object
       self.readImage = ReadImage(self.raw_image_queue, self.data, mean_file, root_path,
               data_mode, batch_size, batch_index, epoch_index )
       self.processImage = ProcessImage(self.data,
          self.raw_image_queue, self.batch_data_queue, crop_width, crop_height,
          self.readImage.data_mean, data_mode, random_transform)

   def get_data_dims( self, idx ):
      if idx == 0:
         return self.processImage.crop_width * self.processImage.crop_height * 3
      if idx == 1:
         return 1

   def get_next_batch( self ):
       return self.batch_data_queue.get()

   def get_num_classes( self ):
       return self.readImage.num_classes

   def get_num_batches( self ):
       return self.readImage.get_num_batches()

   def print_data_summary( self ):
       self.readImage.print_data_summary()
       print "Transform: ", self.processImage.random_transform

   def stopThread( self ):
       self.stop = True

       self.processImage.stopThread()
       # make sure processImage thread should not
       # be blocked by get method of raw_image_queue
       while self.raw_image_queue.empty():
           sleep( 1 )

       # make sure processImage thread should not
       # be blocked by put method of batch_data_queue
       while not self.batch_data_queue.empty():
           clean_queue( self.batch_data_queue )

       self.readImage.stopThread()
       # make sure readImage thread should not
       # be blocked by put method of raw_image_queue
       while not self.raw_image_queue.empty():
           clean_queue( self.raw_image_queue )

       self.readImage.join()
       self.processImage.join()

   def start( self ):
       self.readImage.start()
       self.processImage.start()

def init_data_providers(op):
   num_class = op.get_value("num_class")
   mini = op.get_value("minibatch_size")
   root_path = op.get_value("data_path")
   root_path = root_path + '/'
   mean_file = root_path + 'python/mean1000.pickle'
   if num_class == 1000: data_file = root_path + 'python/data1000.pickle'
   elif num_class == 100: data_file = root_path + 'python/data100.pickle'
   elif num_class == 10: data_file = root_path + 'python/data10.pickle'
   else: sys.exit()

   print '-----------------------------------------------------'
   train_provider = ImagenetDataProvider(
      data_file, mean_file, root_path, 'train', batch_size = mini,
      crop_width = 224, crop_height = 224,
      random_transform = op.get_value("transform"), buffer_size = 2)
   train_provider.print_data_summary()
   print '-----------------------------------------------------'
   test_provider = ImagenetDataProvider(
      data_file, mean_file, root_path, 'val', batch_size = 128,
      crop_width = 224, crop_height = 224, buffer_size = 2 )
   test_provider.print_data_summary()
   print '-----------------------------------------------------'

   return test_provider,train_provider

def signal_handler( signal, frame):
    if PRINT_DEBUG_INFO:
        print 'You pressed Ctrl+C!'
    sys.exit(0)

def main():
   root_path = '/home/snwiz/data/imagenet12/'
   #root_path = "/rose1.a/data/imagenet12"
   #root_path = "/scratch/imagenet12/"
   #data_file = root_path + 'python/data1000.pickle'
   #data_file = root_path + 'python/data10.pickle'
   data_file = root_path + 'python/data100.pickle'
   mean_file = root_path + 'python/mean1000.pickle'
   provider = ImagenetDataProvider( data_file, mean_file, root_path,
           'val', batch_size = 128, random_transform = True )
   provider.print_data_summary()
   # set up signal handler
   signal.signal( signal.SIGINT, signal_handler )
   # start thread
   provider.start()
   for i in range(50):
       load_time_start = time()
       epoch, batch_index, data = provider.get_next_batch()
       load_time = time() - load_time_start
       if PRINT_DEBUG_INFO:
           print "Get next batch: %.2f sec" % load_time
       print 'epoch: ' + str(epoch) + ' batch_index: ' + str(batch_index) + \
            '/' + str(provider.get_num_batches()) + \
            ' data shape: ' + str( data['data'].shape )
            #' data:  ' + str(data['data'][0:5,0:5]) +\
            #' label: ' + str(data['label'][0:5,0:5] )
   provider.stopThread()

if __name__ == "__main__":
    #for i in range(100):
    #    main()
    PRINT_DEBUG_INFO = True
    main()
