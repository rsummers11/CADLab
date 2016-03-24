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

import re
import cPickle
import os
import numpy as n
from math import sqrt

import gzip
import zipfile

class UnpickleError(Exception):
    pass

VENDOR_ID_REGEX = re.compile('^vendor_id\s+: (\S+)')
GPU_LOCK_NO_SCRIPT = -2
GPU_LOCK_NO_LOCK = -1

try:
    import magic
    ms = magic.open(magic.MAGIC_NONE)
    ms.load()
except ImportError: # no magic module
    ms = None

def get_gpu_lock(id=-1):
    import imp
    lock_script_path = '/u/tang/bin/gpu_lock2.py'
    if os.path.exists(lock_script_path):
        locker = imp.load_source("", lock_script_path)
        if id == -1:
            return locker.obtain_lock_id()
        print id
        got_id = locker._obtain_lock(id)
        return id if got_id else GPU_LOCK_NO_LOCK
    return GPU_LOCK_NO_SCRIPT if id < 0 else id

def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', cPickle.dumps(data, -1))
    else:
        fo = open(filename, "wb")
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def unpickle(filename):
    print 'Unpickling ' + filename
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)
    if ms is not None and ms.file(filename).startswith('gzip'):
        fo = gzip.open(filename, 'rb')
        dict = cPickle.load(fo)
    elif ms is not None and ms.file(filename).startswith('Zip'):
        fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
        dict = cPickle.loads(fo.read('data'))
    else:
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
    
    fo.close()
    return dict

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def is_intel_machine():
    f = open('/proc/cpuinfo')
    for line in f:
        m = VENDOR_ID_REGEX.match(line)
        if m:
            f.close()
            return m.group(1) == 'GenuineIntel'
    f.close()
    return False

def get_cpu():
    if is_intel_machine():
        return 'intel'
    return 'amd'

def is_windows_machine():
    print('os.name: {}'.format(os.name))
    return os.name == 'nt'

def plot_array_image( image, image_size ):
   import matplotlib.pyplot as plt
   image = n.round(image.reshape( image_size )).astype(n.uint8)
   plt.imshow(image)
   #plt.colorbar()
   plt.show()


def rbgImage2col( image ):
    assert( image.shape[2] == 3 )
    w = image.shape[0] 
    h = image.shape[1]
    temp_matrix = n.zeros( (w*h*3), n.float32, order='C' )
    # copy each channel
    for channel in range(3):
        s = w * h * channel 
        e = s + w * h
        x = image[:,:,channel]
        temp_matrix[s:e] = x.reshape( (w*h) )
    return temp_matrix

def plot_col_image( image, w, h, d, text):
   import matplotlib.pyplot as plt
   stride = w * h
   if d==3: #rgb channel
       assert( image.shape[0] == w*h*d )
       image_r = n.round(image[0*stride:1*stride]).reshape( (w,h) ).astype(n.uint8)
       image_g = n.round(image[1*stride:2*stride]).reshape( (w,h) ).astype(n.uint8)
       image_b = n.round(image[2*stride:3*stride]).reshape( (w,h) ).astype(n.uint8)
       im = n.zeros( (w,h,3), n.uint8 )
       im[:,:,0] = image_r
       im[:,:,1] = image_g
       im[:,:,2] = image_b
   else : #show vertical cat image channels
       im = n.zeros( (w*d,h,3), n.uint8 )
       for ii in range(d):
           image_ii = n.round(image[ii*stride:(ii+1)*stride]).reshape( (w,h) ).astype(n.uint8)
           im_ii= n.zeros( (w,h,3), n.uint8 )
           im_ii[:,:,0] = image_ii
           im_ii[:,:,1] = image_ii
           im_ii[:,:,2] = image_ii
           im[ii*w:(ii+1)*w,:,:] = im_ii
 
   plt.imshow(im)
   plt.title( text)
   #plt.colorbar()
   plt.show()

# def print network layers
def print_network_layers( layers ):
    for i in range( len(layers) ):
        print i, ": ", " type=", layers[i]['type'],
        print " name=", layers[i]['name']

