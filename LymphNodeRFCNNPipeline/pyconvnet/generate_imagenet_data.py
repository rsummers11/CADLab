# generate image list
# given class.csv: mapping from class index to class name
# imagenet_list

import os, sys, stat
from os import path
from subprocess import call
from util import pickle,unpickle

root = '/home/snwiz/data/imagenet12/'
data_path = root + '/code/data'
classes_file = data_path + "/classes.csv"
RAWLIST_FILE = data_path + "/imagenet_train_rawlist.txt"
tiny_factor = 1
num_class_factor = 0.001
remove_empty_class = True
validation_labels = root + '/devkit/data/ILSVRC2012_validation_ground_truth.txt'
validation_files = root + '/python/imagenet_val_rawlist.txt'
validation_blacklist = root + '/devkit/data/ILSVRC2012_validation_blacklist.txt'
nclasses = 1000
data_file = root + 'python/data' + str(nclasses) + '.pickle'

# m = unpickle('/home/snwiz/data/imagenet12/python/data10.pickle')
# import pdb; pdb.set_trace()

# build class names tables #####################################################
classes_map = {}
max_class = 0
f = open(classes_file, 'r')
f.readline()  #skip first line
for line in f:
   token = map( str.strip, line.split('|') )
   classe = int(token[0])
   classes_map[token[1]] = [ classe, int(token[-1]), token[2] ] # classIdx, num_train_images
   max_class = max(max_class, classe)
f.close()

classes = max_class * [None]
f = open(classes_file, 'r')
f.readline()  #skip first line
for line in f:
   token = map( str.strip, line.split('|') )
   classe = int(token[0])
   classes[classe-1] = [ token[1], int(token[-1]), token[2] ] # classIdx, num_train_images
f.close()

# create full train set ########################################################
train = []
for i in range(nclasses): train.append([])
ntrain = 0
rawlist_file = open( RAWLIST_FILE, 'r' )
for line in rawlist_file:
   (path, filename) = os.path.split(line.strip())
   class_name = path.split('/')[-1]
   # class index start from 0
   train[classes_map[class_name][0]-1].append( line.strip() )
   ntrain = ntrain + 1
rawlist_file.close()

# print info
s = 'Training samples distribution by class: '
for i,v in enumerate(train): s = s + str(i) + ': ' + str(len(v)) + ', '
print s
print 'Loaded ' + str(ntrain) + ' training images in ' + str(nclasses) + ' classes'

# create full validation set ###################################################
val = []
for i in range(nclasses): val.append([])
print 'Loading validation labels from ' + validation_labels
labels = open(validation_labels, 'r')
val_labels = []
for label in labels: val_labels.append(int(label.strip()))
labels.close()

print 'Loading validation files from ' + validation_files
files = open(validation_files, 'r' )
nval = 0
for fname in files:
   # get image's index from filename
   fname = fname.strip()
   index = val_labels[int(fname.split('.')[0].split('_')[3]) - 1]
   val[index-1].append(fname)
   nval = nval + 1
files.close()

# count blacklisted images
bl = open(validation_blacklist, 'r')
n_blacklist = 0
for i in bl: n_blacklist = n_blacklist + 1

# print info
s = 'Validation samples distribution by class: '
for i,v in enumerate(val): s = s + str(i) + ': ' + str(len(v)) + ', '
print s
print 'Loaded ' + str(nval) + ' validation images in ' + str(nclasses) + ' classes'
print 'Validation blacklist: ' + str(n_blacklist)
print 'Validation total: ' + str(nval + n_blacklist)

# save data ####################################################################
out = {}
out['train'] = train
out['val'] = val
out['classes_map'] = classes_map
out['classes'] = classes
out['num_data'] = ntrain + nval
out['num_data_train'] = ntrain
out['num_data_val'] = nval
pickle(data_file, out)
call(["chmod", "777", data_file]) # change mode so that can be read by others
print 'Saved data to ' + data_file

# save smaller versions ########################################################
for n in [10, 100]:
   print 'Creating smaller dataset with ' + str(n) + ' classes'
   train_small = []
   val_small = []
   classes_small = []
   classes_map_small = []
   ntrain = 0
   nval = 0
   for i in range(0,1000,1000/n):
      train_small.append(train[i])
      val_small.append(val[i])
      classes_small.append(classes[i])
      classes_map_small.append(classes_map[classes[i][0]])
      ntrain = ntrain + len(train[i])
      nval = nval + len(val[i])
   data_file = root + 'python/data' + str(n) + '.pickle'
   out_small = {}
   out_small['train'] = train_small
   out_small['val'] = val_small
   out_small['classes_map'] = classes_map_small
   out_small['classes'] = classes_small
   out_small['num_data'] = ntrain + nval
   out_small['num_data_train'] = ntrain
   out_small['num_data_val'] = nval
   pickle(data_file, out_small)
   call(["chmod", "777", data_file]) # change mode so that can be read by others
   print 'Saved data to ' + data_file
