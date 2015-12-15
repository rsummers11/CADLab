#
# Clean version of training conv-net
#

import sys
import numpy as n
import signal
import os
from time import time, asctime, localtime, strftime
import numpy.random as nr
import numpy as np
from util import *
from options import *
import sys
import math as m
import layer as lay
from os import linesep as NL
#import pylab as pl
import matplotlib.pyplot as plt
import imagenetdata
from imagenetdata import ImagenetDataProvider

def get_options_parser( ):
   op = OptionsParser()
   ###### gpumodel.py##########
   op.add_option("f", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCLUDE_ALL)
   #op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
   #op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
   #op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
   op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
   op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
   op.add_option("data-path", "data_path", StringOptionParser, "Data path")
   op.add_option("save-path", "save_path", StringOptionParser, "Save path")
   op.add_option("max-filesize", "max_filesize_mb", IntegerOptionParser, "Maximum save file size (MB)", default=5000)
   op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
   op.add_option("num-gpus", "num_gpus", IntegerOptionParser, "Number of GPUs", default=1)
   op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
   op.add_option("zip-save", "zip_save", BooleanOptionParser, "Compress checkpoints?", default=0)
   op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
   op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override", default=OptionExpression("[-1] * num_gpus"))

   ###### convnet.py##########
   op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
   op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=True)
   op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
   op.add_option("check-grads", "check_grads", BooleanOptionParser, "Check gradients and quit?", default=0, excuses=['data_path','save_path','train_batch_range','test_batch_range'])
   op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0, requires=['logreg_name'])
   op.add_option("crop-border", "crop_border", IntegerOptionParser, "Cropped DP: crop border size", default=4, set_once=True)
   op.add_option("logreg-name", "logreg_name", StringOptionParser, "Cropped DP: logreg layer name (for --multiview-test)", default="")
   op.add_option("conv-to-local", "conv_to_local", ListOptionParser(StringOptionParser), "Convert given conv layers to unshared local", default=[])
   op.add_option("unshare-weights", "unshare_weights", ListOptionParser(StringOptionParser), "Unshare weight matrices in given layers", default=[])
   op.add_option("conserve-mem", "conserve_mem", BooleanOptionParser, "Conserve GPU memory (slower)?", default=0)

   op.delete_option('max_test_err')
   op.options["max_filesize_mb"].default = 0
   op.options["testing_freq"].default = 50
   op.options["num_epochs"].default = 50000
   #op.options['dp_type'].default = None

   ####### my additionial config #########
   op.add_option("num-class", "num_class", IntegerOptionParser, "Num Classes", default=1000)
   op.add_option("transform", "transform", BooleanOptionParser, "Apply Transformation", default=True )
   op.add_option("scale-rate", "scale_rate", FloatOptionParser, "Learning Rate Scale Factor", default=1 )

   return op


def parse_options(op):
    try:
        load_dic = None
        options = op.parse()
        if options["load_file"].value_given:
            load_dic = load_checkpoint(options["load_file"].value)
            old_op = load_dic["op"]
            old_op.merge_from(op)
            op = old_op
        op.eval_expr_defaults()
        return op, load_dic
    except OptionMissingException, e:
        print e
        op.print_usage()
    except OptionException, e:
        print e
    except UnpickleError, e:
        print "Error loading checkpoint:"
        print e
    sys.exit()

def load_checkpoint(load_dir):
    if os.path.isdir(load_dir):
       return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
    return unpickle(load_dir)

class ModelStateException(Exception):
    pass

class ConvNet:
   def __init__( self, op, load_dic, train_data_provider, test_data_provider = None, dp_params = None ):
      if dp_params is None:
         dp_params = {}

      self.set_dataProvider( train_data_provider,test_data_provider )
      #init_model( op, load_dic, dp_params ):
      filename_options = []
      dp_params['multiview_test'] = op.get_value('multiview_test')
      dp_params['crop_border'] = op.get_value('crop_border')

      # these are input parameters
      self.model_name = "ConvNet"
      self.op = op
      self.options = op.options
      self.load_dic = load_dic
      self.filename_options = filename_options
      self.dp_params = dp_params
      self.get_gpus()
      #self.fill_excused_options()
      #assert self.op.all_values_given()

      for o in op.get_options_list():
          setattr(self, o.name, o.value)

      # these are things that the model must remember but they're not input parameters
      if load_dic:
          self.model_state = load_dic["model_state"]
          self.save_file = self.options["load_file"].value
          if not os.path.isdir(self.save_file):
              self.save_file = os.path.dirname(self.save_file)
          # split file name out
          (pdir,self.save_file) = os.path.split( self.save_file )
          if( len(self.save_file) == 0 ):
              (pdir,self.save_file) = os.path.split( pdir )
          assert( os.path.samefile( pdir, self.save_path ) )
      else:
          self.model_state = {}
          if filename_options is not None and len(filename_options) != 0:
              self.save_file = self.model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')

          else:
             self.save_file = self.model_name + "_" + strftime('%Y-%m-%d_%H.%M.%S')

          self.model_state["train_outputs"] = []
          self.model_state["test_outputs"] = []
          self.model_state["epoch"] = 1
          #self.model_state["batch_index"] = self.train_batch_range[0]
          self.model_state["batch_index"] = 1

      try:
          self.init_model_state()
      except ModelStateException, e:
          print e
          sys.exit(1)
      for var, val in self.model_state.iteritems():
          setattr(self, var, val)

      self.import_model()
      self.init_model_lib()

   def set_dataProvider( self, train_data_provider, test_data_provider = None ):
      self.train_data_provider = train_data_provider
      self.train_batch_range = range( 1, train_data_provider.get_num_batches() + 1)
      self.test_data_provider = test_data_provider
      self.test_batch_range = range( 1, test_data_provider.get_num_batches() + 1)

   def get_gpus(self):
      self.device_ids = self.op.get_value('gpu')

   def import_model(self):
      print "========================="
      print "Importing %s C++ module" % ('_' + self.model_name)
      self.libmodel = __import__('_' + self.model_name)

   def init_model_lib(self):
      self.libmodel.initModel(self.layers, self.minibatch_size, self.device_ids[0])

   def init_model_state(self):
      ms = self.model_state
      if self.load_file:
         ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self, ms['layers'])
      else:
         ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self)
      self.layers_dic = dict(zip([l['name'] for l in ms['layers']], ms['layers']))

      logreg_name = self.op.get_value('logreg_name')
      if logreg_name:
         self.logreg_idx = self.get_layer_idx(logreg_name, check_type='cost.logreg')

      # Convert convolutional layers to local
      if len(self.op.get_value('conv_to_local')) > 0:
         for i, layer in enumerate(ms['layers']):
             if layer['type'] == 'conv' and layer['name'] in self.op.get_value('conv_to_local'):
                 lay.LocalLayerParser.conv_to_local(ms['layers'], i)
      # Decouple weight matrices
      if len(self.op.get_value('unshare_weights')) > 0:
         for name_str in self.op.get_value('unshare_weights'):
             if name_str:
                 name = lay.WeightLayerParser.get_layer_name(name_str)
                 if name is not None:
                     name, idx = name[0], name[1]
                     if name not in self.layers_dic:
                         raise ModelStateException("Layer '%s' does not exist; unable to unshare" % name)
                     layer = self.layers_dic[name]
                     lay.WeightLayerParser.unshare_weights(layer, ms['layers'], matrix_idx=idx)
                 else:
                     raise ModelStateException("Invalid layer name '%s'; unable to unshare." % name_str)
      self.op.set_value('conv_to_local', [], parse=False)
      self.op.set_value('unshare_weights', [], parse=False)

   def get_layer_idx(self, layer_name, check_type=None):
       try:
           layer_idx = [l['name'] for l in self.model_state['layers']].index(layer_name)
           if check_type:
               layer_type = self.model_state['layers'][layer_idx]['type']
               if layer_type != check_type:
                   raise ModelStateException("Layer with name '%s' has type '%s'; should be '%s'." % (layer_name, layer_type, check_type))
           return layer_idx
       except ValueError:
           raise ModelStateException("Layer with name '%s' not defined." % layer_name)

   def start(self):
       if self.test_only:
           self.test_outputs += [self.get_test_error()]
           self.print_test_results()
           sys.exit(0)

       self.train_data_provider.start()
       self.test_data_provider.start()
       self.train()

   def train(self):

       print "========================="
       print "learning rate scale: ", self.scale_rate
       print "========================="
       self.scale_learningRate( self.scale_rate )
       print "========================="
       print "Training %s" % self.model_name
       self.op.print_values()
       print "========================="
       self.print_model_state()
       print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
       print "Current time: %s" % asctime(localtime())
       print "Saving checkpoints to %s" % os.path.join(self.save_path, self.save_file)
       print "=========================\n"

       # test before training as pierre required
       #self.test_outputs += [self.get_test_error()]
       #self.print_test_results()
       #self.print_test_status()

       # timing variables
       time_window = 10 # average time over this many samples
       loading_time = 0
       pp_time = 0
       iteration_time = time()
       total_iteration_time = time()
       total_pp_time = 0
       total_loading_time = 0
       time_steps = 0
       total_steps = 0

       # start run it
       epoch_cost = 0
       self.epoch_cost_buffer = []
       next_data = self.get_next_batch()
       while self.epoch <= self.num_epochs:
           data = next_data
           self.epoch, self.batch_index = data[0], data[1]

           # print network and input data info
           #self.sync_with_host()
           #self.print_network()
           #print "data (min,max,mean): ", np.min(data[2][0]), np.max(data[2][0]), np.mean(data[2][0])
           #print "label: ", data[2][1]

           # process epoch cost
           epoch_cost = self.process_epoch_cost( epoch_cost )

           self.print_iteration()
           sys.stdout.flush()

           # start training
           total_time_start = time()
           self.start_batch(data)

           # load the next batch while the current one is computing
           next_data = self.get_next_batch()

           batch_output = self.finish_batch()

           self.train_outputs += [batch_output]
           # cur cost
           epoch_cost += self.print_train_results()

           # print time
           #self.print_train_time(time() - compute_time_py)
           #print "batch: " + str(self.batch_index) + ' batch time: ' + "%.3f sec" % (time() - total_time_start )

           #  get output, and plot histogram
           #self.sync_with_host()
           #x = self.get_network_output( data[2], 'fc8' )
           #print "%f %f %f " % (np.min(x[2]), np.max(x[2]), np.mean(x[2])),

           ## debuging code
           #y = sum(x[2],1 )
           #plt.clf()
           #plt.plot(y/128.0)
           ##plt.show()
           #plt.savefig( "tmp/activation.png" )
           #import pdb; pdb.set_trace()
           #print data[2][1]


           #import pdb; pdb.set_trace()
           #self.conditional_save()

           if self.batch_index == 1 and self.epoch % self.testing_freq == 0:
               print '\n-- TESTING ----------------------------------------------'
               self.sync_with_host()
               self.test_outputs += [self.get_test_error()]
               self.print_test_results()
               self.print_test_status()
               self.conditional_save()
               print '\n'

           # update and print timing
           loading_time += data[3]
           pp_time += data[4]
           total_loading_time += data[3]
           total_pp_time += data[4]
           time_steps += 1
           total_steps += 1
           if time_steps == time_window:
              print "\n%d.%d..." % (self.epoch, self.batch_index) + ' ' + str(time_window) \
                  + '-AVERAGE load: %d ms' % (1000 * loading_time / time_steps) \
                  + ' pp: %d ms' % (1000 * pp_time / time_steps) \
                  + ' iter: %d ms' % (1000 * (time() - iteration_time) / time_steps) \
                  + ' ___ TOTAL AVERAGE load: %d ms' % (1000 * total_loading_time / total_steps) \
                  + ' pp: %d ms' % (1000 * total_pp_time / total_steps) \
                  + ' iter: %d ms' % (1000 * (time() - total_iteration_time) / total_steps) \
                  + ' ___'
              time_steps = 0
              loading_time = 0
              pp_time = 0
              iteration_time = time()

       self.cleanup()

   def process_epoch_cost( self, epoch_cost ):
       if self.batch_index != 1:
           return epoch_cost
       if (not self.epoch_cost_buffer) and epoch_cost == 0:
           return epoch_cost
       limit_buffer_size = 7
       scale = 0.1
       # reset epoch cost
       print "epoch cost: " + str(epoch_cost)
       ## add into buffer
       #self.epoch_cost_buffer.append( epoch_cost )
       ## analysis buffer decide whether adjust learning rate
       #current_buffer = list( self.epoch_cost_buffer )
       #prev_cost = min( current_buffer[:limit_buffer_size/2] )
       #cur_cost  = min( current_buffer[limit_buffer_size/2:] )
       ## dec learning rate
       #if cur_cost > prev_cost:
       #    print "scale learning rate by factor of: " , scale
       #    self.scale_learningRate( scale )

       #self.epoch_cost_buffer = slef.epoch_cost_buffer[-limit_buffer_size,:]
       # return 0
       epoch_cost = 0
       return epoch_cost;

   def cleanup(self):
       sys.exit(0)

   def sync_with_host(self):
       self.libmodel.syncWithHost()

   def print_model_state(self):
       pass

   def get_num_batches_done(self):
       return len(self.train_batch_range) * (self.epoch - 1) + self.batch_index - self.train_batch_range[0] + 1

   def get_next_batch(self, train=True):
       dp = self.train_data_provider
       if not train: dp = self.test_data_provider
       return dp.get_next_batch()

   def start_batch(self, batch_data, train=True):
       self.libmodel.startBatch(batch_data[2], not train)

   def finish_batch(self):
       return self.libmodel.finishBatch()
   def scale_learningRate( self, eps ):
       self.libmodel.scaleModelEps( eps );

   def set_dropRate( self, dropRate ):
       print "set drop rate: ", dropRate
       self.libmodel.setDropRate( dropRate );

   def print_iteration(self):
       print "%d.%d..." % (self.epoch, self.batch_index),

   #def print_train_time(self, compute_time_py):
   #    print "(%.3f sec)" % (compute_time_py)

   #def print_train_results(self):
   #    batch_error = self.train_outputs[-1][0]
   #    if not (batch_error > 0 and batch_error < 2e20):
   #        print "Crazy train error: %.6f" % batch_error
   #        self.cleanup()

   #    print "Train error: %.6f " % (batch_error),

   def print_train_results(self):
       return self.print_costs(self.train_outputs[-1])

   def print_costs(self, cost_outputs):
       total_cost = 0
       costs, num_cases = cost_outputs[0], cost_outputs[1]
       for errname in costs.keys():
           costs[errname] = [(v/num_cases) for v in costs[errname]]
           print "%s: " % errname,
           print ", ".join("%6f" % v for v in costs[errname])
           if sum(m.isnan(v) for v in costs[errname]) > 0 or sum(m.isinf(v) for v in costs[errname]):
               print "^ got nan or inf!"
               sys.exit(1)
           total_cost += costs[errname][0]
       return total_cost

   def print_test_results(self):
       print ""
       print "======================Test output======================"
       self.print_costs(self.test_outputs[-1])
       print ""
       self.print_network()

   def print_network(self):
       print "-------------------------------------------------------",
       for i,l in enumerate(self.layers): # This is kind of hacky but will do for now.
          if 'weights' in l:
             if type(l['weights']) == n.ndarray:
                print "%sLayer '%s' weights: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['weights'])), n.mean(n.abs(l['weightsInc']))),
             elif type(l['weights']) == list:
                print ""
                print NL.join("Layer '%s' weights[%d]: %e [%e] (%e,%e)" % (l['name'], i, n.mean(n.abs(w)), n.mean(n.abs(wi)), n.min(w), n.max(w) ) for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc']))),
             print "%sLayer '%s' biases: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['biases'])), n.mean(n.abs(l['biasesInc']))),
       print ""

   def print_test_status(self):
       status = (len(self.test_outputs) == 1 or self.test_outputs[-1][0] < self.test_outputs[-2][0]) and "ok" or "WORSE"
       print status,

   def conditional_save(self):
       self.save_state()
       print "-------------------------------------------------------"
       print "Saved checkpoint to %s" % os.path.join(self.save_path, self.save_file)
       print "=======================================================",

   def aggregate_test_outputs(self, test_outputs):
       num_cases = sum(t[1] for t in test_outputs)
       for i in xrange(1 ,len(test_outputs)):
          for k,v in test_outputs[i][0].items():
              for j in xrange(len(v)):
                  test_outputs[0][0][k][j] += test_outputs[i][0][k][j]
       return (test_outputs[0][0], num_cases)

   def get_test_error(self):
       next_data = self.get_next_batch(train=False)
       test_outputs = []
       while True:
           data = next_data
           self.start_batch(data, train=False)
           #load_next = not self.test_one and data[1] <= self.test_batch_range[-1]
           # if data[1] (batch index) is the same as last value in test_batch_range
           # we should NOT continue load, thus, < instead of <=
           load_next = not self.test_one and data[1] < self.test_batch_range[-1]
           if load_next: # load next batch
               next_data = self.get_next_batch(train=False)
           test_outputs += [self.finish_batch()]
           #if self.test_only: # Print the individual batch results for safety
           print "batch %d: %s" % (data[1], str(test_outputs[-1]))
           if not load_next:
               break
           sys.stdout.flush()

       return self.aggregate_test_outputs(test_outputs)

   def set_var(self, var_name, var_val):
       setattr(self, var_name, var_val)
       self.model_state[var_name] = var_val
       return var_val

   def get_var(self, var_name):
       return self.model_state[var_name]

   def has_var(self, var_name):
       return var_name in self.model_state

   def save_state(self):
       for att in self.model_state:
           if hasattr(self, att):
               self.model_state[att] = getattr(self, att)

       dic = {"model_state": self.model_state,
              "op": self.op}

       checkpoint_dir = os.path.join(self.save_path, self.save_file)
       checkpoint_file = "%d.%d" % (self.epoch, self.batch_index)
       checkpoint_file_full_path = os.path.join(checkpoint_dir, checkpoint_file)
       if not os.path.exists(checkpoint_dir):
           os.makedirs(checkpoint_dir)

       print "save to ", checkpoint_file_full_path
       pickle(checkpoint_file_full_path, dic,compress=self.zip_save)

       for f in sorted(os.listdir(checkpoint_dir), key=alphanum_key):
           if sum(os.path.getsize(os.path.join(checkpoint_dir, f2)) for f2 in os.listdir(checkpoint_dir)) > self.max_filesize_mb*1024*1024 and f != checkpoint_file:
               os.remove(os.path.join(checkpoint_dir, f))
           else:
               break

   def get_network_output( self, input_data, layer_name = None, dp = None ):
      if layer_name is None:
         layer_name = 'probs'
      if dp is None:
         dp = self.train_data_provider
      # get layer output dim
      layer_names = [ l['name'] for l in self.layers]
      layer_index = layer_names.index( layer_name )
      layer_output_dim = self.layers[layer_index]['outputs']
      # get num input data
      num_input_data = input_data[0].shape[1]
      # get nn output
      pred = np.zeros( ( num_input_data, layer_output_dim ), dtype = np.single )
      data = input_data + [ pred ]
      self.libmodel.startFeatureWriter( data, layer_index )
      self.finish_batch()
      return data

   @staticmethod
   def load_checkpoint(load_dir):
       if os.path.isdir(load_dir):
           return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
       return unpickle(load_dir)

def signal_handler( signal, frame):
    sys.exit(0)

def main():
   # set up signal handler
   signal.signal( signal.SIGINT, signal_handler )
   #nr.seed(5)
   op = get_options_parser()
   op, load_dic = parse_options(op)
   test_provider,train_provider = imagenetdata.init_data_providers(op)
   model = ConvNet(op, load_dic, train_provider, test_provider )
   model.start()

if __name__ == "__main__":
    main()
