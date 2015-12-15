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

from math import exp
import sys
import ConfigParser as cfg
import os
import numpy as n
import numpy.random as nr
from math import ceil, floor
from ordereddict import OrderedDict
from os import linesep as NL
from options import OptionsParser
import re

class LayerParsingError(Exception):
    pass

# A neuron that doesn't take parameters
class NeuronParser:
    def __init__(self, type, func_str, uses_acts=True, uses_inputs=True):
        self.type = type
        self.func_str = func_str
        self.uses_acts = uses_acts  
        self.uses_inputs = uses_inputs
        
    def parse(self, type):
        if type == self.type:
            return {'type': self.type,
                    'params': {},
                    'usesActs': self.uses_acts,
                    'usesInputs': self.uses_inputs}
        return None
    
# A neuron that takes parameters
class ParamNeuronParser(NeuronParser):
    neuron_regex = re.compile(r'^\s*(\w+)\s*\[\s*(\w+(\s*,\w+)*)\s*\]\s*$')
    def __init__(self, type, func_str, uses_acts=True, uses_inputs=True):
        NeuronParser.__init__(self, type, func_str, uses_acts, uses_inputs)
        m = self.neuron_regex.match(type)
        self.base_type = m.group(1)
        self.param_names = m.group(2).split(',')
        assert len(set(self.param_names)) == len(self.param_names)
        
    def parse(self, type):
        m = re.match(r'^%s\s*\[([\d,\.\s\-e]*)\]\s*$' % self.base_type, type)
        if m:
            try:
                param_vals = [float(v.strip()) for v in m.group(1).split(',')]
                if len(param_vals) == len(self.param_names):
                    return {'type': self.base_type,
                            'params': dict(zip(self.param_names, param_vals)),
                            'usesActs': self.uses_acts,
                            'usesInputs': self.uses_inputs}
            except TypeError:
                pass
        return None

class AbsTanhNeuronParser(ParamNeuronParser):
    def __init__(self):
        ParamNeuronParser.__init__(self, 'abstanh[a,b]', 'f(x) = a * |tanh(b * x)|')
        
    def parse(self, type):
        dic = ParamNeuronParser.parse(self, type)
        # Make b positive, since abs(tanh(bx)) = abs(tanh(-bx)) and the C++ code
        # assumes b is positive.
        if dic:
            dic['params']['b'] = abs(dic['params']['b'])
        return dic

# Subclass that throws more convnet-specific exceptions than the default
class MyConfigParser(cfg.SafeConfigParser):
    def safe_get(self, section, option, f=cfg.SafeConfigParser.get, typestr=None, default=None):
        try:
            return f(self, section, option)
        except cfg.NoOptionError, e:
            if default is not None:
                return default
            raise LayerParsingError("Layer '%s': required parameter '%s' missing" % (section, option))
        except ValueError, e:
            if typestr is None:
                raise e
            raise LayerParsingError("Layer '%s': parameter '%s' must be %s" % (section, option, typestr))
        
    def safe_get_list(self, section, option, f=str, typestr='strings', default=None):
        v = self.safe_get(section, option, default=default)
        if type(v) == list:
            return v
        try:
            return [f(x.strip()) for x in v.split(',')]
        except:
            raise LayerParsingError("Layer '%s': parameter '%s' must be ','-delimited list of %s" % (section, option, typestr))
        
    def safe_get_int(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getint, typestr='int', default=default)
        
    def safe_get_float(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getfloat, typestr='float', default=default)
    
    def safe_get_bool(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getboolean, typestr='bool', default=default)
    
    def safe_get_float_list(self, section, option, default=None):
        return self.safe_get_list(section, option, float, typestr='floats', default=default)
    
    def safe_get_int_list(self, section, option, default=None):
        return self.safe_get_list(section, option, int, typestr='ints', default=default)
    
    def safe_get_bool_list(self, section, option, default=None):
        return self.safe_get_list(section, option, lambda x: x.lower() in ('true', '1'), typestr='bools', default=default)

# A class that implements part of the interface of MyConfigParser
class FakeConfigParser(object):
    def __init__(self, dic):
        self.dic = dic

    def safe_get(self, section, option, default=None):
        return self.dic[option]
        

class LayerParser:
    def __init__(self):
        self.dic = {}
        self.set_defaults()
        
    # Post-processing step -- this is called after all layers have been initialized
    def optimize(self, layers):
        self.dic['actsTarget'] = -1
        self.dic['actsGradTarget'] = -1
    
    # Add parameters from layer parameter file
    def add_params(self, mcp):
        pass
    
    def init(self, dic):
        self.dic = dic
        return self
    
    def set_defaults(self):
        self.dic['outputs'] = 0
        self.dic['parser'] = self
        self.dic['requiresParams'] = False
        # Does this layer use its own activity matrix
        # for some purpose other than computing its output?
        # Usually, this will only be true for layers that require their
        # own activity matrix for gradient computations. For example, layers
        # with logistic units must compute the gradient y * (1 - y), where y is 
        # the activity matrix.
        # 
        # Layers that do not not use their own activity matrix should advertise
        # this, since this will enable memory-saving matrix re-use optimizations.
        #
        # The default value of this property is True, for safety purposes.
        # If a layer advertises that it does not use its own activity matrix when
        # in fact it does, bad things will happen.
        self.dic['usesActs'] = True
        
        # Does this layer use the activity matrices of its input layers
        # for some purpose other than computing its output?
        #
        # Again true by default for safety
        self.dic['usesInputs'] = True
        
        # Force this layer to use its own activity gradient matrix,
        # instead of borrowing one from one of its inputs.
        # 
        # This should be true for layers where the mapping from output
        # gradient to input gradient is non-elementwise.
        self.dic['forceOwnActs'] = True
        
        # Does this layer need the gradient at all?
        # Should only be true for layers with parameters (weights).
        self.dic['gradConsumer'] = False
        
    def parse(self, name, mcp, prev_layers, model=None):
        self.prev_layers = prev_layers
        self.dic['name'] = name
        self.dic['type'] = mcp.safe_get(name, 'type')

        return self.dic  

    def verify_float_range(self, v, param_name, _min, _max):
        self.verify_num_range(v, param_name, _min, _max, strconv=lambda x: '%.3f' % x)

    def verify_num_range(self, v, param_name, _min, _max, strconv=lambda x:'%d' % x):
        if type(v) == list:
            for i,vv in enumerate(v):
                self._verify_num_range(vv, param_name, _min, _max, i, strconv=strconv)
        else:
            self._verify_num_range(v, param_name, _min, _max, strconv=strconv)
    
    def _verify_num_range(self, v, param_name, _min, _max, input=-1, strconv=lambda x:'%d' % x):
        layer_name = self.dic['name'] if input < 0 else '%s[%d]' % (self.dic['name'], input)
        if _min is not None and _max is not None and (v < _min or v > _max):
            raise LayerParsingError("Layer '%s': parameter '%s' must be in the range %s-%s" % (layer_name, param_name, strconv(_min), strconv(_max)))
        elif _min is not None and v < _min:
            raise LayerParsingError("Layer '%s': parameter '%s' must be greater than or equal to %s" % (layer_name, param_name,  strconv(_min)))
        elif _max is not None and v > _max:
            raise LayerParsingError("Layer '%s': parameter '%s' must be smaller than or equal to %s" % (layer_name, param_name,  strconv(_max)))
    
    def verify_divisible(self, value, div, value_name, div_name=None, input_idx=0):
        layer_name = self.dic['name'] if len(self.dic['inputs']) == 0 else '%s[%d]' % (self.dic['name'], input_idx)
        if value % div != 0:
            raise LayerParsingError("Layer '%s': parameter '%s' must be divisible by %s" % (layer_name, value_name, str(div) if div_name is None else "'%s'" % div_name))
        
    def verify_str_in(self, value, lst):
        if value not in lst:
            raise LayerParsingError("Layer '%s': parameter '%s' must be one of %s" % (self.dic['name'], value, ", ".join("'%s'" % s for s in lst)))
        
    def verify_int_in(self, value, lst):
        if value not in lst:
            raise LayerParsingError("Layer '%s': parameter '%s' must be one of %s" % (self.dic['name'], value, ", ".join("'%d'" % s for s in lst)))

    # This looks for neuron=x arguments in various layers, and creates
    # separate layer definitions for them.
    @staticmethod
    def detach_neuron_layers(layers):
        layers_new = []
        for i, l in enumerate(layers):
            layers_new += [l]
            if l['type'] != 'neuron' and 'neuron' in l and l['neuron']:
                NeuronLayerParser().detach_neuron_layer(i, layers, layers_new)
        return layers_new
                
    @staticmethod
    def parse_layers(layer_cfg_path, param_cfg_path, model, layers=[]):
        try:
            if not os.path.exists(layer_cfg_path):
                raise LayerParsingError("Layer definition file '%s' does not exist" % layer_cfg_path)
            if not os.path.exists(param_cfg_path):
                raise LayerParsingError("Layer parameter file '%s' does not exist" % param_cfg_path)
            if len(layers) == 0:
                mcp = MyConfigParser(dict_type=OrderedDict)
                mcp.read([layer_cfg_path])
                for name in mcp.sections():
                    if not mcp.has_option(name, 'type'):
                        raise LayerParsingError("Layer '%s': no type given" % name)
                    ltype = mcp.safe_get(name, 'type')
                    if ltype not in layer_parsers:
                        raise LayerParsingError("Layer '%s': Unknown layer type: '%s'" % (name, ltype))
                    layers += [layer_parsers[ltype]().parse(name, mcp, layers, model)]
                
                layers = LayerParser.detach_neuron_layers(layers)
                for l in layers:
                    lp = layer_parsers[l['type']]()
                    l['parser'].optimize(layers)
                    del l['parser']
                    
                for l in layers:
                    if not l['type'].startswith('cost.'):
                        found = max(l['name'] in [layers[n]['name'] for n in l2['inputs']] for l2 in layers if 'inputs' in l2)
                        if not found:
                            raise LayerParsingError("Layer '%s' of type '%s' is unused" % (l['name'], l['type']))
            
            mcp = MyConfigParser(dict_type=OrderedDict)
            mcp.read([param_cfg_path])
            
            for l in layers:
                if not mcp.has_section(l['name']) and l['requiresParams']:
                    raise LayerParsingError("Layer '%s' of type '%s' requires extra parameters, but none given in file '%s'." % (l['name'], l['type'], param_cfg_path))
                lp = layer_parsers[l['type']]().init(l)
                lp.add_params(mcp)
                lp.dic['conserveMem'] = model.op.get_value('conserve_mem')
        except LayerParsingError, e:
            print e
            sys.exit(1)
        return layers
        
    @staticmethod
    def register_layer_parser(ltype, cls):
        if ltype in layer_parsers:
            raise LayerParsingError("Layer type '%s' already registered" % ltype)
        layer_parsers[ltype] = cls

# Any layer that takes an input (i.e. non-data layer)
class LayerWithInputParser(LayerParser):
    def __init__(self, num_inputs=-1):
        LayerParser.__init__(self)
        self.num_inputs = num_inputs

    def verify_num_params(self, params):
        for param in params:
            if len(self.dic[param]) != len(self.dic['inputs']):
                raise LayerParsingError("Layer '%s': %s list length does not match number of inputs" % (self.dic['name'], param))        
    
    def optimize(self, layers):
        LayerParser.optimize(self, layers)
        dic = self.dic
        # Check if I have an input that no one else uses.
        if not dic['forceOwnActs']:
            for i, inp in enumerate(dic['inputs']):
                l = layers[inp]
                if l['outputs'] == dic['outputs'] and sum('inputs' in ll and inp in ll['inputs'] for ll in layers) == 1:
                    # I can share my activity matrix with this layer
                    # if it does not use its activity matrix, and I 
                    # do not need to remember my inputs.
                    if not l['usesActs'] and not dic['usesInputs']:
                        dic['actsTarget'] = i
#                        print "Layer '%s' sharing activity matrix with layer '%s'" % (dic['name'], l['name'])
                    # I can share my gradient matrix with this layer.
                    dic['actsGradTarget'] = i
#                    print "Layer '%s' sharing activity gradient matrix with layer '%s'" % (dic['name'], l['name'])
            
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['inputs'] = [inp.strip() for inp in mcp.safe_get(name, 'inputs').split(',')]
        prev_names = [p['name'] for p in prev_layers]
        for inp in dic['inputs']:
            if inp not in prev_names:
                raise LayerParsingError("Layer '%s': input layer '%s' not defined" % (name, inp))
        dic['inputs'] = [prev_names.index(inp) for inp in dic['inputs']]
        dic['inputLayers'] = [prev_layers[inp] for inp in dic['inputs']]
        for inp in dic['inputs']:
            if prev_layers[inp]['outputs'] == 0:
                raise LayerParsingError("Layer '%s': input layer '%s' does not produce any output" % (name, prev_names[inp]))
        dic['numInputs'] = [prev_layers[i]['outputs'] for i in dic['inputs']]
        
        # Layers can declare a neuron activation function to apply to their output, as a shortcut
        # to avoid declaring a separate neuron layer above themselves.
        dic['neuron'] = mcp.safe_get(name, 'neuron', default="")
        if self.num_inputs > 0 and len(dic['numInputs']) != self.num_inputs:
            raise LayerParsingError("Layer '%s': number of inputs must be %d", name, self.num_inputs)
        
#        input_layers = [prev_layers[i] for i in dic['inputs']]
#        dic['gradConsumer'] = any(l['gradConsumer'] for l in dic['inputLayers'])
#        dic['usesActs'] = dic['gradConsumer'] # A conservative setting by default for layers with input
        return dic
    
    def verify_img_size(self):
        dic = self.dic
        if dic['numInputs'][0] % dic['imgPixels'] != 0 or dic['imgSize'] * dic['imgSize'] != dic['imgPixels']:
            raise LayerParsingError("Layer '%s': has %-d dimensional input, not interpretable as %d-channel images" % (dic['name'], dic['numInputs'][0], dic['channels']))
    
    @staticmethod
    def grad_consumers_below(dic):
        if dic['gradConsumer']:
            return True
        if 'inputLayers' in dic:
            return any(LayerWithInputParser.grad_consumers_below(l) for l in dic['inputLayers'])
        
    def verify_no_grads(self):
        if LayerWithInputParser.grad_consumers_below(self.dic):
            raise LayerParsingError("Layer '%s': layers of type '%s' cannot propagate gradient and must not be placed over layers with parameters." % (self.dic['name'], self.dic['type']))

class NailbedLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['stride'] = mcp.safe_get_int(name, 'stride')

        self.verify_num_range(dic['channels'], 'channels', 1, None)
        
        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        dic['outputsX'] = (dic['imgSize'] + dic['stride'] - 1) / dic['stride']
        dic['start'] = (dic['imgSize'] - dic['stride'] * (dic['outputsX'] - 1)) / 2
        dic['outputs'] = dic['channels'] * dic['outputsX']**2
        
        self.verify_num_range(dic['outputsX'], 'outputsX', 0, None)
        
        self.verify_img_size()
        
        print "Initialized bed-of-nails layer '%s', producing %dx%d %d-channel output" % (name, dic['outputsX'], dic['outputsX'], dic['channels'])
        return dic
    
class GaussianBlurLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        dic['outputs'] = dic['numInputs'][0]
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['filterSize'] = mcp.safe_get_int(name, 'filterSize')
        dic['stdev'] = mcp.safe_get_float(name, 'stdev')

        self.verify_num_range(dic['channels'], 'channels', 1, None)
        self.verify_int_in(dic['filterSize'], [3, 5, 7, 9])
        
        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        dic['filter'] = n.array([exp(-(dic['filterSize']/2 - i)**2 / float(2 * dic['stdev']**2)) 
                                 for i in xrange(dic['filterSize'])], dtype=n.float32).reshape(1, dic['filterSize'])
        dic['filter'] /= dic['filter'].sum()

        self.verify_img_size()
        
        if dic['filterSize'] > dic['imgSize']:
            raise LayerParsingError("Later '%s': filter size (%d) must be smaller than image size (%d)." % (dic['name'], dic['filterSize'], dic['imgSize']))
        
        print "Initialized Gaussian blur layer '%s', producing %dx%d %d-channel output" % (name, dic['imgSize'], dic['imgSize'], dic['channels'])
        
        return dic
    
class ResizeLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        dic['scale'] = mcp.safe_get_float(name, 'scale')
        dic['tgtSize'] = int(floor(dic['imgSize'] / dic['scale']))
        dic['tgtPixels'] = dic['tgtSize']**2
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        # Really not recommended to use this for such severe scalings
        self.verify_float_range(dic['scale'], 'scale', 0.5, 2) 

        dic['outputs'] = dic['channels'] * dic['tgtPixels']
        
        self.verify_img_size()
        self.verify_no_grads()
        
        print "Initialized resize layer '%s', producing %dx%d %d-channel output" % (name, dic['tgtSize'], dic['tgtSize'], dic['channels'])
        
        return dic
    
class RandomScaleLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        
        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        dic['maxScale'] = mcp.safe_get_float(name, 'maxScale')
        dic['tgtSize'] = int(floor(dic['imgSize'] / dic['maxScale']))
        dic['tgtPixels'] = dic['tgtSize']**2
        
        self.verify_float_range(dic['maxScale'], 'maxScale', 1, 2) 

        dic['outputs'] = dic['channels'] * dic['tgtPixels']
        
        self.verify_img_size()
        self.verify_no_grads()
        
        print "Initialized random scale layer '%s', producing %dx%d %d-channel output" % (name, dic['tgtSize'], dic['tgtSize'], dic['channels'])
        
        return dic
    
class ColorTransformLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
    
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False

        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / 3
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        dic['channels'] = 3
        dic['outputs'] = dic['numInputs'][0]
        
        self.verify_img_size()
        self.verify_no_grads()
        
        return dic
    
class RGBToYUVLayerParser(ColorTransformLayerParser):
    def __init__(self):
        ColorTransformLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = ColorTransformLayerParser.parse(self, name, mcp, prev_layers, model)
        print "Initialized RGB --> YUV layer '%s', producing %dx%d %d-channel output" % (name, dic['imgSize'], dic['imgSize'], dic['channels'])
        return dic
    
class RGBToLABLayerParser(ColorTransformLayerParser):
    def __init__(self):
        ColorTransformLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = ColorTransformLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['center'] = mcp.safe_get_bool(name, 'center', default=False)
        print "Initialized RGB --> LAB layer '%s', producing %dx%d %d-channel output" % (name, dic['imgSize'], dic['imgSize'], dic['channels'])
        return dic

class NeuronLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
    
    @staticmethod
    def get_unused_layer_name(layers, wish):
        layer_names = set([l['name'] for l in layers])
        if wish not in layer_names:
            return wish
        for i in xrange(1, 100):
            name = '%s.%d' % (wish, i)
            if name not in layer_names:
                return name
        raise LayerParsingError("This is insane.")
    
    def parse_neuron(self, neuron_str):
        for n in neuron_parsers:
            p = n.parse(neuron_str)
            if p: # Successfully parsed neuron, return it
                self.dic['neuron'] = p
                self.dic['usesActs'] = self.dic['neuron']['usesActs']
                self.dic['usesInputs'] = self.dic['neuron']['usesInputs']
                
                return
        # Could not parse neuron
        # Print available neuron types
        colnames = ['Neuron type', 'Function']
        m = max(len(colnames[0]), OptionsParser._longest_value(neuron_parsers, key=lambda x:x.type)) + 2
        ntypes = [OptionsParser._bold(colnames[0].ljust(m))] + [n.type.ljust(m) for n in neuron_parsers]
        fnames = [OptionsParser._bold(colnames[1])] + [n.func_str for n in neuron_parsers]
        usage_lines = NL.join(ntype + fname for ntype,fname in zip(ntypes, fnames))
        
        raise LayerParsingError("Layer '%s': unable to parse neuron type '%s'. Valid neuron types: %sWhere neurons have parameters, they must be floats." % (self.dic['name'], neuron_str, NL + usage_lines + NL))
    
    def detach_neuron_layer(self, idx, layers, layers_new):
        dic = self.dic
        self.set_defaults()
        dic['name'] = NeuronLayerParser.get_unused_layer_name(layers, '%s_neuron' % layers[idx]['name'])
        dic['type'] = 'neuron'
        dic['inputs'] = layers[idx]['name']
        dic['neuron'] = layers[idx]['neuron']

        dic = self.parse(dic['name'], FakeConfigParser(dic), layers_new)
        
        # Link upper layers to this new one
        for l in layers[idx+1:]:
            if 'inputs' in l:
                l['inputs'] = [i + (i >= len(layers_new) - 1) for i in l['inputs']]
            if 'weightSourceLayerIndices' in l:
                l['weightSourceLayerIndices'] = [i + (i >= len(layers_new)) for i in l['weightSourceLayerIndices']]
        layers_new += [dic]
        
#        print "Initialized implicit neuron layer '%s', producing %d outputs" % (dic['name'], dic['outputs'])
    
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['outputs'] = dic['numInputs'][0]
        self.parse_neuron(dic['neuron'])
        dic['forceOwnActs'] = False
        print "Initialized neuron layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class EltwiseSumLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        
        if len(set(dic['numInputs'])) != 1:
            raise LayerParsingError("Layer '%s': all inputs must have the same dimensionality. Got dimensionalities: %s" % (name, ", ".join(str(s) for s in dic['numInputs'])))
        dic['outputs'] = dic['numInputs'][0]
        dic['usesInputs'] = False
        dic['usesActs'] = False
        dic['forceOwnActs'] = False
        
        dic['coeffs'] = mcp.safe_get_float_list(name, 'coeffs', default=[1.0] * len(dic['inputs']))
        
        print "Initialized elementwise sum layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic
    
class EltwiseMaxLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        if len(dic['inputs']) < 2:
            raise LayerParsingError("Layer '%s': elementwise max layer must have at least 2 inputs, got %d." % (name, len(dic['inputs'])))
        if len(set(dic['numInputs'])) != 1:
            raise LayerParsingError("Layer '%s': all inputs must have the same dimensionality. Got dimensionalities: %s" % (name, ", ".join(str(s) for s in dic['numInputs'])))
        dic['outputs'] = dic['numInputs'][0]

        print "Initialized elementwise max layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class WeightLayerParser(LayerWithInputParser):
    LAYER_PAT = re.compile(r'^\s*([^\s\[]+)(?:\[(\d+)\])?\s*$') # matches things like layername[5], etc
    
    def __init__(self):
        LayerWithInputParser.__init__(self)
    
    @staticmethod
    def get_layer_name(name_str):
        m = WeightLayerParser.LAYER_PAT.match(name_str)
        if not m:
            return None
        return m.group(1), m.group(2)
    
    def add_params(self, mcp):
        dic, name = self.dic, self.dic['name']
        dic['epsW'] = mcp.safe_get_float_list(name, 'epsW')
        dic['epsB'] = mcp.safe_get_float(name, 'epsB')
        dic['momW'] = mcp.safe_get_float_list(name, 'momW')
        dic['momB'] = mcp.safe_get_float(name, 'momB')
        dic['wc'] = mcp.safe_get_float_list(name, 'wc')
        
        self.verify_num_params(['epsW', 'momW', 'wc'])
        
        dic['gradConsumer'] = dic['epsB'] > 0 or any(w > 0 for w in dic['epsW'])
        
    @staticmethod
    def unshare_weights(layer, layers, matrix_idx=None):
        def unshare(layer, layers, indices):
            for i in indices:
                if layer['weightSourceLayerIndices'][i] >= 0:
                    src_name = layers[layer['weightSourceLayerIndices'][i]]['name']
                    src_matrix_idx = layer['weightSourceMatrixIndices'][i]
                    layer['weightSourceLayerIndices'][i] = -1
                    layer['weightSourceMatrixIndices'][i] = -1
                    layer['weights'][i] = layer['weights'][i].copy()
                    layer['weightsInc'][i] = n.zeros_like(layer['weights'][i])
                    print "Unshared weight matrix %s[%d] from %s[%d]." % (layer['name'], i, src_name, src_matrix_idx)
                else:
                    print "Weight matrix %s[%d] already unshared." % (layer['name'], i)
        if 'weightSourceLayerIndices' in layer:
            unshare(layer, layers, range(len(layer['inputs'])) if matrix_idx is None else [matrix_idx])

    # Load weight/biases initialization module
    def call_init_func(self, param_name, shapes, input_idx=-1):
        dic = self.dic
        func_pat = re.compile('^([^\.]+)\.([^\(\)]+)\s*(?:\(([^,]+(?:,[^,]+)*)\))?$')
        m = func_pat.match(dic[param_name])
        if not m:
            raise LayerParsingError("Layer '%s': '%s' parameter must have format 'moduleName.functionName(param1,param2,...)'; got: %s." % (dic['name'], param_name, dic['initWFunc']))
        module, func = m.group(1), m.group(2)
        params = m.group(3).split(',') if m.group(3) is not None else []
        try:
            mod = __import__(module)
            return getattr(mod, func)(dic['name'], input_idx, shapes, params=params) if input_idx >= 0 else getattr(mod, func)(dic['name'], shapes, params=params)
        except (ImportError, AttributeError, TypeError), e:
            raise LayerParsingError("Layer '%s': %s." % (dic['name'], e))
        
    def make_weights(self, initW, rows, cols, order='C'):
        dic = self.dic
        dic['weights'], dic['weightsInc'] = [], []
        if dic['initWFunc']: # Initialize weights from user-supplied python function
            # Initialization function is supplied in the format
            # module.func
            for i in xrange(len(dic['inputs'])):
                dic['weights'] += [self.call_init_func('initWFunc', (rows[i], cols[i]), input_idx=i)]

                if type(dic['weights'][i]) != n.ndarray:
                    raise LayerParsingError("Layer '%s[%d]': weight initialization function %s must return numpy.ndarray object. Got: %s." % (dic['name'], i, dic['initWFunc'], type(dic['weights'][i])))
                if dic['weights'][i].dtype != n.float32:
                    raise LayerParsingError("Layer '%s[%d]': weight initialization function %s must weight matrices consisting of single-precision floats. Got: %s." % (dic['name'], i, dic['initWFunc'], dic['weights'][i].dtype))
                if dic['weights'][i].shape != (rows[i], cols[i]):
                    raise LayerParsingError("Layer '%s[%d]': weight matrix returned by weight initialization function %s has wrong shape. Should be: %s; got: %s." % (dic['name'], i, dic['initWFunc'], (rows[i], cols[i]), dic['weights'][i].shape))
                # Convert to desired order
                dic['weights'][i] = n.require(dic['weights'][i], requirements=order)
                dic['weightsInc'] += [n.zeros_like(dic['weights'][i])]
                print "Layer '%s[%d]' initialized weight matrices from function %s" % (dic['name'], i, dic['initWFunc'])
        else:
            for i in xrange(len(dic['inputs'])):
                if dic['weightSourceLayerIndices'][i] >= 0: # Shared weight matrix
                    src_layer = self.prev_layers[dic['weightSourceLayerIndices'][i]] if dic['weightSourceLayerIndices'][i] < len(self.prev_layers) else dic
                    dic['weights'] += [src_layer['weights'][dic['weightSourceMatrixIndices'][i]]]
                    dic['weightsInc'] += [src_layer['weightsInc'][dic['weightSourceMatrixIndices'][i]]]
                    if dic['weights'][i].shape != (rows[i], cols[i]):
                        raise LayerParsingError("Layer '%s': weight sharing source matrix '%s' has shape %dx%d; should be %dx%d." 
                                                % (dic['name'], dic['weightSource'][i], dic['weights'][i].shape[0], dic['weights'][i].shape[1], rows[i], cols[i]))
                    print "Layer '%s' initialized weight matrix %d from %s" % (dic['name'], i, dic['weightSource'][i])
                else:
                    dic['weights'] += [n.array(initW[i] * nr.randn(rows[i], cols[i]), dtype=n.single, order=order)]
                    dic['weightsInc'] += [n.zeros_like(dic['weights'][i])]
        
    def make_biases(self, rows, cols, order='C'):
        dic = self.dic
        if dic['initBFunc']:
            dic['biases'] = self.call_init_func('initBFunc', (rows, cols))
            if type(dic['biases']) != n.ndarray:
                raise LayerParsingError("Layer '%s': bias initialization function %s must return numpy.ndarray object. Got: %s." % (dic['name'], dic['initBFunc'], type(dic['biases'])))
            if dic['biases'].dtype != n.float32:
                raise LayerParsingError("Layer '%s': bias initialization function %s must return numpy.ndarray object consisting of single-precision floats. Got: %s." % (dic['name'], dic['initBFunc'], dic['biases'].dtype))
            if dic['biases'].shape != (rows, cols):
                raise LayerParsingError("Layer '%s': bias vector returned by bias initialization function %s has wrong shape. Should be: %s; got: %s." % (dic['name'], dic['initBFunc'], (rows, cols), dic['biases'].shape))

            dic['biases'] = n.require(dic['biases'], requirements=order)
            print "Layer '%s' initialized bias vector from function %s" % (dic['name'], dic['initBFunc'])
        else:
            dic['biases'] = dic['initB'] * n.ones((rows, cols), order='C', dtype=n.single)
        dic['biasesInc'] = n.zeros_like(dic['biases'])
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        dic['gradConsumer'] = True
        dic['initW'] = mcp.safe_get_float_list(name, 'initW', default=0.01)
        dic['initB'] = mcp.safe_get_float(name, 'initB', default=0)
        dic['initWFunc'] = mcp.safe_get(name, 'initWFunc', default="")
        dic['initBFunc'] = mcp.safe_get(name, 'initBFunc', default="")
        # Find shared weight matrices
        
        dic['weightSource'] = mcp.safe_get_list(name, 'weightSource', default=[''] * len(dic['inputs']))
        self.verify_num_params(['initW', 'weightSource'])
        
        prev_names = map(lambda x: x['name'], prev_layers)
        dic['weightSourceLayerIndices'] = []
        dic['weightSourceMatrixIndices'] = []
        for i, src_name in enumerate(dic['weightSource']):
            src_layer_idx = src_layer_matrix_idx = -1
            if src_name != '':
                src_layer_match = WeightLayerParser.get_layer_name(src_name)
                if src_layer_match is None:
                    raise LayerParsingError("Layer '%s': unable to parse weight sharing source '%s'. Format is layer[idx] or just layer, in which case idx=0 is used." % (name, src_name))
                src_layer_name = src_layer_match[0]
                src_layer_matrix_idx = int(src_layer_match[1]) if src_layer_match[1] is not None else 0

                if prev_names.count(src_layer_name) == 0 and src_layer_name != name:
                    raise LayerParsingError("Layer '%s': weight sharing source layer '%s' does not exist." % (name, src_layer_name))
                
                src_layer_idx = prev_names.index(src_layer_name) if src_layer_name != name else len(prev_names)
                src_layer = prev_layers[src_layer_idx] if src_layer_name != name else dic
                if src_layer['type'] != dic['type']:
                    raise LayerParsingError("Layer '%s': weight sharing source layer '%s' is of type '%s'; should be '%s'." % (name, src_layer_name, src_layer['type'], dic['type']))
                if src_layer_name != name and len(src_layer['weights']) <= src_layer_matrix_idx:
                    raise LayerParsingError("Layer '%s': weight sharing source layer '%s' has %d weight matrices, but '%s[%d]' requested." % (name, src_layer_name, len(src_layer['weights']), src_name, src_layer_matrix_idx))
                if src_layer_name == name and src_layer_matrix_idx >= i:
                    raise LayerParsingError("Layer '%s': weight sharing source '%s[%d]' not defined yet." % (name, name, src_layer_matrix_idx))

            dic['weightSourceLayerIndices'] += [src_layer_idx]
            dic['weightSourceMatrixIndices'] += [src_layer_matrix_idx]
                
        return dic
        
class FCLayerParser(WeightLayerParser):
    def __init__(self):
        WeightLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = WeightLayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['usesActs'] = False
        dic['outputs'] = mcp.safe_get_int(name, 'outputs')
        
        self.verify_num_range(dic['outputs'], 'outputs', 1, None)
        self.make_weights(dic['initW'], dic['numInputs'], [dic['outputs']] * len(dic['numInputs']), order='F')
        self.make_biases(1, dic['outputs'], order='F')
        print "Initialized fully-connected layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class FCDropOutLayerParser( FCLayerParser ):
    def __init__(self):
        FCLayerParser.__init__(self)
    def parse(self, name, mcp, prev_layers, model):
        dic = FCLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['rate'] = mcp.safe_get_float( name, 'rate' )
        assert( dic['rate'] >= 0 and dic['rate'] <= 1 )
        print "Using DropOut, Output Drop rate: ", dic['rate']
        return dic

class FCDropConnectLayerParser( FCLayerParser ):
    def __init__(self):
        FCLayerParser.__init__(self)
    def parse(self, name, mcp, prev_layers, model):
        dic = FCLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['rate'] = mcp.safe_get_float( name, 'rate' )
        assert( dic['rate'] >= 0 and dic['rate'] <= 1 )
        print "Using DropConnect, Connection Drop rate: ", dic['rate']
        return dic

class FCDropConnectFastLayerParser( FCLayerParser ):
    def __init__(self):
        FCLayerParser.__init__(self)
    def parse(self, name, mcp, prev_layers, model):
        dic = FCLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['rate'] = mcp.safe_get_float( name, 'rate' )
        assert( dic['rate'] >= 0 and dic['rate'] <= 1 )
        print "Connection Drop rate(fast): ", dic['rate']
        return dic

class LocalLayerParser(WeightLayerParser):
    def __init__(self):
        WeightLayerParser.__init__(self)
        
    # Convert convolutional layer to unshared, locally-connected layer
    @staticmethod
    def conv_to_local(layers, idx):
        layer = layers[idx]
        if layer['type'] == 'conv':
            layer['type'] = 'local'
            for inp in xrange(len(layer['inputs'])):
                src_layer_idx = layer['weightSourceLayerIndices'][inp]
                if layer['weightSourceLayerIndices'][inp] >= 0:
                    src_layer = layers[src_layer_idx]
                    src_matrix_idx = layer['weightSourceMatrixIndices'][inp]
                    LocalLayerParser.conv_to_local(layers, src_layer_idx)
                    for w in ('weights', 'weightsInc'):
                        layer[w][inp] = src_layer[w][src_matrix_idx]
                else:
                    layer['weights'][inp] = n.require(n.reshape(n.tile(n.reshape(layer['weights'][inp], (1, n.prod(layer['weights'][inp].shape))), (layer['modules'], 1)),
                                                        (layer['modules'] * layer['filterChannels'][inp] * layer['filterPixels'][inp], layer['filters'])),
                                                      requirements='C')
                    layer['weightsInc'][inp] = n.zeros_like(layer['weights'][inp])
            if layer['sharedBiases']:
                layer['biases'] = n.require(n.repeat(layer['biases'], layer['modules'], axis=0), requirements='C')
                layer['biasesInc'] = n.zeros_like(layer['biases'])
            
            print "Converted layer '%s' from convolutional to unshared, locally-connected" % layer['name']
            
            # Also call this function on any layers sharing my weights
            for i, l in enumerate(layers):
                if 'weightSourceLayerIndices' in l and idx in l['weightSourceLayerIndices']:
                    LocalLayerParser.conv_to_local(layers, i)
        return layer
        
    # Returns (groups, filterChannels) array that represents the set
    # of image channels to which each group is connected
    def gen_rand_conns(self, groups, channels, filterChannels, inputIdx):
        dic = self.dic
        overSample = groups * filterChannels / channels
        filterConns = [x for i in xrange(overSample) for x in nr.permutation(range(channels))]
        
        if dic['initCFunc']: # Initialize connectivity from outside source
            filterConns = self.call_init_func('initCFunc', (groups, channels, filterChannels), input_idx=inputIdx)
            if len(filterConns) != overSample * channels:
                raise LayerParsingError("Layer '%s[%d]': random connectivity initialization function %s must return list of length <groups> * <filterChannels> = %d; got: %d" % (dic['name'], inputIdx, dic['initCFunc'], len(filterConns)))
            if any(c not in range(channels) for c in filterConns):
                raise LayerParsingError("Layer '%s[%d]': random connectivity initialization function %s must return list of channel indices in the range 0-<channels-1> = 0-%d." % (dic['name'], inputIdx, dic['initCFunc'], channels-1))
            # Every "channels" sub-slice should be a permutation of range(channels)
            if any(len(set(c)) != len(c) for c in [filterConns[o*channels:(o+1)*channels] for o in xrange(overSample)]):
                raise LayerParsingError("Layer '%s[%d]': random connectivity initialization function %s must return list of channel indices such that every non-overlapping sub-list of <channels> = %d elements is a permutation of the integers 0-<channels-1> = 0-%d." % (dic['name'], inputIdx, dic['initCFunc'], channels, channels-1))

        elif dic['weightSourceLayerIndices'][inputIdx] >= 0: # Shared weight matrix
            src_layer = self.prev_layers[dic['weightSourceLayerIndices'][inputIdx]] if dic['weightSourceLayerIndices'][inputIdx] < len(self.prev_layers) else dic
            src_inp = dic['weightSourceMatrixIndices'][inputIdx]
            if 'randSparse' not in src_layer or not src_layer['randSparse']:
                raise LayerParsingError("Layer '%s[%d]': randSparse is true in this layer but false in weight sharing source layer '%s[%d]'." % (dic['name'], inputIdx, src_layer['name'], src_inp))
            if (groups, channels, filterChannels) != (src_layer['groups'][src_inp], src_layer['channels'][src_inp], src_layer['filterChannels'][src_inp]):
                raise LayerParsingError("Layer '%s[%d]': groups, channels, filterChannels set to %d, %d, %d, respectively. Does not match setting in weight sharing source layer '%s[%d]': %d, %d, %d." % (dic['name'], inputIdx, groups, channels, filterChannels, src_layer['name'], src_inp, src_layer['groups'][src_inp], src_layer['channels'][src_inp], src_layer['filterChannels'][src_inp]))
            filterConns = src_layer['filterConns'][src_inp]
        return filterConns
        
    def parse(self, name, mcp, prev_layers, model):
        dic = WeightLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        dic['usesActs'] = False
        # Supplied values
        dic['channels'] = mcp.safe_get_int_list(name, 'channels')
        dic['padding'] = mcp.safe_get_int_list(name, 'padding', default=[0]*len(dic['inputs']))
        dic['stride'] = mcp.safe_get_int_list(name, 'stride', default=[1]*len(dic['inputs']))
        dic['filterSize'] = mcp.safe_get_int_list(name, 'filterSize')
        dic['filters'] = mcp.safe_get_int_list(name, 'filters')
        dic['groups'] = mcp.safe_get_int_list(name, 'groups', default=[1]*len(dic['inputs']))
        dic['randSparse'] = mcp.safe_get_bool_list(name, 'randSparse', default=[False]*len(dic['inputs']))
        dic['initW'] = mcp.safe_get_float_list(name, 'initW')
        dic['initCFunc'] = mcp.safe_get(name, 'initCFunc', default='')
        
        self.verify_num_params(['channels', 'padding', 'stride', 'filterSize', \
                                                     'filters', 'groups', 'randSparse', 'initW'])
        
        self.verify_num_range(dic['stride'], 'stride', 1, None)
        self.verify_num_range(dic['filterSize'],'filterSize', 1, None)  
        self.verify_num_range(dic['padding'], 'padding', 0, None)
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        self.verify_num_range(dic['groups'], 'groups', 1, None)
        
        # Computed values
        dic['imgPixels'] = [numInputs/channels for numInputs,channels in zip(dic['numInputs'], dic['channels'])]
        dic['imgSize'] = [int(n.sqrt(imgPixels)) for imgPixels in dic['imgPixels']]
        self.verify_num_range(dic['imgSize'], 'imgSize', 1, None)
        dic['filters'] = [filters*groups for filters,groups in zip(dic['filters'], dic['groups'])]
        dic['filterPixels'] = [filterSize**2 for filterSize in dic['filterSize']]
        dic['modulesX'] = [1 + int(ceil((2 * padding + imgSize - filterSize) / float(stride))) for padding,imgSize,filterSize,stride in zip(dic['padding'], dic['imgSize'], dic['filterSize'], dic['stride'])]

        dic['filterChannels'] = [channels/groups for channels,groups in zip(dic['channels'], dic['groups'])]
        if max(dic['randSparse']): # When randSparse is turned on for any input, filterChannels must be given for all of them
            dic['filterChannels'] = mcp.safe_get_int_list(name, 'filterChannels', default=dic['filterChannels'])
            self.verify_num_params(['filterChannels'])
        
        if len(set(dic['modulesX'])) != 1 or len(set(dic['filters'])) != 1:
            raise LayerParsingError("Layer '%s': all inputs must produce equally-dimensioned output. Dimensions are: %s." % (name, ", ".join("%dx%dx%d" % (filters, modulesX, modulesX) for filters,modulesX in zip(dic['filters'], dic['modulesX']))))

        dic['modulesX'] = dic['modulesX'][0]
        dic['modules'] = dic['modulesX']**2
        dic['filters'] = dic['filters'][0]
        dic['outputs'] = dic['modules'] * dic['filters']
        dic['filterConns'] = [[]] * len(dic['inputs'])
        for i in xrange(len(dic['inputs'])):
            if dic['numInputs'][i] % dic['imgPixels'][i] != 0 or dic['imgSize'][i] * dic['imgSize'][i] != dic['imgPixels'][i]:
                raise LayerParsingError("Layer '%s[%d]': has %-d dimensional input, not interpretable as square %d-channel images" % (name, i, dic['numInputs'][i], dic['channels'][i]))
            if dic['channels'][i] > 3 and dic['channels'][i] % 4 != 0:
                raise LayerParsingError("Layer '%s[%d]': number of channels must be smaller than 4 or divisible by 4" % (name, i))
            if dic['filterSize'][i] > 2 * dic['padding'][i] + dic['imgSize'][i]:
                raise LayerParsingError("Layer '%s[%d]': filter size (%d) greater than image size + 2 * padding (%d)" % (name, i, dic['filterSize'][i], 2 * dic['padding'][i] + dic['imgSize'][i]))
        
            if dic['randSparse'][i]: # Random sparse connectivity requires some extra checks
                if dic['groups'][i] == 1:
                    raise LayerParsingError("Layer '%s[%d]': number of groups must be greater than 1 when using random sparse connectivity" % (name, i))
                self.verify_divisible(dic['channels'][i], dic['filterChannels'][i], 'channels', 'filterChannels', input_idx=i)
                self.verify_divisible(dic['filterChannels'][i], 4, 'filterChannels', input_idx=i)
                self.verify_divisible( dic['groups'][i]*dic['filterChannels'][i], dic['channels'][i], 'groups * filterChannels', 'channels', input_idx=i)
                dic['filterConns'][i] = self.gen_rand_conns(dic['groups'][i], dic['channels'][i], dic['filterChannels'][i], i)
            else:
                if dic['groups'][i] > 1:
                    self.verify_divisible(dic['channels'][i], 4*dic['groups'][i], 'channels', '4 * groups', input_idx=i)
                self.verify_divisible(dic['channels'][i], dic['groups'][i], 'channels', 'groups', input_idx=i)

            self.verify_divisible(dic['filters'], 16*dic['groups'][i], 'filters * groups', input_idx=i)
        
            dic['padding'][i] = -dic['padding'][i]
        dic['overSample'] = [groups*filterChannels/channels for groups,filterChannels,channels in zip(dic['groups'], dic['filterChannels'], dic['channels'])]

        return dic    

class ConvLayerParser(LocalLayerParser):
    def __init__(self):
        LocalLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LocalLayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['partialSum'] = mcp.safe_get_int(name, 'partialSum')
        dic['sharedBiases'] = mcp.safe_get_bool(name, 'sharedBiases', default=True)

        if dic['partialSum'] != 0 and dic['modules'] % dic['partialSum'] != 0:
            raise LayerParsingError("Layer '%s': convolutional layer produces %dx%d=%d outputs per filter, but given partialSum parameter (%d) does not divide this number" % (name, dic['modulesX'], dic['modulesX'], dic['modules'], dic['partialSum']))

        num_biases = dic['filters'] if dic['sharedBiases'] else dic['modules']*dic['filters']

        eltmult = lambda list1, list2: [l1 * l2 for l1,l2 in zip(list1, list2)]
        self.make_weights(dic['initW'], eltmult(dic['filterPixels'], dic['filterChannels']), [dic['filters']] * len(dic['inputs']), order='C')
        self.make_biases(num_biases, 1, order='C')

        print "Initialized convolutional layer '%s', producing %dx%d %d-channel output" % (name, dic['modulesX'], dic['modulesX'], dic['filters'])
        return dic    
    
class LocalUnsharedLayerParser(LocalLayerParser):
    def __init__(self):
        LocalLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LocalLayerParser.parse(self, name, mcp, prev_layers, model)

        eltmult = lambda list1, list2: [l1 * l2 for l1,l2 in zip(list1, list2)]
        scmult = lambda x, lst: [x * l for l in lst]
        self.make_weights(dic['initW'], scmult(dic['modules'], eltmult(dic['filterPixels'], dic['filterChannels'])), [dic['filters']] * len(dic['inputs']), order='C')
        self.make_biases(dic['modules'] * dic['filters'], 1, order='C')
        
        print "Initialized locally-connected layer '%s', producing %dx%d %d-channel output" % (name, dic['modulesX'], dic['modulesX'], dic['filters'])
        return dic  
    
class DataLayerParser(LayerParser):
    def __init__(self):
        LayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerParser.parse(self, name, mcp, prev_layers, model)
        dic['dataIdx'] = mcp.safe_get_int(name, 'dataIdx')
        dic['outputs'] = model.train_data_provider.get_data_dims(idx=dic['dataIdx'])
        
        print "Initialized data layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class SoftmaxLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['outputs'] = prev_layers[dic['inputs'][0]]['outputs']
        print "Initialized softmax layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class PoolLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['sizeX'] = mcp.safe_get_int(name, 'sizeX')
        dic['start'] = mcp.safe_get_int(name, 'start', default=0)
        dic['stride'] = mcp.safe_get_int(name, 'stride')
        dic['outputsX'] = mcp.safe_get_int(name, 'outputsX', default=0)
        dic['pool'] = mcp.safe_get(name, 'pool')
        
        # Avg pooler does not use its acts or inputs
        dic['usesActs'] = 'pool' != 'avg'
        dic['usesInputs'] = 'pool' != 'avg'
        
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        self.verify_num_range(dic['sizeX'], 'sizeX', 1, dic['imgSize'])
        self.verify_num_range(dic['stride'], 'stride', 1, dic['sizeX'])
        self.verify_num_range(dic['outputsX'], 'outputsX', 0, None)
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        
        if LayerWithInputParser.grad_consumers_below(dic):
            self.verify_divisible(dic['channels'], 16, 'channels')
        self.verify_str_in(dic['pool'], ['max', 'avg'])
        
        self.verify_img_size()

        if dic['outputsX'] <= 0:
            dic['outputsX'] = int(ceil((dic['imgSize'] - dic['start'] - dic['sizeX']) / float(dic['stride']))) + 1;
        dic['outputs'] = dic['outputsX']**2 * dic['channels']
        
        print "Initialized %s-pooling layer '%s', producing %dx%d %d-channel output" % (dic['pool'], name, dic['outputsX'], dic['outputsX'], dic['channels'])
        return dic
    
class NormLayerParser(LayerWithInputParser):
    RESPONSE_NORM = 'response'
    CONTRAST_NORM = 'contrast'
    CROSSMAP_RESPONSE_NORM = 'cross-map response'
    
    def __init__(self, norm_type):
        LayerWithInputParser.__init__(self, num_inputs=1)
        self.norm_type = norm_type
        
    def add_params(self, mcp):
        dic, name = self.dic, self.dic['name']
        dic['scale'] = mcp.safe_get_float(name, 'scale')
        dic['scale'] /= dic['size'] if self.norm_type == self.CROSSMAP_RESPONSE_NORM else dic['size']**2
        dic['pow'] = mcp.safe_get_float(name, 'pow')
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['size'] = mcp.safe_get_int(name, 'size')
        dic['blocked'] = mcp.safe_get_bool(name, 'blocked', default=False)
        
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        # Contrast normalization layer does not use its inputs
        dic['usesInputs'] = self.norm_type != self.CONTRAST_NORM
        
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        if self.norm_type == self.CROSSMAP_RESPONSE_NORM: 
            self.verify_num_range(dic['size'], 'size', 2, dic['channels'])
            if dic['channels'] % 16 != 0:
                raise LayerParsingError("Layer '%s': number of channels must be divisible by 16 when using crossMap" % name)
        else:
            self.verify_num_range(dic['size'], 'size', 1, dic['imgSize'])
        
        if self.norm_type != self.CROSSMAP_RESPONSE_NORM and dic['channels'] > 3 and dic['channels'] % 4 != 0:
            raise LayerParsingError("Layer '%s': number of channels must be smaller than 4 or divisible by 4" % name)

        self.verify_img_size()

        dic['outputs'] = dic['imgPixels'] * dic['channels']
        print "Initialized %s-normalization layer '%s', producing %dx%d %d-channel output" % (self.norm_type, name, dic['imgSize'], dic['imgSize'], dic['channels'])
        return dic

class CostParser(LayerWithInputParser):
    def __init__(self, num_inputs=-1):
        LayerWithInputParser.__init__(self, num_inputs=num_inputs)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        del dic['neuron']
        return dic

    def add_params(self, mcp):
        dic, name = self.dic, self.dic['name']
        dic['coeff'] = mcp.safe_get_float(name, 'coeff')
            
class LogregCostParser(CostParser):
    def __init__(self):
        CostParser.__init__(self, num_inputs=2)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)
        if dic['numInputs'][0] != 1: # first input must be labels
            raise LayerParsingError("Layer '%s': dimensionality of first input must be 1" % name)
        if prev_layers[dic['inputs'][1]]['type'] != 'softmax':
            raise LayerParsingError("Layer '%s': second input must be softmax layer" % name)
        if dic['numInputs'][1] != model.train_data_provider.get_num_classes():
            raise LayerParsingError("Layer '%s': softmax input '%s' must produce %d outputs, because that is the number of classes in the dataset" \
                                    % (name, prev_layers[dic['inputs'][1]]['name'], model.train_data_provider.get_num_classes()))
        
        print "Initialized logistic regression cost '%s'" % name
        return dic
    
class SumOfSquaresCostParser(CostParser):
    def __init__(self):
        CostParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)
        print "Initialized sum-of-squares cost '%s'" % name
        return dic

# All the layer parsers
layer_parsers = {'data': lambda : DataLayerParser(),
                 'fc': lambda : FCLayerParser(),
                 'fcdropo': lambda : FCDropOutLayerParser(),
                 'fcdropc': lambda : FCDropConnectLayerParser(),
                 'fcdropcf': lambda : FCDropConnectFastLayerParser(),
                 'conv': lambda : ConvLayerParser(),
                 'local': lambda : LocalUnsharedLayerParser(),
                 'softmax': lambda : SoftmaxLayerParser(),
                 'eltsum': lambda : EltwiseSumLayerParser(),
                 'eltmax': lambda : EltwiseMaxLayerParser(),
                 'neuron': lambda : NeuronLayerParser(),
                 'pool': lambda : PoolLayerParser(),
                 'rnorm': lambda : NormLayerParser(NormLayerParser.RESPONSE_NORM),
                 'cnorm': lambda : NormLayerParser(NormLayerParser.CONTRAST_NORM),
                 'cmrnorm': lambda : NormLayerParser(NormLayerParser.CROSSMAP_RESPONSE_NORM),
                 'nailbed': lambda : NailbedLayerParser(),
                 'blur': lambda : GaussianBlurLayerParser(),
                 'resize': lambda : ResizeLayerParser(),
                 'rgb2yuv': lambda : RGBToYUVLayerParser(),
                 'rgb2lab': lambda : RGBToLABLayerParser(),
                 'rscale': lambda : RandomScaleLayerParser(),
                 'cost.logreg': lambda : LogregCostParser(),
                 'cost.sum2': lambda : SumOfSquaresCostParser()}
 
# All the neuron parsers
# This isn't a name --> parser mapping as the layer parsers above because neurons don't have fixed names.
# A user may write tanh[0.5,0.25], etc.
neuron_parsers = sorted([NeuronParser('ident', 'f(x) = x', uses_acts=False, uses_inputs=False),
                         NeuronParser('logistic', 'f(x) = 1 / (1 + e^-x)', uses_acts=True, uses_inputs=False),
                         NeuronParser('abs', 'f(x) = |x|', uses_acts=False, uses_inputs=True),
                         NeuronParser('relu', 'f(x) = max(0, x)', uses_acts=True, uses_inputs=False),
                         NeuronParser('softrelu', 'f(x) = log(1 + e^x)', uses_acts=True, uses_inputs=False),
                         NeuronParser('square', 'f(x) = x^2', uses_acts=False, uses_inputs=True),
                         NeuronParser('sqrt', 'f(x) = sqrt(x)', uses_acts=True, uses_inputs=False),
                         ParamNeuronParser('tanh[a,b]', 'f(x) = a * tanh(b * x)', uses_acts=True, uses_inputs=False),
                         ParamNeuronParser('brelu[a]', 'f(x) = min(a, max(0, x))', uses_acts=True, uses_inputs=False),
                         ParamNeuronParser('linear[a,b]', 'f(x) = a * x + b', uses_acts=True, uses_inputs=False)],
                        key=lambda x:x.type)
