# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file containing a dataset envelope that handles saving the dataset into
# an hdf5 file, in case it hasn't been saved yet, to speed up training speed

from torch.utils.data import Dataset
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import os
import dill
import signal
import hashlib
import shutil
import h5py
import pathlib
import multiprocessing
from joblib import Parallel, delayed
from dill.source import getsource
import PIL
import PIL.Image
import types
import argparse
import copy
import gc


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

dill.settings['recurse'] = True

class MyKeyboardInterruptionException(Exception):
    "Keyboard Interrupt activate signal handler"
    pass
    
def interupt_handler(signum, frame):
    raise MyKeyboardInterruptionException

class CustomError(Exception):
    pass

class StringContextManager(object):
    def __init__(self, input_string):
        self.string = input_string
    def __enter__(self):
        return self.string
    def __exit__(self, type, value, traceback):
        pass

def get_one_sample(list_index,element_index, original_dataset_):
    print(f'{element_index}-/{len(original_dataset_[0])}')
    return original_dataset_[0][element_index]

def delete_files(files):
    for file in files:
        if os.path.exists(file):
            if not os.path.isdir(file):
                os.remove(file)
            else:
                shutil.rmtree(file)

# class used to create a savable preprocessing or postprocessing function. 
class H5ProcessingFunction:
    def __init__(self):
        # make the instance be represented by the code in its __call__method
        # when saving the function
        self.__code__ = self.__call__.__code__
    
    def sourcecode(self):
        # get a representation of the function source code, for comparison purposes
        return getsource(self.__class__, builtin=True, enclosing=True, force = True)
    
    #this is the function that will be called when pre or post processing is executed
    def __call__(self, name_, assignment_value, fixed_args, joined_structures):
        # this method should be implemented by any preprocessing or postprocessing function
        raise NotImplementedError


class H5ToInt(H5ProcessingFunction):        
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        return assignment_value.astype(int)
    
# class used to create a function that is a sequence of H5ProcessingFunction
# functions. 
class H5ComposeFunction(H5ProcessingFunction):
    def __init__(self, list_functions):
        super().__init__()
        # define its own source code as the sequence of source codes of the
        # component functions
        source = ''
        for function in list_functions:
            source+=function.sourcecode()
            source+='\n'
        self.source =source
        
        # saves properties of each component function
        self.selfs = []
        for f in list_functions:
            properties = copy.deepcopy(vars(f))
            del properties["__code__"]
            self.selfs.append(properties)
        
        #converts the provided instances into True functions for saving
        self.list_functions = [types.FunctionType(f.__code__,{}) for f in list_functions]
    
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        # iterate throgh the component functions in order, providing their "self"
        # (argparse.Namespace(**self.selfs[index_function])) and other arguments
        import argparse
        for index_function,function in enumerate(self.list_functions):
            assignment_value = function(argparse.Namespace(**self.selfs[index_function]), name_, assignment_value, fixed_args, joined_structures)
        return assignment_value
    
    def sourcecode(self):
        return self.source

# function used to convert boolean array from using 8 bits per pixel to 1 bit per pixel.
# The saved count_value represents how many bits to ignore when loading the array
class PackBitArray(H5ProcessingFunction):
    def __init__(self, label_name):
        super().__init__()
        self.label_name = label_name
    
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        import numpy as np
        count_value = -int((-assignment_value.shape[1])%8)
        fixed_args['self'].create_individual_dataset_with_data(fixed_args['h5f'], f"{name_}_{self.label_name}_saved_count_/@{fixed_args['index']}", count_value)
        return np.packbits(assignment_value, axis = 1)

# function used to convert a packed boolean array that is using 1 bit per pixel to
# an usable array in python (8 bits per pixel)
# The loaded count_values represents how many bits to ignore when loading the array
class UnpackBitArray(H5ProcessingFunction):
    def __init__(self, label_name):
        super().__init__()
        self.label_name = label_name
    
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        import numpy as np
        count_values = int(fixed_args['self'].load_variable_to_memory(joined_structures['load_to_memory'], f"{name_}_{self.label_name}_saved_count_", fixed_args['index'],
            lambda: np.zeros([len(fixed_args['self'])]), 
            lambda: fixed_args['self'].get_individual_from_name(fixed_args['h5f'], f"{name_}_{self.label_name}_saved_count_/@{fixed_args['index']}")))
        if count_values==0:
            return np.unpackbits(assignment_value, axis = 1)
        else:
            return np.unpackbits(assignment_value, axis = 1, count = count_values)

# function used to convert types of arrays when saving and loading them such that
# they occupy less space on disk. It also makes loading them faster.
# Changing the range of the array is important such that the whole range of the 
# data type is used.
class change_np_type_fn(H5ProcessingFunction):
    def __init__(self, nptype, multiplier):
        super().__init__()
        self.multiplier = multiplier
        self.nptype = nptype
    
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        # this function assumes that the range will have will grow when saving
        # and shring whem loading. I multiplier is 1, there is no difference
        # between loading and saving
        if self.multiplier>=1:
            # saving
            return (self.multiplier*assignment_value).astype(self.nptype)
        else:
            #loading
            return self.multiplier*(assignment_value.astype(self.nptype))

# internal class used to store disk-savable functions that also contain their source code
# The source code is used to compare the saved functions with newly provided functions
class FunctionWithSource():
    def __init__(self, the_function):
        if the_function is None:
            self.dict_ = {'function':None, 'source':''}
        else:
            self.self = copy.deepcopy(vars(the_function))
            del self.self['__code__']
            self.dict_ = {'function':types.FunctionType(the_function.__code__,{}) , 'source':the_function.sourcecode()}

    def getsource(self):
        return str(self)

    def __call__(self, *args):
        to_return = self.dict_['function'](argparse.Namespace(**self.self), *args)
        return to_return

    def __bool__(self):
        return self.dict_['function'] is not None 

    def __str__(self):
        return self.dict_['source']

    def __repr__(self):
        return str(self.dict_['function'])

    def __eq__(self, obj):
        return str(self)==str(obj)

    def __ne__(self, obj):
        return str(self)!=str(obj)

# generic class to save a pytorch dataset to H5 files, and load them for fast use if
# they have already been saved.

# how to use this file:
# 1. extract the library
# 2. install it running `pip install .` on the root of the extracted folder
# 3. import the library using `from h5_dataset.h5_dataset import H5Dataset`
# 4. use the class with 
# `fast_dataset` = H5Dataset( path = '/scratch/username/',
#                                filename = 'some dataset filename',
#                               fn_create_dataset = lambda: some dataset)

# class arguments:
# path: location where to store the hdf5 files. Avoid using network filesystems
# since they may be overloaded by the use of this class.
# filename: a unique filename for that specific dataset. Any changes to the
# dataset or preprocessing functions should make you use a new filename, 
# or delete the old files. In other words, the class assumes that filename 
# is a unique identifier for the dataset content, uch that if the provided
# filename exists, it will directly load the h5 file and not use original_dataset.
# If no filename is provided, the class will calculate
# a unique filename for your dataset, based on the length of the dataset, first 
# case of the dataset, preprocessed functions provided and the value of individual_datasets.
# * fn_create_dataset = function used to instantiate the dataset that should be 
# stored to disk, e.g., lambda: some_dataset. It is received as a function because
# some datasets might have slow instantiation, and the class does not need to 
# instantiate the dataset when just loading it from disk.
# * n_processes - during the creation of a dataset, it represents the number of 
# parallel processes opened to load the dataset to memory before writing it to 
# disk. 0 means parallel computing is not used.
# * batch_multiplier - during the creation of a dataset, it represents how many 
# batches of cases are loaded to memory before saving them to disk 
# (total cases written to disk at a time = batch_multiplier * n_processes)

# * The following arguments accept either a single value, which will then be applied
# to all parts of your dataset, or a variable with the same format of your dataset
# case, which will then define the property for specific parts of your dataset.
# For example, if, for each case, your dataset returns a tuple list of two elements,
# one image and one image-level label, as in [image, label], if you set load_to_memory
# to [False, True], only the image-level labels will be loaded to memory. 
# The arguments postprocessing_functions and load_to_memoryare defined in the 
# construction of the dataset and saved in their own specific pickle/dill files.
# If you would like to change their values for an already
# constructed file, delete the saved pickle/dill file for that variable.
# * individual_datasets (default False) - defines if the parts of the dataset should
# be stored in individual arrays for each case, or as a single array for the 
# whole dataset. If all cases provide arrays of the same size, you should set
# this to False. For parts of the case that might have different shapes, 
# depending on the case, set it to True.
# * preprocessing_functions - Defines functions that should be used to optimize 
# the disk size of arrays when saving the data. Functions should be an instance 
# of a class that inherits, H5ProcessingFunction, call its initialization, and
# defines its procedure in the __call__ method (see for example the 
# change_np_type_fn above). To allow for the saving of the function into a file,
# all classes should be self-contained: all imports,
# variables and functions used within the class should be defined within the class.
# Use None if there is no reprocessing to be done to that part of the dataset. 
# * postprocessing_functions - Defines functions that undo the preprocessing functions
# when the dataset is being loaded
# * load_to_memory (default False) - defines if parts of your dataset will be 
# loaded to memory when loading a hdf5 file. Loading to memory is done when each
# case of the dataset is first read from disk, distributing the computation and disk
# use of the loading of the dataset throughout the first epoch.

# * Allowed types of data for the stored dataset in current version: np.ndarray,
#  int, bytes, float, bool, PIL.Image, str, None, list, tuple, dict, nested lists,
# nested dicts, nested tuples. For type string, if strings can have different lengths, 
# set individual_datasets to True for that part of the dataset.
class H5Dataset(Dataset):
    def __init__(self, path = '.' , filename = None, fn_create_dataset = None, individual_datasets = None, preprocessing_functions = None, postprocessing_functions = None, n_processes = 0, batch_multiplier = 4, load_to_memory = None):
        super().__init__()
        original_dataset = None
        first_element = None
        self.path = path
        self.memory_variables = {}
        self.loaded_to_memory = {}
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
        
        if filename is None:
            if first_element is None:
                if original_dataset is None:
                    original_dataset = fn_create_dataset()
                first_element = original_dataset[0]
            individual_datasets = H5Dataset.treat_boolean_tree_inputs(individual_datasets, first_element)
            preprocessing_functions = H5Dataset.treat_fn_inputs(preprocessing_functions, first_element)
            # if filename is not provided, try to get a hash key for the dataset to characterize its content
            # only uses the length, first element,preprocessing_functions and individual_datasets
            # of the dataset to build the hash, so this
            # class may confuse two datasets that have only other elements changing
            def hash_example(name_, structure, fixed_args):
                element = np.array(structure['structure'])
                element.flags.writeable = False
                fixed_args['sha1'].update(element.data)
                fixed_args['sha1'].update(name_.encode())
                fixed_args['sha1'].update(str(structure['individual_dataset']).encode())
                if structure['preprocessing_function'] is not None:
                    fixed_args['sha1'].update(getsource(structure['preprocessing_function']).encode())
            sha1 = hashlib.sha1()
            H5Dataset.apply_function_to_nested_iterators(
                H5Dataset.join_structures(
                    {'structure':first_element,
                    'individual_dataset':individual_datasets,
                    'preprocessing_function':preprocessing_functions}), 
                {'sha1': sha1, 'contains_joined_structure':True}, hash_example)
            sha1.update(str(len(original_dataset)).encode())
            filename = str(sha1.hexdigest())
        filename = filename + self.get_extension()
        self.filepath_h5 = path + '/' + filename
        structure_file = path + '/' + filename + '_structure.pkl'
        individual_datasets_file = path + '/' + filename + '_individual_datasets.pkl'
        preprocessing_functions_file = path + '/' + filename + '_preprocessing_functions.dill'
        postprocessing_functions_file = path + '/' + filename + '_postprocessing_functions.dill'
        load_to_memory_file = path + '/' + filename + '_load_to_memory.pkl'
        length_file = path + '/' + filename + '_length.pkl'
        
        
        # creating a representation of the format of each case of the dataset
        if not os.path.exists(structure_file):
            if first_element is None:
                if original_dataset is None:
                    original_dataset = fn_create_dataset()
                first_element = original_dataset[0]
            structure = self.create_structure(first_element)
            with open(structure_file, 'wb') as output:
                pickle.dump(structure, output, pickle.HIGHEST_PROTOCOL)
        with open(structure_file, 'rb') as input:
            structure = pickle.load(input)
        
        individual_datasets = H5Dataset.get_savable_input(individual_datasets,
            individual_datasets_file, H5Dataset.treat_boolean_tree_inputs, structure)
        load_to_memory = H5Dataset.get_savable_input(load_to_memory,
            load_to_memory_file, H5Dataset.treat_boolean_tree_inputs, structure)
        preprocessing_functions = H5Dataset.get_savable_input(preprocessing_functions,
            preprocessing_functions_file, H5Dataset.treat_fn_inputs, structure)
        postprocessing_functions = H5Dataset.get_savable_input(postprocessing_functions,
            postprocessing_functions_file, H5Dataset.treat_fn_inputs, structure)
        
        if not os.path.exists(length_file):
            if original_dataset is None:
                original_dataset = fn_create_dataset()
            with open(length_file, 'wb') as output:
                pickle.dump(len(original_dataset), output, pickle.HIGHEST_PROTOCOL)
        
        with open(length_file, 'rb') as input:
            self.len_ = pickle.load(input)
        
        self.joint_structures = H5Dataset.join_structures(
                {'structure':structure,
                'individual_dataset':individual_datasets, 
                'postprocessing_function':postprocessing_functions, 
                'load_to_memory':load_to_memory})
        
        if not os.path.exists(self.filepath_h5):
            original_sigint_handler = signal.getsignal(signal.SIGINT)
            # catch ctrl+C commands so that files are still deleted if ctrl+C
            # is repeatedly pressed 
            signal.signal(signal.SIGINT, interupt_handler)
            try:
                if original_dataset is None:
                    original_dataset = fn_create_dataset()
                if first_element is None:
                    first_element = original_dataset[0]
                with self.get_file_write() as h5f:
                    self.create_h5_datasets(H5Dataset.join_structures(
                            {'structure':first_element, 
                            'individual_dataset':individual_datasets, 
                            'preprocessing_function':preprocessing_functions}) , 
                        h5f, self.len_)
                    
                    if n_processes == 0:
                        for index in range(self.len_):
                            element = original_dataset[index]
                            print(f'{index}/{self.len_}')
                            self.pack_h5(H5Dataset.join_structures(
                                    {'structure':element,
                                    'individual_dataset':individual_datasets, 
                                    'preprocessing_function':preprocessing_functions}),
                                 index, h5f)
                            del element
                            gc.collect()
                    else:
                        
                        indices_iterations = np.arange(len(original_dataset))
                        manager = multiprocessing.Manager()
                        
                        for indices_batch in [indices_iterations[i:i + n_processes*batch_multiplier] for i in range(0, len(indices_iterations), n_processes*batch_multiplier)]:
                            # numpys = manager.list([original_dataset])
                            elements = Parallel(n_jobs=n_processes, batch_size = 1, require='sharedmem')(delayed(get_one_sample)(list_index,element_index, [original_dataset]) for list_index, element_index in enumerate(indices_batch))
                            for element_index, index in enumerate(indices_batch): 
                                self.pack_h5(H5Dataset.join_structures(
                                        {'structure':elements[element_index],
                                        'individual_dataset':individual_datasets, 
                                        'preprocessing_function':preprocessing_functions}),
                                     index, h5f)
                            del elements
                            # numpys[:] = []
                            # del numpys
                            gc.collect()
            except Exception as err:
                # if there is an error in the middle of writing, delete the generated files
                # to not have corrupted files
                delete_files([self.filepath_h5, structure_file, individual_datasets_file,
                    load_to_memory_file, preprocessing_functions_file, postprocessing_functions_file, length_file])
                raise type(err)(f'Error while writing hash {filename}. Deleting files {self.filepath_h5} and related auxiliary files.').with_traceback(err.__traceback__) 
            # return ctrl+C handling to python
            signal.signal(signal.SIGINT, original_sigint_handler)
    
    @staticmethod
    def get_savable_input(savable_input, input_file, treat_input_fn, structure, fn_input_to_str = None):
        test_fn = False
        if fn_input_to_str is None:
            fn_input_to_str = lambda x: x
        input_is_none = savable_input is None
        treated_input = treat_input_fn(savable_input, structure)
        if not os.path.exists(input_file):
            with open(input_file, 'wb') as output:
                dill.dump(treated_input, output, dill.HIGHEST_PROTOCOL, recurse = True, byref=False)
        with open(input_file, 'rb') as input:
            savable_input = dill.load(input)
        if not input_is_none:
            def compare_all_elements(name_, joined_structure, fixed_args):
                if joined_structure['saved_value'] is not None or joined_structure['input_value'] is not None:
                    if (joined_structure['saved_value'] is not None and joined_structure['input_value'] is not None) \
                      and joined_structure['saved_value']!=joined_structure['input_value']:
                        raise CustomError(f"The element {name_} had different values in the file \
                        {input_file} and its input: \n{joined_structure['saved_value']}\n \
                         and\n{joined_structure['input_value']}.")
            H5Dataset.apply_function_to_nested_iterators(H5Dataset.join_structures(
                    {'structure':structure, 
                    'saved_value': savable_input, 
                    'input_value':treated_input}),
                 {'contains_joined_structure':True}, compare_all_elements)
        return savable_input
    
    @staticmethod
    def treat_fn_inputs(fn_inputs, structure):
        if fn_inputs is None or callable(fn_inputs):
            fn_inputs = H5Dataset.get_structure_with_fixed_value(structure, fn_inputs)
        def function_(name_, value, fixed_args):
            return FunctionWithSource(value)
        fn_inputs = H5Dataset.apply_function_to_nested_iterators(fn_inputs, {'contains_joined_structure':False},function_)
        return fn_inputs
    
    @staticmethod
    def treat_boolean_tree_inputs(boolean_tree_input, structure):
        if boolean_tree_input is None:
            boolean_tree_input = False
        if isinstance(boolean_tree_input, (bool)): 
            boolean_tree_input = H5Dataset.get_structure_with_fixed_value(structure, boolean_tree_input)
        return boolean_tree_input
    
    def get_extension(self):
        return '.h5'
    
    def get_individual_with_fn(self, index, fn_get):
        with self.get_file_read() as file:
            to_return = self.generic_open_individual_h5(self.joint_structures, index, file, fn_get)
        return to_return
    
    def __getitem__(self, index):
        with self.get_file_read() as file:
            to_return = self.unpack_h5(self.joint_structures, index, file)
        return to_return
    
    def __len__(self):
        return self.len_
    
    @staticmethod
    def get_structure_with_fixed_value(one_case, fixed_value):
        def function_(name_, value, fixed_args):
            return fixed_args['fixed_value']
        return H5Dataset.apply_function_to_nested_iterators(one_case, {'fixed_value':fixed_value, 'contains_joined_structure':False}, function_)
    
    @staticmethod
    def join_structures(structures):
        structure = structures['structure']
        if structure is None or callable(structure) or isinstance(structure, (FunctionWithSource, np.ndarray, int, bytes, float, bool, PIL.Image.Image, str,type(type(None)))):
            return structures
        elif isinstance(structure, list) or isinstance(structure, tuple):
            return [H5Dataset.join_structures({key_structure:structures[key_structure][index] for key_structure in structures}) for index in range(len(structure))]
        elif isinstance(structure, dict):
            return {key: H5Dataset.join_structures({key_structure: structures[key_structure][key] for key_structure in structures}) for key in structure}
        else:
            raise ValueError('Unsuported type: ' + str(type(structure)))
    
    def create_structure(self, one_case):
        def function_(name_, value, fixed_args):
            return type(value)
        return H5Dataset.apply_function_to_nested_iterators(one_case, {'contains_joined_structure':False, 'self':self}, function_)
    
    def create_shared_dataset(self, file, name_, shape, type):
        if type is None:
            file.create_dataset(name_, shape = shape)
        else:
            file.create_dataset(name_, shape = shape, dtype = type)
    
    def create_h5_datasets(self, one_case, h5f, n_images):
        def function_(name_, joined_structures, fixed_args):
            if joined_structures['individual_dataset']:
                pass
            else:
                assignment_value = joined_structures['structure']
                if joined_structures['preprocessing_function']:
                    assignment_value = joined_structures['preprocessing_function'](name_, assignment_value, fixed_args, joined_structures)
                if type(assignment_value) == type('a'):
                    self.create_shared_dataset(fixed_args['h5f'], name_, [fixed_args['n_images']] + [len(assignment_value)], 'S'+str(len(assignment_value)))
                elif type(assignment_value) == type(np.array([0])):
                    self.create_shared_dataset(fixed_args['h5f'], name_, [fixed_args['n_images']] + list(np.array(assignment_value).shape), assignment_value.dtype)
                else:
                    self.create_shared_dataset(fixed_args['h5f'], name_, [fixed_args['n_images']] + list(np.array(assignment_value).shape), None)
            return None
        return H5Dataset.apply_function_to_nested_iterators(one_case, {'n_images':n_images, 'h5f': h5f, 'contains_joined_structure':True, 'self': self}, function_)
    
    def write_case_on_shared_dataset(self, file, name_, index, assignment_value):
        file[name_][index,...] = assignment_value
    
    def pack_h5(self, structure, index, h5f):
        def function_(name_, joined_structures, fixed_args):
            assignment_value = joined_structures['structure']
            if joined_structures['preprocessing_function']:
                assignment_value = joined_structures['preprocessing_function'](name_, assignment_value, fixed_args, joined_structures)
            if joined_structures['individual_dataset']:
                self.create_individual_dataset_with_data(fixed_args['h5f'], name_+f"/@{fixed_args['index']}", assignment_value)
                return None
            if type(assignment_value) == type('a'):
                assignment_value = np.array(assignment_value).astype('S'+str(len(assignment_value)))
            self.write_case_on_shared_dataset(fixed_args['h5f'], name_, fixed_args['index'], assignment_value)
            return None
        return H5Dataset.apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'contains_joined_structure':True, 'self':self}, function_)
    
    def generic_open_individual_h5(self, structure, index, h5f, fn_get):
        def function_(name_, joined_structures, fixed_args):
            assert(joined_structures['individual_dataset'])
            return fn_get(self.get_individual_from_name_without_loading(fixed_args['h5f'], name_+f"/@{fixed_args['index']}", False))
        return H5Dataset.apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'contains_joined_structure':True, 'self': self},function_)

    def shape_h5(self, structure, index, h5f):
        def function_(name_, joined_structures, fixed_args):
            if joined_structures['individual_dataset']:
                return self.get_shape_from_name(fixed_args['h5f'], name_+f"/@{fixed_args['index']}")
            else:
                return_value = self.get_shape_from_name(fixed_args['h5f'], name_)
            return return_value
        return H5Dataset.apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'contains_joined_structure':True, 'self': self},function_)
    
    def load_variable_to_memory(self, load_to_memory, name_, index_in_dataset, fn_generate_first_time, fn_get_from_h5):
        if load_to_memory:
            if not name_ in self.memory_variables:
                self.memory_variables[name_] = fn_generate_first_time()
                self.loaded_to_memory[name_] = [0] * len(self)
            if not self.loaded_to_memory[name_][index_in_dataset]:
                self.memory_variables[name_][index_in_dataset] = fn_get_from_h5()
                self.loaded_to_memory[name_][index_in_dataset] = 1
            return_value = self.memory_variables[name_][index_in_dataset]
        else:
            return_value = fn_get_from_h5()
        return return_value
    
    def unpack_h5(self, structure, index, h5f):
        def function_(name_, joined_structures, fixed_args):
            if joined_structures['individual_dataset']:
                return_value = self.load_variable_to_memory(joined_structures['load_to_memory'], name_, fixed_args['index'],
                    lambda: [None] * len(self), 
                    lambda: self.get_individual_from_name(fixed_args['h5f'], name_+f"/@{fixed_args['index']}"))
            else:
                return_value = self.load_variable_to_memory(joined_structures['load_to_memory'], name_, fixed_args['index'],
                    lambda: np.zeros(self.get_shape_from_name(fixed_args['h5f'], name_)),
                    lambda: self.get_from_name_index(fixed_args['h5f'], name_, fixed_args['index']))
            if joined_structures['structure'] == type(None):
                return None
            if joined_structures['structure'] == type('a') and type(return_value)==np.bytes_ and not joined_structures['individual_dataset']:
                return_value = return_value.decode('utf-8')
            if joined_structures['postprocessing_function']:
                return_value = joined_structures['postprocessing_function'](name_, return_value, fixed_args, joined_structures)
            return return_value
        return H5Dataset.apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'contains_joined_structure':True, 'self': self},function_)

    #auxiliary function to iterate and apply functions to all elements of a variable composed
    # of nested variable of these types: list, tuple, dict
    # leafs have to be of kind: np.ndarray, int, float, bool, PIL.Image.Image
    @staticmethod
    def apply_function_to_nested_iterators(structure, fixed_args, function_, name_ = "root"):
        if fixed_args['contains_joined_structure'] and isinstance(structure, dict) and 'structure' in structure:
            value_to_test = structure['structure']
        else:
            value_to_test = structure
        if value_to_test is None or callable(value_to_test) or isinstance(value_to_test, (FunctionWithSource, np.ndarray, int, bytes, float, bool, PIL.Image.Image, str,type(type(None)))):
            return function_(name_, structure, fixed_args)
        elif isinstance(structure, list) or isinstance(structure, tuple):
            return [H5Dataset.apply_function_to_nested_iterators(item, fixed_args, function_, name_ = name_ + "/" + '_index_' + str(index)) for index, item in enumerate(structure)]
        elif isinstance(structure, dict):
            return {key: H5Dataset.apply_function_to_nested_iterators(item, fixed_args, function_, name_ = f'{name_}/{key}') for key, item in structure.items()}
        else:
            print(structure)
            raise ValueError('Unsuported type: ' + str(type(structure['structure'])))
    
    def get_from_name_index(self, file, name_, index):
        return file[name_][index]
    
    def get_shape_from_name(self, file, name_):
        return file[name_].shape
    
    def get_individual_from_name_without_loading(self, file, name_):
        return file[name_]
    
    def get_individual_from_name(self, file, name_):
        return file[name_][()]
    
    def create_individual_dataset_with_data(self, file, name_, value):
        if value is None:
            file.create_dataset(name_, dtype="f") 
        else:
            file.create_dataset(name_, data = value)
    
    def get_file_read(self):
        return h5py.File(self.filepath_h5, 'r', swmr = True, rdcc_nbytes = 0)
    
    def get_file_write(self):
        return h5py.File(self.filepath_h5, 'w')

# Using a more modern format than HDF5
class ZarrDataset(H5Dataset):
    def get_extension(self):
        return '.zarr'
    
    def get_file_write(self):
        return zarr.open(self.filepath_h5, 'w')
    
    def get_file_read(self):
        return zarr.open(self.filepath_h5, 'r')

# not finished. Do not use.
class MMapDataset(H5Dataset):
    def get_individual_from_name(self, file, name_):
        # TODO: if string?
        return np.memmap(f"{file}/{name_}.arr", mode="r")
    
    def create_individual_dataset_with_data(self, file, name_, value):
        pathlib.Path(os.path.dirname(os.path.abspath(f"{file}/{name_}.arr"))).mkdir(parents=True, exist_ok=True) 
        # TODO: if string?
        array = np.memmap(f"{file}/{name_}.arr", mode="w+",
                      dtype=value.dtype, shape=value.shape)
        array = value
    
    def get_file_read(self):
        return StringContextManager(self.filepath_h5)
        
    def get_extension(self):
        return '.arr'
    
    def get_file_write(self):
        return StringContextManager(self.filepath_h5)
    
    def write_case_on_shared_dataset(self, file, name_, index, assignment_value):
        array = np.memmap(f"{file}/{name_}.arr", mode="w+")
        array[index] = assignment_value
    
    def create_shared_dataset(self, file, name_, shape, type):
        pathlib.Path(os.path.dirname(os.path.abspath(f"{file}/{name_}.arr"))).mkdir(parents=True, exist_ok=True) 
        if type is None:
            array = np.memmap(f"{file}/{name_}.arr", mode="w+", shape=tuple(shape))
        else:
            array = np.memmap(f"{file}/{name_}.arr", mode="w+",
                      dtype=type, shape=tuple(shape))
    
    def get_from_name_index(self, file, name_, index):
        array = np.memmap(f"{file}/{name_}.arr", mode="r")
        return array[index]
    
    def get_shape_from_name(self, file, name_):
        raise NotImplementedError
    
    def get_individual_from_name_without_loading(self, file, name_):
        raise NotImplementedError

# Class used to save images as png to calculate the impact of using HDF5 datasets.
# individual_dataset should be set to True for this dataset. 
class PNGDataset(H5Dataset):
    def get_individual_from_name(self, file, name_):
        if os.path.exists(f"{file}/{name_}_0.png"):
            i = 0
            values = []
            while os.path.exists(f"{file}/{name_}_{i}.png"):
                value = np.array(PIL.Image.open(f"{file}/{name_}_{i}.png", mode='r'))
                values.append(value)
                i = i + 1
            return np.stack(values)
        elif os.path.exists(f"{file}/{name_}.csv"):
            return np.loadtxt(f"{file}/{name_}.csv", delimiter=',')
        elif os.path.exists(f"{file}/{name_}.onevalue"):
            with open(f"{file}/{name_}.onevalue", "r") as file:
                value = file.readlines()[0]
            return float(value)
        elif os.path.exists(f"{file}/{name_}.shape"):
            return np.zeros(np.loadtxt(f"{file}/{name_}.shape", delimiter=',').astype(np.int32))
        elif os.path.exists(f"{file}/{name_}.txt"):
            with open(f"{file}/{name_}.txt", "w") as file:
                value = file.readlines()[0].strip()
            return value
    
    def create_individual_dataset_with_data(self, file, name_, value):
        pathlib.Path(os.path.dirname(os.path.abspath(f"{file}/{name_}.png"))).mkdir(parents=True, exist_ok=True) 
        if type(value)==type('a'):
            with open(f"{file}/{name_}.txt", "w") as file:
                file.write(value)
        elif isinstance(value,float):
            with open(f"{file}/{name_}.onevalue", "w") as file:
                file.write(str(value))
        else:
            if len(value.shape)==3:
                if value.shape[0]==0:
                    np.savetxt(f"{file}/{name_}.shape", value.shape, delimiter=',')
                for i in range(value.shape[0]):
                    im = PIL.Image.fromarray(value[i])
                    im.save(f"{file}/{name_}_{i}.png")
            else:
                if len(value.shape)==0:
                    with open(f"{file}/{name_}.onevalue", "w") as file:
                        file.write(str(value))
                else:
                    np.savetxt(f"{file}/{name_}.csv", value, delimiter=',')
    
    def get_file_read(self):
        return StringContextManager(self.filepath_h5)
        
    def get_extension(self):
        return '.png'
    
    def get_file_write(self):
        return StringContextManager(self.filepath_h5)
    
    def write_case_on_shared_dataset(self, file, name_, index, assignment_value):
        raise CustomError(f"All elements from a PNGDataset, should be stored in individual datasets.")
    
    def create_shared_dataset(self, file, name_, shape, type):
        raise CustomError(f"All elements from a PNGDataset, should be stored in individual datasets.")
    
    def get_from_name_index(self, file, name_, index):
        raise CustomError(f"All elements from a PNGDataset, should be stored in individual datasets.")
    
    def get_shape_from_name(self, file, name_):
        raise NotImplementedError
    
    def get_individual_from_name_without_loading(self, file, name_):
        raise NotImplementedError
