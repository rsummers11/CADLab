#!/bin/sh

# Fill in these environment variables.
# Make sure you're using CUDA 4.0 to ensure compatibility.
# Only use Fermi-generation cards. Older cards won't work.

# CUDA toolkit installation directory.
export CUDA_INSTALL_PATH=/usr/local/pkg/cuda/current/cuda

# CUDA SDK installation directory.
#export CUDA_SDK_PATH=/home/wanli/NVIDIA_GPU_Computing_SDK
export CUDA_SDK_PATH=/home/snwiz/data/imagenet12/code/NVIDIA_GPU_Computing_SDK 

# Python include directory. This should contain the file Python.h, among others.
export PYTHON_INCLUDE_PATH=/usr/include/python2.6
#export PYTHON_INCLUDE_PATH=~/include/python2.6

# Numpy include directory. This should contain the file arrayobject.h, among others.
export NUMPY_INCLUDE_PATH=/usr/lib64/python2.6/site-packages/numpy/core/include/numpy
#export NUMPY_INCLUDE_PATH=~/lib/python2.6/site-packages/numpy/core/include/numpy

# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=/usr/lib64/atlas

make $*

