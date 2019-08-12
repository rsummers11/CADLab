#!/bin/bash
# set up a virtual environment to avoid library conflict
virtualenv venv --python=python3.6
source venv/bin/activate

# install pytorch 1.1. You may need to go to https://pytorch.org/ to find the best way to install on your machine (OS, CUDA etc.)
pip install torch torchvision

# necessary python libraries
pip install -r requirements.txt

sudo apt install python3-tk

#git clone https://github.com/cocodataset/cocoapi.git
#cd cocoapi/PythonAPI
#python setup.py build_ext install

python setup.py build develop
