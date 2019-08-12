#!/bin/bash
# set up a virtual environment to avoid library conflict
virtualenv venv --python=python3.6
source venv/bin/activate

echo "installing pytorch 1.1. You may need to go to https://pytorch.org/ to find the best way to install on your machine (OS, CUDA etc.)"
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu100/stable
pip install torchvision==0.3.0
pip install --upgrade cython

# necessary python libraries
pip install -r requirements.txt

sudo apt install python3-tk

python setup.py build develop

rm -rf build