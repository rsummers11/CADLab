# virtualenv .
# source ../bin/activate
pip install --upgrade pip
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl  # modify this according to your CUDA
pip install torchvision
# sudo apt-get install python-tk
pip install -r requirements.txt
