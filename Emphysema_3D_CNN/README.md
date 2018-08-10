# Emphysema 3D CNN

Developed by Wes Caldwell.
Code from my 2018 Summer fellowship to score emphysema in CT scans from the National Lung Screening Trial.


## Requirements

1. CUDA 8.0
2. cuDNN 6
3. Python 2.7

## Important Files

* Preprocessing Files
	* preprocess.sh
	* segment.sh
* 3D CNN File
	* emphnet.py

## Usage

### Preprocessing

First, install the required Python libraries by running

>pip install -r requirements.txt

Then, run

>./preprocess.sh pid_list.lst NLST_ROOT_DIR TARGET_DATA_DIR
>./segment.sh pid_list.lst PHNN_ROOT_DIR TARGET_DATA_DIR

where pid_list.lst is the list of PIDs and labels for the data (I've provided the lists I used in my project in the folder final_fold for example), NLST_ROOT_DIR is the root directory of the dataset, TARGET_DATA_DIR is the root directory of where you want the preprocessed files to go, and PHNN_ROOT_DIR is the root directory of Adam Harrison's P-HNN segmenter (https://gitlab.com/adampharrison/p-hnn). If all goes well, then you're ready to train the network (or just test using the model file I've provided in model/).

### Training/Testing

As of right now, emphnet.py lacks the command line support available in the shell scripts, but the class instance provided should be sufficiently general and well-documented. The interface provides both training and testing functions with data input fed through the data_generator function.

## Acknowledgements

dcm2niix has been developed for research purposes only and is not a clinical tool
Copyright (c) 2014-2016 Chris Rorden. All rights reserved.
