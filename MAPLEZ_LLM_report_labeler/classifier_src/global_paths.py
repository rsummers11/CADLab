# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# File containing the location of a few files loaded by other pytohn files in this folder


import os

h5_path = './scratch/' #Location where hdf5 files with preprocessed datasets will be saved

jpg_path = './MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/'  # Location of the files folder of the MIMIC-CXR dataset
mimic_dir = './MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/' # Location of the tables mimic-cxr-2.0.0-chexpert.csv and mimic-cxr-2.0.0-split.csv from the MIMIC-CXR-JPG dataset
nih_dataset_location = './datasets/nih_chestxray14/' # where the /images/ folder is, from the ChestXray14 dataset

# location of the CheXpert test; used only for testing on the human-labeled test set from CheXpert
# Inside this folder, there are both contents of https://github.com/rajpurkarlab/cheXpert-test-set-labels and https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c
#, so there is a groundtruth.csv file and a CheXpert folder, together with other folders and files.
chexpert_dataset_location = './datasets/chexpert_test/' 
