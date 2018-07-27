## Noraml and 90-degree Rotated Chest X-ray Classification/Split)

*Developed by Yuxing Tang (yuxing.tang@nih.gov), Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
National Institutes of Health Clinical Center*

This software provides a trained model to seperate the frontal view chest x-ray (CXR) into two categories: 
normal vertical and anti-clock wise 90-degree rotated. 

For example, a large number of CXRs in the PLCO dataset (https://biometry.nci.nih.gov/cdas/plco/) are left (anti-clock wise) 90-degree rotated, however, 
no meta data is available with the CXR image describing this. Here I provide a trained CNN model (ResNet18) to automatic seperate normal view CXRs and 
rotated ones.

### Prerequistites
- Linux or OSX
- NVIDIA GPU
- Python 2.7
- PyTorch v0.3 or later
- Numpy

*Usage*:
