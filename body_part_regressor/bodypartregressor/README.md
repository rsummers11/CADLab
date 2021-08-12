# Self-supervised body part regressor (SSBR)

Developed by Ke Yan (ke.yan@nih.gov, [yanke23.com]()), Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
National Institutes of Health Clinical Center

Update 7/2020 by Daniel C. Elton:

* Converted from Python 2 to Python 3
* added nifti_inference.py, to allow for runs on NiFTI images.


If you use this code, please read and cite the following works:

* Ke Yan, Le Lu, Ronald Summers, "Unsupervised Body Part Regression via Spatially Self-ordering Convolutional Neural Networks", IEEE ISBI, 2018, https://arxiv.org/abs/1707.03891
* Ke Yan, Xiaosong Wang, Le Lu, Ling Zhang, Adam Harrison, Mohammadhad Bagheri, Ronald M. Summers, "Deep Lesion Graphs in the Wild: Relationship Learning and Organization of Significant Radiology Image Findings in a Diverse Large-scale Lesion Database", IEEE CVPR, 2018, https://arxiv.org/abs/1711.10535


## Function
Predict a **continuous score** for an axial slice in a CT volume which indicates its relative position in the body, e.g. the figure below. The actual correspondence between values and positions needs to be observed when using. See the paper.

Samples of unsupervisedly learned body-part scores:
![sample body-part scores](sample_results.png)


## Requirements
1. Caffe - use "conda install caffe-gpu" for GPU caffe or "conda install caffe" for CPU
2. easydict - install with "pip install easydict"
3. nibabel (for NiFTI inference)
4. (for training only) VGG-16 pretrained caffemodel (optional, because the
algorithm works well even if trained from scratch given enough data).
5. (for training only) Unlabeled training volumes, each volume stored in a folder
of 2D slices named by <slice_index>.png. List the names of volume folders in a
list file and put the list file's name in TRAIN_IMDB of train.sh. Specify the name
of the folder containing all volumes in DATA_DIR of config.yml. If you want to use
different data format, change data_layer.py.

## Inference with .nii.gz
Run `python nifti_inference.py filename.nii.gz`

Results will be written to "slice_scores*.txt". There are two columns - first is raw scores, 2nd is the linear regression fit.

If you need to add the -1024 HU offset, set the OFFSET variable in line 99 of nifti_inference.py to -1024 before running. 

## Inference with .png
To do inference with .png put images in bodypartregressor/test_data/, then run `python bodypartregressor/deploy.py`. The trained model is in snapshots folder.

## Installation
Minimal packaging of this repository was performed to allow for installation of nifti_inference.
To install in your system, run `python setup.py install` from this directory

Then you can use the NiFTI inference capability of the package as follows:
```python

from bodypartregressor.nifti_inference import nifti_inference

ctnib = nib.load(file)  #read CT using Nibabel
ct = ctnib.get_data().astype(np.float32)  #get CT as numpy array

slice_scores = nifti_inference(ct)  #get a list of scores
```




## Notes
* When training, see the requirements below and run train.sh.
* The provided trained model was trained on 4400 unlabeled CT volumes with various reconstruction filters,
scan ranges, and pathological conditions. Random 2D patch cropping was used when training.
It is expected to be more accurate in shoulder, chest, abdomen, and pelvis because of the
training data.
* Input soft tissue window (-175~275 HU) 8-bit images with size 128x128. If your data
are different in windowing, image size, scan range etc., it is
easy to retrain the algorithm to get a better model for your application. It is also possible to extend
the algorithm to sagittal/coronal planes, MR volumes, etc.
* The output of SSBR can be used to roughly locate slices of certain body-parts, input to other CAD
algorithms as features, detect abnormal volumes, and so on. See paper.


We utilize open source code from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).
