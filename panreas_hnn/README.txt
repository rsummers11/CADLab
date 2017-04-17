Developed by Holger Roth (h.roth@ucl.ac.uk, www.holgerroth.com)
Contact: Le Lu (le.lu@nih.gov), Ronald Summers (rms@nih.gov)

The content of this code is partially covered by US Patent Applications of 62/345,606# and 62/450,681#

Please cite our papers if you end up using this code:

@article{roth2017spatial,
  title={Spatial Aggregation of Holistically-Nested Convolutional Neural Networks for Automated Pancreas Localization and Segmentation},
  author={Roth, Holger R and Lu, Le and Lay, Nathan and Harrison, Adam P and Farag, Amal and Sohn, Andrew and Summers, Ronald M},
  journal={arXiv preprint arXiv:1702.00045},
  year={2017}
}

@inproceedings{roth2016spatial,
  title={Spatial aggregation of holistically-nested networks for automated pancreas segmentation},
  author={Roth, Holger R and Lu, Le and Farag, Amal and Sohn, Andrew and Summers, Ronald M},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={451--459},
  year={2016},
  organization={Springer International Publishing}
}

Code includes modifications of open-source packages:
1. https://github.com/s9xie/hed (modification: balancing weight is computed on entire training dataset)
2. various submission of the Matlab file exchange (https://www.mathworks.com/matlabcentral/fileexchange/)
3. Modification of ITK examples files (https://itk.org/)

Requirements:
	1. Matlab
	2. CUDA-compatible graphics card
	3. CUDA toolkit
	4. Python 2.7 (3 not tested)
	6. ITK (https://itk.org/)
	5. Optional: MITK for viewing result (http://www.mitk.org/Download)
	
Execution:

1. compile hed-globalweight following the Caffe instructions: http://caffe.berkeleyvision.org/installation.html#prequequisites
compile/install ITK (with Module_ITKReview ON) and compile nihApps using cmake. Choose 'nihApps-release' as output folder. 
This was tested with Ubuntu 16.04, CUDA 8.0, Python 2.7, ITK v4.10.0, and Matlab 2015b.

2. Download the dicom image archive from https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT

3. Run run_pancreas_hnn.m 
    CAFFE_LD_LIBRARY_PATH needs to be updated to show location of libcaffe.so and cuda libraries
    input file paths need adjustment to show inupt dicomdir or nifti image file)

The results are saved as *meanmaxAxCoSa.nii.gz in the stage1/stage2 subfolders of the output directory. All resulting images should overlay correctly when using a viewer that respects image offset and orientation (e.g. MitkWorkbench).

