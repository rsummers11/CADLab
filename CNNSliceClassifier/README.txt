Developed by Holger Roth (holger.roth@nih.gov, h.roth@ucl.ac.uk)
Acknowledgement: my-cuda-convnet2 is derived and modified from cuda-convnet2: https://code.google.com/p/cuda-convnet2
The code uses various files from the Matlab File Exchange (http://www.mathworks.com/matlabcentral/fileexchange/): niftiIO, rdir, cvsimport
The slice extraction is based on ITK (http://itk.org/)

Requirements:
    1. Download trained ConvNet from https://drive.google.com/file/d/0Byig_cLsHdU0SlVGQi1wNzlsMkk/view?usp=sharing and unpack to 'CNNSliceClassifier' root directory
	1. Linux (tested: Ubuntu 12.04 LTS)
	2. Matlab (tested: 2010a)
	3. CUDA-compatible graphics card (tested: NVIDIA TITAN Z)
	4. Same requirements as for cuda-convnet2: https://code.google.com/p/cuda-convnet2/wiki/Compiling
	5. ITK v4.6.0 (others are untested): http://sourceforge.net/projects/itk/files/itk/4.6/InsightToolkit-4.6.0.tar.gz/download
	
Run example (Windows 7, 64-bit):

1. Open Matlab
2. CD to where 'run_CNNSlices_Pipeline.m' and specify input image (*.nii and *.nii.gz tested. itkExtractImageSlices should work with all ITK-supported formats, but other than nii/img will not be plotted as a result figure in the matlab code matlab will require the readers for any specific format).

Compiling:
All libraries and executables are provided pre-compiled on github for Ubuntu 12.04 LTS. However, if your system is different, you might want to compile the code yourself:  

		1. compiling my-cuda-convnet2:
			Same as for https://code.google.com/p/cuda-convnet2/wiki/Compiling
			Modify my-cuda-convnet2/build.sh to fit your system
		2. compiling itkExtractImageSlices:
			Use the CMakeLists.txt (cmake 2.8.12 tested) in CNNSliceClassifier/itkApps
			Set CNNSliceClassifier/itkApps_build/release as build directory and compile in release mode
			
			
