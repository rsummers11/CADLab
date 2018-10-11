Developed by Kevin Cherry and Holger Roth (holger.roth@nih.gov, www.holgerroth.com)

Please cite our papers if you end up using this code:

[1] HR Roth, L Lu, J Liu, J Yao, A Seff, C Kevin, L Kim, RM Summers: Improving Computer-aided Detection using Convolutional Neural Networks and Random View Aggregation. Medical Imaging, IEEE Transactions on. Year: 2015, Volume: PP, Issue: 99, Pages: 1 - 1, DOI: 10.1109/TMI.2015.2482920

[2] KM Cherry ; S Wang ; EB Turkbey ; RM Summers: Abdominal lymphadenopathy detection using random forest. Proc. SPIE 9035, Medical Imaging 2014: Computer-Aided Diagnosis, 90351G, doi:10.1117/12.2043837

Code includes open-source packages:
1. http://cs.nyu.edu/~wanli/dropc/ based on https://code.google.com/p/cuda-convnet/
2. https://code.google.com/p/randomforest-matlab/
3. NiftyReg http://sourceforge.net/projects/niftyreg

Requirements:
	1. Matlab
	2. CUDA-compatible graphics card
	3. Install CUDA 4.2 64-bit (https://developer.nvidia.com/cuda-toolkit-42-archive), CUDA 7.0 works as well.
	4. Python 2.7 64-bit with Numpy and Matplotlib!
		Recommended to use Anaconda python distribution: http://continuum.io/downloads
		Make sure it's added to path (and remove any conflicting python versions from path)
	5. Optional: MITK for viewing result (http://www.mitk.org/Download)
	
Run example (Windows or Linux):

1. Download example data from https://wiki.cancerimagingarchive.net/display/Public/CT+Lymph+Nodes
2. Open Matlab
3. CD to where 'run_LymphNodeRFCNNPipeline.m' is located and run it (it will automatically add necessary sub-folders to Matlab path)
4. Select ".\data\CTimage" as input folder
5. Select a suitable output folder

Compile yourself (should work on platforms other than Windows 7, 64-bit and Visual Studio 10, but links have to be updated. Ubuntu 14.04 and CUDA 7.0 seems to work as well):

	Requirements:
		1. ITK v4.5.0 (others are untested) http://sourceforge.net/projects/itk/files/itk/4.5/InsightToolkit-4.5.0.zip/download
		2. CUDA 4.2 (others are untested) https://developer.nvidia.com/cuda-toolkit-42-archive
		3. BOOST 1.55 (others are untested): http://sourceforge.net/projects/boost/files/boost-binaries/1.55.0/boost_1_55_0-msvc-10.0-64.exe/download
		
	1. Locate ".\pyconvnet\CMakeLists.txt", run CMake and produce Visual Studio or makefile (untested)
	2. Compile pyconvnet
	3. compiling PACKAGE should include all binaries in a zip file, which can be installed in Matlab using 'install_LymphNodeRFCNNPipeline.m', called by 'run_LymphNodeRFCNNPipeline.m')
	4. NiftyReg http://sourceforge.net/projects/niftyreg/files/nifty_reg-1.3.9/, Pre-build version with VS2010 is provided. Otherwise rebuild the code in 'niftyreg-git_prebuild_vs10_x64r' and modify m-scripts accordingly. This NiftyReg version from git:
	$ git log
	commit 390df2baaf809a625ed5afe0dbc81ca6a3f7c647
	Author: Marc Modat <m.modat@ucl.ac.uk>
	Date:   Fri Nov 14 13:06:52 2014 +0000
    Modified the bending energy computation so that it does not over-regularise


