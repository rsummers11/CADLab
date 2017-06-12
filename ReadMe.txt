Developed by Jiamin Liu (liujiamin@cc.nih.gov)
https://github.com/rsummers11/CADLab

Please cite our papers if you end up using this code:

Liu, J., Wang, D., Wei, Z., Lu, L., Kim, L., Turkbey, E., & Summers, R. M. (2016, April). 
Colitis detection on computed tomography using regional convolutional neural networks.
In Biomedical Imaging (ISBI), 2016 IEEE 13th International Symposium on (pp. 863-866). IEEE.

Liu, J., Wang, D., Lu, L., Wei, Z., Kim, L., Turkbey, E., Sahiner, B., Petrick, N. & Summers, R. M.
(2017, May), Detection and diagnosis of colitis on computed tomography using deep convolutional neural
networks, Medical Physics, Accepted. 

Code includes modifications of open-source packages:

1. https://github.com/ShaoqingRen/faster_rcnn
2. Various submission of the Matlab file exchange
(https://www.mathworks.com/matlabcentral/fileexchange/29344-read-medical-data-3d?focused=5186757&tab=function)

Requirements:
1. Matlab
2. CUDA-compatible graphics card
3. CUDA toolkit
	
Execution:

1. Download and install Faster RCNN following the instructions:
https://github.com/ShaoqingRen/faster_rcnn
2. Run dicom_to_jpgs.m to convert dicoms to jpgs or download example jpgs.
3. Down load trained colitis detection models.
4. Run script_faster_rcnn_run_colitis_detection.m.
