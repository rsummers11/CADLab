#!/bin/sh

############################################################################
###################### PARAMETERS THAT CAN BE CHANGED ######################
############################################################################
# Array that contains the input images to create the atlas
export IMG_INPUT=(`ls /path/to/all/your/images_*.nii`)
export IMG_INPUT_MASK= # leave empty to not use floating masks

# template image to use to initialise the atlas creation
export TEMPLATE=`ls ${IMG_INPUT[0]}` 
 export TEMPLATE_MASK= # leave empty to not use a reference mask

# folder where the result images will be saved
export RES_FOLDER="/path/to/a/folder/to/store/all/result" 

# argument to use for the affine (reg_aladin)
export AFFINE_args=""
# argument to use for the non-rigid registration (reg_f3d)
export NRR_args=""

# number of affine loop to perform
export AFF_IT_NUM=10

# number of non-rigid loop to perform
export NRR_IT_NUM=10

# grid engine arguments
export QSUB_CMD="qsub -l h_rt=05:00:00 -l tmem=3.5G -l h_vmem=3.5G -l vf=3.5G -l s_stack=10240  -j y -S /bin/csh -b y -cwd -V"
############################################################################