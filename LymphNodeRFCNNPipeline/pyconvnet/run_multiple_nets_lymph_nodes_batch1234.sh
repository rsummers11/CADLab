#!/bin/bash

#### postMICCAI2014 ####
## MEDIASTINAL
###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced_and_cross_valid'
###OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_and_cross_valid'

## ABDOMEN
#DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced_and_cross_valid'
#OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_and_cross_valid'

## MED AND ABDOMEN TOGETHER
###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced_AND_cross_valid'
###OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_AND_cross_valid'

## RF CAD v2
###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST_BATCHES\train_AxCoSa_balanced_6batches'
###OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST_RESULTS\train_AxCoSa_6batches_balanced'

## Bone lesions
###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_NEAREST_BATCHES\train_AxCoSa_balanced_6batches'
###OUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_NEAREST_RESULTS\train_AxCoSa_balanced_6batches'
###NUMBER_BATCHES=6

###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_BATCHES\train_AxCoSa_balanced_6batches'
###OUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_RESULTS\train_AxCoSa_balanced_6batches'
###NUMBER_BATCHES=6

###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered_BATCHES\train_AxRGB_balanced_6batches'
###OUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered_RESULTS\train_AxRGB_balanced_6batches'
###NUMBER_BATCHES=6

#DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\train_AX_balanced_Gray_6batches'
#OUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\train_AX_balanced_Gray_6batches_EarlyStop'
#NUMBER_BATCHES=6

# PostMICCAI full 3-folded cross-validation
## ABDOMEN
DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch12_AxCoSa_balanced'
OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch12_AxCoSa_balanced'
###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch23_AxCoSa_balanced'
###OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch23_AxCoSa_balanced'
###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch13_AxCoSa_balanced'
###OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch13_AxCoSa_balanced'
NUMBER_BATCHES=5

# PostMICCAI full 3-folded cross-validation
## JOINT ABDOMEN AND MEDIASTINUM
###DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\MedAbd_batch12_AxCoSa_balanced_5batches'
###OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\MedAbd_batch12_AxCoSa_balanced_5batches'
#DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\MedAbd_batch23_AxCoSa_balanced_5batches'
#OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\MedAbd_batch23_AxCoSa_balanced_5batches'
#DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\MedAbd_batch13_AxCoSa_balanced_5batches'
#OUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\MedAbd_batch13_AxCoSa_balanced_5batches'
#NUMBER_BATCHES=5
###IMG_SIZE=32
###IMG_CHANNELS=3
###TYPE='fc512-11pct'

# BONE LESIONS full 5-folded cross-validation
DATA_SET_BATCHES_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch1234_AX_balanced_Gray_5batches'
OUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch1234_AX_balanced_Gray_5batches'
NUMBER_BATCHES=5
IMG_SIZE=32
IMG_CHANNELS=1
LAYERS_REL_PATH='bonelesion-layers'
DATA_PROVIDER='general-cropped'
TYPE='fc32'
#TYPE='fc128'

############ CONSTANT NET PARAMS ####################
#LAYERS_REL_PATH='lymphnode-layers'
#DATA_PROVIDER='cifar-cropped'
		
############ RUN ###########################
mkdir -p ${OUT_MODEL_DIR}	
	
# RUN NET	
###s./run_net_lymph_nodes.sh 0 1 $DATA_SET_BATCHES_PATH $OUT_MODEL_DIR $LAYERS_REL_PATH $DATA_PROVIDER $TYPE&	
./run_net_lymph_nodes_Nbatches.sh 0 1 $DATA_SET_BATCHES_PATH $OUT_MODEL_DIR $LAYERS_REL_PATH $DATA_PROVIDER $TYPE $NUMBER_BATCHES $IMG_SIZE $IMG_CHANNELS&	

# RUN NET	
#TYPE='fc-13pct'
#./run_net_lymph_nodes_Nbatches.sh 0 1 $DATA_SET_BATCHES_PATH $OUT_MODEL_DIR $LAYERS_REL_PATH $DATA_PROVIDER $TYPE $NUMBER_BATCHES $IMG_SIZE $IMG_CHANNELS&	

# RUN NET	
#TYPE='fc-conv32'
#./run_net_lymph_nodes_Nbatches.sh 0 1 $DATA_SET_BATCHES_PATH $OUT_MODEL_DIR $LAYERS_REL_PATH $DATA_PROVIDER $TYPE $NUMBER_BATCHES $IMG_SIZE $IMG_CHANNELS&	

# RUN NET	
#TYPE='fc-11pct'
#./run_net_lymph_nodes_Nbatches.sh 0 1 $DATA_SET_BATCHES_PATH $OUT_MODEL_DIR $LAYERS_REL_PATH $DATA_PROVIDER $TYPE $NUMBER_BATCHES $IMG_SIZE $IMG_CHANNELS&	

# RUN NET	
#TYPE='2fc128-11pct'
#./run_net_lymph_nodes_Nbatches.sh 0 1 $DATA_SET_BATCHES_PATH $OUT_MODEL_DIR $LAYERS_REL_PATH $DATA_PROVIDER $TYPE $NUMBER_BATCHES $IMG_SIZE $IMG_CHANNELS&	

#################################################################################
# ## RUN WITH 2 OUTPUTS ##
# LAYERS_REL_PATH='lymphnode-layers' # 2 outputs as in cifar challenge
# BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_2outputNodes'
# RESULST_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_RESULTSout2'

# ## batch permutation 1-7
# PERMUTATIONS=( '1-7' '7-6' '6-5' '5-4' '4-3' '3-2' '2-1' )
# for PERMUTATION in ${PERMUTATIONS[@]} 
# do
	# DATA_SET_PATH=$BATCHES_BASE'/'$PERMUTATION
	# MODEL_DIR=$RESULST_BASE'/'$PERMUTATION
	
	# ./run_net_lymph_nodes_cifar-cropped_fc-13pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH  $NUMBER_BATCHES&
	# ./run_net_lymph_nodes_cifar-cropped_fc-conv32-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH  $NUMBER_BATCHES&
	# ./run_net_lymph_nodes_cifar-cropped_2fc128-11pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH  $NUMBER_BATCHES&
	# ./run_net_lymph_nodes_cifar-cropped_fc-11pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH  $NUMBER_BATCHES&
	# ./run_net_lymph_nodes_cifar-cropped_fc512-11pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH  $NUMBER_BATCHES&
	
	# wait
	# echo "All 5 complete"
# done


########################### CALL HISTORY #######################################
#################################################################################
## RUN WITH 10 OUTPUTS (as in the beginning) ## first experiment ~ 2014-02-07
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_RESULTSout10'

## RUN WITH 10 OUTPUTS (as in the beginning) ## second experiment with randomly shuffled images in batches ~ 2014-02-14 (during SPIE) (RESULT: factor 2 *better* than un-shuffeld)
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_random_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_random_RESULTSout10'

## ABDOMEN: RUN WITH 10 OUTPUTS (as in the beginning) ## third experiment with randomly shuffled images in batches per patient subdivisions ~ 2014-02-25
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/train_and_crossvalid_6batch_random_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_RESULTS_train_and_crossvalid_6batch_random_10outputNodes'

## MEDIASTINAL: RUN WITH 10 OUTPUTS (as in the beginning) ## third experiment with randomly shuffled images in batches per patient subdivisions ~ 2014-02-25
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/train_and_cross_valid_6batches_random_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/Med_LymphNodeData_LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_6batches_random_10outputNodes_RESULTS'

