#!/bin/bash
START=$(date +%s) # timer

# CIFAR
INPUT_MODEL_DIR='D:\HolgerRoth\DropConnect\MyDropConnect\data\cifar-10-py-colmajor_RESULTS'
INPUT_MODEL_NAME='model_fc-11pct-dc'
DATA_SET_PATH='D:\HolgerRoth\DropConnect\MyDropConnect\data\cifar-10-py-colmajor'
TEST_RANGE=1-5
USE_MULTIVIEW=0
###FEATURE_LAYER='fc128'
FEATURE_LAYER='pool2'

# INPUT PARAMS (ABDOMEN MICCAI 2014)
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\MICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_RESULTS_train_and_crossvalid_6batch_random_10outputNodes\1-7'
#INPUT_MODEL_NAME='model_fc512-11pct-dc'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\MICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_per_patient_unsee_valid_1batch_random_10outputNodes'

# INPUT PARAMS (MEDIASTINAL MICCAI 2014)
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\MICCAI2014\Med\Med_LymphNodeData_LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_6batches_random_10outputNodes_RESULTS\1-7'
#INPUT_MODEL_NAME='model_fc512-11pct-dc'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\MICCAI2014\Med\MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs\unseen_valid_1batch_random_10outputNodes'

### CORRECTED MICCAI 2014 VALIDATION ####
# INPUT PARAMS (ABDOMEN MICCAI 2014)
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\MICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_RESULTS_train_and_crossvalid_6batch_random_10outputNodes\1-7'
#INPUT_MODEL_NAME='model_fc512-11pct-dc'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\unseen_valid_posauto_AND_negauto_AxCoSa'

# INPUT PARAMS (MEDIASTINAL MICCAI 2014)
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\MICCAI2014\Med\Med_LymphNodeData_LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_6batches_random_10outputNodes_RESULTS\1-7'
#INPUT_MODEL_NAME='model_fc512-11pct-dc'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\unseen_valid_posauto_AND_negauto_AxCoSa'

### postMICCAI2014 VALIDATION ####
# INPUT PARAMS (MEDIASTINAL MICCAI 2014)
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_and_cross_valid'
#INPUT_MODEL_NAME='fc-13pct'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa\'

#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\MICCAI2014\Med\Med_LymphNodeData_LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_6batches_random_10outputNodes_RESULTS\1-7'
#INPUT_MODEL_NAME='model_fc512-11pct-dc'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa\'

### postMICCAI2014 VALIDATION ####
# INPUT PARAMS (ABDOMEN MICCAI 2014)
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_AND_cross_valid'
###INPUT_MODEL_NAME='fc512-11pct'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced_AND_cross_valid\'
#OUTPUT_NOTE='MedAbdprobs1-5'
#TEST_RANGE=1-5 # exclude cross-validation set (6)
###OUTPUT_NOTE='MedAbdprobs6'
###TEST_RANGE=6 # exclude cross-validation set (6)

###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa_20batches\'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa_20batches\'
###TEST_RANGE=1-20
###USE_MULTIVIEW=0
###FEATURE_LAYER='probs'
###FEATURE_LAYER='fc2'
###FEATURE_LAYER='fc512'

## BONE LESIONS
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\train_AX_balanced_Gray_6batches'
###INPUT_MODEL_NAME='fc512-11pct'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\train_AX_balanced_Gray_6batches'
###TEST_RANGE=1-6
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\test_AX_Gray_20batches'
###TEST_RANGE=1-20
###USE_MULTIVIEW=0
###FEATURE_LAYER='fc512'

#########################################################################################
# PROCESS
INPUT_NN=${INPUT_MODEL_DIR}'/'${INPUT_MODEL_NAME}
FEATURES_OUT_DIR=${DATA_SET_PATH}'/features_'${FEATURE_LAYER}

mkdir -p ${FEATURES_OUT_DIR}
# RUN SCRIPTS
echo 'RECONFIGURE' & echo ''
python ./convnet.py -f $INPUT_NN --logreg-name=logprob --test-range=$TEST_RANGE --test-only=1 --data-path=$DATA_SET_PATH --save-path=$INPUT_MODEL_DIR --test-freq=1 --multiview-test=$USE_MULTIVIEW

echo 'WRITE FEATURES' & echo ''
python ./shownet.py -f $INPUT_NN --feature-path=$FEATURES_OUT_DIR --write-features=$FEATURE_LAYER --test-range=$TEST_RANGE --multiview-test=$USE_MULTIVIEW

echo 'CONVERT FEATURES TO TEXT FILE' & echo ''
python ./lymph-nodes/write_features.py $FEATURES_OUT_DIR $TEST_RANGE
   
# ELAPSED TIME
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours" | tee --append $LOG_FILE_NAME
   