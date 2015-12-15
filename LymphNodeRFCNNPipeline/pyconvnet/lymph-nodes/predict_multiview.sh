#!/bin/bash
START=$(date +%s) # timer

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
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa_20batches\'
###TEST_RANGE=1-20

## BONE LESIONS
#INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\train_AX_balanced_Gray_6batches'
#OUTPUT_NOTE='BoneLesionsBsplineAx'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\train_AX_balanced_Gray_6batches_EarlyStop'
###INPUT_MODEL_NAME='fc512-11pct'
###OUTPUT_NOTE='BoneLesionsBsplineAx_EarlyStop'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\train_AX_balanced_Gray_6batches'
###TEST_RANGE=1-6
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\test_AX_Gray_20batches'
###TEST_RANGE=1-20
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\test_AX_CADe_only_Gray_20batches'
###TEST_RANGE=1-20
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\train_unbalanced_CADe_only_Gray_20batches'
###TEST_RANGE=1-20

### postMICCAI2014 VALIDATION ####
# 3-folded cross-validation ABDOMEN MICCAI 2014
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch12_AxCoSa_balanced'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch3_AxCoSa_valid_20batches'
###OUTPUT_NOTE='3folded_Abd_AxCoSa_trainedBatch12'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch23_AxCoSa_balanced'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch1_AxCoSa_valid_20batches'
###OUTPUT_NOTE='3folded_Abd_AxCoSa_trainedBatch23'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch13_AxCoSa_balanced'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch2_AxCoSa_valid_20batches'
###OUTPUT_NOTE='3folded_Abd_AxCoSa_trainedBatch13'
###INPUT_MODEL_NAME='fc512-11pct'
###TEST_RANGE=1-20 

# 3-folded cross-validation MEDASTINUM MICCAI 2014
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_RESULTS\train_batch12_AxCoSa_balanced_5batches'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch3_AxCoSa_valid_20batches'
#OUTPUT_NOTE='3folded_Med_AxCoSa_trainedBatch12'
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_RESULTS\train_batch23_AxCoSa_balanced_5batches'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch1_AxCoSa_valid_20batches'
#OUTPUT_NOTE='3folded_Med_AxCoSa_trainedBatch23'
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_RESULTS\train_batch13_AxCoSa_balanced_5batches'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch2_AxCoSa_valid_20batches'
#OUTPUT_NOTE='3folded_Med_AxCoSa_trainedBatch13'
#INPUT_MODEL_NAME='fc512-11pct'
#TEST_RANGE=1-20 

# 3-folded cross-validation JOINT MED AND ABD TRAINED MICCAI 2014 (NOT FINISHED YET!!!!!)
# ABDOMEN VALIDATION
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\MedAbd_batch12_AxCoSa_balanced_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch3_AxCoSa_valid_20batches'
###OUTPUT_NOTE='3folded_jointMedAbd_AxCoSa_trainedBatch12'
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch23_AxCoSa_balanced'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch1_AxCoSa_valid_20batches'
#OUTPUT_NOTE='3folded_jointMedAbd_Abd_AxCoSa_trainedBatch23'
#INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch13_AxCoSa_balanced'
#DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch2_AxCoSa_valid_20batches'
#OUTPUT_NOTE='3folded_jointMedAbd_Abd_AxCoSa_trainedBatch13'
#INPUT_MODEL_NAME='fc512-11pct'
#TEST_RANGE=1-20 

### postMICCAI2014 VALIDATION ####
# 3-folded cross-validation ABDOMEN MICCAI 2014 -- JUST CADe MARKS
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch12_AxCoSa_balanced'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch3_AxCoSa_CADe_only_20batches'
###OUTPUT_NOTE='3folded_Abd_AxCoSa_trainedBatch12_CADeOnly'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch23_AxCoSa_balanced'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch1_AxCoSa_CADe_only_20batches'
###OUTPUT_NOTE='3folded_Abd_AxCoSa_trainedBatch23_CADeOnly'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_batch13_AxCoSa_balanced'
###DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch2_AxCoSa_CADe_only_20batches'
###OUTPUT_NOTE='3folded_Abd_AxCoSa_trainedBatch13_CADeOnly'
###INPUT_MODEL_NAME='fc512-11pct'
###TEST_RANGE=1-20 

# BONE LESIONS: 5-fold cross-validation #
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch1234_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch5_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch1234_CADeOnly'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch2345_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch1_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch2345_CADeOnly'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch3451_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch2_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch3451_CADeOnly'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch4512_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch3_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch4512_CADeOnly'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch5123_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch4_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch5123_CADeOnly'
###INPUT_MODEL_NAME='fc32'
###TEST_RANGE=1-20 

# BONE LESIONS: 5-fold training #
INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch1234_AX_balanced_Gray_5batches'
DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch1_unbalanced_CADe_only_Gray_20batches'
OUTPUT_NOTE='5folded_AX_trainedBatch1234_CADeOnly_TRAIN2'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch2345_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch2_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch2345_CADeOnly_TRAIN2'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch3451_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch3_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch3451_CADeOnly_TRAIN2'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch4512_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch4_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch4512_CADeOnly_TRAIN2'
###INPUT_MODEL_DIR='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\batch5123_AX_balanced_Gray_5batches'
###DATA_SET_PATH='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch5_unbalanced_CADe_only_Gray_20batches'
###OUTPUT_NOTE='5folded_AX_trainedBatch5123_CADeOnly_TRAIN2'
###INPUT_MODEL_NAME='fc32'
###TEST_RANGE=1-20

# TEST PIPELINE #
INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\MedAbd_batch12_AxCoSa_balanced_5batches'
DATA_SET_PATH='D:\HolgerRoth\data\LymphNodes\RandForestCAD\PQRVBUG4\PQRVBUG4_HU_CNN_images_BSPLINE_CNNpredictions'
OUTPUT_NOTE='5folded_AX_trainedBatch1234_CADeOnly_TRAIN2'
INPUT_MODEL_NAME='fc512-11pct'
TEST_RANGE=1-5

#########################################################################################
# PROCESS
INPUT_NN=${INPUT_MODEL_DIR}'/'${INPUT_MODEL_NAME}
PREDICTIONS_BATCH_FILE=${DATA_SET_PATH}'/'${INPUT_MODEL_NAME}'_'${OUTPUT_NOTE}'_predictions'
PREDICTIONS_TEXT_FILE=${PREDICTIONS_BATCH_FILE}'.txt'
USE_MULTIVIEW=1
# RUN SCRIPTS
echo 'RECONFIGURE'
python ./convnet.py -f $INPUT_NN --logreg-name=logprob --multiview-test=$USE_MULTIVIEW --test-range=$TEST_RANGE --test-only=1 --data-path=$DATA_SET_PATH --save-path=$INPUT_MODEL_DIR --test-freq=1

echo 'WRITE PREDICTIONS'
python ./shownet.py -f $INPUT_NN --write-predictions=$PREDICTIONS_BATCH_FILE --multiview-test=$USE_MULTIVIEW --test-range=$TEST_RANGE

echo 'CONVERT PREDICTIONS TO TEXT FILE'
python ./lymph-nodes/predict_multiview.py $PREDICTIONS_BATCH_FILE $PREDICTIONS_TEXT_FILE $USE_MULTIVIEW

# ELAPSED TIME
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours" | tee --append $LOG_FILE_NAME
   