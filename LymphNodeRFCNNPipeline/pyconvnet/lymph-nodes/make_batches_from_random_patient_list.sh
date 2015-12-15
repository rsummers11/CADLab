#!/bin/bash
START=$(date +%s) # timer

# postMICCAI2014 MED and ABD: training
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced'
###START_IDX=1
###NUMBER_BATCHES=5
###SEARCH_STR='_AxCoSa.png'
# postMICCAI2014 MED and ABD: cross-validation
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\cross_valid'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\cross_valid'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\cross_valid_AxCoSa'
###START_IDX=6
###NUMBER_BATCHES=1
###SEARCH_STR='_AxCoSa.png'

#### MICCAI2014 REGUTTAL MED and ABD: unseen-validation
## ABDOMEN
INPUT_FOLDER1=''#'D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\train'
INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\cross_valid'
INPUT_FOLDER3='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\unseen_valid'
OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\Random_CrossValANDUnseenTest\Round5'
START_IDX=1
NUMBER_BATCHES=20
SEARCH_STR='_AxCoSa.png'
PATIENTNAMES_FILE='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\Abd_all_26_cross_valid_AND_testing_patientnames.txt'
NUMBER_RANDOM_PATIENTS=14

## MEDIASTINUM
###INPUT_FOLDER1=''#'D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\train'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\cross_valid'
###INPUT_FOLDER3='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\unseen_valid'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\Random_CrossValANDUnseenTest\Round5'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxCoSa.png'
###PATIENTNAMES_FILE='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_all_25_cross_valid_AND_testing_patientnames.txt'
###NUMBER_RANDOM_PATIENTS=15

################# RUN #####################
mkdir -p $OUT_BATCH_FOLDER
python ./lymph-nodes/make_batches_from_random_patient_list.py $INPUT_FOLDER1 $INPUT_FOLDER2 $INPUT_FOLDER3 $OUT_BATCH_FOLDER $START_IDX $NUMBER_BATCHES $SEARCH_STR $PATIENTNAMES_FILE $NUMBER_RANDOM_PATIENTS

############### ELAPSED TIME ##############################
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours" | tee --append $LOG_FILE_NAME
   
