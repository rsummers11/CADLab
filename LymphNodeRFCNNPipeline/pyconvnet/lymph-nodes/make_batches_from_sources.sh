#!/bin/bash

# postMICCAI2014 MED and ABD: training
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced'
###START_IDX=1
###NUMBER_BATCHES=5
###SEARCH_STR='_AxCoSa.png'
# postMICCAI2014 MED and ABD: training
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\cross_valid'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\cross_valid'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\cross_valid_AxCoSa'
###START_IDX=6
###NUMBER_BATCHES=1
###SEARCH_STR='_AxCoSa.png'

# postMICCAI2014 MED AND ABD: training for 3-folded cross-validaiton
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales\MedAbd_batch12_AxCoSa_balanced\Abd\batch12_AxCoSa_balanced'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales\MedAbd_batch12_AxCoSa_balanced\Med\batch12_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\MedAbd_batch12_AxCoSa_balanced'
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales\MedAbd_batch23_AxCoSa_balanced\Abd\batch23_AxCoSa_balanced'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales\MedAbd_batch23_AxCoSa_balanced\Med\batch23_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\MedAbd_batch23_AxCoSa_balanced'
INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales\MedAbd_batch13_AxCoSa_balanced\Abd\batch13_AxCoSa_balanced'
INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales\MedAbd_batch13_AxCoSa_balanced\Med\batch13_AxCoSa_balanced'
OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\MedAbd\MedAbd_LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\MedAbd_batch13_AxCoSa_balanced_5batches'
START_IDX=1
NUMBER_BATCHES=5
SEARCH_STR='_AxCoSa.png'

################# RUN #####################
mkdir -p $OUT_BATCH_FOLDER
python ./lymph-nodes/make_batches_from_sources.py $INPUT_FOLDER1 $INPUT_FOLDER2 $OUT_BATCH_FOLDER $START_IDX $NUMBER_BATCHES $SEARCH_STR

################### CALL HISTORY #######################################################################
# MICCAI2014 ABD: training (10 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/train'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/train_5batch_random_10outputNodes'
#START_IDX=1
#NUMBER_BATCHES=5
# MICCAI2014 ABD: cross-validation (10 outputs) (best ConvNet search)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/crossvalid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/crossvalid_1batch_random_10outputNodes'
#START_IDX=1
#NUMBER_BATCHES=1
# MICCAI2014 ABD: unseen-validation (10 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_per_patient_unsee_valid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_per_patient_unsee_valid_1batch_random_10outputNodes'
#START_IDX=1
#NUMBER_BATCHES=1

# MICCAI2014 MED: training (10 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/train'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/train_5batch_random_10outputNodes'
#START_IDX=1
#NUMBER_BATCHES=5
# MICCAI2014 MED: cross-valid (10 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/cross_valid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/cross_valid_1batch_random_10outputNodes'
#START_IDX=1
#NUMBER_BATCHES=1
# MICCAI2014 MED: unseen-validation (10 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/unseen_valid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/unseen_valid_1batch_random_10outputNodes'
#START_IDX=1
#NUMBER_BATCHES=1

###################### 2 OUTPUTS #########################################################
# MICCAI2014 training (2 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/train'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/train_5batch_random_2outputNodes'
#START_IDX=1
#NUMBER_BATCHES=5
# MICCAI2014 cross-validation (2 outputs) (best ConvNet search)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/crossvalid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/crossvalid_1batch_random_2outputNodes'
#START_IDX=1
#NUMBER_BATCHES=1

# MICCAI2014 MED: training (2 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/train'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/train_5batch_random_2outputNodes'
#START_IDX=1
#NUMBER_BATCHES=5
# MICCAI2014 MED: cross-valid (2 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/cross_valid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/cross_valid_1batch_random_2outputNodes'
#START_IDX=1
#NUMBER_BATCHES=1
