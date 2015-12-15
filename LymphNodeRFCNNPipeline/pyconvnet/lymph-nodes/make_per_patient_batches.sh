#!/bin/bash

# postMICCAI2014 ABD: corrected unseen-validation 
INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced'
#START_IDX=1
#NUMBER_BATCHES=5
#SEARCH_STR='_AxCoSa.png'
#INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\cross_valid'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\cross_valid_AxCoSa'
#START_IDX=6
#NUMBER_BATCHES=1
#SEARCH_STR='_AxCoSa.png'
# postMICCAI2014 ABD: unseen-validation
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\unseen_valid'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa_20batches'
###START_IDX=1 
###NUMBER_BATCHES=20 # Needs to be dividable by the number of patch images
###SEARCH_STR='_AxCoSa.png'

################# RUN #####################
mkdir -p $OUT_BATCH_FOLDER
python ./lymph-nodes/make_general_per_patient_batches.py $INPUT_FOLDER $OUT_BATCH_FOLDER $START_IDX $NUMBER_BATCHES $SEARCH_STR

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

