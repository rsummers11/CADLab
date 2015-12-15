#!/bin/bash

# postMICCAI2014 ABD: training 
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales/train_AX_balanced'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES/train_AX_balanced'
#START_IDX=1
#NUMBER_BATCHES=5
#SEARCH_STR='_AX.png'
# MICCAI2014 ABD: cross-validation (10 outputs) (best ConvNet search)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales/crossvalid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES/crossvalid_AX'
#START_IDX=1
#NUMBER_BATCHES=1
#SEARCH_STR='_AX.png'
# MICCAI2014 ABD: unseen-validation (10 outputs)
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales/unseen_valid'
#OUT_BATCH_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES/unseen_valid_AX'
#START_IDX=1
#NUMBER_BATCHES=1
#SEARCH_STR='_AX.png'

# postMICCAI2014 MED: training
#INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced'
#START_IDX=1
#NUMBER_BATCHES=5
#SEARCH_STR='_AxCoSa.png'
# postMICCAI2014 MED: cross-validation
#INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\cross_valid'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\cross_valid_AxCoSa'
#START_IDX=6
#NUMBER_BATCHES=1
#SEARCH_STR='_AxCoSa.png'
# postMICCAI2014 MED: unseen-validation
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\unseen_valid'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxCoSa.png'

# postMICCAI2014 ABD: corrected unseen-validation 
#INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
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

# Kevin RF CAD v2
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST\train_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST_BATCHES\train_AxCoSa_balanced_6batches'
###START_IDX=1
###NUMBER_BATCHES=6
###SEARCH_STR='_AxCoSa.png'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST\unseen_valid'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST\unseen_valid_60batches'
###START_IDX=1
###NUMBER_BATCHES=60
###SEARCH_STR='_AxCoSa.png'

# Bone Lesions
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered\train_AxRGB_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered_BATCHES\train_AxRGB_balanced_6batches'
###START_IDX=1
###NUMBER_BATCHES=6
###SEARCH_STR='_AxRGB.png'
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered\test'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered_BATCHES\test_AxRGB_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxRGB.png'

###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\train_AX_balanced_Gray'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\train_AX_balanced_Gray_6batches'
###START_IDX=1
###NUMBER_BATCHES=6
###SEARCH_STR='_AX.png'
###IMG_SIZE=32
###IMG_CHANNELS=1
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\test_AX_Gray'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\test_AX_Gray_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AX.png'
###IMG_SIZE=32
###IMG_CHANNELS=1
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\test_AX_CADe_only_Gray'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\test_AX_CADe_only_Gray_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AX.png'
###IMG_SIZE=32
###IMG_CHANNELS=1

# Bone lesions
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\train_unbalanced_CADe_only_Gray'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\train_unbalanced_CADe_only_Gray_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AX.png'
###IMG_SIZE=32
###IMG_CHANNELS=1

# postMICCAI2014 MED: training
#INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\train_AxCoSa_balanced'
#START_IDX=1
#NUMBER_BATCHES=5
#SEARCH_STR='_AxCoSa.png'
# postMICCAI2014 MED: cross-validation
#INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\cross_valid'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\cross_valid_AxCoSa'
#START_IDX=6
#NUMBER_BATCHES=1
#SEARCH_STR='_AxCoSa.png'
# postMICCAI2014 MED: unseen-validation
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\unseen_valid'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\unseen_valid_AxCoSa_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxCoSa.png'

## postMICCAI2014 ABD: corrected unseen-validation 3-folded cross-validation
# TRAINING
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch12_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch12_AxCoSa_balanced'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch23_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch23_AxCoSa_balanced'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch13_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch13_AxCoSa_balanced'
###START_IDX=1
###NUMBER_BATCHES=5
# VALIDATION
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch1'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch1_AxCoSa_valid_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch2'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch2_AxCoSa_valid_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch3'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\batch3_AxCoSa_valid_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxCoSa.png'
###IMG_SIZE=32
###IMG_CHANNELS=3

## postMICCAI2014 MED: corrected unseen-validation 3-folded cross-validation
# TRAINING
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\batch12_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch12_AxCoSa_balanced'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\batch23_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch23_AxCoSa_balanced'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\batch13_AxCoSa_balanced'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch13_AxCoSa_balanced'
###START_IDX=1
###NUMBER_BATCHES=5
# VALIDATION
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch1'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch1_AxCoSa_valid_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch2'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch2_AxCoSa_valid_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch3'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch3_AxCoSa_valid_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxCoSa.png'
###IMG_SIZE=32
###IMG_CHANNELS=3

# VALIDATION -- JUST CADe MARKS
# ABDOMEN
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch1_CADe_only'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch1_AxCoSa_CADe_only_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch2_CADe_only'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch2_AxCoSa_CADe_only_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch3_CADe_only'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch3_AxCoSa_CADe_only_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxCoSa.png'
###IMG_SIZE=32
###IMG_CHANNELS=3
# MEDIASTINUM
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch1_CADe_only'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch1_CADe_only_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch2_CADe_only'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch2_CADe_only_20batches'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch3_CADe_only'
###OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\batch3_CADe_only_20batches'
###START_IDX=1
###NUMBER_BATCHES=20
###SEARCH_STR='_AxCoSa.png'
###IMG_SIZE=32
###IMG_CHANNELS=3

# BONE LESIONS - 5-folded cross-validation
INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2345_AX_balanced_Gray'
OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch2345_AX_balanced_Gray_5batches'
#INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3451_AX_balanced_Gray'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch3451_AX_balanced_Gray_5batches'
#INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4512_AX_balanced_Gray'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch4512_AX_balanced_Gray_5batches'
#INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5123_AX_balanced_Gray'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch5123_AX_balanced_Gray_5batches'
#INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1234_AX_balanced_Gray'
#OUT_BATCH_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_BATCHES\batch1234_AX_balanced_Gray_5batches'
START_IDX=1
NUMBER_BATCHES=5
SEARCH_STR='_AX.png'
IMG_SIZE=32
IMG_CHANNELS=1

################# RUN #####################
function pause(){
   read -p "$*"
}

mkdir -p $OUT_BATCH_FOLDER
python ./lymph-nodes/make_general_batches.py $INPUT_FOLDER $OUT_BATCH_FOLDER $START_IDX $NUMBER_BATCHES $SEARCH_STR $IMG_SIZE $IMG_CHANNELS

#pause 'Press [Crtl+C] to exit (avoids double execution)...'

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

