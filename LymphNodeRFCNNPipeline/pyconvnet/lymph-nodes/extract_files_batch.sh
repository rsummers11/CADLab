#!/bin/bash

# postMICCAI2014 MED: training 
#INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Med/Med_LymphNodeData_win_iso_trans_rot_2scales/unseen_valid'
#OUT_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Med/Med_LymphNodeData_win_iso_trans_rot_2scales/unseen_valid_posauto_AND_negauto_AxCoSa'
#SEARCH_STR='_AxCoSa.png'
#SEARCH_STR='_neg_ROI'

# postMICCAI2014 ABD: training 
###INPUT_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales/unseen_valid'
###OUT_FOLDER='D:/HolgerRoth/data/LymphNodes/postMICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales/unseen_valid_posauto_AND_negauto_AxCoSa'
###SEARCH_STR='_AxCoSa.png'
###SEARCH_STR='_neg_ROI'

# Bone Lesions:
# Just CADe test images (no gold standard)
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\test'
###OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\test_CADe_only'
###SEARCH_STR1='_pos_boneCADe_'
#SEARCH_STR1='_neg_boneCADe_'
###SEARCH_STR2='_AX.png'
# Just CADe training images (no gold standard)
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\train'
###OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\train_unbalanced_CADe_only'
###SEARCH_STR1='_pos_boneCADe_'
###SEARCH_STR1='_neg_boneCADe_'
###SEARCH_STR2='_AX.png'


# postMICCAI2014 ABD: training -- JUST CAD DETECTIONS
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch1'
###OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch1_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch2'
###OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch2_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch3'
###OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch3_CADe_only'
# postMICCAI2014 MED: training -- JUST CAD DETECTIONS
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch1'
###OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch1_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch2'
###OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch2_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch3'
###OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch3_CADe_only'
###SEARCH_STR1='_neg_'
###SEARCH_STR2='_negauto_'
###SEARCH_STR3='_pos_'
###SEARCH_STR4='_posauto_'
###SEARCH_STR_ENDING='_AxCoSa.png'

# Bone Lesions: 5-folded cross-validation
# Just CADe images (no gold standard)
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1'
###OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1_unbalanced_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2'
###OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2_unbalanced_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3'
###OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3_unbalanced_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4'
###OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4_unbalanced_CADe_only'
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5'
###OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5_unbalanced_CADe_only'
SEARCH_STR1='_pos_boneCADe_'
SEARCH_STR2='_neg_boneCADe_'
SEARCH_STR_ENDING='_AX.png'

################# RUN #####################
function pause(){
   read -p "$*"
}

mkdir -p $OUT_FOLDER
python ./lymph-nodes/extract_files.py $INPUT_FOLDER $OUT_FOLDER $SEARCH_STR1 $SEARCH_STR_ENDING & 
python ./lymph-nodes/extract_files.py $INPUT_FOLDER $OUT_FOLDER $SEARCH_STR2 $SEARCH_STR_ENDING & 
#python ./lymph-nodes/extract_files.py $INPUT_FOLDER $OUT_FOLDER $SEARCH_STR3 $SEARCH_STR_ENDING & 
#python ./lymph-nodes/extract_files.py $INPUT_FOLDER $OUT_FOLDER $SEARCH_STR4 $SEARCH_STR_ENDING & 

wait
echo "All complete"
pause 'Press [Crtl+C] to exit (avoids double execution)...'

