# postMICCAI2014 ABD: training 
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\train'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\train_balanced'
###SEARCH_STR='_AxCoSa.png'

# postMICCAI2014 ABD: training 
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\train'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\train_AxCoSa_balanced'
###SEARCH_STR='_AxCoSa.png'

# RF CAD v2: training 
###INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST\train\batches3'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\KevinRFCAD_v2\LymphNodeData_highRes_win_iso_trans_rot_4scales_NEAREST\train_AxCoSa_balanced\batches3'
###SEARCH_STR='_AxCoSa.png'

# Bone lesions
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered\train'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered\train_AxRGB_balanced'
###SEARCH_STR='_AxRGB.png'
###INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\train'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\train_AX_balanced'
###SEARCH_STR='_AX.png'

# postMICCAI2014 ABD: corrected unseen-validation 3-folded cross-validation
# train: batch1/2, test: batch3
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch1'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch2'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch12_AxCoSa_balanced'
###SEARCH_STR='_AxCoSa.png'
# train: batch2/3, test: batch1
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch2'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch3'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch23_AxCoSa_balanced'
###SEARCH_STR='_AxCoSa.png'
# train: batch1/3, test: batch2
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch1'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch3'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales\batch13_AxCoSa_balanced'
###SEARCH_STR='_AxCoSa.png'

# postMICCAI2014 MED: corrected unseen-validation 3-folded cross-validation
# train: batch1/2, test: batch3
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch1'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch2'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch12_AxCoSa_balanced'
###SEARCH_STR='_AxCoSa.png'
# train: batch2/3, test: batch1
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch2'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch3'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch23_AxCoSa_balanced'
###SEARCH_STR='_AxCoSa.png'
# train: batch1/3, test: batch2
###INPUT_FOLDER1='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch1'
###INPUT_FOLDER2='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch3'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales\batch13_AxCoSa_balanced'
###SEARCH_STR='_AxCoSa.png'

# BONE LESIONS - 5-folded cross-validation
# train: batch2,3,4,5, test: batch1
###INPUT_FOLDER1='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2'
###INPUT_FOLDER2='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3'
###INPUT_FOLDER3='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4'
###INPUT_FOLDER4='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2345_AX_balanced'
###SEARCH_STR='_AX.png'
# train: batch3,4,5,1 test: batch2
###INPUT_FOLDER1='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3'
###INPUT_FOLDER2='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4'
###INPUT_FOLDER3='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5'
###INPUT_FOLDER4='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3451_AX_balanced'
###SEARCH_STR='_AX.png'
# train: batch4,5,1,2 test: batch3
###INPUT_FOLDER1='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4'
###INPUT_FOLDER2='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5'
###INPUT_FOLDER3='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1'
###INPUT_FOLDER4='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4512_AX_balanced'
###SEARCH_STR='_AX.png'
# train: batch5,1,2,3 test: batch4
###INPUT_FOLDER1='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5'
###INPUT_FOLDER2='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1'
###INPUT_FOLDER3='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2'
###INPUT_FOLDER4='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch5123_AX_balanced'
###SEARCH_STR='_AX.png'
# train: batch1,2,3,4 test: batch5
###INPUT_FOLDER1='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1'
###INPUT_FOLDER2='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch2'
###INPUT_FOLDER3='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch3'
###INPUT_FOLDER4='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4'
###BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch1234_AX_balanced'
###SEARCH_STR='_AX.png'

# PANCREAS
INPUT_FOLDER1='D:\HolgerRoth\data\Pancreas\rois\all_images'
INPUT_FOLDER2='notvalid'
INPUT_FOLDER3='notvalid'
INPUT_FOLDER4='notvalid'
BALANCED_OUT_FOLDER='D:\HolgerRoth\data\Pancreas\rois\balanced'
SEARCH_STR='_AX.png'

################# RUN #####################
function pause(){
   read -p "$*"
}
mkdir -p $BALANCED_OUT_FOLDER
python ./lymph-nodes/balance_files_from_sources.py $INPUT_FOLDER1 $INPUT_FOLDER2 $INPUT_FOLDER3 $INPUT_FOLDER4 $BALANCED_OUT_FOLDER $SEARCH_STR

pause 'Press [Crtl+C] to exit (avoids double execution)...'
