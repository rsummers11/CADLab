#!/bin/bash

INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4512_AX_balanced_Gray'
CLEANED_FILES_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\batch4512_AX_balanced_Gray_zeros'
SEARCH_STR='_AX.png'

################# RUN #####################
mkdir -p $CLEANED_FILES_FOLDER
python ./lymph-nodes/clean_zero_files.py $INPUT_FOLDER $CLEANED_FILES_FOLDER $SEARCH_STR

