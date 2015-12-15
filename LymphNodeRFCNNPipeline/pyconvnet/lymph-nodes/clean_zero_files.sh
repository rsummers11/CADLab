#!/bin/bash

INPUT_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\normal_cases_8bit'
CLEANED_FILES_FOLDER='D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered\normal_cases_8bit_zeros'
SEARCH_STR='_AX.png'

################# RUN #####################
mkdir -p $CLEANED_FILES_FOLDER
python ./lymph-nodes/clean_zero_files.py $INPUT_FOLDER $CLEANED_FILES_FOLDER $SEARCH_STR

