#!/bin/bash

INPUT_FOLDER='D:\HolgerRoth\data\LymphNodes\test\U12GV2G1_testRegion_win_iso_30mm'
OUT_BATCH_FOLDER='D:\HolgerRoth\data\LymphNodes\test\U12GV2G1_testRegion_win_iso_30mm_batch'
mkdir -p $OUT_BATCH_FOLDER
python ./lymph-nodes/make_one_volume_batch.py $INPUT_FOLDER $OUT_BATCH_FOLDER
