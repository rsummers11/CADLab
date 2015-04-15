#!/bin/bash
INPUT_FOLDER_ROOT='C:\HolgerRoth\data\Liver\Liver_GrTrSeg_Slices_256' 
OUT_BATCH_FOLDER_ROOT=$INPUT_FOLDER_ROOT'_batches'
PATIENT_START=1002
PATIENT_END=1018

########### RUN ##############
mkdir -p $OUT_BATCH_FOLDER_ROOT
for ((x=$PATIENT_START; x<=$PATIENT_END; x++))
{
   echo "......................... PREDICTING PATIENT $x ....................................."
   INPUT_FOLDER=$INPUT_FOLDER_ROOT'/'$x'_Raw' 
   OUT_BATCH_FOLDER=$OUT_BATCH_FOLDER_ROOT'/'$x'_Raw' 
   ./liver/make_general_data_liver.sh $INPUT_FOLDER $OUT_BATCH_FOLDER
}
