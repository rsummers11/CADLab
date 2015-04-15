#!/bin/bash
PATIENT_START=1002
PATIENT_END=1018

for ((x=$PATIENT_START; x<=$PATIENT_END; x++))
{
   echo "......................... PREDICTING PATIENT $x ....................................."
   ./liver/predict_multiview_liver.sh $x
}
