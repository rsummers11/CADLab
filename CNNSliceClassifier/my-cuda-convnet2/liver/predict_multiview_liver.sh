#!/bin/bash
START=$(date +%s) # timer

#LIVER: testing
#NET='/home/rothhr/Data/Liver/LiverImages/Training256_originals_5batches_ConvNet2/ConvNet__2014-10-09_19.01.49/1200.4' # 14% error
NET='/home/rothhr/Data/Liver/LiverImages/Training512_t3_r3_d3_256_500batches_ConvNet2/ConvNet__2014-10-10_13.42.26/17.400' # 7% error with transformations
##DATA_SET_PATH='/home/rothhr/Data/Liver/LiverImages/Training256_originals_1batches'
DATA_SET_PATH='/home/rothhr/Data/Liver/Liver_GrTrSeg_Slices_256_batches/'$1'_Raw/'
OUTPUT_NOTE='Test'
TEST_RANGE=1 # ONLY WORKS WITH 1 BATCH!!!!!!!!!!!!! (see predict_multiview_prostate for multi-batch predictions)
LABEL_TYPE='liver'

#########################################################################################
# PROCESS
DATA_BATCH_TEXT_FILE=${DATA_SET_PATH}'/data_batch_'${TEST_RANGE}'.txt'
PREDICTIONS_DIR=${DATA_SET_PATH}'/'${OUTPUT_NOTE}'_predictions'
PREDICTIONS_BATCH_FILE=${PREDICTIONS_DIR}'/data_batch_'${TEST_RANGE}
PREDICTIONS_TEXT_FILE=${PREDICTIONS_BATCH_FILE}'.txt'
#USE_MULTIVIEW=1 # multi-view doesn't seem to work together with write features

echo 'WRITE PREDICTIONS'
mkdir -p $PREDICTIONS_DIR
python ./convnet.py --load-file $NET --data-path=$DATA_SET_PATH --write-feature probs --feature-path $PREDICTIONS_DIR --train-range $TEST_RANGE --test-range $TEST_RANGE --test-only 1 
#--multiview-test $USE_MULTIVIEW

echo 'CONVERT PREDICTIONS TO TEXT FILE'
python ./predict_multiview.py $PREDICTIONS_BATCH_FILE $PREDICTIONS_TEXT_FILE $DATA_BATCH_TEXT_FILE $LABEL_TYPE
#$USE_MULTIVIEW

# ELAPSED TIME
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes"
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours"
   
