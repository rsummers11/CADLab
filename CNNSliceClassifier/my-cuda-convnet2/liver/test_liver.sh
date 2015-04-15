#!/bin/bash
# ISBI 2015 (original images)
#BATCHES_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_256_5batches"
#TRAINED_NET_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_256_5batches_ConvNet2/ConvNet__2014-11-14_16.16.20/101.1"
#TEST_RANGE="5"

# ISBI 2015 (augmentated images)
BATCHES_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_def_t2_r2_balanced_256_500batches"
TRAINED_NET_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_def_t2_r2_balanced_256_500batches_ConvNet2/ConvNet__2014-11-14_21.13.15/101.1"
TEST_RANGE="401-500"

#### RUN NET ####
START=$(date +%s) # timer
NOW=$(date +"%Y-%m-%d_%H.%M.%S")
LOG_FILE=$TRAINED_NET_PATH'_TESTRANGE'$TEST_RANGE'_'$NOW'_LOG.txt' 

# single-view
echo "SINGLE-VIEW TEST"
python ./convnet.py \
	--load-file $TRAINED_NET_PATH \
	--data-path $BATCHES_PATH \
	--train-range 1-1 \
	--test-range $TEST_RANGE \
	--test-only 1 \
	--multiview-test 0 \
        | tee $LOG_FILE

### multi-view ###
echo "MULTI-VIEW TEST"
python ./convnet.py \
	--load-file $TRAINED_NET_PATH \
	--data-path $BATCHES_PATH \
	--train-range 1-1 \
	--test-range $TEST_RANGE \
	--test-only 1 \
	--multiview-test 1 \
        | tee --append $LOG_FILE

# ELAPSED TIME
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" | tee --append $LOG_FILE
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes" | tee --append $LOG_FILE
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours" | tee --append $LOG_FILE

