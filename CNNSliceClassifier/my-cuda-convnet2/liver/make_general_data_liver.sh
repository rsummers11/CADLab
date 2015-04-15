#!/bin/bash

###LIVER: TRAINING
#INPUT_FOLDER="/home/rothhr/Data/Liver/LiverImages/Train256"
#OUT_BATCH_FOLDER="/home/rothhr/Data/Liver/LiverImages/Train256_5batches"

#INPUT_FOLDER="/home/rothhr/Data/Liver/LiverImages/Train512_test"
#OUT_BATCH_FOLDER="/home/rothhr/Data/Liver/LiverImages/Train512_test_5batches"

INPUT_FOLDER="/home/rothhr/Data/Liver/LiverImages/Training512_t3_r3_d3_256"
OUT_BATCH_FOLDER="/home/rothhr/Data/Liver/LiverImages/Training512_t3_r3_d3_256_500batches"
NUMBER_BATCHES=500
DO_SHUFFLE=1
SEARCH_STR='.png'

#INPUT_FOLDER="/home/rothhr/Data/Liver/LiverImages/Training256_originals"
#OUT_BATCH_FOLDER="/home/rothhr/Data/Liver/LiverImages/Training256_originals_5batches"
#NUMBER_BATCHES=5
#DO_SHUFFLE=1
#SEARCH_STR='.png'

###LIVER: TESTING
#INPUT_FOLDER="/home/rothhr/Data/Liver/LiverImages/Testing_FullCT/U12GV2G1_slices"
#OUT_BATCH_FOLDER="/home/rothhr/Data/Liver/LiverImages/Testing_FullCT/U12GV2G1_slices_1batches"
#NUMBER_BATCHES=1
#DO_SHUFFLE=0

###INPUT_FOLDER="/home/rothhr/Data/Liver/LiverImages/Training256_originals"
###OUT_BATCH_FOLDER="/home/rothhr/Data/Liver/LiverImages/Training256_originals_1batches"
###NUMBER_BATCHES=1
###DO_SHUFFLE=0

### NUMERICAL EVALUATION (ISBI 2015)
#INPUT_FOLDER=$1
#OUT_BATCH_FOLDER=$2
#NUMBER_BATCHES=1
#DO_SHUFFLE=0
#SEARCH_STR='.jpg'

#INPUT_FOLDER=$1
#OUT_BATCH_FOLDER=$2
#NUMBER_BATCHES=1
#DO_SHUFFLE=0

# TESTING FOR ISBI2015 figure
INPUT_FOLDER="/home/rothhr/Data/Liver/ISBI2015_figs/organs256"
OUT_BATCH_FOLDER="/home/rothhr/Data/Liver/ISBI2015_figs/organs256_1batch"
NUMBER_BATCHES=1
DO_SHUFFLE=0
SEARCH_STR='.png'
EXISTING_MEAN='/home/rothhr/Data/Liver/LiverImages/Training512_t3_r3_d3_256_500batches/data_mean.png'

START_IDX=1
IMG_SIZE=256
IMG_CHANNELS=1
LABEL_TYPE="liver"

################# RUN #####################
LOG_FILE_NAME=$OUT_BATCH_FOLDER'/LOG_BATCHES_'$START_IDX'-'$NUMBER_BATCHES'.txt'
START=$(date +%s) # timer

function pause(){
   read -p "$*"
}

mkdir -p $OUT_BATCH_FOLDER
python ./make_data/make_general_data.py $INPUT_FOLDER $OUT_BATCH_FOLDER $START_IDX $NUMBER_BATCHES $SEARCH_STR $IMG_SIZE $IMG_CHANNELS $DO_SHUFFLE $LABEL_TYPE $EXISTING_MEAN\
 | tee $LOG_FILE_NAME

# ELAPSED TIME
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours" | tee --append $LOG_FILE_NAME

#pause 'Press [Crtl+C] to exit (avoids double execution)...'


