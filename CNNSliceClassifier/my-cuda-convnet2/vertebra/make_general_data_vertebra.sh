#!/bin/bash
# Training/Testing set
INPUT_FOLDER="C:\HR\Data\Spine\PostElemFractures\CT_post_Elem_Healthy_mLabel\orbits128"
OUT_BATCH_FOLDER="C:\HR\Data\Spine\PostElemFractures\CT_post_Elem_Healthy_mLabel\orbits128_500batches"
NUMBER_BATCHES=500
DO_SHUFFLE=1
SEARCH_STR='.jpg'
EXISTING_MEAN=''

START_IDX=1
IMG_SIZE=256
IMG_CHANNELS=3
LABEL_TYPE="vertebra"

################# RUN #####################
LOG_FILE_NAME=$OUT_BATCH_FOLDER'/LOG_BATCHES_'$START_IDX'-'$NUMBER_BATCHES'.txt'
START=$(date +%s) # timer

function pause(){
   read -p "$*"
}

mkdir -p $OUT_BATCH_FOLDER
python ./make_data/make_general_data.py $INPUT_FOLDER $OUT_BATCH_FOLDER $START_IDX $NUMBER_BATCHES $SEARCH_STR $IMG_SIZE $IMG_CHANNELS $DO_SHUFFLE $LABEL_TYPE $EXISTING_MEAN \
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


