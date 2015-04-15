#!/bin/bash
# ISBI 2015 (original images)
#BATCHES_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_256_5batches"
#TRAINED_NET_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_256_5batches_ConvNet2"
#TRAIN_RANGE="1-4"
#TEST_RANGE="5"
# ISBI 2015 (augmentated)
BATCHES_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_def_t2_r2_balanced_256_500batches"
TRAINED_NET_PATH="/home/rothhr/Data/Liver/AnatomyImages/All512_to_def_t2_r2_balanced_256_500batches_ConvNet2"
TRAIN_RANGE="1-400"
TEST_RANGE="401-410"
# common
LAYER_DEF="layers/liver/layers-imagenet-1gpu.cfg"
LAYER_PARAMS="layers/liver/layer-params-imagenet-1gpu.cfg"
DATA_PROVIDER="general"
INNER_SIZE="224" # crop images
LOG_STRING="imagenet-1gpu"
EPOCHS="100"
TEST_FREQ="100"

#### RUN NET ####
START=$(date +%s) # timer
NOW=$(date +"%Y-%m-%d_%H.%M.%S")
LOG_FILE=$TRAINED_NET_PATH'/'$LOG_STRING'_'$NOW'_LOG.txt' 

mkdir -p $TRAINED_NET_PATH

python ./convnet.py \
	--data-path $BATCHES_PATH \
	--train-range $TRAIN_RANGE \
	--test-range $TEST_RANGE \
	--save-path $TRAINED_NET_PATH \
	--layer-def $LAYER_DEF \
	--layer-params $LAYER_PARAMS \
	--data-provider $DATA_PROVIDER \
	--inner-size $INNER_SIZE \
	--epochs $EPOCHS \
	--gpu 0 \
	--mini 128 \
        --test-freq $TEST_FREQ \
        --force-save 1 \
        | tee $LOG_FILE

#python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-1gpu.cfg --layer-params layers/layer-params-imagenet-1gpu.cfg --data-provider image --inner-size 224 --gpu 0 --mini 128 --test-freq 201 --color-noise 0.1

# ELAPSED TIME
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" | tee --append $LOG_FILE
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes" | tee --append $LOG_FILE
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours" | tee --append $LOG_FILE

