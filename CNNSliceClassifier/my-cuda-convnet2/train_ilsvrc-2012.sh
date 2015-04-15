#!/bin/bash

BATCHES_PATH="/home/rothhr/Data/IMAGENET/ilsvrc-2012-batches"
TRAIN_RANGE="0-417"
TEST_RANGE="1000-1016"
LAYER_DEF="layers/layers-imagenet-1gpu.cfg"
LAYER_PARAMS="layers/layer-params-imagenet-1gpu.cfg"
DATA_PROVIDER="image"
INNER_SIZE="224"
LOG_STRING="imagenet-1gpu"

#### OUTPUT PATH ####
TRAINED_NET_PATH="/home/rothhr/Data/IMAGENET/ilsvrc-2012-batches_ConvNet2"


#### RUN NET ####
LOG_FILE=$TRAINED_NET_PATH'/'$LOG_STRING'_LOG.txt' 

mkdir -p $TRAINED_NET_PATH

python convnet.py \
	--data-path $BATCHES_PATH \
	--train-range $TRAIN_RANGE \
	--test-range $TEST_RANGE \
	--save-path $TRAINED_NET_PATH \
	--layer-def $LAYER_DEF \
	--layer-params $LAYER_PARAMS \
	--data-provider $DATA_PROVIDER \
	--inner-size $INNER_SIZE \
	--gpu 0 \
	--mini 128 \
	--epochs 90 \
        --test-freq 201 \
        --color-noise 0.1 | tee $LOG_FILE

#python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-1gpu.cfg --layer-params layers/layer-params-imagenet-1gpu.cfg --data-provider image --inner-size 224 --gpu 0 --mini 128 --test-freq 201 --color-noise 0.1
