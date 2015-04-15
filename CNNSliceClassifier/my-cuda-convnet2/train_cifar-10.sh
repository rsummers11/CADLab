#!/bin/bash

BATCHES_PATH="/home/rothhr/Data/CIFAR-10/cifar-10-py-colmajor"
TRAIN_RANGE="1-5"
TEST_RANGE="6"
LAYER_DEF="layers/layers-cifar10-11pct.cfg"
LAYER_PARAMS="layers/layer-params-cifar10-11pct.cfg"
INNER_SIZE="24"
DATA_PROVIDER="cifar"

#### OUTPUT PATH ####
TRAINED_NET_PATH="/home/rothhr/Data/CIFAR-10/cifar-10-py-colmajor_ConvNet2"

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
	--gpu 0 | tee $LOG_FILE

#python convnet.py --data-provider cifar --test-range 6 --train-range 1-5 --data-path /usr/local/storage/akrizhevsky/cifar-10-py-colmajor --inner-size 24 --save-path /usr/local/storage/akrizhevsky/ --gpu 0 --layer-def layers/layers-cifar10-11pct.cfg --layer-params layers/layer-params-cifar10-11pct.cfg
