#!/bin/bash
# for training, just run ./train.sh

set -x
set -e

DIR=`pwd`/`dirname $0`
export PYTHONPATH=caffe/python:${PYTHONPATH}
echo ${PYTHONPATH}
export PYTHONUNBUFFERED="True"

### parameters ###
NAME=""
GPU_ID=0

SOLVERFILE=${DIR}/solver.prototxt
TESTFILE=${DIR}/test.prototxt
TRAIN_IMDB=${DIR}/train_volume_list_example.txt
CFG_FILE=${DIR}/config.yml
NET=VGG16.v2

SNAPSHOT_PATH=${DIR}/snapshots/
mkdir -p $SNAPSHOT_PATH
LOG="${DIR}/log.traintest.`date +'%m-%d_%H-%M-%S'`_${NAME}"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python ${DIR}/python/train_SSBR.py \
	--name "${NAME}" \
	--gpu ${GPU_ID} \
	--solver ${SOLVERFILE} \
	--train_imdb ${TRAIN_IMDB} \
	--test_prototxt ${TESTFILE} \
	--cfg ${CFG_FILE} \
#--weights ${DIR}/${NET}.caffemodel  # comment this if training from scratch
