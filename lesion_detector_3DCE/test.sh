#!/usr/bin/env bash
# test existing model. Modify exp_name and begin_epoch in default.yml to choose which model to test
# config all the other things in default.yml and config.yml

set -x
set -e

note='just a test'
DIR=`pwd`/`dirname $0`

read -r line < $DIR/default.yml
exp_name=`echo $line | cut -d "'" -f 2`

LOG="$DIR/log.test_`date +'%m-%d_%H-%M-%S'`.$exp_name"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python $DIR/rcnn/tools/test.py