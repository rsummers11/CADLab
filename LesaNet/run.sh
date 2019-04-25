#!/usr/bin/env bash
# do the whole process of training, validation and test
# config all things in default.yml and config.yml
set -x
set -e

DIR=`pwd`/`dirname $0`

#LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
#PATH=/usr/local/cuda-9.0/bin:$PATH

read -r line < $DIR/default.yml  # the first line of default.yml should be the name of the experiment
exp_name=`echo $line | cut -d "'" -f 2`

LOG="$DIR/log/log.`date +'%m-%d_%H-%M-%S'`.$exp_name.log"  # use the exp_name and time to name the log file
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python $DIR/main.py