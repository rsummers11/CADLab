#!/bin/bash
START=$(date +%s) # timer

# training definitions
NN_ROOT=D:/HolgerRoth/DropConnect/MyDropConnect/src

GPU=$1 #input from command line
INIT_SCALE=$2

DATA_SET_PATH=$3
MODEL_DIR=$4

MODEL_SCRIPT_PATH=./$5

DATA_PROVIDER=$6
#DATA_PROVIDER=cifar-cropped-rand

FILE_NAME=$7
NN_DEF='layers-'${FILE_NAME}'-dc.cfg'
NN_PARAMS='params-'${FILE_NAME}'.cfg'

ADP_DROP=0
RESET_MOM=0

# input params

# drop mode:
# 0: no drop out, general less epoch of training
# 1: drop output
# 2: drop connection, dec mini 
DROP_MODE=2

# exec file
EXEC_CMD=python
#EXEC_CMD=echo


#---------------------------------------
# dependent variables
#---------------------------------------
mkdir $MODEL_DIR
LOG_FILE_NAME=$MODEL_DIR/${FILE_NAME}_log.txt


# set epoch
if [ $DROP_MODE -eq 0 ]
then
    EPOCH_1=350
    EPOCH_2=150
    EPOCH_3=50
    EPOCH_4=50
else  
    EPOCH_1=700
    EPOCH_2=300
    # Experiment F-Table Dec17 and its analysis suggest 
    # epoch 3,4 should be 100,100 rather than 20,20
    # which is just double of non-drop nn
    EPOCH_3=100 
    EPOCH_4=100
fi

#echo =========================================
#echo warning using my own epochs!!!!
#echo =========================================
#EPOCH_1=700
#EPOCH_2=170
#EPOCH_3=10 
#EPOCH_4=10

#EPOCH_1=70
#EPOCH_2=30
#EPOCH_3=10 
#EPOCH_4=10

# set mini-batch
if [ $DROP_MODE -eq 2 ]
then
    MINI_1=64
    MINI_2=64
    MINI_3=32
    MINI_4=16
else
    MINI_1=128
    MINI_2=128
    MINI_3=128
    MINI_4=128
fi

# scale mode:
SCALE_1=$(echo $INIT_SCALE 1    | awk '{printf "%4.3f",$1*$2}' )
SCALE_2=$(echo $INIT_SCALE 1    | awk '{printf "%4.3f",$1*$2}' )
SCALE_3=$(echo $INIT_SCALE 0.1  | awk '{printf "%4.3f",$1*$2}' )
SCALE_4=$(echo $INIT_SCALE 0.01 | awk '{printf "%4.3f",$1*$2}' )

# print summary information
echo 
echo '----------------------------------------'
echo 'log file : ' ${LOG_FILE_NAME} 
echo 'epoch    : ' $EPOCH_1 $EPOCH_2 $EPOCH_3 $EPOCH_4
echo 'mini     : ' $MINI_1 $MINI_2 $MINI_3 $MINI_4
echo 'scale    : ' $SCALE_1 $SCALE_2 $SCALE_3 $SCALE_4
echo 'reset-mom : ' $RESET_MOM '   GPU      : ' ${GPU}
echo '----------------------------------------'
echo

#---------------------------------------
# run network
#---------------------------------------
# stage 1: train range 1-4 test on 5
EPOCH=$EPOCH_1
$EXEC_CMD $NN_ROOT/convnet.py \
    --data-path=$DATA_SET_PATH \
    --save-path=$MODEL_DIR \
    --model-file=$FILE_NAME \
    --test-range=5 --train-range=1-4 \
    --layer-def=$MODEL_SCRIPT_PATH/$NN_DEF  \
    --layer-params=$MODEL_SCRIPT_PATH/$NN_PARAMS \
    --data-provider=${DATA_PROVIDER} --test-freq=20 \
    --crop-border=4 --epoch=$EPOCH --adp-drop=$ADP_DROP \
    --gpu=$GPU --mini=$MINI_1\
    --scale=$SCALE_1 | tee $LOG_FILE_NAME
   
# stage 2: fold 5th batch 
EPOCH=`expr $EPOCH + $EPOCH_2`
$EXEC_CMD $NN_ROOT/convnet.py -f $MODEL_DIR/$FILE_NAME  \
   --train-range=1-5 --epoch=$EPOCH\
   --adp-drop=0 --mini=$MINI_2\
   --scale=$SCALE_2 | tee --append $LOG_FILE_NAME

if [ $RESET_MOM -eq 1 ] 
then
    EPOCH=`expr $EPOCH + $EPOCH_2`
    $EXEC_CMD $NN_ROOT/convnet.py -f $MODEL_DIR/$FILE_NAME  \
        --train-range=1-5 --epoch=$EPOCH\
        --adp-drop=0 --mini=$MINI_2\
        --reset-mom=$RESET_MOM \
        --scale=$SCALE_2 | tee --append $LOG_FILE_NAME
fi

# stage 3: dec learning rate by 10 and 20 iterations more
EPOCH=`expr $EPOCH + $EPOCH_3`
$EXEC_CMD $NN_ROOT/convnet.py -f $MODEL_DIR/$FILE_NAME  \
   --epoch=$EPOCH \
   --adp-drop=0 --mini=$MINI_3\
   --test-range=5 \
   --reset-mom=$RESET_MOM \
   --scale=$SCALE_3 | tee --append $LOG_FILE_NAME

# stage 4: dec learning rate by 10 and 20 iterations more
EPOCH=`expr $EPOCH + $EPOCH_4`
$EXEC_CMD $NN_ROOT/convnet.py -f $MODEL_DIR/$FILE_NAME  \
   --epoch=$EPOCH \
   --adp-drop=0 --mini=$MINI_4\
   --reset-mom=$RESET_MOM \
   --multiview-test=1 \
   --logreg-name=logprob --test-range=5 \
   --scale=$SCALE_4 | tee --append $LOG_FILE_NAME

# stage 5: test on batch 6
$EXEC_CMD $NN_ROOT/convnet.py -f $MODEL_DIR/$FILE_NAME  \
   --multiview-test=1 --test-only=1 \
   --logreg-name=logprob --test-range=6 \
   --reset-mom=$RESET_MOM | tee --append $LOG_FILE_NAME

# print summary information again
echo 
echo '----------------------------------------'
echo 'log file : ' ${LOG_FILE_NAME} 
echo 'epoch    : ' $EPOCH_1 $EPOCH_2 $EPOCH_3 $EPOCH_4
echo 'mini     : ' $MINI_1 $MINI_2 $MINI_3 $MINI_4
echo 'scale    : ' $SCALE_1 $SCALE_2 $SCALE_3 $SCALE_4
echo 'reset-mom : ' $RESET_MOM '   GPU      : ' ${GPU}
echo '----------------------------------------'
echo

# Look at the trained net:
#$EXEC_CMD $NN_ROOT/shownet.py -f $MODEL_DIR/$FILE_NAME --show-cost=logprob

# ELAPSED TIME
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes" | tee --append $LOG_FILE_NAME
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours" | tee --append $LOG_FILE_NAME

