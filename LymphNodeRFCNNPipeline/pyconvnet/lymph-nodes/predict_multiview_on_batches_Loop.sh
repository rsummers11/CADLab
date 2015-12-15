#!/bin/bash
START=$(date +%s) # timer

# MICCAI 2014 REBUTTAL
### ABDOMEN ###
INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_and_cross_valid'
INPUT_MODEL_NAME='fc512-11pct'
DATA_SET_PATH_PREFIX='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_BATCHES\Random_CrossValANDUnseenTest\Round'
TEST_RANGE=1-20
OUT_SUFFIX='ABD_MICCAI2014rev'
USE_MULTIVIEW=1

### MEDIASTINUM ###
###INPUT_MODEL_DIR='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_and_cross_valid'
###INPUT_MODEL_NAME='fc512-11pct'
###DATA_SET_PATH_PREFIX='D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Med\Med_LymphNodeData_win_iso_trans_rot_2scales_BATCHES\Random_CrossValANDUnseenTest\Round'
###TEST_RANGE=1-20
###OUT_SUFFIX='MED_MICCAI2014rev'
###USE_MULTIVIEW=1

# ## PREDICT ACROSS ROUNDS
ROUNDS=( '1' '2' '3' '4' '5' )
for ROUND in ${ROUNDS[@]} 
do
	START_LOOP=$(date +%s) # timer
	DATA_SET_PATH=${DATA_SET_PATH_PREFIX}${ROUND}

	#########################################################################################
	# PROCESS
	INPUT_NN=${INPUT_MODEL_DIR}'/'${INPUT_MODEL_NAME}
	PREDICTIONS_BATCH_FILE=${DATA_SET_PATH}'/'${INPUT_MODEL_NAME}'_'${OUT_SUFFIX}'_predictions'
	PREDICTIONS_TEXT_FILE=${PREDICTIONS_BATCH_FILE}'.txt'

	# RUN SCRIPTS
	echo 'Generating Prediction file at: ' ${PREDICTIONS_TEXT_FILE}
	echo 'RECONFIGURE'
###	python ./convnet.py -f $INPUT_NN --logreg-name=logprob --multiview-test=$USE_MULTIVIEW --test-range=$TEST_RANGE --test-only=1 --data-path=$DATA_SET_PATH --save-path=$INPUT_MODEL_DIR --test-freq=1

	echo 'WRITE PREDICTIONS'
###	python ./shownet.py -f $INPUT_NN --write-predictions=$PREDICTIONS_BATCH_FILE --multiview-test=$USE_MULTIVIEW --test-range=$TEST_RANGE

	echo 'CONVERT PREDICTIONS TO TEXT FILE'
	python ./lymph-nodes/predict_multiview.py $PREDICTIONS_BATCH_FILE $PREDICTIONS_TEXT_FILE $USE_MULTIVIEW
   
	wait
	echo "***********************************************************"
	echo "ROUND "${ROUND}" COMPLETED."   
	echo "***********************************************************"
	
	############### ELAPSED TIME IN CURRENT LOOP ##############################
	END=$(date +%s) # timer
	DIFF=$(( $END - $START_LOOP ))
	echo "It took $DIFF seconds"
	DIFF=$(( $END/60 - $START_LOOP/60 ))
	echo "     or $DIFF minutes"
	DIFF=$(( $END/3600 - $START_LOOP/3600 ))
	echo "     or $DIFF hours"	
done
   
############### ELAPSED TIME ##############################
END=$(date +%s) # timer
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
DIFF=$(( $END/60 - $START/60 ))
echo "     or $DIFF minutes"
DIFF=$(( $END/3600 - $START/3600 ))
echo "     or $DIFF hours"
   