#!/bin/bash

## MEDIASTINAL: RUN WITH
LAYERS_REL_PATH='cifar-layers' # 2 outputs as in cifar challenge
BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/train_and_cross_valid_6batches_random_10outputNodes'
RESULST_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/Med_LymphNodeData_LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_6batches_random_10outputNodes_RESULTS'
DATA_PROVIDER=cifar-cropped	

mkdir ${RESULST_BASE}

# #PERMUTATIONS=( '1-7' '7-6' '6-5' '5-4' '4-3' '3-2' '2-1' )
PERMUTATIONS=( '1-7' )

## run 5 nets on 7 batch permutations 
for PERMUTATION in ${PERMUTATIONS[@]} 
do
	DATA_SET_PATH=$BATCHES_BASE'/'$PERMUTATION
	MODEL_DIR=$RESULST_BASE'/'$PERMUTATION
	
	# RUN NET	
	TYPE='fc512-11pct'
	LAYER_CFG='layers-'{$TYPE}'-dc.cfg'
	PARAMS_CFG='params-'{$TYPE}'.cfg'
	./run_net_lymph_nodes.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH $LAYER_CFG $PARAMS_CFG $DATA_PROVIDER $TYPE&
	
	wait
	echo "All 5 complete"
done

#################################################################################
# ## RUN WITH 2 OUTPUTS ##
# LAYERS_REL_PATH='lymphnode-layers' # 2 outputs as in cifar challenge
# BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_2outputNodes'
# RESULST_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_RESULTSout2'

# ## batch permutation 1-7
# PERMUTATIONS=( '1-7' '7-6' '6-5' '5-4' '4-3' '3-2' '2-1' )
# for PERMUTATION in ${PERMUTATIONS[@]} 
# do
	# DATA_SET_PATH=$BATCHES_BASE'/'$PERMUTATION
	# MODEL_DIR=$RESULST_BASE'/'$PERMUTATION
	
	# ./run_net_lymph_nodes_cifar-cropped_fc-13pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH &
	# ./run_net_lymph_nodes_cifar-cropped_fc-conv32-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH &
	# ./run_net_lymph_nodes_cifar-cropped_2fc128-11pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH &
	# ./run_net_lymph_nodes_cifar-cropped_fc-11pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH &
	# ./run_net_lymph_nodes_cifar-cropped_fc512-11pct-dc.sh 0 1 $DATA_SET_PATH $MODEL_DIR $LAYERS_REL_PATH &
	
	# wait
	# echo "All 5 complete"
# done


########################### CALL HISTORY #######################################
#################################################################################
## RUN WITH 10 OUTPUTS (as in the beginning) ## first experiment ~ 2014-02-07
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_RESULTSout10'

## RUN WITH 10 OUTPUTS (as in the beginning) ## second experiment with randomly shuffled images in batches ~ 2014-02-14 (during SPIE) (RESULT: factor 2 *better* than un-shuffeld)
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_random_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_random_RESULTSout10'

## ABDOMEN: RUN WITH 10 OUTPUTS (as in the beginning) ## third experiment with randomly shuffled images in batches per patient subdivisions ~ 2014-02-25
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_pngs/train_and_crossvalid_6batch_random_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Abd/LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_RESULTS_train_and_crossvalid_6batch_random_10outputNodes'

## MEDIASTINAL: RUN WITH 10 OUTPUTS (as in the beginning) ## third experiment with randomly shuffled images in batches per patient subdivisions ~ 2014-02-25
#LAYERS_REL_PATH='cifar-layers' # 10 outputs as in cifar challenge
#BATCHES_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/MedMed_LymphNodeData_win_iso_trans_rot_2scales_per_patient_pngs/train_and_cross_valid_6batches_random_10outputNodes'
#RESULST_BASE='D:/HolgerRoth/data/LymphNodes/MICCAI2014/Med/Med_LymphNodeData_LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_6batches_random_10outputNodes_RESULTS'

