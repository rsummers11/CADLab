#!/bin/bash
## 
## segment.sh
## 
## Description: This script does the second and final pre-processing step
## required to run EmphNet. They could probably be one script, but it's
## easier to identify errors when they're split up, and boy are there
## errors to identify. This script generates the segmented images from
## the output of preprocess.sh using Adam Harrison's p-hnn segmenter and
## places the segmented images in the same directory as preprocess.sh.
## 
## Author: Wes Caldwell
## Created: July 2018
## Version: 1.0
## 
## Usage: segment.sh pid_list.lst PHNN_ROOT_DIR TARGET_DATA_DIR
##        Here, TARGET_DATA_DIR should be the same as it was for preprocess.sh
##
##


# Parse command line arguments
if (($# != 3)); then
	echo "Incorrect usage."
	echo "Usage: segment.sh pid_list.lst PHNN_ROOT_DIR TARGET_DATA_DIR"
	exit 1
fi

pid_file=$1 # final_fold/all.lst
phnn_root=$2 # /home/caldwellwg/p-hnn/
target_root=$3 # /data2/wes/nlst-ct/

# Extract pids from pid_file
pids=$(cut -d' ' -f1 $pid_file)
for pid in $pids; do
	if ! [[ $pid =~ ^[0-9]+$ ]]; then
		echo "Error in parsing PID file."
		echo "Verify each line is of the form PID LABEL."
		exit 1
	fi
done

# Segment PID list
for pid in $pids
do
	echo -n "Working on PID #$pid..."
	for year in "T0" "T1" "T2"; do
		folder="$target_root/$pid/$year"
		if ! [[ -d $folder ]]; then
			echo -n " No $year file."
		else
			file=$(ls $folder | grep "^$year")
			python $phnn_root/segment_lung.py --caffe_root $phnn_root/caffe_pls/ --file_in $folder/$file --file_out $folder/segmented_$file > /dev/null 2>&1
			python $phnn_root/threshold_pmap.py --file_in $folder/segmented_$file --file_out $folder/segmented_$file > /dev/null 2>&1
			echo -n " $year success!"
		fi
	done
	echo ""
done

