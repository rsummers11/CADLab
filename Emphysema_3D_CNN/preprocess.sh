##!/bin/bash
## 
## preprocess.sh
## 
## Description : This script does the first pre-processing step
## required to run EmphNet. That includes converting the NLST images
## to DICOM and choosing from an individuals screens the one
## with the most slices (to prevent X-rays and localizers and
## stuff like that).
## 
## Author: Wes Caldwell
## Created: July 2018
## Version: 1.0
## 
## Usage: preprocess.sh pid_list.lst NLST_ROOT_DIR TARGET_DATA_DIR
## 
## 


# Parse command line arguments
if (($# != 3)); then
	echo "Incorrect usage."
	echo "Usage: preprocess.sh pid_list.lst NLST_ROOT_DIR TARGET_DATA_DIR"
	exit 1
fi

pid_file=$1 # final_fold/all.lst
nlst_root=$2 # /media/erdos_wes/nlst-ct/
target_root=$3 # /data2/wes/nlst-ct/

# Extract pids from pid_file
pids=$(cut -d' ' -f1 $pid_file)
for pid in $pids; do
	if ! [[ $pid =~ ^[0-9]+$ ]]; then
		echo "Error in parsing PID file."
		echo "Verify each line is of the form PID LABEL"
		exit 1
	fi
done

# Pre-process PID list
for pid in $pids; do
	echo -n "Working on pid #$pid..."
	for year in "T0" "T1" "T2"; do
		mkdir -p "$target_root/$pid/$year"
		./dcm2niix -o "$target_root/$pid/$year" -z y -i y "$nlst_root/$pid/$year" > /dev/null
		best_file=$(python << HEREDOC


import os, sys, numpy, nibabel
files=filter(lambda f: f.endswith('.nii.gz'), os.listdir('$target_root/$pid/$year'))
if files:
	n_slices=map(lambda file: nibabel.load('$target_root/$pid/$year/' + file).get_header()['dim'][3], files)
	print(files[numpy.argmax(n_slices)])
HEREDOC
)

		if [[ -z "$best_file" ]]; then
			echo -n " $year failure."
			rmdir "$target_root/$pid/$year"
		else
			find "$target_root/$pid/$year" ! -name $best_file -type f -exec rm {} \;
			echo -n " $year success!"
		fi
	done
	echo ""
done

