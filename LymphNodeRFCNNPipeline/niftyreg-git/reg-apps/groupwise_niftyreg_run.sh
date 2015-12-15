#!/bin/sh

#### What could be done ##################################################################
# - add a preprocessing step in order to intensity normalise all the input images ???
# - Any other ?
##########################################################################################

if [ $# -lt 1 ]
then
	echo ""
	echo "*******************************************************************************"
	echo "One argument is expected to run this script:"
	echo "- File with contains the altas creation parameters"
	echo "example: $0 param_groupwise_niftyreg.sh "
	echo "*******************************************************************************"
	echo ""
	exit
fi


#############################################################################
# read the input parameters
. $1

#############################################################################
## the argument value are checked
if [ ${#IMG_INPUT[@]} -lt 2 ]
then
	echo "Less than 2 images have been specified"
	echo "Exit ..."
	exit
fi

if [ ! -e ${TEMPLATE} ]
then
	echo "The template image (${TEMPLATE}) does not exist"
	echo "Exit ..."
	exit
fi

if [ "${TEMPLATE_MASK}" != "" ] && [ ! -f ${TEMPLATE_MASK} ]
then
	echo "The template image mask (${TEMPLATE_MASK}) does not exist"
	echo "Exit ..."
fi

IMG_NUMBER=${#IMG_INPUT[@]}
MASK_NUMBER=${#IMG_INPUT_MASK[@]}
if [ ${MASK_NUMBER} -gt 0 ] && [ ! -f ${IMG_INPUT_MASK[0]} ] \
	&& [ ${MASK_NUMBER} != ${IMG_NUMBER} ]
then
	echo "The number of images is different from the number of floating masks"
	echo "Exit ..."
	exit
fi

#############################################################################
## SET UP THE NIFTYREG EXECUTABLES
AFFINE=reg_aladin
NRR=reg_f3d
RES=reg_resample
AVERAGE=reg_average
TRANS=reg_transform
TOOLS=reg_tools

#############################################################################
echo ""
echo "************************************************************"
echo ">>> There are ${IMG_NUMBER} input images to groupwise register <<<"
echo ">>> The template image to initialise the registration is ${TEMPLATE} <<<"
echo "************************************************************"
echo ""
#############################################################################
# CREATE THE RESULT FOLDER
if [ ! -d ${RES_FOLDER} ]
then
	echo "The output image folder (${RES_FOLDER}) does not exist"
	mkdir ${RES_FOLDER}
	if [ ! -d ${RES_FOLDER} ]
	then
		echo "Unable to create the ${RES_FOLDER} folder"
		echo "Exit ..."
		exit
	else
		echo "The output image folder (${RES_FOLDER}) has been created"
	fi
fi

#############################################################################
#############################################################################
# PERFORM THE RIGID/AFFINE REGISTRATION

# The initial average image is as specified by the user
averageImage=${TEMPLATE}

# Loop over all iterations
for (( CUR_IT=1; CUR_IT<=${AFF_IT_NUM}; CUR_IT++ ))
do
	# Check if the iteration has already been performed
	if [ ! -f ${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz ]
	then
		#############################
		# Create a folder to store the result
		if [ ! -d ${RES_FOLDER}/aff_${CUR_IT} ]
		then
			mkdir ${RES_FOLDER}/aff_${CUR_IT}
		fi

		#############################
		# Run the rigid or affine registration
		if [ "`which qsub 2> /dev/null`" == "" ]
		then
			# All registration are performed serially
			for (( i=0 ; i<${IMG_NUMBER}; i++ ))
			do
				name=`basename ${IMG_INPUT[${i}]} .gz`
				name=`basename ${name} .nii`
				name=`basename ${name} .hdr`
				name=`basename ${name} .img`
				# Check if the registration has already been performed
				if [ ! -f ${RES_FOLDER}/aff_${CUR_IT}/aff_mat_${name}_it${CUR_IT}.txt ]
				then
					aladin_args=""
					# Registration is forced to be rigid for the first step
					if [ ${CUR_IT} == 1 ]
					then
						aladin_args="-rigOnly"
					else
						# Check if a previous affine can be use for initialisation
						if [ -f ${RES_FOLDER}/aff_`expr ${CUR_IT} - 1`/aff_mat_${name}_it`expr ${CUR_IT} - 1`.txt ]
						then
							aladin_args="-inaff \
								${RES_FOLDER}/aff_`expr ${CUR_IT} - 1`/aff_mat_${name}_it`expr ${CUR_IT} - 1`.txt"
						fi
					fi
					# Check if a mask has been specified for the reference image
					if [ "${TEMPLATE_MASK}" != "" ]
					then
						aladin_args="${aladin_args} -rmask ${TEMPLATE_MASK}"
					fi
					# Check if a mask has been specified for the floating image
					if [ ${MASK_NUMBER} == ${IMG_NUMBER} ]
					then
						aladin_args="${aladin_args} -fmask ${IMG_INPUT_MASK[${i}]}"
					fi
					result="/dev/null"
					if [ "${CUR_IT}" == "${AFF_IT_NUM}" ]
					then
						result="${RES_FOLDER}/aff_${CUR_IT}/aff_res_${name}_it${CUR_IT}.nii.gz"
					fi
					# Perform the registration
					reg_aladin ${AFFINE_args} ${aladin_args} \
						-ref ${averageImage} \
						-flo ${IMG_INPUT[${i}]} \
						-aff ${RES_FOLDER}/aff_${CUR_IT}/aff_mat_${name}_it${CUR_IT}.txt \
						-res ${result} > ${RES_FOLDER}/aff_${CUR_IT}/aff_log_${name}_it${CUR_IT}.txt
					if [ ! -f ${RES_FOLDER}/aff_${CUR_IT}/aff_mat_${name}_it${CUR_IT}.txt ]
					then
						echo "Error when creating \
							${RES_FOLDER}/aff_${CUR_IT}/aff_mat_${name}_it${CUR_IT}.txt"
						exit
					fi
				fi
			done
		else
			# Create shell script to run all jobs in an array
			echo \#\!/bin/sh > ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			# Define the current image index
			echo "img_number=\`expr \$SGE_TASK_ID - 1\`" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo ". `readlink -f $1`"  \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			# Extract the name of the file without the path and the extension
			echo "name=\`basename \${IMG_INPUT[\$img_number]} .gz\`" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo "name=\`basename \$name .nii\`" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo "name=\`basename \$name .hdr\`" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo "name=\`basename \$name .img\`" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			# Check that the registration has not already been performed
			echo "if [ ! -e ${RES_FOLDER}/aff_${CUR_IT}/aff_mat_\${name}_it${CUR_IT}.txt ]" >> \
				${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo "then" >> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			# Check if an input affine is available
			echo "trans_affine=${RES_FOLDER}/aff_`expr ${CUR_IT} - 1`/aff_mat_\${name}_it`expr ${CUR_IT} - 1`.txt" >> \
				${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			# Set up the registration argument
			echo "${AFFINE} ${AFFINE_args} \\" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo "-ref ${averageImage} \\" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo "-flo \${IMG_INPUT[img_number]} \\" \
				>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			echo "-aff ${RES_FOLDER}/aff_${CUR_IT}/aff_mat_\${name}_it${CUR_IT}.txt \\" >> \
				${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			result="/dev/null"
			if [ "${CUR_IT}" == "${AFF_IT_NUM}" ]
			then
			result="${RES_FOLDER}/aff_${CUR_IT}/aff_res_\${name}_it${CUR_IT}.nii.gz"
			fi
			echo "-res ${result} \\" >> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			if [ "${TEMPLATE_MASK}" != "" ]; then
				echo "-rmask ${TEMPLATE_MASK} \\" \
					>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			fi
			if [ ${MASK_NUMBER} == ${IMG_NUMBER} ]; then
				echo "-fmask \${IMG_INPUT_MASK[\$img_number]} \\" \
					>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			fi
			# If this is the first iteration. The registration is forced to be rigid
			# Otherwise the previous affine is used for initialisation
			if [ ${CUR_IT} == 1 ]
			then
				echo "-rigOnly" \
					>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			else
				echo "-inaff \${trans_affine}" \
					>> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			fi
			echo "fi"  >> ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
			## Run the rigid/affine registration - Submit the job array
			# Wait to see if the previous iteration average has been created
			${QSUB_CMD} \
				-hold_jid avg_aff_`expr ${CUR_IT} - 1`_${$} \
				-o ${RES_FOLDER}/aff_${CUR_IT} \
				-N aladin_${CUR_IT}_${$} \
				-t 1-${IMG_NUMBER} \
				sh ${RES_FOLDER}/aff_${CUR_IT}/run_gw_niftyReg_aladin_${CUR_IT}_${$}.sh
		fi

		#############################
		if [ "${CUR_IT}" != "${AFF_IT_NUM}" ]
		then
		# The transformation are demean'ed to create the average image
		# Note that this is not done for the last iteration step
			list_average=""
			for img in ${IMG_INPUT[@]}
			do
				name=`basename ${img} .gz`
				name=`basename ${name} .nii`
				name=`basename ${name} .hdr`
				name=`basename ${name} .img`
				list_average="${list_average} \
					${RES_FOLDER}/aff_${CUR_IT}/aff_mat_${name}_it${CUR_IT}.txt ${img}"
			done
			if [ "`which qsub 2> /dev/null`" == "" ]
			then
				# The average is created on the host
				reg_average \
					${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz \
					-demean1 ${averageImage} \
					${list_average}
				if [ ! -f ${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz ]
				then
					echo "Error when creating \
						${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz"
					exit
				fi
			else # if [ "`which qsub 2> /dev/null`" == "" ]
				# The average is performed through the cluster
				${QSUB_CMD} \
					-hold_jid aladin_${CUR_IT}_${$} \
					-o ${RES_FOLDER}/aff_${CUR_IT} \
					-N avg_aff_${CUR_IT}_${$} \
					reg_average \
					${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz \
					-demean1 ${averageImage} \
					${list_average}
			fi # if [ "`which qsub 2> /dev/null`" == "" ]
		else # if [ "${CUR_IT}" != "${AFF_IT_NUM}" ]
			# All the result images are directly averaged during the last step
			if [ "`which qsub 2> /dev/null`" == "" ]
			then
				reg_average \
					${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz \
					-avg \
					`ls ${RES_FOLDER}/aff_${CUR_IT}/aff_res_*_it${CUR_IT}.nii*`
				if [ ! -f ${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz ]
				then
					echo "Error when creating \
						${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz"
					exit
				fi
			else # if [ "`which qsub 2> /dev/null`" == "" ]
				# The average is performed through the cluster
				${QSUB_CMD} \
					-hold_jid aladin_${CUR_IT}_${$} \
					-o ${RES_FOLDER}/aff_${CUR_IT} \
					-N avg_aff_${CUR_IT}_${$} \
					reg_average \
					${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz \
					-avg ${RES_FOLDER}/aff_${CUR_IT}/aff_res_*_it${CUR_IT}.nii*
			fi # if [ "`which qsub 2> /dev/null`" == "" ]
		fi # if [ "${CUR_IT}" != "${AFF_IT_NUM}" ]
	else # if [ ! -f ${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz ]
		echo "${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz already exists"
	fi # if [ ! -f ${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz ]
	# Update the average image used as a reference
	averageImage=${RES_FOLDER}/aff_${CUR_IT}/average_affine_it_${CUR_IT}.nii.gz
done # Loop over affine iteration


#############################################################################
#############################################################################
### Non rigid registration loop

for (( CUR_IT=1; CUR_IT<=${NRR_IT_NUM}; CUR_IT++ ))
do

	#############################
	# Check if the current average image has already been created
	if [ ! -f ${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz ]
	then

		#############################
		# Create a folder to store the current results
		if [ ! -d ${RES_FOLDER}/nrr_${CUR_IT} ]
		then
			mkdir ${RES_FOLDER}/nrr_${CUR_IT}
		fi

		#############################
		# Run the nonrigid registrations
		if [ "`which qsub 2> /dev/null`" == "" ]
		then
			for (( i=0 ; i<${IMG_NUMBER}; i++ ))
			do
				name=`basename ${IMG_INPUT[${i}]} .gz`
				name=`basename ${name} .nii`
				name=`basename ${name} .hdr`
				name=`basename ${name} .img`
				# Check if the registration has already been performed
				if [ ! -f ${RES_FOLDER}/nrr_${CUR_IT}/nrr_cpp_${name}_it${CUR_IT}.nii* ]
				then
					f3d_args=""
					# Check if a mask has been specified for the reference image
					if [ "${TEMPLATE_MASK}" != "" ]
					then
						f3d_args="${f3d_args} -rmask ${TEMPLATE_MASK}"
					fi
					# Check if a mask has been specified for the floating image
					if [ ${MASK_NUMBER} == ${IMG_NUMBER} ]
					then
						f3d_args="${f3d_args} -fmask ${IMG_INPUT_MASK[${i}]}"
					fi
					if [ ${AFF_IT_NUM} -gt 0 ]
					then
						f3d_args="${f3d_args} -aff \
							${RES_FOLDER}/aff_${AFF_IT_NUM}/aff_mat_${name}_it${AFF_IT_NUM}.txt"
					fi
					result="/dev/null"
					if [ "${CUR_IT}" == "${NRR_IT_NUM}" ]
					then
						result="${RES_FOLDER}/nrr_${CUR_IT}/nrr_res_${name}_it${CUR_IT}.nii.gz"
					fi
					# Perform the registration
					reg_f3d ${NRR_args} ${f3d_args} \
						-ref ${averageImage} \
						-flo ${IMG_INPUT[${i}]} \
						-cpp ${RES_FOLDER}/nrr_${CUR_IT}/nrr_cpp_${name}_it${CUR_IT}.nii.gz \
						-res ${result} > ${RES_FOLDER}/nrr_${CUR_IT}/nrr_log_${name}_it${CUR_IT}.txt
				fi
			done
		else
			echo \#\!/bin/sh > ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			# Define the current image index
			echo "img_number=\`expr \$SGE_TASK_ID - 1\`" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			echo ". `readlink -f $1`"  \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			# Extract the name of the file without the path and the extension
			echo "name=\`basename \${IMG_INPUT[\$img_number]} .gz\`" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			echo "name=\`basename \$name .nii\`" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			echo "name=\`basename \$name .hdr\`" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			echo "name=\`basename \$name .img\`" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			# Check that the registration has not already been performed
			echo "if [ ! -e ${RES_FOLDER}/nrr_${CUR_IT}/nrr_cpp_\${name}_it${CUR_IT}.nii* ]" >> \
				${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			echo "then" >> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			# Set up the registration argument
			echo "${NRR} ${NRR_args} \\" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			echo "-ref ${averageImage} \\" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			if [ "${TEMPLATE_MASK}" != "" ]; then
				echo "-rmask ${TEMPLATE_MASK} \\" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			fi
			echo "-flo \${IMG_INPUT[\$img_number]} \\" \
			>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			if [ ${AFF_IT_NUM} -gt 0 ]
			then
				echo "-aff ${RES_FOLDER}/aff_${AFF_IT_NUM}/aff_mat_\${name}_it${AFF_IT_NUM}.txt \\" >> \
					${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			fi
			if [ ${MASK_NUMBER} == ${IMG_NUMBER} ]; then
				echo "-fmask \${IMG_INPUT_MASK[\$img_number]} \\" \
				>> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			fi
			echo "-cpp ${RES_FOLDER}/nrr_${CUR_IT}/nrr_cpp_\${name}_it${CUR_IT}.nii.gz \\" >> \
				${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			result="/dev/null"
			if [ "${CUR_IT}" == "${NRR_IT_NUM}" ]
			then
				result="${RES_FOLDER}/nrr_${CUR_IT}/nrr_res_\${name}_it${CUR_IT}.nii.gz"
			fi
			echo "-res ${result}" >> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			echo "fi"  >> ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
			## Run the nonrigid registrations - Submit the job array
			${QSUB_CMD} \
				-hold_jid avg_aff_${AFF_IT_NUM}_${$},avg_nrr_`expr ${CUR_IT} - 1`_${$} \
				-o ${RES_FOLDER}/nrr_${CUR_IT} \
				-N f3d_${CUR_IT}_${$} \
				-t 1-${IMG_NUMBER} \
				sh ${RES_FOLDER}/nrr_${CUR_IT}/run_gw_niftyReg_f3d_${CUR_IT}_${$}.sh
		fi

		#############################
		# The transformation are demean'ed to create the average image
		# Note that this is not done for the last iteration step
		if [ "${CUR_IT}" != "${NRR_IT_NUM}" ]
		then
			list_average=""
			for img in ${IMG_INPUT[@]}
			do
				name=`basename ${img} .gz`
				name=`basename ${name} .nii`
				name=`basename ${name} .hdr`
				name=`basename ${name} .img`
				list_average="${list_average} \
					${RES_FOLDER}/aff_${AFF_IT_NUM}/aff_mat_${name}_it${AFF_IT_NUM}.txt \
					${RES_FOLDER}/nrr_${CUR_IT}/nrr_cpp_${name}_it${CUR_IT}.nii.gz ${img}"
			done
			if [ "`which qsub 2> /dev/null`" == "" ]
			then
				# The average is created on the host
				reg_average \
					${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz \
					-demean3 ${averageImage} \
					${list_average}
				if [ ! -f ${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz ]
				then
					echo "Error when creating \
						${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz"
					exit
				fi
			else # if [ "`which qsub 2> /dev/null`" == "" ]
				# The average is performed through the cluster
				${QSUB_CMD} \
					-hold_jid f3d_${CUR_IT}_${$} \
					-o ${RES_FOLDER}/nrr_${CUR_IT} \
					-N avg_nrr_${CUR_IT}_${$} \
					reg_average \
					${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz \
					-demean3 ${averageImage} \
					${list_average}
			fi # if [ "`which qsub 2> /dev/null`" == "" ]
		else # if [ "${CUR_IT}" != "${NRR_IT_NUM}" ]
			# All the result images are directly averaged during the last step
			if [ "`which qsub 2> /dev/null`" == "" ]
			then
				reg_average \
					${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz \
					-avg \
					`ls ${RES_FOLDER}/nrr_${CUR_IT}/nrr_res_*_it${CUR_IT}.nii*`
				if [ ! -f ${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz ]
				then
					echo "Error when creating \
						${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz"
					exit
				fi
			else # if [ "`which qsub 2> /dev/null`" == "" ]
				# The average is performed through the cluster
				${QSUB_CMD} \
					-hold_jid f3d_${CUR_IT}_${$} \
					-o ${RES_FOLDER}/nrr_${CUR_IT} \
					-N avg_nrr_${CUR_IT}_${$} \
					reg_average \
					${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz \
					-avg ${RES_FOLDER}/nrr_${CUR_IT}/nrr_res_*_it${CUR_IT}.nii*
			fi # if [ "`which qsub 2> /dev/null`" == "" ]
		fi # if [ "${CUR_IT}" != "${NRR_IT_NUM}" ]
	else # if [ ! -f ${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz ]
		echo "${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz already exists"
	fi # if [ ! -f ${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz ]
	# Update the average image
	averageImage=${RES_FOLDER}/nrr_${CUR_IT}/average_nonrigid_it_${CUR_IT}.nii.gz
done
#############################################################################
