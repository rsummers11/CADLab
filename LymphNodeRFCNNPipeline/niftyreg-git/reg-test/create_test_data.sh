#!/bin/sh

function display_usage()
{
	echo "usage $0\n\t<referenceImage2D> <floatingImage2D>\n\t<referenceImage3D> <floatingImage3D>\n\t<code_git_hashkey> <outputFolder>"
	exit 1
}

function file_exists()
{
	local f=$1
	if [ -f ${f} ]
	then
		return 0
	else
		echo "[ERROR] File ${f} does not exist"
		exit 1
	fi
}

function folder_exists()
{
	local f=$1
	if [ -d ${f} ]
	then
		return 0
	else
		echo "[ERROR] Folder ${f} does not exist"
		exit 1
	fi
}

function file_zipped()
{
	local f=$1
	if [ "`echo ${f} | awk -F . '{print $NF}`" == "gz" ]
	then
		return 0
	else
		return 1
	fi
}

function copy_compress()
{
	local input=$1
	local output=$2
	local temp_folder=$3
	if [ "`echo ${input} | awk -F . '{print $NF}'`" == "gz" ]
	then
		cp ${input} ${output}
	else
		cp ${input} ${temp_folder}/temp.nii
		gzip ${temp_folder}/temp.nii
		mv ${temp_folder}/temp.nii.gz ${output}
		rm ${temp_folder}/temp.nii.gz
	fi
}

## MAIN

CURRENTDIR=`pwd`

# Check the number of argument
if [ $# -lt 6 ]
then
	display_usage
fi

# Check if the specified files exist
file_exists $1
file_exists $2
file_exists $3
file_exists $4
folder_exists $6

# Extract the full path of the input images
input_ref2D="$(cd "$(dirname "${1}")"; pwd)/`basename ${1}`"
input_flo2D="$(cd "$(dirname "${2}")"; pwd)/`basename ${2}`"
input_ref3D="$(cd "$(dirname "${3}")"; pwd)/`basename ${3}`"
input_flo3D="$(cd "$(dirname "${4}")"; pwd)/`basename ${4}`"
input_output="$(cd ${6}; pwd)"

echo "Input reference file 2D ${input_ref2D}"
echo "Input floating file 2D ${input_flo2D}"
echo "Input reference file 3D ${input_ref3D}"
echo "Input floating file 3D ${input_flo3D}"
echo "Output folder ${input_output}"

# Create a temp folder
TEMPFOLDER=""
if [ "${TMPDIR}" != "" ]
then
	TEMPFOLDER=${TMPDIR}/temp_nr_test_$$
else
	TEMPFOLDER=/tmp/temp_nr_test_$$
fi
mkdir ${TEMPFOLDER}


# Create a temporary folder to host the data
mkdir ${TEMPFOLDER}/niftyreg-git_test
cd ${TEMPFOLDER}/niftyreg-git_test

# Copy, rename and compress if required the input images
refImg2D=${TEMPFOLDER}/niftyreg-git_test/refImg2D.nii.gz
floImg2D=${TEMPFOLDER}/niftyreg-git_test/floImg2D.nii.gz
refImg3D=${TEMPFOLDER}/niftyreg-git_test/refImg3D.nii.gz
floImg3D=${TEMPFOLDER}/niftyreg-git_test/floImg3D.nii.gz
copy_compress ${input_ref2D} ${refImg2D} ${TEMPFOLDER}/niftyreg-git_test
copy_compress ${input_flo2D} ${floImg2D} ${TEMPFOLDER}/niftyreg-git_test
copy_compress ${input_ref3D} ${refImg3D} ${TEMPFOLDER}/niftyreg-git_test
copy_compress ${input_flo3D} ${floImg3D} ${TEMPFOLDER}/niftyreg-git_test

# Checkout the code
git clone git://git.code.sf.net/p/niftyreg/git ${TEMPFOLDER}/niftyreg-git
cd ${TEMPFOLDER}/niftyreg-git
git checkout -b $5

NR_OPENMP=OFF
NR_SSE=ON
NR_BUILD=Debug

# Build the code
mkdir ${TEMPFOLDER}/niftyreg-git_build
cd ${TEMPFOLDER}/niftyreg-git_build
cmake \
	-D BUILD_ALL_DEP=ON \
	-D BUILD_TESTING=OFF \
	-D CMAKE_BUILD_TYPE=${NR_BUILD} \
	-D USE_OPENMP=${NR_OPENMP} \
	-D USE_SSE=${NR_SSE} \
	-D USE_CUDA=OFF \
	${TEMPFOLDER}/niftyreg-git
make

# Define some 2D variables
affine_mat2D=${TEMPFOLDER}/niftyreg-git_test/affine_mat2D.txt
affine_def2D=${TEMPFOLDER}/niftyreg-git_test/affine_def2D.nii.gz
nonlin_cpp2D=${TEMPFOLDER}/niftyreg-git_test/nonlin_cpp2D.nii.gz
nonlin_vel2D=${TEMPFOLDER}/niftyreg-git_test/nonlin_vel2D.nii.gz
nonlin_def2D=${TEMPFOLDER}/niftyreg-git_test/nonlin_def2D.nii.gz
nonlin_com2D=${TEMPFOLDER}/niftyreg-git_test/nonlin_com2D.nii.gz
warped_nea2D=${TEMPFOLDER}/niftyreg-git_test/warped_nea2D.nii.gz
warped_lin2D=${TEMPFOLDER}/niftyreg-git_test/warped_lin2D.nii.gz
warped_cub2D=${TEMPFOLDER}/niftyreg-git_test/warped_cub2D.nii.gz
warped_sin2D=${TEMPFOLDER}/niftyreg-git_test/warped_sin2D.nii.gz

# Define some 3D variables
affine_mat3D=${TEMPFOLDER}/niftyreg-git_test/affine_mat3D.txt
affine_def3D=${TEMPFOLDER}/niftyreg-git_test/affine_def3D.nii.gz
nonlin_cpp3D=${TEMPFOLDER}/niftyreg-git_test/nonlin_cpp3D.nii.gz
nonlin_vel3D=${TEMPFOLDER}/niftyreg-git_test/nonlin_vel3D.nii.gz
nonlin_def3D=${TEMPFOLDER}/niftyreg-git_test/nonlin_def3D.nii.gz
nonlin_com3D=${TEMPFOLDER}/niftyreg-git_test/nonlin_com3D.nii.gz
warped_nea3D=${TEMPFOLDER}/niftyreg-git_test/warped_nea3D.nii.gz
warped_lin3D=${TEMPFOLDER}/niftyreg-git_test/warped_lin3D.nii.gz
warped_cub3D=${TEMPFOLDER}/niftyreg-git_test/warped_cub3D.nii.gz
warped_sin3D=${TEMPFOLDER}/niftyreg-git_test/warped_sin3D.nii.gz

# Run affine registrations
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_aladin \
	-ref ${refImg2D} -flo ${floImg2D} -res /dev/null -aff ${affine_mat2D} &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_aladin \
	-ref ${refImg3D} -flo ${floImg3D} -res /dev/null -aff ${affine_mat3D} &> /dev/null
# Create affine deformation fields
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_transform \
	-ref ${refImg2D} -def ${affine_mat2D} ${affine_def2D} &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_transform \
	-ref ${refImg3D} -def ${affine_mat3D} ${affine_def3D} &> /dev/null
# Run nonlinear registrations - f3d
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_f3d \
	-ref ${refImg2D} -flo ${floImg2D} -aff ${affine_mat2D} -res /dev/null \
	-cpp ${nonlin_cpp2D} &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_f3d \
	-ref ${refImg3D} -flo ${floImg3D} -aff ${affine_mat3D} -res /dev/null \
	-cpp ${nonlin_cpp3D} &> /dev/null
# Run nonlinear registrations - f3d2
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_f3d -vel \
	-ref ${refImg2D} -flo ${floImg2D} -aff ${affine_mat2D} -res /dev/null \
	-cpp ${nonlin_vel2D} &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_f3d -vel \
	-ref ${refImg3D} -flo ${floImg3D} -aff ${affine_mat3D} -res /dev/null \
	-cpp ${nonlin_vel3D} &> /dev/null
# Generate nonlinear deformation fields
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_transform \
	-ref ${refImg2D} -def ${nonlin_cpp2D} ${nonlin_def2D} &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_transform \
	-ref ${refImg3D} -def ${nonlin_cpp3D} ${nonlin_def3D} &> /dev/null
# Generate composed transformations
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_transform \
	-comp ${nonlin_def2D} ${nonlin_def2D} ${nonlin_com2D} &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_transform \
	-comp ${nonlin_def3D} ${nonlin_def3D} ${nonlin_com3D} &> /dev/null
# Resample 2D image
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg2D} -flo ${floImg2D} -trans ${nonlin_def2D} \
	-res ${warped_nea2D} -inter 0 &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg2D} -flo ${floImg2D} -trans ${nonlin_def2D} \
	-res ${warped_lin2D} -inter 1 &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg2D} -flo ${floImg2D} -trans ${nonlin_def2D} \
	-res ${warped_cub2D} -inter 3 &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg2D} -flo ${floImg2D} -trans ${nonlin_def2D} \
	-res ${warped_sin2D} -inter 4 &> /dev/null
# Resample 3D image
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg3D} -flo ${floImg3D} -trans ${nonlin_def3D} \
	-res ${warped_nea3D} -inter 0 &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg3D} -flo ${floImg3D} -trans ${nonlin_def3D} \
	-res ${warped_lin3D} -inter 1 &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg3D} -flo ${floImg3D} -trans ${nonlin_def3D} \
	-res ${warped_cub3D} -inter 3 &> /dev/null
${TEMPFOLDER}/niftyreg-git_build/reg-apps/reg_resample \
	-ref ${refImg3D} -flo ${floImg3D} -trans ${nonlin_def3D} \
	-res ${warped_sin3D} -inter 4 &> /dev/null

# Compress all images into a single tar ball
cd ${TEMPFOLDER}/niftyreg-git_test

echo "test data for niftyreg" > README.txt
echo "USE_OPENMP set to ${NR_OPENMP}" >> README.txt
echo "USE_SSE set to ${NR_SSE}" >> README.txt
echo "CMAKE_BUILD_TYPE set to ${NR_BUILD}" >> README.txt
echo "Git hash key ${5}" >> README.txt
echo "os ${OS}" >> README.txt
echo "`cmake --version`" >> README.txt
c_compiler=`cat ${TEMPFOLDER}/niftyreg-git_build/CMakeCache.txt | grep CMAKE_C_COMPILER | awk -F = '{print $2}'`
echo "C compiler: `${c_compiler} --version | head -n 1`" >> README.txt
cxx_compiler=`cat ${TEMPFOLDER}/niftyreg-git_build/CMakeCache.txt | grep CMAKE_CXX_COMPILER | awk -F = '{print $2}'`
echo "CXX compiler: `${cxx_compiler} --version | head -n 1`" >> README.txt


tar czvf ${input_output}/niftyreg_test_data.tar.gz \
	$(basename ${refImg2D}) $(basename ${floImg2D}) \
	$(basename ${refImg3D}) $(basename ${floImg3D}) \
	$(basename ${affine_mat2D}) $(basename ${affine_mat3D}) \
	$(basename ${affine_def2D}) $(basename ${affine_def3D}) \
	$(basename ${nonlin_cpp2D}) $(basename ${nonlin_cpp3D}) \
	$(basename ${nonlin_vel2D}) $(basename ${nonlin_vel3D}) \
	$(basename ${nonlin_def2D}) $(basename ${nonlin_def3D}) \
	$(basename ${nonlin_com2D}) $(basename ${nonlin_com3D}) \
	$(basename ${warped_nea2D}) $(basename ${warped_nea3D}) \
	$(basename ${warped_lin2D}) $(basename ${warped_lin3D}) \
	$(basename ${warped_cub2D}) $(basename ${warped_cub3D}) \
	$(basename ${warped_sin2D}) $(basename ${warped_sin3D}) \
	README.txt
	 
cd ${CURRENTDIR}

rm -rf ${TEMPFOLDER}/niftyreg-git
rm -rf ${TEMPFOLDER}/niftyreg-git_test
rm -rf ${TEMPFOLDER}/niftyreg-git_build