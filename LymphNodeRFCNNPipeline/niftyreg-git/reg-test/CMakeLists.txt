#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
	# Build the coverage test
	option(WITH_COVERAGE "Set up the C, CXX and linker flags to run the coverage test" OFF)
	if(WITH_COVERAGE)
		set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY_ONCE TRUE)
		set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
				"Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
				FORCE)
		set(CMAKE_C_FLAGS_DEBUG
				"-g -O0 -Wall -W -fprofile-arcs -ftest-coverage"
				CACHE STRING "Force the debug C flags for the coverage test" FORCE)
		set(CMAKE_CXX_FLAGS_DEBUG
				"-g -O0 -Wall -W -Wunused-variable -Wunused-parameter -Wunused-function -Wunused -Wno-system-headers -Wno-deprecated -Woverloaded-virtual -Wwrite-strings -fprofile-arcs -ftest-coverage"
				CACHE STRING "Force the debug CXX flags for the coverage test" FORCE)
		set(CMAKE_EXE_LINKER_FLAGS_DEBUG
				"-fprofile-arcs -ftest-coverage"
				CACHE STRING "Force the debug linker flags for the coverage test" FORCE)
		set(BUILD_ALL_DEP ON CACHE BOOL "All the dependencies are build" FORCE)
		configure_file(${CMAKE_SOURCE_DIR}/reg-test/CTestCustom.cmake.in
									 ${CMAKE_BINARY_DIR}/CTestCustom.cmake)
	endif(WITH_COVERAGE)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
	# Set the build name
	unset(BUILDNAME CACHE)
	set(CTEST_BUILD_NAME "${CMAKE_SYSTEM}_${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}_cmake-${CMAKE_VERSION}_${CMAKE_BUILD_TYPE}")
	if(USE_SSE)
		set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}_sse")
	endif(USE_SSE)
	if(USE_OPENMP)
		set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}_openmp")
	endif(USE_OPENMP)
	if(USE_CUDA)
		set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}_cuda-${CUDA_VERSION}")
	endif(USE_CUDA)
	unset(BUILDNAME)
	set(BUILDNAME ${CTEST_BUILD_NAME} CACHE STRING "Build name variable for CDash" FORCE)
	mark_as_advanced(BUILDNAME)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
	# The test data are dowloaded
	set(DATA_FOLDER ${CMAKE_BINARY_DIR}/reg-test/test-data)
	if(NOT EXISTS "${DATA_FOLDER}/test_data.tar.gz")
		file(MAKE_DIRECTORY ${DATA_FOLDER})
		set(test_data_url "http://www.cs.ucl.ac.uk/staff/mmodat/niftyreg_test_data.tar.gz")
		message(STATUS "Downloading the test data: ${test_data_url}")
		file(DOWNLOAD
				 ${test_data_url}
				 "${DATA_FOLDER}/test_data.tar.gz"
				 SHOW_PROGRESS STATUS download_status)
		list(GET download_status 0 test_data_download_status)
		if(${test_data_download_status})
			message(WARNING "Error when downloading the test data, forcing BUILD_TESTING to OFF")
			set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
			endif(${test_data_download_status})
	endif(NOT EXISTS "${DATA_FOLDER}/test_data.tar.gz")

	if(NOT EXISTS "${DATA_FOLDER}/refImg2D.nii.gz")
		message(STATUS "Extracting test data in ${DATA_FOLDER}")
		execute_process(
			COMMAND ${CMAKE_COMMAND} -E tar xzf ${DATA_FOLDER}/test_data.tar.gz
			WORKING_DIRECTORY ${DATA_FOLDER}
			ERROR_VARIABLE test_data_extract_status
		)
		if(NOT ${test_data_extract_status} STREQUAL "")
			message(WARNING "Error when extracting the test data, forcing BUILD_TESTING to OFF")
			set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
		endif(NOT ${test_data_extract_status} STREQUAL "")
	endif(NOT EXISTS "${DATA_FOLDER}/refImg2D.nii.gz")
#-----------------------------------------------------------------------------
	set(EXEC reg_test_affine_deformation_field)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/affine_mat2D.txt ${DATA_FOLDER}/affine_def2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt ${DATA_FOLDER}/affine_def3D.nii.gz)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_nonlinear_deformation_field)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/nonlin_cpp2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/nonlin_cpp3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_interpolation)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_0_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_nea2D.nii.gz 0)
	add_test(${EXEC}_0_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_nea3D.nii.gz 0)
	add_test(${EXEC}_1_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_lin2D.nii.gz 1)
	add_test(${EXEC}_1_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_lin3D.nii.gz 1)
	add_test(${EXEC}_3_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_cub2D.nii.gz 3)
	add_test(${EXEC}_3_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_cub3D.nii.gz 3)
	add_test(${EXEC}_4_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz 4)
	add_test(${EXEC}_4_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz 4)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_compose_deformation_field)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/nonlin_com2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/nonlin_com3D.nii.gz)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_fullAffine)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_aladin)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/affine_mat2D.txt)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_fullNonlinear)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/affine_mat2D.txt ${DATA_FOLDER}/nonlin_cpp2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt ${DATA_FOLDER}/nonlin_cpp3D.nii.gz)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_fullSymNonlinear)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/affine_mat2D.txt ${DATA_FOLDER}/nonlin_vel2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt ${DATA_FOLDER}/nonlin_vel3D.nii.gz)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_matrix_operation)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_maths reg_nifti z)
	add_test(${EXEC} ${EXEC})
#-----------------------------------------------------------------------------
	set(EXEC reg_test_svd)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_maths)
	add_test(${EXEC} ${EXEC})
#-----------------------------------------------------------------------------
	set(EXEC reg_test_measure)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_LNCC_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz LNCC)
	add_test(${EXEC}_NMI_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz NMI)
	add_test(${EXEC}_SSD_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz SSD)
	add_test(${EXEC}_LNCC_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz LNCC)
	add_test(${EXEC}_NMI_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz NMI)
	add_test(${EXEC}_SSD_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz SSD)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

