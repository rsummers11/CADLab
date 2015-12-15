#The name of the project: python scripts expect "pyconvnet" for windows!
# requires Python 2.7 64-bit with Numpy and Matplotlib!
# recommended to use Anaconda python distribution: http://continuum.io/downloads
message(WARNING "requires Python 2.7 64-bit with Numpy and Matplotlib!")
message(WARNING "recommended to use Anaconda python distribution: http://continuum.io/downloads")

project(pyconvnet)
 
#The FindCUDA script is distributed since version 2.8
cmake_minimum_required(VERSION 2.8)

#Find Python
find_package(PythonLibs REQUIRED)
include_directories(
  ${PYTHON_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIR}/../Lib/site-packages/numpy/core/include/numpy
)
link_directories(
  ${PYTHON_INCLUDE_DIR}/../libs
)
#target_link_libraries(<your exe or lib> ${PYTHON_LIBRARIES})
 
#Find CUDA
find_package(CUDA)
mark_as_advanced(CUDA_SDK_ROOT_DIR)
if(CUDA_FOUND)
	# Display the CUDA version
	message(STATUS "The CUDA version you have is v${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
	# Compile an executable to check if there is at least one suitable graphical card
	try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
			${CMAKE_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}/cuda/checkCudaCard.cpp
			CMAKE_FLAGS
			-DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS}
			-DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
			COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
			RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR
	)
	# Check if the executable could not compile
	if(NOT COMPILE_RESULT_VAR)
		message(STATUS "The code to check the presence of a CUDA-enabled card failed.")
		message(STATUS "The USE_CUDA flag has been turned OFF.")
		set(USE_CUDA false)
	else(NOT COMPILE_RESULT_VAR)
		# Check if the executable return failure
		if(RUN_RESULT_VAR)
			message(STATUS "No CUDA-enabled card has been detected.")
			message(STATUS "Result message: ${RUN_RESULT_VAR}")
			message(STATUS "Error message: ${RUN_OUTPUT_VAR}")
			message(STATUS "The USE_CUDA flag has been turned OFF.")
			set(USE_CUDA false)
		else(RUN_RESULT_VAR)
			# Check if ptxas information should be displayed
			set(CUDA_CAPABILITY "" CACHE STRING "CUDA capability to use to generate the ptxas information (1.0, 2.0, 3.0) [0]")
			if(CUDA_CAPABILITY EQUAL "1.0")
				set(CUDA_NVCC_FLAGS
					"${CUDA_NVCC_FLAGS} --ptxas-options=-v --generate-code arch=compute_10,code=sm_10")
			elseif(CUDA_CAPABILITY EQUAL "2.0")
				set(CUDA_NVCC_FLAGS
					"${CUDA_NVCC_FLAGS} --ptxas-options=-v --generate-code arch=compute_20,code=sm_20")
			elseif(CUDA_CAPABILITY EQUAL "3.0")
				set(CUDA_NVCC_FLAGS
					"${CUDA_NVCC_FLAGS} --ptxas-options=-v --generate-code arch=compute_30,code=sm_30")
			endif(CUDA_CAPABILITY EQUAL "1.0")
			# The CUDA implementation is used
			add_definitions(-D_USE_CUDA)
			message(STATUS "Found CUDA and a CUDA-enabled card - the GPU code will be compiled.")
		endif(RUN_RESULT_VAR)
	endif(NOT COMPILE_RESULT_VAR)
else(CUDA_FOUND)
	message(FATAL_ERROR "CUDA not found. Please turn the USE_CUDA flag off or set the variables manually.")
endif(CUDA_FOUND)
	 
#cuda-convnet files
include_directories( 
  include
  include/common
  include/cudaconv2
  include/nvmatrix
  LIBS/include
  ${CUDA_SDK_ROOT_DIR}/common/inc
)
 
set(CUDA_FILES
  include/common/matrix.h
  include/common/matrix_funcs.h
  include/common/queue.h
  include/common/thread.h
  src/common/matrix.cpp
  src/convnet.cu
  src/cost.cu
  src/cudaconv2/conv_util.cu
  src/cudaconv2/filter_acts.cu
  src/cudaconv2/img_acts.cu
  src/cudaconv2/weight_acts.cu
  src/data.cu
  src/layer.cu
  src/layer_kernels.cu
  src/neuron.cu
  src/nvmatrix/nvmatrix.cu
  src/nvmatrix/nvmatrix_kernels.cu
  src/pyconvnet.cu
  src/util.cu
  src/weights.cu
  src/worker.cu
)

# DROPCONNECT modifications
set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost 1.54.0 COMPONENTS date_time filesystem system regex) # add components here if neccessary

include_directories( 
  include/tdata
  ${Boost_INCLUDE_DIRS}
)

set(DROPCONNECT_FILES
  include/dropc/dropc_dev.hpp
  include/dropc/dropc_bit_dev.hpp
  include/dropc/cuda_common.hpp
  src/dropc/dropc_dev.cu
  src/dropc/dropc_bit_dev.cu
  include/tdata/CImg.h
  include/tdata/cimg_extension.h
  include/tdata/preprocess.hpp
  src/tdata/preprocess.cpp
)

link_directories(
  ${CMAKE_SOURCE_DIR}/LIBS
  ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
)
set(TARGET_LIBRARIES
  libopenblas.lib
  pthreadVC2_x64.lib
  cublas.lib
  cudart.lib
  curand.lib
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
cuda_add_library(${PROJECT_NAME} SHARED ${CUDA_FILES} ${DROPCONNECT_FILES})
target_link_libraries(${PROJECT_NAME} ${TARGET_LIBRARIES} ${CUDA_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_NAME})

# Python requires dll to be name *.pyd (this is not supported by cmake and therefore have to manually rename the resulting dll to a pyd file as a post-build event)
message(WARNING "RENAME_DLL is currently hardcoded for Release mode!")
add_custom_target(RENAME_DLL ALL
	DEPENDS ${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
        "${LIBRARY_OUTPUT_PATH}/Release/${PROJECT_NAME}.dll" # <--this is in-file
        "${CMAKE_SOURCE_DIR}/${PROJECT_NAME}.pyd" # <--this is out-file path		
)
		
# if you need to pass options/flags
#set( custom_cuda_flags "-arch=sm_20" CACHE STRING "Set the cuda flags like --arch=sm_20 --ptxas-options=-v #--use_fast_math")
#list(APPEND CUDA_NVCC_FLAGS ${custom_cuda_flags})
 
#list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-some flag)
#here -Xcompiler means that the flag that fallow is for GCC and not nvcc

### ITK-depending command line apps ###
find_package(ITK REQUIRED) 
if( ITK_FOUND )
  include( ${ITK_USE_FILE} ) 
endif( ITK_FOUND )

set(ITK_APP_DIR "ITKapps")
# Libs
set( LIB_SRC 
	${ITK_APP_DIR}/Libs/itkIsotropicResampleImageFilter.h
	${ITK_APP_DIR}/Libs/itkIsotropicWindowingResampleImageFilter.h
	${ITK_APP_DIR}/Libs/nihHelperFunctions.h
)
include_directories( ${ITK_APP_DIR}/Libs )

# add the executables
set(APP_NAME itkDicomSeriesReadImageWrite2)
add_executable (${APP_NAME} ${ITK_APP_DIR}/${APP_NAME}.cxx)
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})
install(TARGETS ${APP_NAME} DESTINATION ${PROJECT_NAME})

set(APP_NAME itkConvertIndexToPhysicalPoints)
add_executable (${APP_NAME} ${ITK_APP_DIR}/${APP_NAME}.cxx)
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})
install(TARGETS ${APP_NAME} DESTINATION ${PROJECT_NAME})

set(APP_NAME itkResampleRegionOfInterestFromCoordinateList)
add_executable (${APP_NAME} ${ITK_APP_DIR}/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})
install(TARGETS ${APP_NAME} DESTINATION ${PROJECT_NAME})

set(APP_NAME itkGetRegionOfInterestCentersFromList)
add_executable (${APP_NAME} ${ITK_APP_DIR}/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})
install(TARGETS ${APP_NAME} DESTINATION ${PROJECT_NAME})

set(APP_NAME itkReadWriteImage)
add_executable (${APP_NAME} ${ITK_APP_DIR}/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})
install(TARGETS ${APP_NAME} DESTINATION ${PROJECT_NAME})

### PACKAGING ###
set(CPACK_GENERATOR "ZIP")
#set(CPACK_DEBIAN_PACKAGE_MAINTAINER "David Doria") #required
 
include(CPack)

