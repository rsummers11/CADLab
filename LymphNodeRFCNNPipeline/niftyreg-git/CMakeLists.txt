project(NiftyReg)
#-----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8.8)
if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.8$")
 mark_as_advanced(FORCE CMAKE_BACKWARDS_COMPATIBILITY)
else("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.8$")
 mark_as_advanced(CLEAR CMAKE_BACKWARDS_COMPATIBILITY)
endif("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.8$")

#-----------------------------------------------------------------------------
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message("In-source builds not allowed by NiftyReg police.")
	message("Please create a new directory (called a build directory) and run CMake from there.")
	message(FATAL_ERROR "You may need to remove CMakeCache.txt and CMakeFiles.")
endif(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})

#-----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
#-----------------------------------------------------------------------------
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(NOT cmake_build_type_tolower STREQUAL "debug"
	 AND NOT cmake_build_type_tolower STREQUAL "release"
	 AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
	message("Unknown build type \"${CMAKE_BUILD_TYPE}\".")
	message(FATAL_ERROR "Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif(NOT cmake_build_type_tolower STREQUAL "debug"
	 AND NOT cmake_build_type_tolower STREQUAL "release"
	 AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
if(cmake_build_type_tolower STREQUAL "debug")
	set(DEBUG_MODE ON)
elseif(cmake_build_type_tolower STREQUAL "release")
	set(DEBUG_MODE OFF)
endif(cmake_build_type_tolower STREQUAL "debug")

#-----------------------------------------------------------------------------
set(NiftyReg_VERSION_MAJOR 1)
set(NiftyReg_VERSION_MINOR 4)
set(NiftyReg_VERSION_PATCH 0)

#-----------------------------------------------------------------------------
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
		add_definitions(/W1)
else(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
		add_definitions(-fPIC)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

#-----------------------------------------------------------------------------
option(BUILD_TESTING "To build the unit tests" OFF)
option(BUILD_NR_SLICER_EXT "Where NiftyReg meets 3DSlicer" ${NiftyRegExtension_BUILD_SLICER_EXTENSION})
mark_as_advanced(BUILD_NR_SLICER_EXT)
option(USE_SSE "To enable SEE computation in some case" ON)
option(USE_DOUBLE "To force double precision instead of single precision" OFF)
option(USE_OPENMP "To use openMP for multi-CPU processing" ON)
option(BUILD_SHARED_LIBS "Build the libraries as shared" OFF)
option(USE_CUDA "To enable CUDA for a GPU implementation of the code" OFF)
option(BUILD_ALL_DEP "All the dependencies are build" OFF)

#-----------------------------------------------------------------------------
if(WIN32)
		set(BUILD_ALL_DEP ON CACHE BOOL "All the dependencies are build" FORCE)
endif(WIN32)

#-----------------------------------------------------------------------------
# All dependencies are build to create the 3DSlicer package
if(BUILD_NR_SLICER_EXT)
		set(BUILD_ALL_DEP ON)
		mark_as_advanced(FORCE BUILD_ALL_DEP)
else(BUILD_NR_SLICER_EXT)
		mark_as_advanced(CLEAR BUILD_ALL_DEP)
endif(BUILD_NR_SLICER_EXT)

#-----------------------------------------------------------------------------
# Z library
# Try first to find the z library on the system and built is from the sources if it can not be find
if(NOT BUILD_ALL_DEP)
		find_package(ZLIB)
		if(ZLIB_FOUND)
				include_directories(${ZLIB_INCLUDE_DIR})
						message(STATUS "Found zlib - the z library will not be built")
		else(ZLIB_FOUND)
				include_directories(${CMAKE_SOURCE_DIR}/reg-io/zlib)
				message(STATUS "zlib not found - the z library will be built")
		endif(ZLIB_FOUND)
else(NOT BUILD_ALL_DEP)
		include_directories(${CMAKE_SOURCE_DIR}/reg-io/zlib)
endif(NOT BUILD_ALL_DEP)

#-----------------------------------------------------------------------------
# Try to find the png library and header on the system
if(NOT BUILD_ALL_DEP)
		## PNG support - First try to find the PNG library on the system and build it if it is not found
		## I did not use the FindPNG.cmake here as the zlib is also included into the project
		if(CYGWIN)
				if(NOT BUILD_SHARED_LIBS)
						set (PNG_DEFINITIONS -DPNG_STATIC)
				endif(NOT BUILD_SHARED_LIBS)
		endif(CYGWIN)
		set(PNG_NAMES ${PNG_NAMES} png libpng png15 libpng15 png15d libpng15d png14 libpng14 png14d libpng14d png12 libpng12 png12d libpng12d)
		find_library(PNG_LIBRARY NAMES ${PNG_NAMES})
		find_path(PNG_INCLUDE_DIR png.h
				/usr/local/include/libpng
				/sw/include
		)
		# If the png library and header can not be found, it is build from the sources
		if(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
				message(STATUS "libpng not found - the png library will be built")
				set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/reg-io/png/lpng1510)
				set(PNG_LIBRARY png)
				set(BUILD_INTERNAL_PNG true)
		else(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
				message(STATUS "Found libpng - the png library will not be built")
				set(BUILD_INTERNAL_PNG false)
		endif(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
else(NOT BUILD_ALL_DEP)
		set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/reg-io/png/lpng1510)
		set(PNG_LIBRARY png)
endif(NOT BUILD_ALL_DEP)
include_directories(${CMAKE_SOURCE_DIR}/reg-io/png)
include_directories(${PNG_INCLUDE_DIR})

#-----------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/reg-lib)
include_directories(${CMAKE_SOURCE_DIR}/reg-io)
include_directories(${CMAKE_SOURCE_DIR}/reg-io/nifti)
include_directories(${CMAKE_SOURCE_DIR}/third-party)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/reg-io/nrrd)
include_directories(${CMAKE_SOURCE_DIR}/reg-io/nrrd/NrrdIO)

#-----------------------------------------------------------------------------
if(USE_DOUBLE)
		if(USE_SSE)
				# Double can not be used with SSE as it has only been implemented for single precision
				message("Can not use double precision and SSE implementation concurrenlty")
				message(FATAL_ERROR "Please turn USE_SSE OFF to use double precision")
		endif(USE_SSE)
		if(USE_CUDA)
				# Double can not be used with CUDA
				message("Can not use double precision and CUDA implementation concurrenlty")
				message(FATAL_ERROR "Please turn USE_CUDA OFF to use double precision")
		endif(USE_CUDA)
		add_definitions(-D_USE_NR_DOUBLE)
endif(USE_DOUBLE)

#-----------------------------------------------------------------------------
if(USE_SSE)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
		add_definitions(-D_USE_SSE)
endif(USE_SSE)

#-----------------------------------------------------------------------------
if(USE_OPENMP)
	find_package(OpenMP)
	if(NOT OPENMP_FOUND)
		set(USE_OPENMP OFF CACHE BOOL "To use openMP for multi-CPU processing" FORCE)
		message(WARNING "OpenMP does not appear to be supported by your compiler, forcing USE_OPENMP to OFF")
	else(NOT OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif(NOT OPENMP_FOUND)
endif(USE_OPENMP)

#-----------------------------------------------------------------------------
if(BUILD_SHARED_LIBS)
	if(USE_CUDA)
		 set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build the libraries as shared." FORCE)
		 message(WARNING "CUDA is not compatible with shared libraries. Forcing BUILD_SHARED_LIBS to OFF")
		 set(NIFTYREG_LIBRARY_TYPE STATIC)
	else(USE_CUDA)
		set(NIFTYREG_LIBRARY_TYPE SHARED)
	endif(USE_CUDA)
else(BUILD_SHARED_LIBS)
	set(NIFTYREG_LIBRARY_TYPE STATIC)
endif(BUILD_SHARED_LIBS)

#-----------------------------------------------------------------------------
if(USE_CUDA)
	# Check if the CUDA driver are available
	find_package(CUDA)
	mark_as_advanced(CUDA_SDK_ROOT_DIR)

	if(CUDA_FOUND)
			# Display the CUDA version
		message(STATUS "The CUDA version you have is v${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
		# Compile an executable to check if there is at least one suitable graphical card
		try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cuda/checkCudaCard.cpp
				CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS} -DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
				COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
				RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR
		)
		# Check if the executable could not compile
		if(NOT COMPILE_RESULT_VAR)
			message(WARNING "The code to check the presence of a CUDA-enabled card failed.")
			message(WARNING "The USE_CUDA flag has been turned OFF.")
			set(USE_CUDA OFF CACHE BOOL "To enable CUDA for a GPU implementation of the code" FORCE)
				# Check if the executable return failure
		elseif(RUN_RESULT_VAR)
			message(WARNING "No CUDA-enabled card has been detected\nResult code: ${RUN_RESULT_VAR}\nError message: ${RUN_OUTPUT_VAR}\nThe USE_CUDA flag has been turned OFF.")
			set(USE_CUDA OFF CACHE BOOL "To enable CUDA for a GPU implementation of the code" FORCE)
		else(NOT COMPILE_RESULT_VAR)
			message(STATUS "Found CUDA and a CUDA-enabled card:  ${RUN_OUTPUT_VAR}")
				set(CUDA_NVCC_FLAGS "")

				#check cuda version and adjust compile flags
				if(RUN_OUTPUT_VAR EQUAL "1.0")
					set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_10,code=sm_10")
				elseif(RUN_OUTPUT_VAR EQUAL "2.0")
					set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20")
				elseif(RUN_OUTPUT_VAR EQUAL "3.0")
					set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
				elseif(RUN_OUTPUT_VAR EQUAL "3.5")
						set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
				elseif(RUN_OUTPUT_VAR EQUAL "5.0")
						set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
				endif(RUN_OUTPUT_VAR EQUAL "1.0")

				#adjust for debug and release versions
				if(DEBUG_MODE)
					#common debug flags for all architectures
					set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v -g -G -lineinfo")
				else(DEBUG_MODE)
						set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -use_fast_math -O3 -Xcompiler")
				endif(DEBUG_MODE)

				message(STATUS "NVCC Debug: ${DEBUG_MODE} | NVCC Flags: ${CUDA_NVCC_FLAGS}")
				# The CUDA implementation is used
				add_definitions(-D_USE_CUDA)
			endif(NOT COMPILE_RESULT_VAR)
		else(CUDA_FOUND)
		set(USE_CUDA OFF CACHE BOOL "To enable CUDA for a GPU implementation of the code" FORCE)
		message(WARNING "CUDA is not compatible with shared libraries. Forcing BUILD_SHARED_LIBS to OFF")
	endif(CUDA_FOUND)
endif(USE_CUDA)

#-----------------------------------------------------------------------------
find_package(Git)
if(GIT_FOUND)
	execute_process(
		COMMAND git rev-parse HEAD
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE GIT_COMMIT_HASH
		RESULT_VARIABLE GIT_FAILED
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	if(NOT GIT_FAILED)
		 add_definitions(-D_GIT_HASH=\"${GIT_COMMIT_HASH}\")
		 message(STATUS "Found Git - the hash is added to the source.")
	else(NOT GIT_FAILED)
		 add_definitions(-D_GIT_HASH="${NiftyReg_VERSION_MAJOR}.${NiftyReg_VERSION_MINOR}.${NiftyReg_VERSION_PATCH}")
	endif(NOT GIT_FAILED)
else(GIT_FOUND)
	 add_definitions(-D_GIT_HASH="${NiftyReg_VERSION_MAJOR}.${NiftyReg_VERSION_MINOR}.${NiftyReg_VERSION_PATCH}")
endif(GIT_FOUND)

#-----------------------------------------------------------------------------
subdirs(reg-io)
subdirs(reg-lib)
subdirs(reg-apps)
subdirs(third-party)
if(BUILD_TESTING)
	enable_testing()
	include(${CMAKE_ROOT}/Modules/Dart.cmake)
	subdirs(reg-test)
endif(BUILD_TESTING)

#-----------------------------------------------------------------------------
# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
		add_custom_target(doc
				${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
				WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
				COMMENT "Generating API documentation with Doxygen" VERBATIM
		)
		message(STATUS "Found doxygen")
endif(DOXYGEN_FOUND)

#-----------------------------------------------------------------------------
# The package is only build if 3DSlicer package is not build as well
if(NOT BUILD_NR_SLICER_EXT)
		# Set some variable about the package
		set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_NAME}: Tools for efficient medical image registration.")
		set(CPACK_PACKAGE_VENDOR "Marc Modat (UCL)")
		set(CPACK_PACKAGE_VERSION_MAJOR "${NiftyReg_VERSION_MAJOR}")
		set(CPACK_PACKAGE_VERSION_MINOR "${NiftyReg_VERSION_MINOR}")
		set(CPACK_PACKAGE_VERSION_PATCH "${NiftyReg_VERSION_PATCH}")
		set(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")

		# Check if the system processor is defined
		if(CMAKE_SYSTEM_PROCESSOR MATCHES "unknown")
			SET (CMAKE_SYSTEM_PROCESSOR "x86")
		endif(CMAKE_SYSTEM_PROCESSOR MATCHES "unknown")
		if(NOT DEFINED CPACK_SYSTEM_NAME)
			set(CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE})
		endif(NOT DEFINED CPACK_SYSTEM_NAME)

		if(WIN32 AND NOT UNIX)
			# Set some variable for a windows-based package
			if(CMAKE_CL_64)
				set(CPACK_SYSTEM_NAME Win64-${CMAKE_SYSTEM_PROCESSOR})
			else(CMAKE_CL_64)
				set(CPACK_SYSTEM_NAME Win32-${CMAKE_SYSTEM_PROCESSOR})
			endif(CMAKE_CL_64)
			set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}\\\\README.txt")
			set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}\\\\LICENSE.txt")
		#  set(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/logo\\\\nifty_reg_logo_128.png")
			set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\${PROJECT_NAME}.exe")
			set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} ${PROJECT_NAME}")
			set(CPACK_NSIS_HELP_LINK "https:\\\\\\\\sourceforge.net/projects/niftyreg/")
			set(CPACK_NSIS_URL_INFO_ABOUT "https:\\\\\\\\sourceforge.net/projects/niftyreg/")
			set(CPACK_NSIS_CONTACT "m.modat@ucl.ac.uk")
			set(CPACK_NSIS_MODIFY_PATH ON)
		else(WIN32 AND NOT UNIX)
			# Set some variables for a unix-based package
			set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
			set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
		#  set(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/logo/nifty_reg_logo_128.png")
			set(CPACK_HELP_LINK "https:/sourceforge.net/projects/niftyreg/")
			set(CPACK_URL_INFO_ABOUT "https:/sourceforge.net/projects/niftyreg/")
			set(CPACK_STRIP_FILES "bin/${PROJECT_NAME}")
			set(CPACK_SOURCE_STRIP_FILES OFF)
			set(CPACK_PACKAGE_CONTACT "m.modat@ucl.ac.uk")
			if(APPLE)
				# MacOs specific variables
#        set(CPACK_PACKAGING_INSTALL_PREFIX "usr")
				set(CPACK_POSTFLIGHT_SCRIPT ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}SetEnvUnix.sh)
				if(${CPACK_SYSTEM_NAME} MATCHES Darwin AND CMAKE_OSX_ARCHITECTURES)
					list(LENGTH CMAKE_OSX_ARCHITECTURES _length)
					if(_length GREATER 1)
						set(CPACK_SYSTEM_NAME Darwin-Universal-${CMAKE_BUILD_TYPE})
					else(_length GREATER 1)
						set(CPACK_SYSTEM_NAME Darwin-${CMAKE_OSX_ARCHITECTURES}-${CMAKE_BUILD_TYPE})
					endif(_length GREATER 1)
				endif(${CPACK_SYSTEM_NAME} MATCHES Darwin AND CMAKE_OSX_ARCHITECTURES)
				set(CPACK_GENERATOR "PackageMaker")
				set(CMAKE_STRIP "${CMAKE_STRIP} -x")
			else(APPLE)
				# Debian based variable
				set(CPACK_GENERATOR "TGZ")
			endif(APPLE)
		endif(WIN32 AND NOT UNIX)
		include(CPack)
endif(NOT BUILD_NR_SLICER_EXT)
#=============================================================================
