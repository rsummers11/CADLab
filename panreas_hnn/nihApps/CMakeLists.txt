cmake_minimum_required (VERSION 2.8)
project (nihApps)

set(COMPILER_OPTIONS -std=c++0x -Wall -pedantic -Werror -Wextra)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/") # For FindOpenMesh and FindEigen3
#-----------------------------------------------------------------------------
# VTK and ITK with VTK dependency
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
#if (ITKVtkGlue_LOADED)
#  find_package(VTK)
#  include(${VTK_USE_FILE})
#endif()
#-----------------------------------------------------------------------------
# Libs
include_directories( Libs )
set( LIB_SRC
#	Libs/itkIsotropicResampleImageFilter.h
#	Libs/itkIsotropicWindowingResampleImageFilter.h
	Libs/nihHelperFunctions.h
#	Libs/nihVTKfunctions.h
#	Libs/antsAllocImage.h
#	Libs/CentrelineExtract.h
#	Libs/CentrelineExtract.cpp
)
#-----------------------------------------------------------------------------
# maxflow
#set( MAXFLOW_SRC_VERSION "maxflow-v3.03.src" )
#set( MAXFLOW_SRC
#	Libs/${MAXFLOW_SRC_VERSION}/block.h
#	Libs/${MAXFLOW_SRC_VERSION}/graph.h
#	Libs/${MAXFLOW_SRC_VERSION}/graph.cpp
#	Libs/${MAXFLOW_SRC_VERSION}/maxflow.cpp
#	Libs/${MAXFLOW_SRC_VERSION}/instances.inc
#)
#include_directories( Libs/${MAXFLOW_SRC_VERSION} )
#-----------------------------------------------------------------------------
# Eigen++
#find_package(Eigen3)
#if (EIGEN3_FOUND)
#  include_directories( ${EIGEN3_INCLUDE_DIR} )
#endif (EIGEN3_FOUND)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#if (ITKVtkGlue_LOADED)
#	set( OBJ_SRC
#		Libs/vtkOBJWriter.h
#		Libs/vtkOBJWriter.cxx
#	)
#endif()
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
# add the executables
#-----------------------------------------------------------------------------
### ITK-based apps ###
set(APP_NAME itkExtractImageSlices)
add_executable (${APP_NAME} Apps/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})

set(APP_NAME itkExtractBoundingBoxRegion)
add_executable (${APP_NAME} Apps/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})

set(APP_NAME itkDicomSeriesReadImageWrite2)
add_executable (${APP_NAME} Apps/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})


