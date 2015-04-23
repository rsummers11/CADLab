cmake_minimum_required (VERSION 2.8)
project (nihApps)

# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Libs
set( LIB_SRC 
	Libs/nihHelperFunctions.h
)

include_directories( Libs )

# add the executables
### ITK based apps ###
set(APP_NAME itkExtractImageSlices)
add_executable (${APP_NAME} Apps/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})

set(APP_NAME itkRandomlyDeformImages2D)
add_executable (${APP_NAME} Apps/${APP_NAME}.cxx ${LIB_SRC})
target_link_libraries (${APP_NAME} ${ITK_LIBRARIES})

