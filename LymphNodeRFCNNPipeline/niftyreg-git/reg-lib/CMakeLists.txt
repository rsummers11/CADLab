#-----------------------------------------------------------------------------
##BUILD THE CPU LIBRARIES
#-----------------------------------------------------------------------------
add_library(_reg_maths ${NIFTYREG_LIBRARY_TYPE}
	_reg_maths.h
	_reg_maths.cpp
	)
target_link_libraries(_reg_maths
	)
install(TARGETS _reg_maths
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES _reg_maths.h DESTINATION include)

#-----------------------------------------------------------------------------
add_library(_reg_tools ${NIFTYREG_LIBRARY_TYPE}
	_reg_tools.h
	_reg_tools.cpp
	)
target_link_libraries(_reg_tools
	_reg_maths
	reg_nifti
	)
install(TARGETS _reg_tools
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES _reg_tools.h DESTINATION include)

#-----------------------------------------------------------------------------
add_library(_reg_globalTransformation ${NIFTYREG_LIBRARY_TYPE}
	_reg_globalTransformation.h
	_reg_globalTransformation.cpp)
target_link_libraries(_reg_globalTransformation
	_reg_tools
	)
install(TARGETS _reg_globalTransformation
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES _reg_globalTransformation.h DESTINATION include)

#-----------------------------------------------------------------------------
set(LIB_LIST
	_reg_localTransformation
	_reg_resampling
	_reg_nmi
	_reg_ssd
	_reg_KLdivergence
	_reg_lncc
	_reg_dti
	_reg_blockMatching
	_reg_femTransformation
)

foreach(NAME ${LIB_LIST})
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
		${NAME}.h
		${NAME}.cpp)
	target_link_libraries(${NAME}
		_reg_globalTransformation
		)
	install(TARGETS ${NAME}
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
	install(FILES ${NAME}.h DESTINATION include)
endforeach(NAME)

install(FILES _reg_measure.h DESTINATION include)

#-----------------------------------------------------------------------------
set(NAME _reg_optimiser)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp)
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cpp)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
target_link_libraries(${NAME})
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)

#-----------------------------------------------------------------------------
##BUILD THE GPU LIBRARIES
if(USE_CUDA)
	#-----------------------------------------------------------------------------
	#    set(NAME _reg_blocksize_gpu)
	#    cuda_add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cu)
	#    install(TARGETS ${NAME}
	#            RUNTIME DESTINATION bin
	#            LIBRARY DESTINATION lib
	#            ARCHIVE DESTINATION lib
	#    )
	#    install(FILES ${NAME}.h DESTINATION include)
	#-----------------------------------------------------------------------------
	set(NAME _reg_common_gpu)
	cuda_add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cu _reg_blocksize_gpu.h _reg_blocksize_gpu.cu)
	if(WIN32)
		target_link_libraries(${NAME} ${CUDA_CUDA_LIBRARY})
	endif(WIN32)
	#	if(APPLE)
	#		FIND_LIBRARY(CUDA_CUDA_LIB cuda)
	#		target_link_libraries(${NAME} ${CUDA_CUDA_LIB})
	#	endif(APPLE)
	install(TARGETS ${NAME}
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
	install(FILES ${NAME}.h DESTINATION include)
	#-----------------------------------------------------------------------------
	set(LIST _reg_globalTransformation)
	set(LIST ${LIST} _reg_blockMatching)
	set(LIST ${LIST} _reg_localTransformation)
	set(LIST ${LIST} _reg_nmi)
	set(LIST ${LIST} _reg_ssd)
	set(LIST ${LIST} _reg_resampling)
	set(LIST ${LIST} _reg_tools)
	set(LIST ${LIST} _reg_optimiser)
	foreach(NAME ${LIST})
		cuda_add_library(${NAME}_gpu ${NIFTYREG_LIBRARY_TYPE} ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
		install(TARGETS ${NAME}_gpu
			RUNTIME DESTINATION bin
			LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
			)
		install(FILES ${NAME}_gpu.h DESTINATION include)
	endforeach(NAME)
	#-----------------------------------------------------------------------------
	install(FILES _reg_blocksize_gpu.h DESTINATION include)
	#-----------------------------------------------------------------------------
endif(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
	SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY CPU OR GPU
set(NAME _reg_f3d)
if(USE_CUDA)
	#-----------------------------------------------------------------------------
	cuda_add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
		_reg_base.h
		_reg_base.cpp
		${NAME}.h
		${NAME}.cpp
		${NAME}_sym.h
		${NAME}_sym.cpp
		${NAME}2.h
		${NAME}2.cpp
		${NAME}_gpu.h
		${NAME}_gpu.cpp)
	target_link_libraries(${NAME}
		_reg_localTransformation
		_reg_resampling
		_reg_globalTransformation
		_reg_KLdivergence
		_reg_ssd
		_reg_nmi
		_reg_dti
		_reg_lncc
		_reg_optimiser
		_reg_tools
		_reg_ReadWriteImage
		_reg_optimiser_gpu
		_reg_localTransformation_gpu
		_reg_resampling_gpu
		_reg_globalTransformation_gpu
		_reg_nmi_gpu
		_reg_ssd_gpu
		_reg_tools_gpu
		_reg_common_gpu)
	install(FILES ${NAME}_gpu.h ${NAME}_gpu.cpp DESTINATION include)
	#-----------------------------------------------------------------------------
else(USE_CUDA)
	#-----------------------------------------------------------------------------
	if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
		add_library(${NAME} SHARED
			_reg_base.h
			_reg_base.cpp
			${NAME}.h
			${NAME}.cpp
			${NAME}2.h
			${NAME}2.cpp
			${NAME}_sym.h
			${NAME}_sym.cpp)
	else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
		add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
			_reg_base.h
			_reg_base.h
			_reg_base.cpp
			${NAME}.h
			${NAME}.cpp
			${NAME}2.h
			${NAME}2.cpp
			${NAME}_sym.h
			${NAME}_sym.cpp)
	endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	target_link_libraries(${NAME}
		_reg_localTransformation
		_reg_resampling
		_reg_globalTransformation
		_reg_lncc
		_reg_dti
		_reg_KLdivergence
		_reg_ssd _reg_nmi
		_reg_optimiser
		_reg_tools
		_reg_ReadWriteImage)
endif(USE_CUDA)
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)
install(FILES ${NAME}2.h ${NAME}2.cpp DESTINATION include)
install(FILES ${NAME}_sym.h ${NAME}_sym.cpp DESTINATION include)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE ALADIN LIBRARY CPU

set(NAME _reg_aladin)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED
		${NAME}.h
		${NAME}.cpp
		${NAME}_sym.h
		${NAME}_sym.cpp
		)
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
		${NAME}.h
		${NAME}.cpp
		${NAME}_sym.h
		${NAME}_sym.cpp
		)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

target_link_libraries(${NAME}
	_reg_localTransformation
	_reg_blockMatching
	_reg_resampling
	_reg_globalTransformation
	_reg_ssd
	_reg_nmi
	_reg_tools
	_reg_ReadWriteImage
	)
install(TARGETS ${NAME}
	RUNTIME DESTINATION lib
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}_sym.h ${NAME}.cpp ${NAME}_sym.cpp _reg_macros.h DESTINATION include)

#-----------------------------------------------------------------------------
## BUILD THE TPS LIBRARY CPU
set(NAME _reg_thinPlateSpline)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp)
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cpp)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
target_link_libraries(${NAME} _reg_tools _reg_ReadWriteImage)
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)
#-----------------------------------------------------------------------------
## BUILD THE POLYAFFINE LIBRARY CPU
#set(NAME _reg_polyAffine)
#if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#  add_library(${NAME} SHARED _reg_base.h _reg_base.cpp ${NAME}.h ${NAME}.cpp)
#else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#  add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} _reg_base.h _reg_base.cpp ${NAME}.h ${NAME}.cpp)
#endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#target_link_libraries(${NAME} _reg_tools _reg_ReadWriteImage)
#install(TARGETS ${NAME}
#  RUNTIME DESTINATION bin
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib
#  )
#install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)
#-----------------------------------------------------------------------------
## Generate install commands for any extra include files
set(ADDITIONAL_INCLUDE_LIST
	_reg_base.h
	_reg_base.cpp
	)
foreach(NAME ${ADDITIONAL_INCLUDE_LIST})
	install(FILES ${NAME} DESTINATION include)
endforeach()

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

