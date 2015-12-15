#-----------------------------------------------------------------------------
add_executable(reg_average reg_average.cpp)
target_link_libraries(reg_average _reg_resampling _reg_globalTransformation _reg_localTransformation _reg_maths _reg_tools _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_tools reg_tools.cpp)
target_link_libraries(reg_tools _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_resample reg_resample.cpp)
target_link_libraries(reg_resample _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_measure reg_measure.cpp)
target_link_libraries(reg_measure _reg_resampling _reg_tools _reg_nmi _reg_ssd _reg_KLdivergence _reg_lncc _reg_dti _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_ppcnr reg_ppcnr.cpp)
target_link_libraries(reg_ppcnr _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_transform reg_transform.cpp)
target_link_libraries(reg_transform _reg_thinPlateSpline _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_maths _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_jacobian reg_jacobian.cpp)
target_link_libraries(reg_jacobian _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
if(USE_CUDA)
	cuda_add_executable(reg_f3d reg_f3d.cpp)
else(USE_CUDA)
	add_executable(reg_f3d reg_f3d.cpp)
endif(USE_CUDA)
target_link_libraries(reg_f3d _reg_f3d)
#-----------------------------------------------------------------------------
if(USE_CUDA)
	cuda_add_executable(reg_aladin reg_aladin.cpp)
else(USE_CUDA)
	add_executable(reg_aladin reg_aladin.cpp)
endif(USE_CUDA)
target_link_libraries(reg_aladin _reg_aladin)
#-----------------------------------------------------------------------------
set(MODULE_LIST
		reg_resample
		reg_jacobian
		reg_aladin
		reg_f3d
		)
#-----------------------------------------------------------------------------
if(BUILD_NR_SLICER_EXT)
		set(EXTENSION_NAME "NiftyRegExtension")
		set(EXTENSION_HOMEPAGE "https://sourceforge.net/projects/niftyreg")
		set(EXTENSION_CATEGORY "Registration")
		set(EXTENSION_CONTRIBUTORS "Marc Modat, Pankaj Daga, David Cash, Sebastien Ourselin (UCL)")
		set(EXTENSION_DESCRIPTION "NiftyReg meets slicer, integration of a NiftyReg module within 3DSlicer")
		set(EXTENSION_ICONURL "https://dl.dropbox.com/u/2962833/nifty_reg_logo_128.png")
		set(EXTENSION_SCREENSHOTURLS "http://www.na-mic.org/Wiki/images/4/4c/Nifty_reg_logo.png")
		set(EXTENSION_ACKNOWLEDGEMENTS "This work is partially funded through CBRC grant 168 (UK)")
		set(EXTENSION_MAJOR_VERSION ${NiftyReg_VERSION_MAJOR})
		set(EXTENSION_MINOR_VERSION ${NiftyReg_VERSION_MINOR})
		set(EXTENSION_PATCH_VERSION ${NiftyReg_VERSION_PATCH})

		find_package(Slicer REQUIRED)
		include(${Slicer_USE_FILE})

		foreach(MODULE_NAME ${MODULE_LIST})
				install(TARGETS ${MODULE_NAME}
						RUNTIME DESTINATION ${Slicer_INSTALL_CLIMODULES_BIN_DIR} COMPONENT Runtime
						LIBRARY DESTINATION ${Slicer_INSTALL_CLIMODULES_LIB_DIR} COMPONENT Runtime
						ARCHIVE DESTINATION ${Slicer_INSTALL_CLIMODULES_LIB_DIR} COMPONENT Runtime
						)
		endforeach(MODULE_NAME)
		include(${Slicer_EXTENSION_CPACK})
else(BUILD_NR_SLICER_EXT)
		set(MODULE_LIST
				${MODULE_LIST}
				reg_average
				reg_tools
				reg_transform
				reg_measure
				)
endif(BUILD_NR_SLICER_EXT)
#-----------------------------------------------------------------------------
foreach(MODULE_NAME ${MODULE_LIST})
		install(TARGETS ${MODULE_NAME}
				RUNTIME DESTINATION bin COMPONENT Runtime
				LIBRARY DESTINATION lib COMPONENT Runtime
				ARCHIVE DESTINATION lib COMPONENT Runtime
				)
endforeach(MODULE_NAME)
#-----------------------------------------------------------------------------
install(PROGRAMS groupwise_niftyreg_params.sh DESTINATION bin COMPONENT Runtime)
install(PROGRAMS groupwise_niftyreg_run.sh DESTINATION bin COMPONENT Runtime)
#-----------------------------------------------------------------------------
