set(DATA_FOLDER ${CMAKE_SOURCE_DIR}/reg-test/reg-test-data)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_interp)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_resampling _reg_maths reg_nifti ${ZLIB})
#-----------------------------------------------------------------------------
add_test(interpolation_3D_NN ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 0
    ${DATA_FOLDER}/test_warped_BW_3D_NN.nii.gz)
add_test(interpolation_3D_LIN ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 1
    ${DATA_FOLDER}/test_warped_BW_3D_LIN.nii.gz)
add_test(interpolation_3D_SPL ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 3
    ${DATA_FOLDER}/test_warped_BW_3D_SPL.nii.gz)
add_test(interpolation_2D_NN ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 0
    ${DATA_FOLDER}/test_warped_BW_2D_NN.nii.gz)
add_test(interpolation_2D_LIN ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 1
    ${DATA_FOLDER}/test_warped_BW_2D_LIN.nii.gz)
add_test(interpolation_2D_SPL ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 3
    ${DATA_FOLDER}/test_warped_BW_2D_SPL.nii.gz)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_mat44_operations)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_maths reg_nifti ${ZLIB})
#-----------------------------------------------------------------------------
add_test(mat44_operations ${EXEC})
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



