#-----------------------------------------------------------------------------
##BUILD THE CPU LIBRARIES
#-----------------------------------------------------------------------------
set(LIST _reg_resampling)
set(LIST ${LIST} _reg_maths)
set(LIST ${LIST} _reg_globalTransformation)
set(LIST ${LIST} _reg_mutualinformation)
set(LIST ${LIST} _reg_ssd)
set(LIST ${LIST} _reg_KLdivergence)
set(LIST ${LIST} _reg_blockMatching)
set(LIST ${LIST} _reg_femTransformation)
foreach(NAME ${LIST})
    add_library(${NAME} ${NAME}.h ${NAME}.cpp)
    target_link_libraries(${NAME})
    install(TARGETS ${NAME}
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    install(FILES ${NAME}.h DESTINATION include COMPONENT Development)
endforeach(NAME)
#-----------------------------------------------------------------------------
set(LIST _reg_tools)
set(LIST ${LIST} _reg_localTransformation)
foreach(NAME ${LIST})
    add_library(${NAME} ${NAME}.h ${NAME}.cpp)
    target_link_libraries(${NAME} _reg_maths)
    install(TARGETS ${NAME}
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    install(FILES ${NAME}.h DESTINATION include COMPONENT Development)
endforeach(NAME)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
##BUILD THE GPU LIBRARIES
if(USE_CUDA)
#-----------------------------------------------------------------------------
    set(LIST _reg_globalTransformation)
    set(LIST ${LIST} _reg_blockMatching)
    set(LIST ${LIST} _reg_localTransformation)
    set(LIST ${LIST} _reg_mutualinformation)
    set(LIST ${LIST} _reg_resampling)
    set(LIST ${LIST} _reg_tools)
    foreach(NAME ${LIST})
        cuda_add_library(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
        install(TARGETS ${NAME}_gpu
                RUNTIME DESTINATION bin COMPONENT Development
                LIBRARY DESTINATION lib COMPONENT Development
                ARCHIVE DESTINATION lib COMPONENT Development
        )
        install(FILES ${NAME}_gpu.h DESTINATION include COMPONENT Development)
    endforeach(NAME)
#-----------------------------------------------------------------------------
    set(NAME _reg_cudaCommon)
    cuda_add_library(${NAME} ${NAME}.h ${NAME}.cu)
    install(TARGETS ${NAME}
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    install(FILES ${NAME}.h DESTINATION include)
    install(FILES _reg_blocksize_gpu.h DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
endif(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY CPU OR GPU
set(NAME _reg_f3d)
if(USE_CUDA)
#-----------------------------------------------------------------------------
    if(WIN32)
        cuda_add_library(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_gpu.h ${NAME}_gpu.cpp)
    else(WIN32)
        cuda_add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_gpu.h ${NAME}_gpu.cpp)
    endif(WIN32)
    target_link_libraries(${NAME} _reg_localTransformation _reg_resampling _reg_globalTransformation _reg_KLdivergence _reg_ssd _reg_mutualinformation _reg_tools _reg_ReadWriteImage _reg_localTransformation_gpu _reg_resampling_gpu _reg_globalTransformation_gpu _reg_mutualinformation_gpu _reg_tools_gpu _reg_cudaCommon)
    install(FILES ${NAME}_gpu.h ${NAME}_gpu.cpp DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
else(USE_CUDA)
#-----------------------------------------------------------------------------
    if(WIN32)
        add_library(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
    else(WIN32)
        add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
    endif(WIN32)
    target_link_libraries(${NAME} _reg_localTransformation _reg_resampling _reg_globalTransformation _reg_KLdivergence _reg_ssd _reg_mutualinformation _reg_tools _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
endif(USE_CUDA)
#-----------------------------------------------------------------------------
install(TARGETS ${NAME}
        RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
        LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
        ARCHIVE DESTINATION lib COMPONENT RuntimeLibraries
)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include COMPONENT Development)
install(FILES ${NAME}2.h ${NAME}2.cpp DESTINATION include COMPONENT Development)
install(FILES ${NAME}_sym.h ${NAME}_sym.cpp DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
## BUILD THE ALADIN LIBRARY CPU
set(NAME _reg_aladin)
if(WIN32)
    add_library(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
else(WIN32)
    add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
endif(WIN32)
target_link_libraries(${NAME} _reg_localTransformation _reg_blockMatching _reg_resampling _reg_globalTransformation _reg_ssd _reg_mutualinformation _reg_tools _reg_ReadWriteImage)
install(TARGETS ${NAME}
        RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
        LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
        ARCHIVE DESTINATION lib COMPONENT RuntimeLibraries
)
install(FILES ${NAME}.h ${NAME}.cpp _reg_macros.h DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
## BUILD THE TPS LIBRARY CPU
set(NAME _reg_thinPlateSpline)
if(WIN32)
    add_library(${NAME} ${NAME}.h ${NAME}.cpp)
else(WIN32)
    add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp)
endif(WIN32)
    target_link_libraries(${NAME} _reg_tools _reg_ReadWriteImage)
install(TARGETS ${NAME}
        RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
        LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
        ARCHIVE DESTINATION lib COMPONENT RuntimeLibraries
)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
