#-----------------------------------------------------------------------------
add_definitions(-DHAVE_ZLIB)
set(NAME reg_nifti)
add_library(${NAME} nifti1.h nifti1_io.c nifti1_io.h znzlib.c znzlib.h)
target_link_libraries(${NAME} z)
install(TARGETS ${NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
install(FILES nifti1_io.h znzlib.h nifti1.h DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
