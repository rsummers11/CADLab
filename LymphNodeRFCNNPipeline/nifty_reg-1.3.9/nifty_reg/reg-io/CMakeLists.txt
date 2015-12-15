# The z lib library is build only if it was not found on the system
if(NOT ZLIB_FOUND OR BUILD_ALL_DEP)
    subdirs(zlib)
endif(NOT ZLIB_FOUND OR BUILD_ALL_DEP)

# Build the nifti file format library
subdirs(nifti)

set(LIBRARIES reg_nifti z)

# Build the png library if required
subdirs(png)
set(LIBRARIES ${LIBRARIES} reg_png)

# Build the NRRD file format library if required
subdirs(nrrd)
set(LIBRARIES ${LIBRARIES} reg_nrrd)

# Create the reg_io library
add_library(_reg_ReadWriteImage _reg_ReadWriteImage.h _reg_ReadWriteImage.cpp)
target_link_libraries(_reg_ReadWriteImage ${LIBRARIES})
install(TARGETS _reg_ReadWriteImage
        RUNTIME DESTINATION bin COMPONENT Development
        LIBRARY DESTINATION lib COMPONENT Development
        ARCHIVE DESTINATION lib COMPONENT Development
)
install(FILES _reg_ReadWriteImage.h DESTINATION include COMPONENT Development)
