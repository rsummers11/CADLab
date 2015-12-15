
if(BUILD_INTERNAL_PNG OR BUILD_ALL_DEP)
		# If the png library is not present on the machine, it is build from the sources
		# Most of the following lines are extracted from the libpng1510 CMakeLists.txt
		set(PNGLIB_MAJOR 1)
		set(PNGLIB_MINOR 5)
		set(PNGLIB_RELEASE 10)
		set(PNGLIB_NAME libpng${PNGLIB_MAJOR}${PNGLIB_MINOR})
		set(PNGLIB_VERSION ${PNGLIB_MAJOR}.${PNGLIB_MINOR}.${PNGLIB_RELEASE})
		# Check if the m library is present
		if(NOT WIN32)
			find_library(M_LIBRARY
				NAMES m
				PATHS /usr/lib /usr/local/lib
			)
			if(NOT M_LIBRARY)
				message(STATUS
					"math library 'libm' not found - floating point support disabled")
			endif(NOT M_LIBRARY)
		else(NOT WIN32)
			# the m library is not needed on windows
			set(M_LIBRARY "")
		endif(NOT WIN32)
		# generate the config file for libpng and set the path to use it
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lpng1510/pnglibconf.h.prebuilt
									 ${CMAKE_BINARY_DIR}/pnglibconf.h)
		include_directories(${CMAKE_BINARY_DIR})
		# Set the libpng sources
		set(png_HDRS
		lpng1510/png.h
		lpng1510/pngconf.h
			lpng1510/pngdebug.h
			lpng1510/pnginfo.h
			lpng1510/pngpriv.h
			lpng1510/pngstruct.h
		)
		set(png_SRCS
				lpng1510/png.c
				lpng1510/pngerror.c
				lpng1510/pngget.c
				lpng1510/pngmem.c
				lpng1510/pngpread.c
				lpng1510/pngread.c
				lpng1510/pngrio.c
				lpng1510/pngrtran.c
				lpng1510/pngrutil.c
				lpng1510/pngset.c
				lpng1510/pngtest.c
				lpng1510/pngtrans.c
				lpng1510/pngwio.c
				lpng1510/pngwrite.c
				lpng1510/pngwtran.c
				lpng1510/pngwutil.c
		)
		# Build the library
		add_library(png STATIC ${png_SRCS} ${png_HDRS})
		target_link_libraries(png z)
		install(TARGETS png
				LIBRARY DESTINATION lib COMPONENT Development
				ARCHIVE DESTINATION lib COMPONENT Development
		)
		install(FILES ${png_HDRS} ${CMAKE_BINARY_DIR}/pnglibconf.h DESTINATION include COMPONENT Development)
endif(BUILD_INTERNAL_PNG OR BUILD_ALL_DEP)

add_library(reg_png reg_png.cpp reg_png.h readpng.cpp readpng.h)
target_link_libraries(reg_png ${PNG_LIBRARY} _reg_tools reg_nifti z)
install(TARGETS reg_png
				RUNTIME DESTINATION bin COMPONENT Development
				LIBRARY DESTINATION lib COMPONENT Development
				ARCHIVE DESTINATION lib COMPONENT Development
)
install(FILES reg_png.h readpng.h DESTINATION include COMPONENT Development)
