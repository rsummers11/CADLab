#-----------------------------------------------------------------------------
if(NOT ZLIB_FOUND)
    add_library(z adler32.c compress.c crc32.c gzio.c uncompr.c deflate.c trees.c zutil.c inflate.c infback.c inftrees.c inffast.c)
    install(TARGETS z
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    install(FILES zlib.h zutil.h DESTINATION include COMPONENT Development)
    set(ZLIB_LIBRARY "z")
endif(NOT ZLIB_FOUND)
#-----------------------------------------------------------------------------
