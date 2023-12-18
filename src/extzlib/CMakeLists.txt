# Create a library with the listed source files.
# Any number of sources could be listed here.
add_library( extz STATIC 
	##OBJZ:
	adler32.c
	crc32.c
	deflate.c
	infback.c
	inffast.c
	inflate.c
	inftrees.c
	trees.c
	zutil.c
	##OBJG:
	compress.c
	uncompr.c
	gzclose.c
	gzlib.c
	gzread.c
	gzwrite.c
)

file(GLOB_RECURSE sourcefls "*.c")
set_source_files_properties(${sourcefls} PROPERTIES LANGUAGE C)
set_target_properties(extz PROPERTIES LINKER_LANGUAGE C)

# Make sure the compiler can find include files for the library
# when other libraries or executables link to it
target_include_directories( extz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
