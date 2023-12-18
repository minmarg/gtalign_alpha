# Add executable that is built from the listed source files.
# The extensions are automatically found.
add_executable( gtalign 
	gtalign.cpp 
)


file(TO_NATIVE_PATH "${CMAKE_INSTALL_PREFIX}/${LOCALSTATEDIR}" GTALIGNSTATEDIRNAT)
string(REGEX REPLACE "([\\])" "\\\\\\1" GTALIGNSTATEDIRNAT "${GTALIGNSTATEDIRNAT}")


## ONE executable
## Specify include directories
target_include_directories( gtalign PUBLIC 
	${CMAKE_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions( gtalign PUBLIC
	GTALIGN_INSTALL_DIR=${CMAKE_INSTALL_PREFIX}
	GTALIGNSTATEDIR=${GTALIGNSTATEDIRNAT}
)

# Link the executable to the libraries 
if(GPUINUSE)
  set_target_properties( gtalign PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties( mycu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries( gtalign PUBLIC mycu mympbase genp util extz Threads::Threads)
else()
  target_link_libraries( gtalign PUBLIC mymp mympbase genp util extz Threads::Threads)
endif()


## install targets
install(TARGETS 
	gtalign 
	DESTINATION ${LOCALBINDIR}
)
#install(FILES 
#	gtalign.sh 
#	DESTINATION ${LOCALBINDIR} 
#	PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ WORLD_READ
#)

