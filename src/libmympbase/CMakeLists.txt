# Create a library with the listed source files.
# Any number of sources could be listed here.
add_library( mympbase STATIC 
	mplayout/CuMemoryBase.cpp
)

# Make sure the compiler can find include files for the library
# when other libraries or executables link to it
if(GPUINUSE)
  target_include_directories( mympbase PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
else()
  target_include_directories( mympbase PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

