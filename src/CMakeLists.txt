# refer to the root source directory of the project as ${<PRJNAME>_SOURCE_DIR} and
# to the root binary directory of the project as ${<PRJNAME>_BINARY_DIR}.
cmake_minimum_required (VERSION 3.9)
if(GPUINUSE)
  project(GTALIGN LANGUAGES CXX CUDA C)
else()
  project(GTALIGN LANGUAGES CXX C)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  if(MSVC)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:experimental")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  else()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif()

## set build configuration
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

if(WIN32)
  add_compile_definitions(_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1)
endif()

if(GPUINUSE)
  add_definitions(-DGPUINUSE)
endif()

## set definitions for configuration DEBUG, required when this 
## configuration is in use
#set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS_DEBUG "__DEBUG__")
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:__DEBUG__>)


if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
##  set_property(DIRECTORY APPEND PROPERTY COMPILE_OPTIONS $<$<CONFIG:Debug>:-O2>)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra -Xptxas -v")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  if(NOT GPUINUSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
  if(FASTMATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  endif()
  ##set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
  ##add_compile_options(-Wall -Wextra)
elseif(MSVC)
  if(FASTMATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(FASTMATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  endif()
endif()



FOREACH(LANG C CXX CUDA CUDACXX)
  ##set(CMAKE_${LANG}_FLAGS_DEBUG 
  ##  "${CMAKE_${LANG}_FLAGS_DEBUG} -D__DEBUG__")
  ## source file extension is replaced when producing object files
  set(CMAKE_${LANG}_OUTPUT_EXTENSION_REPLACE 1)
ENDFOREACH()

## set local directories (bin,state)
set(LOCALBINDIR bin)
set(LOCALSTATEDIR var)

# Recurse into the subdirectories. 
# The same process will walk through the project's entire directory structure.
## libs...
##add_subdirectory (extsp)
add_subdirectory (extzlib)
add_subdirectory (libutil)
add_subdirectory (libgenp)
add_subdirectory (libmympbase)
if(GPUINUSE)
  add_subdirectory (libmycu)
else()
  add_subdirectory (libmymp)
endif()
## execs...
add_subdirectory (gtalign)

