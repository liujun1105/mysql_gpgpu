project(reduction)

cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored"
)
  
 ## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored"
)

if(BUILD_SHARED_LIBS)
  # User wants to build Dynamic Libraries, so change the LIB_TYPE variable to CMake keyword 'SHARED'
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
  set(REDUCTION_STATIC_LIB 1)
endif(BUILD_SHARED_LIBS)

ADD_SUBDIRECTORY(src)
