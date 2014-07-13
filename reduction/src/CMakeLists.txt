MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CCFILES
  reduce.cpp
  reduction_task.cpp
  dispatcher.cpp
)

set (HFILES
  dispatcher.h
  reduction_task.h
  reduction_operator.h
  sharedmem.h
)

set (CUHFILES
  
)

set(CUFILES
  reduction_kernel.cu
)

set(HFILES_PUBLIC
  reduce.h
)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)

set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX

option(REDUCTION_GENCODE_SM10
       "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
       ON)

option(REDUCTION_GENCODE_SM13
       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
       OFF)

option(REDUCTION_GENCODE_SM20
       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
       ON)

option(REDUCTION_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       OFF)

option(REDUCTION_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       OFF)

if (REDUCTION_GENCODE_SM10)
  set(GENCODE ${GENCODE} ${GENCODE_SM10})
endif(REDUCTION_GENCODE_SM10)

if (REDUCTION_GENCODE_SM13)
  set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(REDUCTION_GENCODE_SM13)

if (REDUCTION_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(REDUCTION_GENCODE_SM20)

if (REDUCTION_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(REDUCTION_GENCODE_SM30)

if (REDUCTION_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(REDUCTION_GENCODE_SM35)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)


set(CUDA_ADDITIONAL_OPTIONS --keep --keep-dir)
MESSAGE(STATUS "GENCODE ${GENCODE}")
MESSAGE(STATUS "VERBOSE_PTXAS ${VERBOSE_PTXAS}")
set(CCBIN "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin")
#-ccbin "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin"

CUDA_ADD_LIBRARY(reduction ${LIB_TYPE} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS} ${CUDA_ADDITIONAL_OPTIONS} ${CCBIN}
)

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
)

install(TARGETS reduction
  DESTINATION lib
  EXPORT reduction-targets
)

install(EXPORT reduction-targets
  DESTINATION lib
)