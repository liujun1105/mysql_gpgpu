
MESSAGE(STATUS "Prepare User-Defined Aggregator Functions")
ADD_DEFINITIONS("-DHAVE_DLOPEN -DMYSQL_DYNAMIC_PLUGIN")

set(CCFILES
  reduction/reduce.cpp
  reduction/reduction_task.cpp
  reduction/dispatcher.cpp
)

set(CUFILES
  reduction/reduction_kernel.cu
)

set (HFILES
  reduction/reduce.h
  reduction/dispatcher.h
  reduction/reduction_task.h
  reduction/reduction_operator.h
  reduction/sharedmem.h
)

set (AGGR_SUM_HFILES aggr_udf_sum.h host_buffer.h)
set (AGGR_SUM_CPPFILES aggr_udf_sum.cpp)
set (AGGR_SUM_DEFFILE aggr_udf_sum.def)

set (GSUM_HFILES gsum.h host_buffer.h)
set (GSUM_CPPFILES gsum.cpp)
set (GSUM_DEFFILE gsum.def)

set (CSUM_HFILES csum.h host_buffer.h)
set (CSUM_CPPFILES csum.cpp)
set (CSUM_DEFFILE csum.def)

source_group("Reduction Header Files" FILES ${HFILES})
source_group("Reduction Source Files" FILES ${CCFILES} ${CUFILES})
source_group("Plugin Header Files" FILES ${AGGR_SUM_HFILES} ${GSUM_HFILES} ${CSUM_HFILES})
source_group("Plugin Source Files" FILES ${GSUM_CPPFILES} ${AGGR_SUM_CPPFILES} ${CSUM_CPPFILES})
source_group("Plugin DEF Files" FILES ${GSUM_DEFFILE} ${AGGR_SUM_DEFFILE} ${CSUM_DEFFILE})

MYSQL_ADD_PLUGIN(aggr_udfs ${CCFILES} ${CUFILES} ${HFILES} ${AGGR_SUM_HFILES} ${AGGR_SUM_DEFFILE} ${AGGR_SUM_CPPFILES} MODULE_ONLY)
MYSQL_ADD_PLUGIN(gsum ${CCFILES} ${CUFILES} ${HFILES} ${GSUM_HFILES} ${GSUM_DEFFILE} ${GSUM_CPPFILES} MODULE_ONLY)
MYSQL_ADD_PLUGIN(csum ${CSUM_HFILES} ${CSUM_DEFFILE} ${CSUM_CPPFILES} MODULE_ONLY)

set(CUDA_JDK_PATH $ENV{CUDA_PATH})

if(IS_DIRECTORY ${CUDA_JDK_PATH})
	
	MESSAGE(STATUS "Find CUDA SDK Directory -> ${CUDA_JDK_PATH}")
	MESSAGE(STATUS "${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}")
	set(INCLUDES ${CUDA_JDK_PATH}/include 
				 ${CMAKE_SOURCE_DIR}/include 
				 ${CMAKE_SOURCE_DIR}/sql 
				 ${CMAKE_SOURCE_DIR}/regex 
				 ${CMAKE_SOURCE_DIR}/dbug
				 ${CMAKE_SOURCE_DIR}/mysys
    )
	MESSAGE(STATUS "include directories [${INCLUDES}]")
	INCLUDE_DIRECTORIES(${INCLUDES})
	
	if(WIN32)
		if (CMAKE_CL_64)
			MESSAGE(STATUS "Set CUDA(x64) Library Directory")
			set(CUDA_LIB ${CUDA_JDK_PATH}/lib/x64)
		else()
			MESSAGE(STATUS "Set CUDA(win32) Library Directory")
			set(CUDA_LIB ${CUDA_JDK_PATH}/lib/Win32)
		endif()
	else()
		MESSAGE(FATAL_ERROR "NOT A SUPPORTED PLATFORM")
	endif()
	
	FIND_LIBRARY(CUDART_LIBRARY cudart ${CUDA_LIB})
	if(CUDART_LIBRARY)
		MESSAGE(STATUS "Find CUDA Runtime Library [${CUDART_LIBRARY}]")
		LINK_DIRECTORIES(${CUDA_LIB})
		TARGET_LINK_LIBRARIES(aggr_udfs ${CUDART_LIBRARY} mysqld sql mysys dbug regex)
		TARGET_LINK_LIBRARIES(gsum ${CUDART_LIBRARY} mysqld sql mysys dbug regex)		
	else()
		MESSAGE(FATAL_ERROR "Failed to Find CUDA Runtime Library")
	endif(CUDART_LIBRARY)
	
	TARGET_LINK_LIBRARIES(csum mysqld sql mysys dbug regex)
	
else()
	MESSAGE(FATAL_ERROR "NO CUDA SDK PAHT FOUND")
endif()



