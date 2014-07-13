# Copyright (c) 2006, 2010, Oracle and/or its affiliates. All rights reserved.
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

if(STANDALONE_DEBUG)
	CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
else()
  MESSAGE(STATUS "Prepare GPU_INFO Information Schema Plugin")
  MYSQL_ADD_PLUGIN(gpu_info_is gpu_info_i_s.cpp GPUInfoHelper.cpp 
  MODULE_ONLY MODULE_OUTPUT_NAME "gpu_info_i_s")
endif()

set(CUDA_JDK_PATH $ENV{CUDA_PATH})

if(IS_DIRECTORY ${CUDA_JDK_PATH})
	
	MESSAGE(STATUS "Find CUDA SDK Directory -> ${CUDA_JDK_PATH}")
	
	set(INCLUDES ${CUDA_JDK_PATH}/include ${CMAKE_SOURCE_DIR}/plugin/gpu_info_is)
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
	FIND_LIBRARY(CUDA_LIBRARY cuda ${CUDA_LIB})
	if(CUDA_LIBRARY)
		MESSAGE(STATUS "Find CUDA Library [${CUDA_LIBRARY}]")
		LINK_DIRECTORIES(${CUDA_LIB})
		TARGET_LINK_LIBRARIES(gpu_info_is ${CUDA_LIBRARY})
	else()
		MESSAGE(FATAL_ERROR "Failed to Find CUDA Library")
	endif()
else()
	MESSAGE(FATAL_ERROR "NO CUDA SDK PAHT FOUND")
endif()




