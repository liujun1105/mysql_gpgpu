#include "GPUInfoHelper.h"

CUresult initDriverAPI()
{
	return cuInit(0);
}

int CUDAEnabledDeviceCount() 
{
	CUresult error_id;
	int deviceCount = 0;
	error_id = cuDeviceGetCount(&deviceCount);
	if (error_id == CUDA_SUCCESS) {
		return deviceCount;
	} 
	else {
		// 0 device
		return 0;
	}
}

int getDeviceName(CUdevice dev, char* const device_name, const int max_name_length)
{
	CUresult error_id;
	error_id = cuDeviceGetName(device_name, max_name_length, dev);
	if (error_id == CUDA_SUCCESS) {
		return 0;
	} 
	else {
		return 1;
	}
}

int getDeviceVersion()
{
	int driverVersion;
	cuDriverGetVersion(&driverVersion);
	return driverVersion;
}

size_t getDeviceTotalGlobalMemorySize(CUdevice dev)
{
	size_t totalGlobalMem;
	CUresult error_id = cuDeviceTotalMem(&totalGlobalMem, dev);
	if (error_id != CUDA_SUCCESS) 
	{
		return -1;
	}
	else 
	{
		return totalGlobalMem;
	}
}

float getDeviceTotalGlobalMemoryInMB(size_t totalGlobalMem)
{
	return (float)totalGlobalMem/1048576.0f;
}

int getMultiProcessorCount(CUdevice dev) 
{
	int multiProcessorCount;
	CUresult error_id = cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
	if (error_id != CUDA_SUCCESS)
    {
		return -1;
	}
	return multiProcessorCount;
}