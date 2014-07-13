#ifndef GPUINFOHELPER_H

#define GPUINFOHELPER_H
#include <cuda.h>
#include <cuda_runtime.h>


CUresult initDriverAPI();
int CUDAEnabledDeviceCount();
int getDeviceName(CUdevice dev, char* const device_name, const int max_name_length);

int getDeviceVersion();
size_t getDeviceTotalGlobalMemorySize(CUdevice dev);
float getDeviceTotalGlobalMemoryInMB(size_t totalGlobalMem);
int getMultiProcessorCount(CUdevice dev);

#endif