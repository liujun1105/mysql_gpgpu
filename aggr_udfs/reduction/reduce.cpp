#include "reduce.h"
#include "reduction_task.h"
#include "dispatcher.h"
#include <cuda_runtime.h>
#include <iostream>

template<typename T>
void prealloc(const unsigned int number_of_elements, void** d_in_ptr)
{	
	int input_size = sizeof(T)*number_of_elements;
	cudaMalloc((void **)d_in_ptr, input_size);
	CUDA_CHECK_ERRORS("prealloc");	
}

void free_prealloc(void* d_in) 
{
	cudaFree(d_in);
	CUDA_CHECK_ERRORS("free");
}

void prealloc_with_mapped_memory(const unsigned int number_of_elements, unsigned int unit_size, void** mem_pptr, int flag)
{
	cudaHostAlloc((void**)mem_pptr, number_of_elements*unit_size, flag);
	CUDA_CHECK_ERRORS("prealloc_with_mapped_memory failed");
}

void free_mapped_memory(void* mem_ptr)
{
	cudaFreeHost(mem_ptr);
	CUDA_CHECK_ERRORS("free_mapped_memory failed");
}

template<typename T>
T execReductionWithReuseCache(const void* h_in, void* h_out, void* d_in,	        
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype)
{
	T gpu_result = 0;
	
	gpu_result = reductionWorker<T>(h_in, h_out, d_in, number_of_elements, unit_size, block_size, thread_size_per_block, r_type, r_datatype);

	return gpu_result;
}

template<typename T>
T execReduction(const void* h_in, void* h_out, 	        
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype)
{
	T gpu_result = 0;
	
	gpu_result = reductionWorker<T>(h_in, h_out, number_of_elements, unit_size, block_size, thread_size_per_block, r_type, r_datatype);

	return gpu_result;
}


template<typename T>
T reductionWorker(const void* h_in, void* h_out, void* d_in,	        
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype)
{
	T* d_out = NULL;
	int output_size = block_size * sizeof(T);
	int input_size = sizeof(T)*number_of_elements;
	cudaMalloc((void **)&d_out, output_size);
	CUDA_CHECK_ERRORS("d_out malloc");

	cudaMemcpy(d_in, h_in, input_size, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERRORS("cudaMemcpy -> d_in");
	cudaMemcpy(d_out, h_in, output_size, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERRORS("cudaMemcpy -> d_out");
	
	cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);	

	ReductionTask* reduction_task = new ReductionTask(number_of_elements, sizeof(T),  
		block_size, thread_size_per_block, r_type, r_datatype);
	dispatch(d_in, d_out, reduction_task);
	
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

	float time = 0;
	cudaEventElapsedTime(&time, startEvent, stopEvent);
	//calculate bandwidth in MB/s
    float bandwidthInMBs = (1e3f * input_size) / (time * (float)(1 << 20));
	printf("gpu time (measured using cudaEvent) [%f seconds]; bandwidth (mb) [%f]\n", time/1000.0, bandwidthInMBs);
	fprintf(stderr, "gpu time (measured using cudaEvent) [%f seconds]; bandwidth (mb) [%f]\n", time/1000.0, bandwidthInMBs);
	fflush(stderr); 

	cudaMemcpy(h_out, d_out, output_size, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERRORS("reduction output");

	T gpu_result = 0;
	for (unsigned int i = 0; i < block_size; i++)
	{
		gpu_result += ((double*)h_out)[i];
	}

	cudaFree(d_out);
	CUDA_CHECK_ERRORS("d_out free");

	delete reduction_task;

	return gpu_result;
}

template<typename T>
T reductionWorker(const void* h_in, void* h_out, 	        
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype)
{
	T* d_in = NULL;	
	int input_size = sizeof(T)*number_of_elements;
	
	cudaMalloc((void **)&d_in, input_size);
	CUDA_CHECK_ERRORS("d_in malloc");

	T gpu_result = reductionWorker<T>(h_in, h_out, d_in, number_of_elements, unit_size, block_size, thread_size_per_block, r_type, r_datatype);
	
	cudaFree(d_in);
	CUDA_CHECK_ERRORS("d_in free");

	return gpu_result;
}

template<typename T>
void reductionWorkerUsingMappedMemory(const T* d_in, T* d_out, 	        
									  ReductionTask* reduction_task)
{
	dispatch(d_in, d_out, reduction_task);
}

template<typename T>
void reductionWorkerUsingStreams(const T* d_in, T* d_out, 	        
	                             ReductionTask* reduction_task,
			                     cudaStream_t stream)
{
	dispatchKernalWithStream(d_in, d_out, reduction_task, stream);
}