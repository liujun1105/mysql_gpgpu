#include "reduce.h"
#include "reduction_task.h"
#include "dispatcher.h"
#include <cuda_runtime.h>
#include <iostream>


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
T reductionWorker(const void* h_in, void* h_out, 	        
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype)
{
	T* d_in = NULL;	
	T* d_out = NULL;

	int input_size = sizeof(T)*number_of_elements;
	int output_size = block_size * sizeof(T);
	 
	cudaMalloc((void **)&d_out, output_size);
	CUDA_CHECK_ERRORS("d_out malloc");
	cudaMalloc((void **)&d_in, input_size);
	CUDA_CHECK_ERRORS("d_in malloc");
	
	cudaMemcpy(d_out, h_in, output_size, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERRORS("reduction input -> d_out");
	cudaMemcpy(d_in, h_in, input_size, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERRORS("reduction input -> d_in");
	

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
	printf("gpu time (measured using cudaEvent) [%f seconds]\n", time/1000.0);

	cudaMemcpy(h_out, d_out, output_size, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERRORS("reduction output");

	T gpu_result = 0;
	for (unsigned int i = 0; i < block_size; i++)
	{
		gpu_result += ((double*)h_out)[i];
	}
	
	cudaFree(d_in);
	CUDA_CHECK_ERRORS("d_in free");
	cudaFree(d_out);
	CUDA_CHECK_ERRORS("d_out free");

	delete reduction_task;

	return gpu_result;
}