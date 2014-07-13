#include "reduce.h"
#include <iostream>
#include <ctime>
#include "assert.h"
#include "cuda_runtime_api.h"
#include "reduction_task.h"
using namespace std;

const unsigned int BLOCK_SIZE = 256;
const unsigned int THREAD_SIZE = 256;
const unsigned int N = 102400000;
const unsigned int NUMBER_OF_LOOPS = 1;
const unsigned int NUMBER_OF_STREAMS = 32;


template<class T>
T reduceCPU_Max(T *data, unsigned int size)
{
	T max_value = (T)0;
	T c = (T)0.0;

	for (unsigned int i = 0; i < size; i++)
	{
		if (data[i] > max_value)
		{		
			max_value = data[i];
		}
	}

	return max_value;
}

template<class T>
T reduceCPU(T *data, unsigned int size)
{
	T sum = (T)0;
	T c = (T)0.0;

	for (unsigned int i = 0; i < size; i++)
	{
		//T y = data[i] - c;
		//T t = sum + y;
		//c = (t - sum) - y;
		//sum = t;
		sum += data[i];
	}

	return sum;
}

unsigned int input_size_required(const int number_of_elements, const int unit_size)
{
	return number_of_elements * unit_size;
}

void testPageableMemory() 
{
	printf("testPageableMemory\n");

	const unsigned int input_size = N * sizeof(double);
	const unsigned int output_size = BLOCK_SIZE * sizeof(double);

	double* h_in = (double*) malloc(input_size);
	double* h_out = (double*) malloc(output_size);

	void** d_in_ptr = 0;
	void* ptr = malloc(sizeof(void*));
	d_in_ptr = &ptr;
	prealloc<double>(N, d_in_ptr);

	for (unsigned int i = 0; i < N; i++)
	{        
		h_in[i] = i; 
	}

	for (int i=0; i<NUMBER_OF_LOOPS; i++)
	{

		clock_t gpu_with_cache_clock;
		gpu_with_cache_clock = clock();
		double gpu__with_cache_result = execReductionWithReuseCache<double>(h_in, h_out, *d_in_ptr, N, sizeof(double), BLOCK_SIZE, THREAD_SIZE, R_SUM, R_DOUBLE);
		gpu_with_cache_clock = clock() - gpu_with_cache_clock;
		printf("gpu result [%f], gpu time [%f seconds]\n", gpu__with_cache_result, ((float)gpu_with_cache_clock)/CLOCKS_PER_SEC);

		//clock_t gpu_clock;
		//gpu_clock = clock();
		//double gpu_result = execReduction<double>(h_in, h_out, N, sizeof(double), BLOCK_SIZE, THREAD_SIZE, R_SUM, R_DOUBLE);
		//gpu_clock = clock() - gpu_clock;
		//printf("gpu result [%f], gpu time [%f seconds]\n", gpu_result, ((float)gpu_clock)/CLOCKS_PER_SEC);

		clock_t cpu_clock;
		cpu_clock = clock();
		double cpu_result = reduceCPU<double>(h_in, N);
		cpu_clock = clock() - cpu_clock;
		printf("cpu result [%f], cpu time [%f seconds]\n", cpu_result, ((float)cpu_clock)/CLOCKS_PER_SEC);

		printf("-------------\n");
	}
	free_prealloc(*d_in_ptr);
	free(h_out);
	free(h_in);
}

void testMappedMemory() 
{

	printf("testMappedMemory\n");

	const unsigned int input_size = N * sizeof(double);
	const unsigned int output_size = BLOCK_SIZE * sizeof(double);

	double* h_in;
	double* h_out;
	double* d_in;
	double* d_out;

	cudaHostAlloc((void**)&h_in, input_size, cudaHostAllocWriteCombined | cudaHostAllocMapped );
	CUDA_CHECK_ERRORS("cudaHostAlloc -> h_in");
	cudaHostAlloc((void**)&h_out, output_size, cudaHostAllocWriteCombined | cudaHostAllocMapped );
	CUDA_CHECK_ERRORS("cudaHostAlloc -> h_out");

	double sum = 0;
	for (unsigned int i = 0; i < N; i++)
	{        
		h_in[i] = i;  
		sum += i;
	}
	printf("expected result [%d]\n", sum);

	cudaHostGetDevicePointer((void**)&d_in, h_in, 0);
	cudaHostGetDevicePointer((void**)&d_out, h_out, 0);

	for (unsigned int i = 0; i < NUMBER_OF_LOOPS; i++)
	{
		cudaEvent_t start, stop;
		float elapsedTime;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		ReductionTask reduction_task(N, sizeof(double), BLOCK_SIZE, THREAD_SIZE, R_SUM, R_DOUBLE);
		reductionWorkerUsingMappedMemory<double>(d_in, d_out, &reduction_task);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		double gpu_result = 0;
		for (unsigned int i = 0; i < BLOCK_SIZE; i++)
		{
			gpu_result += ((double*)h_out)[i];
		}

		float bandwidthInMBs = (1e3f * input_size) / (elapsedTime * (float)(1 << 20));
		printf("gpu result [%f], gpu time [%f seconds] bandwidth (mb) [%f]\n", gpu_result, elapsedTime/1000.0, bandwidthInMBs);
	}
	cudaFreeHost(h_in);
	CUDA_CHECK_ERRORS("cudaFreeHost -> h_in");
	cudaFreeHost(h_out);
	CUDA_CHECK_ERRORS("cudaFreeHost -> h_out");
}

void testDeviceInfo() 
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("concurrent kernels [%d]\n", deviceProp.concurrentKernels);
}

void testStream() 
{
	printf("testStream\n");

	const unsigned int input_size = N * sizeof(double);
	const unsigned int output_size = BLOCK_SIZE * sizeof(double);
	const unsigned int input_inc = N/NUMBER_OF_STREAMS;
	const unsigned int output_inc = BLOCK_SIZE/NUMBER_OF_STREAMS;
	const unsigned int input_copy_block_size = input_size/NUMBER_OF_STREAMS;
	const unsigned int output_copy_block_size = output_size/NUMBER_OF_STREAMS;

	double** h_in = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	double** h_out = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	double** d_in = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	double** d_out = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	ReductionTask reduction_task(input_inc, sizeof(double), output_inc, THREAD_SIZE, R_SUM, R_DOUBLE);
	cudaStream_t* stream_ptr = (cudaStream_t*)malloc(NUMBER_OF_STREAMS*sizeof(cudaStream_t));	

	for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{ 
		cudaHostAlloc((void**)&h_in[i], input_copy_block_size, cudaHostAllocDefault);
		CUDA_CHECK_ERRORS("cudaHostAlloc -> h_in");
		cudaHostAlloc((void**)&h_out[i], output_copy_block_size, cudaHostAllocDefault);
		CUDA_CHECK_ERRORS("cudaHostAlloc -> h_out");
		cudaStreamCreate(&stream_ptr[i]);	
		CUDA_CHECK_ERRORS("failed to create stream");
		cudaMalloc((void **)&d_in[i], input_copy_block_size);
		cudaMalloc((void **)&d_out[i], output_copy_block_size);		
	}
	for (unsigned int j=0; j<NUMBER_OF_STREAMS; j++)
	{
		for (unsigned int i = 0; i < input_inc; i++)
		{        		
			h_in[j][i] = 1;
			//rand() * (sizeof(double)*8-1) * N;        
		}
	}

	for (unsigned int k=0; k<NUMBER_OF_LOOPS; k++){

		clock_t	cpu_clock = clock();

		double cpu_result = 0;
		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) {
			cpu_result += reduceCPU(h_in[i], input_inc);
		}
		cpu_clock = clock() - cpu_clock;
		printf("cpu sum [%f], cpu time [%f seconds]\n", cpu_result, ((float)cpu_clock)/CLOCKS_PER_SEC);

		cudaEvent_t start, stop;
		float elapsedTime;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
		{
			cudaMemcpyAsync(d_in[i], h_in[i], input_copy_block_size, cudaMemcpyHostToDevice, stream_ptr[i]);
			//CUDA_CHECK_ERRORS("failed to copy memory (async) from host to device");
		}

		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
		{			
			reductionWorkerUsingStreams<double>(d_in[i], d_out[i], &reduction_task, stream_ptr[i]);
		}

		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
		{	
			cudaMemcpyAsync(h_out[i], d_out[i], output_copy_block_size, cudaMemcpyDeviceToHost, stream_ptr[i]);
			//CUDA_CHECK_ERRORS("failed to copy memory (async) from device to host");
		}

		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
		{ 
			cudaStreamSynchronize(stream_ptr[i]); 
			//CUDA_CHECK_ERRORS("failed to synchronize stream");
		} 
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);

		double gpu_result = 0;
		for (unsigned int j = 0; j < NUMBER_OF_STREAMS; j++) 
		{
			for (unsigned int i = 0; i < output_inc; i++)
			{
				gpu_result += h_out[j][i];
			}
		}

		float bandwidthInMBs = (1e3f * input_size) / (elapsedTime * (float)(1 << 20));
		printf("gpu sum [%f], gpu time [%f seconds] bandwidth (mb) [%f]\n", gpu_result, elapsedTime/1000.0, bandwidthInMBs);
	}
	for (int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{ 
		cudaStreamDestroy(stream_ptr[i]);  
		cudaFree(d_in[i]);
		cudaFree(d_out[i]);
		cudaFreeHost(h_in[i]);
		cudaFreeHost(h_out[i]);
		CUDA_CHECK_ERRORS("failed to destory stream");
	} 

	/*cudaFree(d_in);
	cudaFree(d_out);*/

	free(d_in);
	free(d_out);
	free(h_in);
	free(h_out);
	//cudaFreeHost(h_in);
	//CUDA_CHECK_ERRORS("cudaFreeHost -> h_in");
	//cudaFreeHost(h_out);
	//CUDA_CHECK_ERRORS("cudaFreeHost -> h_out");

}

void testStream_Max() 
{
	printf("testStream\n");

	const unsigned int input_size = N * sizeof(double);
	const unsigned int output_size = BLOCK_SIZE * sizeof(double);
	const unsigned int input_inc = N/NUMBER_OF_STREAMS;
	const unsigned int output_inc = BLOCK_SIZE/NUMBER_OF_STREAMS;
	const unsigned int input_copy_block_size = input_size/NUMBER_OF_STREAMS;
	const unsigned int output_copy_block_size = output_size/NUMBER_OF_STREAMS;
	
	double** h_in = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	double** h_out = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	double** d_in = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	double** d_out = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	ReductionTask reduction_task(input_inc, sizeof(double), output_inc, THREAD_SIZE, R_MAX, R_DOUBLE);
	cudaStream_t* stream_ptr = (cudaStream_t*)malloc(NUMBER_OF_STREAMS*sizeof(cudaStream_t));	

	for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{ 
		cudaHostAlloc((void**)&h_in[i], input_copy_block_size, cudaHostAllocDefault);
		CUDA_CHECK_ERRORS("cudaHostAlloc -> h_in");
		cudaHostAlloc((void**)&h_out[i], output_copy_block_size, cudaHostAllocDefault);
		CUDA_CHECK_ERRORS("cudaHostAlloc -> h_out");
		cudaStreamCreate(&stream_ptr[i]);	
		CUDA_CHECK_ERRORS("failed to create stream");
		cudaMalloc((void **)&d_in[i], input_copy_block_size);
		cudaMalloc((void **)&d_out[i], output_copy_block_size);		
	}
	for (unsigned int j=0; j<NUMBER_OF_STREAMS; j++)
	{
		for (unsigned int i = 0; i < input_inc; i++)
		{        		
			h_in[j][i] = rand() * (sizeof(double)*8-1) * N;        
		}
	}

	for (unsigned int k=0; k<NUMBER_OF_LOOPS; k++){

		clock_t	cpu_clock = clock();

		double cpu_result = 0;
		double temp = 0;
		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) {
			temp = reduceCPU_Max(h_in[i], input_inc);
			printf("[%f]\n",temp);
			if (cpu_result < temp) {
				cpu_result = temp;
			}
			temp = 0;
		}

		cpu_clock = clock() - cpu_clock;
		printf("cpu max [%f], cpu time [%f seconds]\n", cpu_result, ((float)cpu_clock)/CLOCKS_PER_SEC);

		cudaEvent_t start, stop;
		float elapsedTime;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		for (unsigned int i = 0, j = 0; i < NUMBER_OF_STREAMS; i++, j += input_inc) 
		{
			cudaMemcpyAsync(d_in[i], h_in[i], input_copy_block_size, cudaMemcpyHostToDevice, stream_ptr[i]);
			//CUDA_CHECK_ERRORS("failed to copy memory (async) from host to device");
		}

		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
		{			
			reductionWorkerUsingStreams<double>(d_in[i], d_out[i], &reduction_task, stream_ptr[i]);
		}

		for (unsigned int i = 0, h = 0; i < NUMBER_OF_STREAMS; i++, h += output_inc) 
		{	
			cudaMemcpyAsync(h_out[i], d_out[i], output_copy_block_size, cudaMemcpyDeviceToHost, stream_ptr[i]);
			//CUDA_CHECK_ERRORS("failed to copy memory (async) from device to host");
		}

		//cudaDeviceSynchronize();
		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
		{ 
			cudaStreamSynchronize(stream_ptr[i]); 
			//CUDA_CHECK_ERRORS("failed to synchronize stream");
		} 
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);

		double gpu_max = 0;
		for (unsigned int j = 0; j < NUMBER_OF_STREAMS; j++) 
		{
		for (unsigned int i = 0; i < output_inc; i++)
		{
			if (gpu_max < h_out[j][i]) {
				gpu_max = h_out[j][i];
			}
		}
	}

		float bandwidthInMBs = (1e3f * input_size) / (elapsedTime * (float)(1 << 20));
		printf("gpu max [%f], gpu time [%f seconds] bandwidth (mb) [%f]\n", gpu_max, elapsedTime/1000.0, bandwidthInMBs);
	}
	for (int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{ 
		cudaStreamDestroy(stream_ptr[i]);  
		cudaFree(d_in[i]);
		cudaFree(d_out[i]);
		cudaFreeHost(h_in[i]);
		cudaFreeHost(h_out[i]);
		CUDA_CHECK_ERRORS("failed to destory stream");
	} 

	/*cudaFree(d_in);
	cudaFree(d_out);*/

	free(d_in);
	free(d_out);
	free(h_in);
	free(h_out);
	//cudaFreeHost(h_in);
	//CUDA_CHECK_ERRORS("cudaFreeHost -> h_in");
	//cudaFreeHost(h_out);
	//CUDA_CHECK_ERRORS("cudaFreeHost -> h_out");

}

void testCPU() 
{
	printf("testCPU\n");
	const unsigned int input_size = N * sizeof(double);
	const unsigned int output_size = BLOCK_SIZE * sizeof(double);

	double* h_in = (double*) malloc(input_size);

	for (unsigned int i = 0; i < N; i++)
	{        
		h_in[i] = i;      
	}

	clock_t cpu_clock;
	for (unsigned int k=0; k<NUMBER_OF_LOOPS; k++){
		cpu_clock = clock();
		double cpu_result = reduceCPU_Max(h_in, N);
			//reduceCPU<double>(h_in, N);
		cpu_clock = clock() - cpu_clock;
		printf("cpu result [%f], cpu time [%f seconds]\n", cpu_result, ((float)cpu_clock)/CLOCKS_PER_SEC);
	}
	printf("-------------\n");

	free(h_in);
}


int main( void ) { 
	testDeviceInfo();

	//testCPU();
	//testPageableMemory();
	//testMappedMemory();	
	testStream();
	
	//testStream_Max();

	return 0; 
}

