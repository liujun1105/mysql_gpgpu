#ifndef REDUCTION_CU
#define REDUCTION_CU

#include <stdio.h>
#include "reduce.h"
#include "reduction_operator.h"
#include "reduction_task.h"
#include "sharedmem.h"

template <typename T, class Oper, const unsigned blockSize, const bool isPow2>
__global__ void reduce(const T* d_in,           /* input data array */
	                   T* d_out,                /* reduced output array */
	                   const unsigned int N)    /* number of elements in d_in */
	
{
	Oper op;

	/* dynamically create an array on shared memory space */
	volatile T* shared_data = SharedMemory<T>();
	/* get current thread id */
	unsigned int tid = threadIdx.x;

	/* get the index of the element in the d_in array corresponding to the current thread */
	unsigned int d_in_index = blockIdx.x * (blockSize*2) + threadIdx.x;

	/* Populate shared_data array on the shared memory*/
	T tempSum = op.identity();	
	
	const unsigned int inc = gridDim.x * blockSize * 2;
	
	while (d_in_index < N)
	{
		asm ("prefetch.global.L2 [%0];"::"r"(&d_in[d_in_index]));
		//asm volatile ("prefetch.global.L2 [%0];"::"d"(d_in[d_in_index]));
		tempSum = op(tempSum,  d_in[d_in_index]);		
		//tempSum += d_in[d_in_index];	
		if (isPow2, d_in_index + blockSize < N) {
			asm ("prefetch.global.L2 [%0];"::"r"(&d_in[d_in_index+blockSize]));
			tempSum = op(tempSum,  d_in[d_in_index+blockSize]);
		}
			//tempSum += d_in[d_in_index+blockSize];
		d_in_index += inc;
		
	}
	shared_data[tid] = tempSum;	

	/* make sure each thread has done is own work to assign a value to the shared_data array */
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 64]); } __syncthreads(); }

	if (tid < 32)
	{
		if (blockSize >= 64) shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 32]);
		if (blockSize >= 32) shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 16]);
		if (blockSize >= 16) shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 8]);
		if (blockSize >= 8)  shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 4]);
		if (blockSize >= 4)  shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 2]);
		if (blockSize >= 2)  shared_data[tid] = tempSum = op(tempSum, shared_data[tid + 1]);
	}

	// only first thread of each block write data into output data array
	if (tid == 0) {
		d_out[blockIdx.x] = shared_data[0];
	}
}

template <class Oper, typename T>
void reduce(const T *d_in, 	
		    T* d_out,
		    ReductionTask* reduction_task
		   )
{
	dim3 dimBlock(reduction_task->thread_size_per_block, 1, 1);
	dim3 dimGrid(reduction_task->block_size, 1, 1);
	unsigned int shared_memory_size =  reduction_task->shared_memory_size;
	unsigned int N = reduction_task->number_of_elements;
	//printf("shared memory size [%d]\n", shared_memory_size); 
	if (reduction_task->isPowerOfTwo())
	{		
		switch(dimBlock.x) 
		{
		case 512:
			reduce<T, Oper, 512, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 256:
			reduce<T, Oper, 256, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 128:
			reduce<T, Oper, 128, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 64:
			reduce<T, Oper,  64, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 32:
			reduce<T, Oper,  32, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 16:
			reduce<T, Oper,  16, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 8:
			reduce<T, Oper,   8, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 4:
			reduce<T, Oper,   4, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 2:
			reduce<T, Oper,   2, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 1:
			reduce<T, Oper,   1, true><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		}			
	}
	else 
	{
		switch(dimBlock.x) 
		{
		case 512:
			reduce<T, Oper, 512, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 256:
			reduce<T, Oper, 256, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 128:
			reduce<T, Oper, 128, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 64:
			reduce<T, Oper,  64, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 32:
			reduce<T, Oper,  32, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 16:
			reduce<T, Oper,  16, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 8:
			reduce<T, Oper,   8, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 4:
			reduce<T, Oper,   4, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 2:
			reduce<T, Oper,   2, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		case 1:
			reduce<T, Oper,   1, false><<<dimGrid, dimBlock, shared_memory_size>>>(d_in, d_out, N);
			break;
		}	
	}
}

template 
void reduce<ReductionAdd<double>, double>(const double* d_in, double* d_out, ReductionTask* reduction_task);

template 
void reduce<ReductionMax<double>, double>(const double* d_in, double* d_out, ReductionTask* reduction_task);

template <class Oper, typename T>
void reduceUsingStream(const T *d_in, 	
		               T* d_out,
		               ReductionTask* reduction_task,
			           cudaStream_t stream
		              )
{
	dim3 dimBlock(reduction_task->thread_size_per_block, 1, 1);
	dim3 dimGrid(reduction_task->block_size, 1, 1);
	unsigned int shared_memory_size =  reduction_task->shared_memory_size;
	unsigned int N = reduction_task->number_of_elements;
	//printf("shared memory size [%d]\n", shared_memory_size); 
	if (reduction_task->isPowerOfTwo())
	{
		switch(dimBlock.x) 
		{
		case 512:
			reduce<T, Oper, 512, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 256:
			reduce<T, Oper, 256, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 128:
			reduce<T, Oper, 128, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 64:
			reduce<T, Oper,  64, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 32:
			reduce<T, Oper,  32, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 16:
			reduce<T, Oper,  16, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 8:
			reduce<T, Oper,   8, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 4:
			reduce<T, Oper,   4, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 2:
			reduce<T, Oper,   2, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 1:
			reduce<T, Oper,   1, true><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		}			
	}
	else 
	{
		switch(dimBlock.x) 
		{
		case 512:
			reduce<T, Oper, 512, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 256:
			reduce<T, Oper, 256, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 128:
			reduce<T, Oper, 128, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 64:
			reduce<T, Oper,  64, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 32:
			reduce<T, Oper,  32, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 16:
			reduce<T, Oper,  16, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 8:
			reduce<T, Oper,   8, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 4:
			reduce<T, Oper,   4, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 2:
			reduce<T, Oper,   2, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		case 1:
			reduce<T, Oper,   1, false><<<dimGrid, dimBlock, shared_memory_size, stream>>>(d_in, d_out, N);
			break;
		}	
	}
}

template 
void reduceUsingStream<ReductionAdd<double>, double>(const double* d_in, 
                                                     double* d_out, 
													 ReductionTask* reduction_task, 
													 cudaStream_t stream);

template 
void reduceUsingStream<ReductionMax<double>, double>(const double* d_in, 
                                                     double* d_out, 
													 ReductionTask* reduction_task, 
													 cudaStream_t stream);
#endif // REDUCTION_CU



