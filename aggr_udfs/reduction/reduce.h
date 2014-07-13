#ifndef REDUCE_H
#define REDUCE_H
#include "cuda_runtime_api.h"
class ReductionTask;


#define CUDA_CHECK_ERRORS(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

enum ReductionType 
{
	R_UNKNOWN,
	R_SUM, 
	R_COUNT, 
	R_MAX, 
	R_MIN
};

enum ReductionDataType 
{
	R_DOUBLE,
	R_INT,
	R_LONG,
	R_FLOAT
};

#ifdef __cplusplus
}
#endif

template<typename T>
void prealloc(const unsigned int number_of_elements, void** d_in_ptr);

template<typename T>
T  execReductionWithReuseCache(const void* h_in, void* h_out, void* d_in,
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype);

template<typename T>
T  execReduction(const void* h_in, void* h_out, 
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype);

template<typename T>
T reductionWorker(const void* h_in, void* h_out, 	        
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype);

template<typename T>
T reductionWorker(const void* h_in, void* h_out, void* d_in,       
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype);

template<typename T>
void reductionWorkerUsingMappedMemory(const T* d_in, T* d_out, 	        
	                                  ReductionTask* reduction_task);

template<typename T>
void reductionWorkerUsingStreams(const T* d_in, T* d_out,       
	                             ReductionTask* reduction_task,
			                     cudaStream_t stream);

#ifdef __cplusplus
extern "C" {
#endif

template double	
execReduction<double>(const void* h_in, void* h_out, 
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype);

template double	
execReductionWithReuseCache<double>(const void* h_in, void* h_out, void* d_in,
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype);

template void
reductionWorkerUsingMappedMemory(const double* d_in, double* d_out, 	        
								 ReductionTask* reduction_task);

template void
reductionWorkerUsingStreams(const double* d_in, double* d_out, 	        
	                        ReductionTask* reduction_task,
			                cudaStream_t stream);

template
void prealloc<double>(const unsigned int number_of_elements, void** d_in_ptr);
void free_prealloc(void* d_in);

void prealloc_with_mapped_memory(const unsigned int number_of_elements, unsigned int unit_size, void** mem_pptr, int flag);
void free_mapped_memory(void* mem_ptr);

#ifdef __cplusplus
}
#endif

#endif // REDUCE_H