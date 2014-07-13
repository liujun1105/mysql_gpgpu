
#ifndef HOST_BUFFER_H

#define HOST_BUFFER_H

#include "cuda_runtime_api.h"

class ReductionTask;

template <class T>
class HostBuffer
{
public:

	T* ptr;                 // pointer holds the result
	T** h_in;
	T** h_out;
	T** d_in;
	T** d_out;
	unsigned int index;     // zero based
	unsigned int max_size;  // one based	
	unsigned int unit_size; // size of T
	unsigned short in_progress;

	cudaStream_t* stream_buffer;
	ReductionTask* reduction_task;
	cudaEvent_t start, stop;
	clock_t clock;
};

#endif /* HOST_BUFFER_H */