#ifndef REDUCTION_TASK_H
#define REDUCTION_TASK_H

#include "reduce.h"

class ReductionTask
{
public:
	ReductionTask();
	ReductionTask(const unsigned int number_of_elements,
				  const unsigned int element_unit_size,
		          const unsigned int block_size,
				  const unsigned int thread_size_per_block,
				  const ReductionType r_type,
				  const ReductionDataType r_datatype		 
				  );
	~ReductionTask(void);

	void allocStorage();
	void freeStorage();

	bool isPowerOfTwo();
	
	//void* d_out;
	unsigned int number_of_elements;
	unsigned int data_size;
	unsigned int block_size;
	unsigned int thread_size_per_block;
	unsigned int unit_size;
	unsigned int shared_memory_size;
	ReductionType r_type;
	ReductionDataType r_datatype;
};




#endif // REDUCTION_TASK_H