#include "reduction_task.h"

ReductionTask::ReductionTask(
	const unsigned int number_of_elements,
	const unsigned int element_unit_size,
	const unsigned int block_size,
	const unsigned int thread_size_per_block,
	const ReductionType r_type,
	const ReductionDataType r_datatype
	) 
{
	this->number_of_elements = number_of_elements;
	this->block_size = block_size;
	this->thread_size_per_block = thread_size_per_block;
	this->r_type = r_type;
	this->unit_size = element_unit_size;
	this->shared_memory_size = (thread_size_per_block <= 32) ? 2 * thread_size_per_block * unit_size : thread_size_per_block * unit_size;
	this->r_datatype = r_datatype;	
}

ReductionTask::~ReductionTask(void)
{	
}

bool ReductionTask::isPowerOfTwo() 
{
	/* treat 0 is also power of 2 */
	//(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	return (number_of_elements & (number_of_elements - 1)) == 0;

}
