#include "dispatcher.h"
#include "reduction_task.h"

void dispatch(const void* d_in, void* d_out, ReductionTask* reduction_task)
{
	switch(reduction_task->r_datatype)
	{
	case R_DOUBLE:
		switch(reduction_task->r_type)
		{
		case R_SUM:	
			reduce<ReductionAdd<double>, double>((const double*)d_in, (double*)d_out, reduction_task);
			break;
		}
		break;
	default:
		break;
	}
}




