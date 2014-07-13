#ifndef DISPATCHER_H
#define DISPATCHER_H

#include "cuda_runtime_api.h"

class ReductionTask;

/* Template Methods */
template<class T>
class ReductionAdd;

template<class T>
class ReductionMax;

template <class Oper, typename T>
void reduce(const T* d_in, T* d_out, ReductionTask* reduction_task);

void dispatch(const void* d_in, void* d_out, ReductionTask* reduction_task);

template <class Oper, typename T>
void reduceUsingStream(const T* d_in, T* d_out, ReductionTask* reduction_task, cudaStream_t stream);

void dispatchKernalWithStream(const void* d_in, void* d_out, ReductionTask* reduction_task, cudaStream_t stream);

#endif // DISPATCHER_H