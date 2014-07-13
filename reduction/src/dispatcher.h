#ifndef DISPATCHER_H
#define DISPATCHER_H

class ReductionTask;

/* Template Methods */
template<class T>
class ReductionAdd;

template <class Oper, typename T>
void reduce(const T* d_in, T* d_out, ReductionTask* reduction_task);

void dispatch(const void* d_in, void* d_out, ReductionTask* reduction_task);

#endif // DISPATCHER_H