#ifndef REDUCTION_OPERATOR_H
#define REDUCTION_OPERATOR_H

template <typename T>
class ReductionAdd
{
public:
	__device__ T operator()(T opd1, T opd2);

	__device__ T identity();
};

template <typename T>
__device__ T ReductionAdd<T>::operator()(T opd1, T opd2)
{
	return opd1 + opd2;
}

template <typename T>
__device__ T ReductionAdd<T>::identity()
{
	return (T)0;
}

#endif
