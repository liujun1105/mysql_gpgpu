#include "reduce.h"
#include <iostream>
#include <ctime>
using namespace std;

const unsigned int BLOCK_SIZE = 8;
const unsigned int THREAD_SIZE = 256;
const unsigned int N = 90000000;

template<class T>
T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;

    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int input_size_required(const int number_of_elements, const int unit_size)
{
	return number_of_elements * unit_size;
}

int main( void ) { 

	printf("number of input_size required [%d]\n", input_size_required(N, sizeof(double)));
	
	const unsigned int input_size = N * sizeof(double);
	const unsigned int output_size = BLOCK_SIZE * sizeof(double);

	double* h_in = (double*) malloc(input_size);
	double* h_out = (double*) malloc(output_size);
	

	for (unsigned int i = 0; i < N; i++)
	{        
		h_in[i] = 1;       
	}

	clock_t gpu_clock;
	gpu_clock = clock();
	double gpu_result = execReduction<double>(h_in, h_out, N, sizeof(double), BLOCK_SIZE, THREAD_SIZE, R_SUM, R_DOUBLE);
	gpu_clock = clock() - gpu_clock;
	printf("gpu result [%f], gpu time [%f seconds]\n", gpu_result, ((float)gpu_clock)/CLOCKS_PER_SEC);

	clock_t cpu_clock;
	cpu_clock = clock();
	double cpu_result = reduceCPU<double>(h_in, N);
	cpu_clock = clock() - cpu_clock;
	printf("cpu result [%f], cpu time [%f seconds]\n", cpu_result, ((float)cpu_clock)/CLOCKS_PER_SEC);
	
	free(h_out);
	free(h_in);
	
	return 0; 
}

