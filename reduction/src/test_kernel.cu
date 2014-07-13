//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <stdio.h>
//#include <iostream>
//using namespace::std;
//__global__ void test(int* d_in, int* d_out)
//{
//	for (int i=0; i<5; i++)
//	{
//		d_out[i] = d_in[i]+1;
//	}
//}
//
//void test_wrapper(void* d_in, void* d_out)
//{
//	test<<<1,1>>>((int*)d_in, (int*)d_out);
//}
//
//int main( void ) 
//{
//	int* h_out;
//	int* h_in;
//	double* d_in;
//	double* d_out;
//
//	h_in = (int*)malloc(sizeof(int)*5);
//	h_out = (int*)malloc(sizeof(int)*5);
//	cudaMalloc(&d_in, 5*sizeof(int));
//	cudaMalloc(&d_out, 5*sizeof(int));
//
//	for (unsigned int i=0; i<5; i++)
//	{
//		h_in[i] = 1;
//	}
//	
//	cudaMemcpy(d_in, h_in, 5*sizeof(int), cudaMemcpyHostToDevice);
//
//	test_wrapper(d_in, d_out);
//	
//	cudaMemcpy(h_out, d_out, 5*sizeof(int), cudaMemcpyDeviceToHost);
//
//	for (int i=0; i<5; i++)
//	{
//		printf("---> [%d]\n", h_out[i]);
//	}
//
//	cudaFree(d_in);
//	cudaFree(d_out);
//	free(h_in);
//	free(h_out);
//
//	char ch;
//	cin >> ch;
//
//	return 0; 
//}