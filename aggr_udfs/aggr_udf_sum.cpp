#include "aggr_udf_sum.h"
#include <ctime>

my_bool aggr_udf_sum_init(UDF_INIT *initid, UDF_ARGS *args, char *message) 
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum_init");	
	int num_of_args = args->arg_count;
	DBUG_ASSERT(num_of_args >= 1);
	DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("Number of Arguments -> [%d]", num_of_args));

	if (num_of_args > 1)
	{
		strcpy(message, "aggr_udf_sum_init() number of argument should equal to 1");
		DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("Number of arguments [%d > 1]", num_of_args));
		DBUG_RETURN(1);
	}

	// Check whether the argument are numeric types	
	if (args->arg_type[0] == STRING_RESULT || args->arg_type[0] == ROW_RESULT)
	{
		strcpy(message, "aggr_udf_sum_init() argument has to be a numeric number");
		DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("Argument type [%d] is invalid", args->arg_type[0]));
		DBUG_RETURN(1);
	}

	HostBuffer<double> *host_buffer = new HostBuffer<double>();
	
	host_buffer->index = 0;
	host_buffer->max_size = MAXIMUM_ELEMENTS_IN_CACHE;
	host_buffer->unit_size = sizeof(double);

	host_buffer->ptr = (double*) malloc(sizeof(double));
	host_buffer->d_in = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	host_buffer->d_out =(double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	host_buffer->h_in = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	host_buffer->h_out =(double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));

	host_buffer->stream_buffer = (cudaStream_t*)malloc(NUMBER_OF_STREAMS*sizeof(cudaStream_t));
	
	for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{ 
		cudaHostAlloc((void**)&host_buffer->h_in[i], INPUT_COPY_BLOCK_SIZE, cudaHostAllocDefault);
		cudaHostAlloc((void**)&host_buffer->h_out[i], OUTPUT_COPY_BLOCK_SIZE, cudaHostAllocDefault);
		cudaMalloc((void**)&host_buffer->d_in[i], INPUT_COPY_BLOCK_SIZE);
		cudaMalloc((void**)&host_buffer->d_out[i], OUTPUT_COPY_BLOCK_SIZE);
		cudaStreamCreate(&host_buffer->stream_buffer[i]);
	} 
	
	host_buffer->reduction_task;
	ReductionTask reduction_task(INPUT_INC, sizeof(double), OUTPUT_INC, CUDA_THREAD_PER_BLOCK_SIZE, R_SUM, R_DOUBLE);
	host_buffer->reduction_task = &reduction_task;
	
	/* store the host_buffer to the pointer */
	initid->ptr = (char*) host_buffer;

	initid->decimals = 3;
	
	cudaEventCreate(&host_buffer->start);
	cudaEventCreate(&host_buffer->stop);
	cudaEventRecord(host_buffer->start, 0);
	host_buffer->clock = clock();
	DBUG_RETURN(0);
}

void aggr_udf_sum_deinit(UDF_INIT *initid) 
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum_deinit");

	HostBuffer<double>* host_buffer = (HostBuffer<double>*) initid->ptr;
	
	float elapsedTime;
	cudaEventRecord(host_buffer->stop, 0);
	cudaEventSynchronize(host_buffer->stop);
	cudaEventElapsedTime(&elapsedTime, host_buffer->start, host_buffer->stop);
	float bandwidthInMBs = (1e3f * MAXIMUM_ELEMENTS_IN_CACHE*sizeof(double)) / (elapsedTime * (float)(1 << 20));
	fprintf(stderr, "gpu time [%f seconds], sum [%f], bandwidth (mb) [%f]\n", elapsedTime/1000.0, *host_buffer->ptr, bandwidthInMBs);
	fflush(stderr);

	host_buffer->clock  = clock() - host_buffer->clock ;
	fprintf(stderr, "gpu time [%f seconds]\n", ((float)host_buffer->clock)/CLOCKS_PER_SEC);
	fflush(stderr);

	for (int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{ 
		cudaStreamDestroy(host_buffer->stream_buffer[i]);  
		cudaFree(host_buffer->d_in[i]);
		cudaFree(host_buffer->d_out[i]);
		cudaFreeHost(host_buffer->h_in[i]);
		cudaFreeHost(host_buffer->h_out[i]);
	}
	free(host_buffer->stream_buffer);
	free(host_buffer->d_in);
	free(host_buffer->d_out);
	free(host_buffer->h_in);
	free(host_buffer->h_out);	
	free(initid->ptr);
	
	DBUG_VOID_RETURN;
}

void aggr_udf_sum_add(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error)
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum_add");

	HostBuffer<double>* host_buffer = (HostBuffer<double>*) initid->ptr;

	/*if (host_buffer->index == host_buffer->max_size)
	{
		DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("enter::add_not_success_block"));

		if (host_buffer->in_progress == 0) 
		{
			for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
			{ 				
				cudaStreamSynchronize(host_buffer->stream_buffer[i]); 				
			} 
			host_buffer->in_progress = 1;
			double gpu_result = 0;
			for (unsigned int i = 0; i < CUDA_BLOCK_SIZE; i++)
			{
				gpu_result += ((double*)host_buffer->h_out)[i];
			}
			*host_buffer->ptr += gpu_result;
		}

		for (unsigned int i = 0, j = 0; i < NUMBER_OF_STREAMS; i++, j += INPUT_INC) 
		{
			cudaMemcpyAsync(host_buffer->d_in[i], host_buffer->h_in+j, INPUT_COPY_BLOCK_SIZE, cudaMemcpyHostToDevice, 
				              host_buffer->stream_buffer[i]);
		}
		for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
		{			
			reductionWorkerUsingStreams<double>(host_buffer->d_in[i], host_buffer->d_out[i], 
				host_buffer->reduction_task, host_buffer->stream_buffer[i]);
		}
		for (unsigned int i = 0, h = 0; i < NUMBER_OF_STREAMS; i++, h += OUTPUT_INC) 
		{	
			cudaMemcpyAsync(host_buffer->h_out+h, host_buffer->d_out[i], OUTPUT_COPY_BLOCK_SIZE, 
				          cudaMemcpyDeviceToHost, host_buffer->stream_buffer[i]);
		}

		host_buffer->in_progress = 0;
		host_buffer->index = 0;

		DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("partial result [%f]", *host_buffer->ptr));		
		DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("exit::add_not_success_block"));
	}*/

	int y_index = host_buffer->index/INPUT_INC;
	int x_index = host_buffer->index%INPUT_INC;
	//fprintf(stderr, "[%d,%d,%d]\n", y_index, x_index,*(double *)args->args[0]);
	//fflush(stderr);
	host_buffer->h_in[y_index][x_index] = *(double *)args->args[0];
	host_buffer->index++;
	DBUG_VOID_RETURN;
}

void aggr_udf_sum_clear(UDF_INIT *initid, char *is_null, char *error)
{
	DBUG_ENTER("aggr_udf_sum_clear::aggr_udf_sum_clear");

	if (initid->ptr)
	{
		HostBuffer<double>* host_buffer = (HostBuffer<double> *) initid->ptr;
		host_buffer->in_progress = 1;
		host_buffer->index = 0;
		*host_buffer->ptr = 0;
	}
	DBUG_VOID_RETURN;
}

double aggr_udf_sum(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error)
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum");

	HostBuffer<double>* host_buffer = (HostBuffer<double> *) initid->ptr;

	fprintf(stderr, "number of records [%d]\n", host_buffer->index);
	fflush(stderr);

	for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{
		cudaMemcpyAsync(host_buffer->d_in[i], host_buffer->h_in[i], 
			INPUT_COPY_BLOCK_SIZE, cudaMemcpyHostToDevice, 
			host_buffer->stream_buffer[i]);
	}

	for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{		
		reductionWorkerUsingStreams<double>(
			host_buffer->d_in[i], host_buffer->d_out[i], 
			host_buffer->reduction_task, host_buffer->stream_buffer[i]);
	}

	for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{	
		cudaMemcpyAsync(host_buffer->h_out[i], host_buffer->d_out[i], 
			OUTPUT_COPY_BLOCK_SIZE, cudaMemcpyDeviceToHost, 
			host_buffer->stream_buffer[i]);
	}

	for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	{ 
		cudaStreamSynchronize(host_buffer->stream_buffer[i]); 				
	} 

	double gpu_result = 0;
	for (unsigned int j = 0; j < NUMBER_OF_STREAMS; j++) 
	{
		for (unsigned int i = 0; i < OUTPUT_INC; i++)
		{
			gpu_result += host_buffer->h_out[j][i];
		}
	}
	*host_buffer->ptr = gpu_result;

	DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("full result [%f]",*host_buffer->ptr));
	DBUG_RETURN(*host_buffer->ptr);
}

