#include "gsum.h"
#include "table.h"
#include "my_dbug.h"
#include "sql_class.h"
#include "lock.h"
#include "sql_base.h" // open and lock table
#include "reduction/reduction_task.h"

Field *get_field(TABLE *table, const char *name) {
	Field **f;
	for (f = table->field; *f != NULL; f++) {
		if (strcmp((*f)->field_name, name) == 0) 
		{
			return *f;
		}
	}
	return NULL;
}

int number_of_field(TABLE *table) {
	Field **f;
	int count = 0;
	for (f = table->field; *f != NULL; f++) {
		count++;
	}
	return count;
}

extern "C" my_bool gsum_init(UDF_INIT *initid, 
	UDF_ARGS *args, 
	char *message)
{
	DBUG_ENTER("udf_sum::gsum_init");
	if (args->arg_count != 3)
	{
		strcpy(message, "Wrong number of arguments, expected [3]");
		DBUG_RETURN(1);
	}
	initid->decimals = 3;

	//HostBuffer<double> *host_buffer = new HostBuffer<double>();
	//initid->ptr = (char*) host_buffer;
	//clock_t cpu_clock;
	//cpu_clock = clock();
	//
	//host_buffer->index = 0;
	//host_buffer->max_size = MAXIMUM_ELEMENTS_IN_CACHE;
	//host_buffer->unit_size = sizeof(double);
	///*host_buffer->h_in = (double*) malloc(sizeof(double)*host_buffer->max_size);
	//host_buffer->h_out = (double*) malloc(sizeof(double)*CUDA_BLOCK_SIZE);
	//host_buffer->ptr = (double*) malloc(sizeof(double));
	//host_buffer->d_in = malloc(sizeof(void*));
	//host_buffer->d_in_ptr = &host_buffer->d_in;
	//prealloc<double>(MAXIMUM_ELEMENTS_IN_CACHE, host_buffer->d_in_ptr);*/

	//host_buffer->ptr = (double*) malloc(sizeof(double));

	//host_buffer->d_in = (double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	//host_buffer->d_out =(double**)malloc(NUMBER_OF_STREAMS*sizeof(double*));
	//for (unsigned int i = 0; i < NUMBER_OF_STREAMS; i++) 
	//{ 
	//	prealloc<double>(MAXIMUM_ELEMENTS_IN_CACHE/NUMBER_OF_STREAMS, (void**)&host_buffer->d_in[i]);
	//	CUDA_CHECK_ERRORS("failed to prealloc host_buffer->d_in");
	//	prealloc<double>(CUDA_BLOCK_SIZE/NUMBER_OF_STREAMS, (void**)&host_buffer->d_out[i]);
	//	CUDA_CHECK_ERRORS("failed to prealloc host_buffer->d_out");
	//} 
	//prealloc_with_mapped_memory(host_buffer->max_size, host_buffer->unit_size, (void**)&host_buffer->h_in, cudaHostAllocDefault);
	//prealloc_with_mapped_memory(CUDA_BLOCK_SIZE, host_buffer->unit_size, (void**)&host_buffer->h_out, cudaHostAllocDefault);
	//memset(host_buffer->h_in, 0, host_buffer->max_size*host_buffer->unit_size);
	//memset(host_buffer->h_out, 0, CUDA_BLOCK_SIZE*host_buffer->unit_size);

	//cpu_clock = clock() - cpu_clock;
	//fprintf(stderr, "setup time [%f seconds]\n", ((float)cpu_clock)/CLOCKS_PER_SEC);
	//fflush(stderr);

	DBUG_RETURN(0);
}

extern "C" double gsum(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) 
{
	DBUG_ENTER("udf_sum::gsum");

	double* h_in;
	double* h_out;
	double* d_in;
	double* d_out;
	unsigned int index = 0;

	cudaHostAlloc((void**)&h_in, MAXIMUM_ELEMENTS_IN_CACHE*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped );
	CUDA_CHECK_ERRORS("cudaHostAlloc -> h_in");
	cudaHostAlloc((void**)&h_out, CUDA_BLOCK_SIZE*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped );
	CUDA_CHECK_ERRORS("cudaHostAlloc -> h_out");

	cudaHostGetDevicePointer((void**)&d_in, h_in, 0);
	cudaHostGetDevicePointer((void**)&d_out, h_out, 0);


	char* column_name = (char*) args->args[0];
	char* table_name = (char*) args->args[1];
	char* schema_name = (char*) args->args[2];

	DBUG_PRINT("info", ("column_name [%s], table_name [%s], schema_name [%s]", column_name, table_name, schema_name));
	fprintf(stderr, "column_name [%s], table_name [%s], schema_name [%s]\n", column_name, table_name, schema_name);
	fflush(stderr);

	THD *thd = current_thd;

	TABLE_LIST* table_list = new TABLE_LIST;	
	memset((char*) table_list, 0, sizeof(TABLE_LIST));

	DBUG_PRINT("info", ("table_list->init_one_table"));
	table_list->init_one_table(schema_name, strlen(schema_name), table_name, strlen(table_name), table_name, TL_READ);
	DBUG_PRINT("info", ("open_and_lock_tables"));
	open_and_lock_tables(thd, table_list, FALSE, MYSQL_OPEN_IGNORE_FLUSH | MYSQL_LOCK_IGNORE_TIMEOUT);

	TABLE* table = table_list->table;

	clock_t cpu_clock;
	cpu_clock = clock();
	table->file->ha_rnd_init(true);

	while (table->file->ha_rnd_next(table->record[0]) == 0){
		h_in[index++] = table->field[1]->val_real();
	}
	table->file->ha_rnd_end();
	cpu_clock = clock() - cpu_clock;
	fprintf(stderr, "gsum -> index [%d]\n", index);
	fprintf(stderr, "gsum -> fill cache within [%f seconds]\n", ((float)cpu_clock)/CLOCKS_PER_SEC);
	fflush(stderr);
	

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	ReductionTask reduction_task(MAXIMUM_ELEMENTS_IN_CACHE, sizeof(double), CUDA_BLOCK_SIZE, CUDA_THREAD_PER_BLOCK_SIZE, R_SUM, R_DOUBLE);
	reductionWorkerUsingMappedMemory<double>(d_in, d_out, &reduction_task);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	double gpu_sum = 0;
	for (unsigned int i = 0; i < CUDA_BLOCK_SIZE; i++)
	{
		gpu_sum += ((double*)h_out)[i];
	}

	float bandwidthInMBs = (1e3f * MAXIMUM_ELEMENTS_IN_CACHE*sizeof(double)) / (elapsedTime * (float)(1 << 20));
	fprintf(stderr, "gpu result [%f], gpu time [%f seconds] bandwidth (mb) [%f]\n", gpu_sum, elapsedTime/1000.0, bandwidthInMBs);
	fflush(stderr);

	cudaFreeHost(h_in);
	CUDA_CHECK_ERRORS("cudaFreeHost -> h_in");
	cudaFreeHost(h_out);
	CUDA_CHECK_ERRORS("cudaFreeHost -> h_out");

	thd->cleanup_after_query();
	DBUG_PRINT("info", ("about to delete table_list"));
	delete table_list;

	DBUG_RETURN(gpu_sum);
}

extern "C" void gsum_deinit(UDF_INIT *initid)
{
	DBUG_ENTER("udf_sum::gsum_deinit");
	//if (initid->ptr)
	//{
	//	HostBuffer<double>* host_buffer = (HostBuffer<double> *) initid->ptr;	
	//	for (int i = 0; i < NUMBER_OF_STREAMS; i++) 
	//	{	 
	//		free_prealloc(host_buffer->d_in[i]);
	//		CUDA_CHECK_ERRORS("free_prealloc -> d_in");
	//		free_prealloc(host_buffer->d_out[i]);
	//		CUDA_CHECK_ERRORS("free_prealloc -> d_out");
	//	}
	//	host_buffer->index = 0;
	//	*host_buffer->ptr = 0;
	//}
	//delete initid->ptr;
	DBUG_VOID_RETURN;
}

