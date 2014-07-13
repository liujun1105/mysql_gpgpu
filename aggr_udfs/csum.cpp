#include "csum.h"
#include <ctime>
#include "table.h"
#include "my_dbug.h"
#include "sql_class.h"
#include "lock.h"
#include "sql_base.h" // open and lock table

extern "C"  my_bool csum_init(UDF_INIT *initid, UDF_ARGS *args, char *message) 
{
	DBUG_ENTER("csum::csum_init");	

	if (args->arg_count != 3)
	{
		strcpy(message, "Wrong number of arguments, expected [3]");
		DBUG_RETURN(1);
	}
	
	initid->decimals = 3;
	
	DBUG_RETURN(0);
}

extern "C"  void csum_deinit(UDF_INIT *initid) 
{
	DBUG_ENTER("csum::csum_deinit");
	if (initid->ptr)
	{
		HostBuffer<double>* host_buffer = (HostBuffer<double>*) initid->ptr;
		free(host_buffer->h_in);
		delete host_buffer;
	}

	DBUG_VOID_RETURN;
}

extern "C" double csum(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error)
{
	DBUG_ENTER("csum::csum");

	HostBuffer<double> *host_buffer = new HostBuffer<double>();
	initid->ptr = (char*) host_buffer;

	clock_t cpu_clock;
	cpu_clock = clock();
	
	host_buffer->index = 0;
	host_buffer->max_size = MAXIMUM_ELEMENTS_IN_CACHE;
	host_buffer->unit_size = sizeof(double);	
	host_buffer->h_in = (double*) malloc(sizeof(double)*host_buffer->max_size);
	host_buffer->ptr = (double*) malloc(sizeof(double));	
	cpu_clock = clock() - cpu_clock;
	fprintf(stderr, "setup time [%f seconds]\n", ((float)cpu_clock)/CLOCKS_PER_SEC);
	fflush(stderr);
	
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

	cpu_clock = clock();
	table->file->ha_rnd_init(true);

	while (table->file->ha_rnd_next(table->record[0]) == 0){
		host_buffer->h_in[host_buffer->index++] = table->field[1]->val_real();
	}
	table->file->ha_rnd_end();
	cpu_clock = clock() - cpu_clock;
	fprintf(stderr, "csum -> index [%d]\n", host_buffer->index);
	fprintf(stderr, "csum -> fill cache within [%f seconds]\n", ((float)cpu_clock)/CLOCKS_PER_SEC);
	fflush(stderr);
	
	cpu_clock = clock();	
	double cpu_sum = 0;
	for (unsigned int i = 0; i < host_buffer->index; i++)
	{
		cpu_sum += host_buffer->h_in[i];
	}	
	cpu_clock = clock() - cpu_clock;
	fprintf(stderr, "csum -> full result [%f]\n", cpu_sum);
	fprintf(stderr, "csum -> total time [%f seconds]\n", ((float)cpu_clock)/CLOCKS_PER_SEC);
	fflush(stderr);

	DBUG_PRINT("info", ("full result [%f]", cpu_sum));
	DBUG_RETURN(cpu_sum);
}
