#include "aggr_udf_sum.h"

my_bool aggr_udf_sum_init(UDF_INIT *initid, UDF_ARGS *args, char *message) 
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum_init");	
	
	/* Allocate memory from the pool of the current thread descriptor.
	   The memory allocated use this method is freed when the query is
	   finished.
	 */
	double* float_total = (double*) sql_alloc(sizeof(double));
	*float_total = 0;
	initid->ptr = (char*) float_total;

	int num_of_args = args->arg_count;
	DBUG_ASSERT(num_of_args >= 1);
	DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("Number of Arguments -> [%d]", num_of_args));

	// Check whether all arguments are numeric types
	for (int i=0; i<num_of_args; i++)
	{
		if (args->arg_type[i] == STRING_RESULT || args->arg_type[i] == ROW_RESULT)
		{
			strcpy(message, "aggr_udf_sum_init() argument has to be a numeric number");
			DBUG_PRINT(AGGR_UDF_SUM_PRINT_TAG, ("Argument [%d] is invalid", i));
			DBUG_RETURN(1);
		}
	}

	initid->decimals = 3;

	DBUG_RETURN(0);
}

void aggr_udf_sum_deinit(UDF_INIT *initid) 
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum_deinit");
	DBUG_VOID_RETURN;
}

void aggr_udf_sum_add(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error)
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum_add");
	double* float_total;
	float_total = (double*)initid->ptr;

	int num_of_args = args->arg_count;
	DBUG_ASSERT(num_of_args >= 1);

	for (int i=0; i<num_of_args; i++) 
	{
		if (args->args[i]) {
			*float_total += *(double *)args->args[i];
		}
	}
	DBUG_VOID_RETURN;
}

void aggr_udf_sum_clear(UDF_INIT *initid, char *is_null, char *error)
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum_clear");
	double* float_total;
	float_total = (double*) initid->ptr;
	*float_total = 0;
	DBUG_VOID_RETURN;
}

double aggr_udf_sum(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error)
{
	DBUG_ENTER("aggr_udf_sum::aggr_udf_sum");
	double* float_total;
	float_total = (double*)initid->ptr;
	DBUG_RETURN(*float_total);
}

