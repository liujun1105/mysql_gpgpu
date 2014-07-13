/*
   DROP FUNCTION IF EXISTS aggr_udf_sum;
   CREATE AGGREGATE FUNCTION aggr_udf_sum RETURNS REAL SONAME 'aggr_udfs.dll';
   SELECT aggr_udf_sum(price) FROM test.rates;
   SELECT aggr_udf_sum(ase_double_col) FROM ase_project.ase_table;
 */

#ifndef AGGR_UDF_SUM_H

#define AGGR_UDF_SUM_H


#if defined(MYSQL_SERVER)
#include "m_string.h"		/* To get strmov() */
#else
/* when compiled as standalone */
#include <string.h>
#define strmov(a,b) stpcpy(a,b)
#endif

#include "my_sys.h"
#include "mysql.h"
#include "thr_malloc.h"
#include "my_dbug.h"
#include "reduction/reduce.h"
#include "host_buffer.h"
#include "reduction/reduction_task.h"

void *sql_alloc(size_t);

#define AGGR_UDF_SUM_PRINT_TAG "aggr_udf_sum"

#define MAXIMUM_ELEMENTS_IN_CACHE 102400000
#define CUDA_BLOCK_SIZE 256
#define CUDA_THREAD_PER_BLOCK_SIZE 256
#define NUMBER_OF_STREAMS 16

const unsigned int INPUT_INC = MAXIMUM_ELEMENTS_IN_CACHE/NUMBER_OF_STREAMS;
const unsigned int OUTPUT_INC = CUDA_BLOCK_SIZE/NUMBER_OF_STREAMS;
const unsigned int INPUT_COPY_BLOCK_SIZE = MAXIMUM_ELEMENTS_IN_CACHE/NUMBER_OF_STREAMS*sizeof(double);
const unsigned int OUTPUT_COPY_BLOCK_SIZE = CUDA_BLOCK_SIZE/NUMBER_OF_STREAMS*sizeof(double);

#ifdef  __cplusplus
extern "C" {
#endif

my_bool aggr_udf_sum_init(UDF_INIT *initid, UDF_ARGS *args, char *message) ;
void aggr_udf_sum_deinit(UDF_INIT *initid);
void aggr_udf_sum_add(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error);
void aggr_udf_sum_clear(UDF_INIT *initid, char *is_null, char *error);
double aggr_udf_sum(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error);

template double	
execReduction<double>(const void* h_in, void* h_out, 
	        const unsigned int number_of_elements,
			const unsigned int unit_size,
		    const unsigned int block_size,
			const unsigned int thread_size_per_block,
			const ReductionType r_type,
			const ReductionDataType r_datatype);

#ifdef	__cplusplus
}
#endif

#endif /* AGGR_UDF_SUM_H */