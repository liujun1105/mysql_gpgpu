/*
   CREATE AGGREGATE FUNCTION aggr_udf_sum RETURNS REAL SONAME 'aggr_udfs.dll';
   DROP FUNCTION aggr_udf_sum;

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

void *sql_alloc(size_t);

#define AGGR_UDF_SUM_PRINT_TAG "AGGR_UDF_SUM"

#ifdef  __cplusplus
extern "C" {
#endif

my_bool aggr_udf_sum_init(UDF_INIT *initid, UDF_ARGS *args, char *message) ;
void aggr_udf_sum_deinit(UDF_INIT *initid);
void aggr_udf_sum_add(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error);
void aggr_udf_sum_clear(UDF_INIT *initid, char *is_null, char *error);
double aggr_udf_sum(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error);

#ifdef	__cplusplus
}
#endif

#endif /* AGGR_UDF_SUM_H */