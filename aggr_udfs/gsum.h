/*
  DROP FUNCTION IF EXISTS gsum;
  CREATE FUNCTION gsum RETURNS REAL SONAME 'gsum.dll';

 */
#ifndef MYSQL_SERVER
#define MYSQL_SERVER
#endif

#ifndef GSUM_H

#define GSUM_H
//#define NDEBUG
#include "mysys_priv.h"
#include "reduction/reduce.h"
#include "host_buffer.h"

#define MAXIMUM_ELEMENTS_IN_CACHE 102400000
#define CUDA_BLOCK_SIZE 256
#define CUDA_THREAD_PER_BLOCK_SIZE 256
#define NUMBER_OF_STREAMS 16

const unsigned int INPUT_INC = MAXIMUM_ELEMENTS_IN_CACHE/NUMBER_OF_STREAMS;
const unsigned int OUTPUT_INC = CUDA_BLOCK_SIZE/NUMBER_OF_STREAMS;
const unsigned int INPUT_COPY_BLOCK_SIZE = MAXIMUM_ELEMENTS_IN_CACHE/NUMBER_OF_STREAMS*sizeof(double);
const unsigned int OUTPUT_COPY_BLOCK_SIZE = CUDA_BLOCK_SIZE/NUMBER_OF_STREAMS*sizeof(double);

class THD;
struct TABLE_LIST;
class Prelocking_strategy;
struct TABLE;

void *my_malloc(size_t size, myf my_flags);
void bzero(char* ptr, size_t size);
void simple_open_n_lock_tables(THD* thd, TABLE_LIST* tables);
void close_thread_tables(THD* thd);
bool open_and_lock_tables(THD *thd, TABLE_LIST *tables,
                          bool derived, uint flags,
                          Prelocking_strategy *prelocking_strategy);
#endif // GSUM_H