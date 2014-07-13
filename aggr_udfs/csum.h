/*
   DROP FUNCTION IF EXISTS csum;
   CREATE FUNCTION csum RETURNS REAL SONAME 'csum.dll';   
 */
#ifndef MYSQL_SERVER
#define MYSQL_SERVER
#endif

#ifndef CSUM_H

#define CSUM_H

#include "my_sys.h"
#include "mysql.h"
#include "my_dbug.h"
#include "host_buffer.h"
#define MAXIMUM_ELEMENTS_IN_CACHE 92160000

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

#endif /* CSUM_H */