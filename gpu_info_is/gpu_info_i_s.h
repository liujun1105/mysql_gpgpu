#ifndef MYSQL_SERVER
#define MYSQL_SERVER
#endif

#ifndef GPU_INFO_I_S_H

#define GPU_INFO_I_S_H

#include <stdio.h>
#include "GPUInfoHelper.h"

#include "mysql/plugin.h"
#include "m_ctype.h"
#include "my_dbug.h"
#include "sql_class.h"
#include "sql_show.h"

bool schema_table_store_record(THD *thd, TABLE *table);

#endif