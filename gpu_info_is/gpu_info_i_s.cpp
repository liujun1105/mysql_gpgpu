#include "gpu_info_i_s.h"

/*
Field array, which contains the content of the GPU_INFO_I_S_TABLE.
The array contains one element per column plus a terminating element

-- ST_FIELD_INFO is in the table.h
*/
static ST_FIELD_INFO gpu_info_table_fields[]=
{
	/* field_name, field_length, field_type, value (unused), field_flags, old_name (corresponding to SHOW statement), open_method (not important)*/
	{"DEVICE_ID", 10, MYSQL_TYPE_LONG, 0, MY_I_S_UNSIGNED | MY_I_S_MAYBE_NULL, 0, 0},
	{"GPU_DEVICE_NAME", 20, MYSQL_TYPE_STRING, 0, MY_I_S_MAYBE_NULL, 0, 0},
	{"DEVICE_VERSION", 3, MYSQL_TYPE_FLOAT, 0, MY_I_S_MAYBE_NULL, 0, 0},
	{"GLOBAL_MEMORY_IN_MB", 10, MYSQL_TYPE_FLOAT, 0, MY_I_S_MAYBE_NULL, 0, 0},
	{"MULTI_PROCESSORS", 3, MYSQL_TYPE_TINY, 0, MY_I_S_UNSIGNED | MY_I_S_MAYBE_NULL, 0, 0},
	// terminating element
	{0, 0, MYSQL_TYPE_NULL, 0, 0, 0, 0}
};

static int gpu_info_fill_table(THD *thd, TABLE_LIST *tables, Item *cond)
{
	DBUG_ENTER("gpu_info_i_s::gpu_info_fill_table");

	CUresult error_id = initDriverAPI();
	if (error_id == CUDA_SUCCESS) {
		fprintf(stdout, "CUDA Driver API Initialisation Successfully\n");
		TABLE *table= tables->table;
		CHARSET_INFO *cs = system_charset_info;
		/* call store() to install the row
		store(const char *to, uint length, CHARSET_INFO *cs);
		*/
		int deviceCount = CUDAEnabledDeviceCount();
		if (deviceCount > 0) 
		{
			CUdevice dev;
			for (dev = 0; dev < deviceCount; dev++) {
				
				// set device id
				table->field[0]->store(dev, true);
				DBUG_PRINT("GPU_INFO_IS_DEBUG", ("device id [%d]", dev));

				// set device name
				char device_name[20];
				int error_code = getDeviceName(dev, device_name, 20);
				if (error_code == 0) {
					DBUG_PRINT("GPU_INFO_IS_DEBUG", ("device name [%s]", device_name));
					table->field[1]->set_notnull();
					table->field[1]->store(device_name, strlen(device_name), cs);
				}
				else {
					table->field[1]->set_null();
				}

				// set device version
				int device_version = getDeviceVersion();				
				if (device_version > 0) {
					DBUG_PRINT("GPU_INFO_IS_DEBUG", ("device version [%d.%d]", device_version/1000, (device_version%100)/10));
					float device_version_float =  ((float)device_version/1000.0f+ (float)(device_version%100)/10.0f);
					fprintf(stderr, "device version [%f]\n", device_version_float);
					table->field[2]->set_notnull();
					table->field[2]->store(device_version_float);
				}
				

				// set global memory size
				size_t gmem_in_bytes = getDeviceTotalGlobalMemorySize(dev);
				float gmem_in_mbytes = getDeviceTotalGlobalMemoryInMB(gmem_in_bytes);
				DBUG_PRINT("GPU_INFO_IS_DEBUG", ("device global memory [%f]", gmem_in_mbytes));
				table->field[3]->set_notnull();
				table->field[3]->store(gmem_in_mbytes);

				int multi_processor_count = getMultiProcessorCount(dev);
				table->field[4]->set_notnull();
				table->field[4]->store(multi_processor_count, true);

				bool ret = schema_table_store_record(thd, table);
				if(ret) {
					DBUG_RETURN(1);
				}
				DBUG_RETURN(0);
			}			
		} 
		else {
			DBUG_PRINT("GPU_INFO_IS_ERROR", ("No CUDA Device Found"));
			DBUG_RETURN(1);
		}							
		
	} else {
		DBUG_PRINT("GPU_INFO_IS_ERROR", ("CUDA Driver API Initialisation Failed"));
		// still return 0 as we should display empty table
	}

	DBUG_RETURN(0);
}


/*
This function returns 0 for sucess, 1 if error occurs.
This function receives a generic pointer, 
which it should interpret as a pointer to the table structure
*/
static int gpu_info_table_init(void *ptr)
{
	ST_SCHEMA_TABLE *schema_table= (ST_SCHEMA_TABLE*)ptr;

	/* An array of ST_FIELD_INFO structures that contain information about each column */
	schema_table->fields_info= gpu_info_table_fields; 
	/* A function that populates the table */
	schema_table->fill_table= gpu_info_fill_table; 
	return 0;
}

static struct st_mysql_information_schema gpu_info_table_info =
{ MYSQL_INFORMATION_SCHEMA_INTERFACE_VERSION };

mysql_declare_plugin(gpu_info_i_s_library)
{
	MYSQL_INFORMATION_SCHEMA_PLUGIN,
		&gpu_info_table_info,                /* type-specific descriptor */
		"GPU_INFO",                          /* table name */
		"Jun Liu (jun.liu@dcu.ie)",          /* author */
		"GPU Info INFORMATION_SCHEMA table", /* description */
		PLUGIN_LICENSE_GPL,                  /* license type */
		gpu_info_table_init,                 /* init function */
		NULL,
		0x0100,                              /* version = 1.0 */
		NULL,                                /* no status variables */
		NULL,                                /* no system variables */
		NULL,                                /* no reserved information */
		0                                    /* no flags */
}
mysql_declare_plugin_end;

