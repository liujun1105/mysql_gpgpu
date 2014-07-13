# GPU Aware Column-Based storage engine plugin

SET(SPARTAN_PLUGIN_STATIC "gpucdb")
SET(SPARTAN_PLUGIN_DYNAMIC "gpucdb")

SET(GPUCDB_SOURCES ha_gpucdb.cc ha_gpucdb.h gpucdb_data.cc gpucdb_data.h)

MYSQL_ADD_PLUGIN(gpucdb ${GPUCDB_SOURCES} STORAGE_ENGINE MODULE_ONLY)

TARGET_LINK_LIBRARIES(gpucdb mysys)
