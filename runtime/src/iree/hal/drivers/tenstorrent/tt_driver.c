// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_driver.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/tenstorrent/tt_device.h"

//===----------------------------------------------------------------------===//
// iree_hal_tt_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  
  // Driver identifier for logging/debugging
  iree_string_view_t identifier;
  
  // TODO: Add TT-Metal system handle when not in mock mode
  // #ifndef TT_IREE_ENABLE_MOCK
  // tt_metal_system_t* system;
  // #endif
} iree_hal_tt_driver_t;

static const iree_hal_driver_vtable_t iree_hal_tt_driver_vtable;

static iree_hal_tt_driver_t* iree_hal_tt_driver_cast(
    iree_hal_driver_t* base_driver) {
  IREE_HAL_ASSERT_TYPE(base_driver, &iree_hal_tt_driver_vtable);
  return (iree_hal_tt_driver_t*)base_driver;
}

//===----------------------------------------------------------------------===//
// Driver creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_driver_create(
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  iree_hal_tt_driver_t* driver = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*driver), (void**)&driver);
  
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_tt_driver_vtable, &driver->resource);
    driver->host_allocator = host_allocator;
    driver->identifier = identifier;
    
#ifndef TT_IREE_ENABLE_MOCK
    // TODO: Initialize TT-Metal
    // status = tt_metal_system_init(&driver->system);
#endif
  }
  
  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    if (driver) {
      iree_allocator_free(host_allocator, driver);
    }
  }
  
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Driver vtable implementation
//===----------------------------------------------------------------------===//

static void iree_hal_tt_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_tt_driver_t* driver = iree_hal_tt_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
#ifndef TT_IREE_ENABLE_MOCK
  // TODO: Cleanup TT-Metal
  // tt_metal_system_shutdown(driver->system);
#endif
  
  iree_allocator_free(host_allocator, driver);
  
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_tt_driver_query_available_devices(
    iree_hal_driver_t* base_driver,
    iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  
#ifdef TT_IREE_ENABLE_MOCK
  // Mock mode: return one fake device
  *out_device_info_count = 1;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(iree_hal_device_info_t), (void**)&device_infos));
  
  device_infos[0].device_id = 0;
  device_infos[0].name = IREE_SVL("Tenstorrent P100A (Mock)");
  
  *out_device_infos = device_infos;
  return iree_ok_status();
#else
  // Real mode: query TT-Metal for available devices
  // TODO: Implement real device enumeration
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "real device enumeration not yet implemented");
#endif
}

static iree_status_t iree_hal_tt_driver_dump_device_info(
    iree_hal_driver_t* base_driver,
    iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  // TODO: Add device info dumping
  iree_string_builder_append_cstring(builder, "Tenstorrent Device\n");
  iree_string_builder_append_cstring(builder, "  Architecture: Wormhole\n");
  iree_string_builder_append_cstring(builder, "  Cores: 8x8 Tensix grid\n");
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_driver_create_device_by_id(
    iree_hal_driver_t* base_driver,
    iree_hal_device_id_t device_id,
    iree_host_size_t param_count,
    const iree_string_pair_t* params,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_tt_driver_t* driver = iree_hal_tt_driver_cast(base_driver);
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  iree_status_t status = iree_hal_tt_device_create(
      driver,
      device_id,
      host_allocator,
      out_device);
  
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_tt_driver_create_device_by_path(
    iree_hal_driver_t* base_driver,
    iree_string_view_t driver_name,
    iree_string_view_t device_path,
    iree_host_size_t param_count,
    const iree_string_pair_t* params,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  // For now, just use device ID 0 if path is empty or "0"
  if (iree_string_view_is_empty(device_path) ||
      iree_string_view_equal(device_path, IREE_SV("0"))) {
    return iree_hal_tt_driver_create_device_by_id(
        base_driver, 0, param_count, params, host_allocator, out_device);
  }
  
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "device path '%.*s' not supported",
                         (int)device_path.size, device_path.data);
}

//===----------------------------------------------------------------------===//
// vtable
//===----------------------------------------------------------------------===//

static const iree_hal_driver_vtable_t iree_hal_tt_driver_vtable = {
    .destroy = iree_hal_tt_driver_destroy,
    .query_available_devices = iree_hal_tt_driver_query_available_devices,
    .dump_device_info = iree_hal_tt_driver_dump_device_info,
    .create_device_by_id = iree_hal_tt_driver_create_device_by_id,
    .create_device_by_path = iree_hal_tt_driver_create_device_by_path,
};
