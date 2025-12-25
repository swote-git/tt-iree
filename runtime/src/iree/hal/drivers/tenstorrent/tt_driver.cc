// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_driver.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "iree/hal/drivers/tenstorrent/tt_device.h"

#ifndef TT_IREE_ENABLE_MOCK
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#endif

//===----------------------------------------------------------------------===//
// iree_hal_tenstorrent_driver_t
//===----------------------------------------------------------------------===//

struct iree_hal_tenstorrent_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
  
#ifndef TT_IREE_ENABLE_MOCK
  std::vector<std::string> device_names;  // Persistent storage for device names
#endif
};

static const iree_hal_driver_vtable_t iree_hal_tenstorrent_driver_vtable;

static iree_hal_tenstorrent_driver_t* iree_hal_tenstorrent_driver_cast(
    iree_hal_driver_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tenstorrent_driver_vtable);
  return (iree_hal_tenstorrent_driver_t*)base;
}

//===----------------------------------------------------------------------===//
// Driver creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tenstorrent_driver_create(
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;
  
  iree_hal_tenstorrent_driver_t* driver = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*driver), (void**)&driver));
  
  new (driver) iree_hal_tenstorrent_driver_t();
  
  iree_hal_resource_initialize(&iree_hal_tenstorrent_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  driver->identifier = identifier;
  
  fprintf(stderr, "tt-iree: Creating Tenstorrent driver\n");
  
  *out_driver = (iree_hal_driver_t*)driver;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Driver vtable
//===----------------------------------------------------------------------===//

static void iree_hal_tenstorrent_driver_destroy(iree_hal_driver_t* base) {
  auto* driver = iree_hal_tenstorrent_driver_cast(base);
  iree_allocator_t host_allocator = driver->host_allocator;
  
  fprintf(stderr, "tt-iree: Destroying Tenstorrent driver\n");
  
  driver->~iree_hal_tenstorrent_driver_t();
  iree_allocator_free(host_allocator, driver);
}

static iree_status_t iree_hal_tenstorrent_driver_query_available_devices(
    iree_hal_driver_t* base,
    iree_allocator_t host_allocator,
    iree_host_size_t* out_count,
    iree_hal_device_info_t** out_infos) {
  auto* driver = iree_hal_tenstorrent_driver_cast(base);
  IREE_ASSERT_ARGUMENT(out_count);
  IREE_ASSERT_ARGUMENT(out_infos);

#ifdef TT_IREE_ENABLE_MOCK
  *out_count = 1;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(iree_hal_device_info_t), (void**)out_infos));
  (*out_infos)[0].device_id = 0;
  (*out_infos)[0].name = iree_make_cstring_view("Tenstorrent P100A (Mock)");
#else
  try {
    size_t device_count = tt::tt_metal::GetNumAvailableDevices();
    
    if (device_count == 0) {
      *out_count = 0;
      *out_infos = nullptr;
      return iree_ok_status();
    }
    
    *out_count = device_count;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, sizeof(iree_hal_device_info_t) * device_count,
        (void**)out_infos));
    
    driver->device_names.resize(device_count);
    
    for (size_t i = 0; i < device_count; i++) {
      (*out_infos)[i].device_id = i;
      
      // Create a temporary device to get info
      // this is expensive but only done once
      try {
        auto* device = tt::tt_metal::CreateDevice(i);
        auto grid = device->compute_with_storage_grid_size();
        auto arch = device->arch();
        const char* arch_name = (arch == tt::ARCH::BLACKHOLE) ? "Blackhole" :
                                (arch == tt::ARCH::WORMHOLE_B0) ? "Wormhole" : "Unknown";
        
        char name_buf[128];
        std::snprintf(name_buf, sizeof(name_buf), "Tenstorrent %s (%ux%u cores)",
                     arch_name, grid.x, grid.y);
        driver->device_names[i] = name_buf;
        
        tt::tt_metal::CloseDevice(device);
      } catch (...) {
        driver->device_names[i] = "Tenstorrent Device";
      }
      
      (*out_infos)[i].name = iree_make_cstring_view(driver->device_names[i].c_str());
    }
  } catch (const std::exception& e) {
    *out_count = 0;
    *out_infos = nullptr;
    return iree_make_status(IREE_STATUS_INTERNAL,
                           "failed to enumerate devices: %s", e.what());
  }
#endif
  
  return iree_ok_status();
}

static iree_status_t iree_hal_tenstorrent_driver_dump_device_info(
    iree_hal_driver_t* base,
    iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_string_builder_append_cstring(builder, "Tenstorrent Device\n");
  
#ifndef TT_IREE_ENABLE_MOCK
  try {
    auto* device = tt::tt_metal::CreateDevice(device_id);
    auto grid = device->compute_with_storage_grid_size();
    auto arch = device->arch();
    const char* arch_name = (arch == tt::ARCH::BLACKHOLE) ? "Blackhole" :
                            (arch == tt::ARCH::WORMHOLE_B0) ? "Wormhole" : "Unknown";
    
    uint64_t dram_size = device->num_dram_channels() * device->dram_size_per_channel();
    
    char buf[256];
    std::snprintf(buf, sizeof(buf), "  Architecture: %s\n", arch_name);
    iree_string_builder_append_cstring(builder, buf);
    std::snprintf(buf, sizeof(buf), "  Cores: %ux%u (%u total)\n",
                 grid.x, grid.y, grid.x * grid.y);
    iree_string_builder_append_cstring(builder, buf);
    std::snprintf(buf, sizeof(buf), "  DRAM: %lu MB\n",
                 (unsigned long)(dram_size / (1024*1024)));
    iree_string_builder_append_cstring(builder, buf);
    
    tt::tt_metal::CloseDevice(device);
  } catch (const std::exception& e) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "  Error: %s\n", e.what());
    iree_string_builder_append_cstring(builder, buf);
  }
#else
  iree_string_builder_append_cstring(builder, "  Architecture: Blackhole (Mock)\n");
  iree_string_builder_append_cstring(builder, "  Cores: 11x10 (110 total)\n");
#endif
  
  return iree_ok_status();
}

static iree_status_t iree_hal_tenstorrent_driver_create_device_by_id(
    iree_hal_driver_t* base,
    iree_hal_device_id_t device_id,
    iree_host_size_t param_count,
    const iree_string_pair_t* params,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  auto* driver = iree_hal_tenstorrent_driver_cast(base);
  return iree_hal_tt_device_create(driver, device_id, host_allocator, out_device);
}

static iree_status_t iree_hal_tenstorrent_driver_create_device_by_path(
    iree_hal_driver_t* base,
    iree_string_view_t driver_name,
    iree_string_view_t device_path,
    iree_host_size_t param_count,
    const iree_string_pair_t* params,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  if (iree_string_view_is_empty(device_path) ||
      iree_string_view_equal(device_path, IREE_SV("0"))) {
    return iree_hal_tenstorrent_driver_create_device_by_id(
        base, 0, param_count, params, host_allocator, out_device);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "device path '%.*s' not supported",
                         (int)device_path.size, device_path.data);
}

static const iree_hal_driver_vtable_t iree_hal_tenstorrent_driver_vtable = {
    .destroy = iree_hal_tenstorrent_driver_destroy,
    .query_available_devices = iree_hal_tenstorrent_driver_query_available_devices,
    .dump_device_info = iree_hal_tenstorrent_driver_dump_device_info,
    .create_device_by_id = iree_hal_tenstorrent_driver_create_device_by_id,
    .create_device_by_path = iree_hal_tenstorrent_driver_create_device_by_path,
};
