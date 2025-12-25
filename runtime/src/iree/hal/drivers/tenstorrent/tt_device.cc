// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_device.h"

#include <cstdio>
#include <cstring>

#include "iree/hal/drivers/tenstorrent/tt_allocator.h"

#ifndef TT_IREE_ENABLE_MOCK
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#endif

//===----------------------------------------------------------------------===//
// iree_hal_tt_device_t
//===----------------------------------------------------------------------===//

struct iree_hal_tt_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  
  iree_string_view_t identifier;
  iree_hal_device_id_t device_id;
  iree_hal_allocator_t* device_allocator;
  
#ifndef TT_IREE_ENABLE_MOCK
  tt::tt_metal::Device* tt_device;
  tt::tt_metal::CommandQueue* compute_queue;
#endif
};

static const iree_hal_device_vtable_t iree_hal_tt_device_vtable;

static iree_hal_tt_device_t* iree_hal_tt_device_cast(iree_hal_device_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_device_vtable);
  return (iree_hal_tt_device_t*)base;
}

//===----------------------------------------------------------------------===//
// Internal accessors
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK
tt::tt_metal::Device* iree_hal_tt_device_handle(iree_hal_tt_device_t* device) {
  return device ? device->tt_device : nullptr;
}

tt::tt_metal::CommandQueue* iree_hal_tt_device_queue(iree_hal_tt_device_t* device) {
  return device ? device->compute_queue : nullptr;
}
#endif

//===----------------------------------------------------------------------===//
// Device creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_device_create(
    iree_hal_tenstorrent_driver_t* driver,
    iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  iree_hal_tt_device_t* device = nullptr;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*device), (void**)&device);
  
  if (iree_status_is_ok(status)) {
    std::memset(device, 0, sizeof(*device));
    iree_hal_resource_initialize(&iree_hal_tt_device_vtable, &device->resource);
    device->host_allocator = host_allocator;
    device->identifier = iree_make_cstring_view("tenstorrent");
    device->device_id = device_id;
  }

#ifndef TT_IREE_ENABLE_MOCK
  // Open TT-Metal device
  if (iree_status_is_ok(status)) {
    try {
      device->tt_device = tt::tt_metal::CreateDevice(device_id);
      if (!device->tt_device) {
        status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                 "failed to open device %d", (int)device_id);
      }
    } catch (const std::exception& e) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                               "TT-Metal error: %s", e.what());
    }
  }
  
  // Get command queue
  if (iree_status_is_ok(status)) {
    try {
      device->compute_queue = &device->tt_device->command_queue();
      
      auto grid = device->tt_device->compute_with_storage_grid_size();
      auto arch = device->tt_device->arch();
      const char* arch_name = (arch == tt::ARCH::BLACKHOLE) ? "Blackhole" :
                              (arch == tt::ARCH::WORMHOLE_B0) ? "Wormhole" : "Unknown";
      
      fprintf(stderr, "tt-iree: Device %d opened (%s, %ux%u cores, %lu MB DRAM)\n",
              (int)device_id, arch_name, grid.x, grid.y,
              (unsigned long)(device->tt_device->num_dram_channels() *
                             device->tt_device->dram_size_per_channel() / (1024*1024)));
    } catch (const std::exception& e) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                               "failed to get queue: %s", e.what());
    }
  }
#else
  if (iree_status_is_ok(status)) {
    fprintf(stderr, "tt-iree: Device %d opened (MOCK MODE)\n", (int)device_id);
  }
#endif
  
  // Create allocator
  if (iree_status_is_ok(status)) {
    status = iree_hal_tt_allocator_create(device, host_allocator,
                                          &device->device_allocator);
  }
  
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    if (device) {
#ifndef TT_IREE_ENABLE_MOCK
      if (device->tt_device) {
        try { tt::tt_metal::CloseDevice(device->tt_device); } catch (...) {}
      }
#endif
      if (device->device_allocator) {
        iree_hal_allocator_release(device->device_allocator);
      }
      iree_allocator_free(host_allocator, device);
    }
  }
  
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Device vtable
//===----------------------------------------------------------------------===//

static void iree_hal_tt_device_destroy(iree_hal_device_t* base) {
  auto* device = iree_hal_tt_device_cast(base);
  iree_allocator_t host_allocator = device->host_allocator;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  fprintf(stderr, "tt-iree: Closing device %d\n", (int)device->device_id);
  
  if (device->device_allocator) {
    iree_hal_allocator_release(device->device_allocator);
  }
  
#ifndef TT_IREE_ENABLE_MOCK
  if (device->tt_device) {
    try { tt::tt_metal::CloseDevice(device->tt_device); } catch (...) {}
  }
#endif
  
  iree_allocator_free(host_allocator, device);
  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_tt_device_id(iree_hal_device_t* base) {
  return iree_hal_tt_device_cast(base)->identifier;
}

static iree_allocator_t iree_hal_tt_device_host_allocator(iree_hal_device_t* base) {
  return iree_hal_tt_device_cast(base)->host_allocator;
}

static iree_hal_allocator_t* iree_hal_tt_device_allocator(iree_hal_device_t* base) {
  return iree_hal_tt_device_cast(base)->device_allocator;
}

static void iree_hal_tt_device_replace_allocator(
    iree_hal_device_t* base, iree_hal_allocator_t* new_allocator) {
  auto* device = iree_hal_tt_device_cast(base);
  if (device->device_allocator) iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
  iree_hal_allocator_retain(new_allocator);
}

static void iree_hal_tt_device_replace_channel_provider(
    iree_hal_device_t*, iree_hal_channel_provider_t*) {}

static iree_status_t iree_hal_tt_device_trim(iree_hal_device_t*) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_query_i64(
    iree_hal_device_t* base, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  auto* device = iree_hal_tt_device_cast(base);
  *out_value = 0;
  
  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value = device->device_id;
    return iree_ok_status();
  }
  
#ifndef TT_IREE_ENABLE_MOCK
  if (iree_string_view_equal(category, IREE_SV("hal.device")) && device->tt_device) {
    auto grid = device->tt_device->compute_with_storage_grid_size();
    if (iree_string_view_equal(key, IREE_SV("core_count_x"))) {
      *out_value = grid.x;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("core_count_y"))) {
      *out_value = grid.y;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("dram_size"))) {
      *out_value = device->tt_device->num_dram_channels() *
                   device->tt_device->dram_size_per_channel();
      return iree_ok_status();
    }
  }
#endif
  
  return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown key '%.*s::%.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

// Stub implementations
static iree_status_t iree_hal_tt_device_create_channel(
    iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_channel_params_t,
    iree_hal_channel_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "channel not implemented");
}

static iree_status_t iree_hal_tt_device_create_command_buffer(
    iree_hal_device_t*, iree_hal_command_buffer_mode_t, iree_hal_command_category_t,
    iree_hal_queue_affinity_t, iree_host_size_t, iree_hal_command_buffer_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "command buffer not implemented");
}

static iree_status_t iree_hal_tt_device_create_event(
    iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_event_flags_t,
    iree_hal_event_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not implemented");
}

static iree_status_t iree_hal_tt_device_create_executable_cache(
    iree_hal_device_t*, iree_string_view_t, iree_loop_t, iree_hal_executable_cache_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "executable cache not implemented");
}

static iree_status_t iree_hal_tt_device_import_file(
    iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_memory_access_t,
    iree_io_file_handle_t*, iree_hal_external_file_flags_t, iree_hal_file_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "file import not implemented");
}

static iree_status_t iree_hal_tt_device_create_semaphore(
    iree_hal_device_t*, iree_hal_queue_affinity_t, uint64_t,
    iree_hal_semaphore_flags_t, iree_hal_semaphore_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "semaphore not implemented");
}

static iree_hal_semaphore_compatibility_t
iree_hal_tt_device_query_semaphore_compatibility(iree_hal_device_t*, iree_hal_semaphore_t*) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_tt_device_queue_alloca(
    iree_hal_device_t*, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_allocator_pool_t, iree_hal_buffer_params_t, iree_device_size_t,
    iree_hal_alloca_flags_t, iree_hal_buffer_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue alloca not implemented");
}

static iree_status_t iree_hal_tt_device_queue_dealloca(
    iree_hal_device_t*, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_buffer_t*, iree_hal_dealloca_flags_t) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue dealloca not implemented");
}

static iree_status_t iree_hal_tt_device_queue_read(
    iree_hal_device_t*, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_file_t*, uint64_t, iree_hal_buffer_t*, iree_device_size_t,
    iree_device_size_t, iree_hal_read_flags_t) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue read not implemented");
}

static iree_status_t iree_hal_tt_device_queue_write(
    iree_hal_device_t*, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_buffer_t*, iree_device_size_t, iree_hal_file_t*, uint64_t,
    iree_device_size_t, iree_hal_write_flags_t) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue write not implemented");
}

static iree_status_t iree_hal_tt_device_queue_execute(
    iree_hal_device_t*, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_command_buffer_t*, iree_hal_buffer_binding_table_t,
    iree_hal_execute_flags_t) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue execute not implemented");
}

static iree_status_t iree_hal_tt_device_queue_flush(
    iree_hal_device_t* base, iree_hal_queue_affinity_t) {
#ifndef TT_IREE_ENABLE_MOCK
  auto* device = iree_hal_tt_device_cast(base);
  if (device->compute_queue) {
    try { tt::tt_metal::Finish(*device->compute_queue); } catch (...) {}
  }
#endif
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_wait_semaphores(
    iree_hal_device_t*, iree_hal_wait_mode_t, const iree_hal_semaphore_list_t,
    iree_timeout_t, iree_hal_wait_flags_t) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "wait semaphores not implemented");
}

static iree_status_t iree_hal_tt_device_profiling_begin(
    iree_hal_device_t*, const iree_hal_device_profiling_options_t*) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_profiling_flush(iree_hal_device_t*) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_profiling_end(iree_hal_device_t*) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_tt_device_vtable = {
    .destroy = iree_hal_tt_device_destroy,
    .id = iree_hal_tt_device_id,
    .host_allocator = iree_hal_tt_device_host_allocator,
    .device_allocator = iree_hal_tt_device_allocator,
    .replace_device_allocator = iree_hal_tt_device_replace_allocator,
    .replace_channel_provider = iree_hal_tt_device_replace_channel_provider,
    .trim = iree_hal_tt_device_trim,
    .query_i64 = iree_hal_tt_device_query_i64,
    .create_channel = iree_hal_tt_device_create_channel,
    .create_command_buffer = iree_hal_tt_device_create_command_buffer,
    .create_event = iree_hal_tt_device_create_event,
    .create_executable_cache = iree_hal_tt_device_create_executable_cache,
    .import_file = iree_hal_tt_device_import_file,
    .create_semaphore = iree_hal_tt_device_create_semaphore,
    .query_semaphore_compatibility = iree_hal_tt_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_tt_device_queue_alloca,
    .queue_dealloca = iree_hal_tt_device_queue_dealloca,
    .queue_read = iree_hal_tt_device_queue_read,
    .queue_write = iree_hal_tt_device_queue_write,
    .queue_execute = iree_hal_tt_device_queue_execute,
    .queue_flush = iree_hal_tt_device_queue_flush,
    .wait_semaphores = iree_hal_tt_device_wait_semaphores,
    .profiling_begin = iree_hal_tt_device_profiling_begin,
    .profiling_flush = iree_hal_tt_device_profiling_flush,
    .profiling_end = iree_hal_tt_device_profiling_end,
};
