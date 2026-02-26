// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_device.h"

#include <cstdio>
#include <cstring>

#include "iree/hal/drivers/tenstorrent/tt_allocator.h"
#include "iree/hal/drivers/tenstorrent/tt_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/hal/drivers/tenstorrent/tt_semaphore.h"
#include "iree/hal/drivers/tenstorrent/tt_executable_cache.h"

#ifndef TT_IREE_ENABLE_MOCK
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/distributed.hpp"
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
  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
  tt::tt_metal::distributed::MeshCommandQueue* mesh_cq;
#endif
};

// Forward declarations of vtable functions
static void iree_hal_tt_device_destroy(iree_hal_device_t* base);
static iree_string_view_t iree_hal_tt_device_id(iree_hal_device_t* base);
static iree_allocator_t iree_hal_tt_device_host_allocator(iree_hal_device_t* base);
static iree_hal_allocator_t* iree_hal_tt_device_allocator(iree_hal_device_t* base);
static void iree_hal_tt_device_replace_allocator(iree_hal_device_t*, iree_hal_allocator_t*);
static void iree_hal_tt_device_replace_channel_provider(iree_hal_device_t*, iree_hal_channel_provider_t*);
static iree_status_t iree_hal_tt_device_trim(iree_hal_device_t*);
static iree_status_t iree_hal_tt_device_query_i64(iree_hal_device_t*, iree_string_view_t, iree_string_view_t, int64_t*);
static iree_status_t iree_hal_tt_device_create_channel(iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_channel_params_t, iree_hal_channel_t**);
// Implemented in tt_command_buffer.cc
extern iree_status_t iree_hal_tt_device_create_command_buffer(iree_hal_device_t*, iree_hal_command_buffer_mode_t, iree_hal_command_category_t, iree_hal_queue_affinity_t, iree_host_size_t, iree_hal_command_buffer_t**);
static iree_status_t iree_hal_tt_device_create_event(iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_event_flags_t, iree_hal_event_t**);
static iree_status_t iree_hal_tt_device_create_executable_cache(iree_hal_device_t*, iree_string_view_t, iree_loop_t, iree_hal_executable_cache_t**);
static iree_status_t iree_hal_tt_device_import_file(iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_memory_access_t, iree_io_file_handle_t*, iree_hal_external_file_flags_t, iree_hal_file_t**);
static iree_status_t iree_hal_tt_device_create_semaphore(iree_hal_device_t*, iree_hal_queue_affinity_t, uint64_t, iree_hal_semaphore_flags_t, iree_hal_semaphore_t**);
static iree_hal_semaphore_compatibility_t iree_hal_tt_device_query_semaphore_compatibility(iree_hal_device_t*, iree_hal_semaphore_t*);
static iree_status_t iree_hal_tt_device_queue_alloca(iree_hal_device_t*, iree_hal_queue_affinity_t, const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t, iree_hal_allocator_pool_t, iree_hal_buffer_params_t, iree_device_size_t, iree_hal_alloca_flags_t, iree_hal_buffer_t**);
static iree_status_t iree_hal_tt_device_queue_dealloca(iree_hal_device_t*, iree_hal_queue_affinity_t, const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t, iree_hal_buffer_t*, iree_hal_dealloca_flags_t);
static iree_status_t iree_hal_tt_device_queue_read(iree_hal_device_t*, iree_hal_queue_affinity_t, const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t, iree_hal_file_t*, uint64_t, iree_hal_buffer_t*, iree_device_size_t, iree_device_size_t, iree_hal_read_flags_t);
static iree_status_t iree_hal_tt_device_queue_write(iree_hal_device_t*, iree_hal_queue_affinity_t, const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t, iree_hal_buffer_t*, iree_device_size_t, iree_hal_file_t*, uint64_t, iree_device_size_t, iree_hal_write_flags_t);
static iree_status_t iree_hal_tt_device_queue_execute(iree_hal_device_t*, iree_hal_queue_affinity_t, const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t, iree_hal_command_buffer_t*, iree_hal_buffer_binding_table_t, iree_hal_execute_flags_t);
static iree_status_t iree_hal_tt_device_queue_flush(iree_hal_device_t*, iree_hal_queue_affinity_t);
static iree_status_t iree_hal_tt_device_wait_semaphores(iree_hal_device_t*, iree_hal_wait_mode_t, const iree_hal_semaphore_list_t, iree_timeout_t, iree_hal_wait_flags_t);
static iree_status_t iree_hal_tt_device_profiling_begin(iree_hal_device_t*, const iree_hal_device_profiling_options_t*);
static iree_status_t iree_hal_tt_device_profiling_flush(iree_hal_device_t*);
static iree_status_t iree_hal_tt_device_profiling_end(iree_hal_device_t*);

// Define vtable early so it can be used by cast function
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

static iree_hal_tt_device_t* iree_hal_tt_device_cast(iree_hal_device_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_device_vtable);
  return (iree_hal_tt_device_t*)base;
}

//===----------------------------------------------------------------------===//
// Internal accessors
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK

tt::tt_metal::distributed::MeshDevice* iree_hal_tt_device_mesh_handle(iree_hal_tt_device_t* device) {
  return device ? device->mesh_device.get() : nullptr;
}

tt::tt_metal::distributed::MeshCommandQueue* iree_hal_tt_device_mesh_queue(iree_hal_tt_device_t* device) {
  return device ? device->mesh_cq : nullptr;
}

tt::tt_metal::IDevice* iree_hal_tt_device_idevice(iree_hal_tt_device_t* device) {
  if (!device || !device->mesh_device) return nullptr;
  // For unit mesh, get the single device at coordinate {0,0}
  return device->mesh_device->get_device({0, 0});
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
  // Create TT-Metal MeshDevice (v0.65 API - even for single device)
  if (iree_status_is_ok(status)) {
    try {
      // Create a 1x1 mesh (unit mesh) for single device
      device->mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

      if (!device->mesh_device) {
        status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                 "failed to create mesh device %d", (int)device_id);
      }
    } catch (const std::exception& e) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                               "TT-Metal error: %s", e.what());
    }
  }

  // Get mesh command queue and print device info
  if (iree_status_is_ok(status)) {
    try {
      device->mesh_cq = &device->mesh_device->mesh_command_queue();

      // Get underlying IDevice for queries
      tt::tt_metal::IDevice* idevice = device->mesh_device->get_device({0, 0});
      if (idevice) {
        auto grid = idevice->compute_with_storage_grid_size();
        auto arch = idevice->arch();
        const char* arch_name = (arch == tt::ARCH::BLACKHOLE) ? "Blackhole" :
                                (arch == tt::ARCH::WORMHOLE_B0) ? "Wormhole" : "Unknown";

        fprintf(stderr, "tt-iree: MeshDevice %d opened (%s, %ux%u cores, %lu MB DRAM)\n",
                (int)device_id, arch_name, grid.x, grid.y,
                (unsigned long)(idevice->num_dram_channels() *
                               idevice->dram_size_per_channel() / (1024*1024)));
      }
    } catch (const std::exception& e) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                               "failed to get mesh queue: %s", e.what());
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
      if (device->mesh_device) {
        try { device->mesh_device->close(); } catch (...) {}
        // shared_ptr will handle deletion
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

  fprintf(stderr, "tt-iree: Closing mesh device %d\n", (int)device->device_id);

  if (device->device_allocator) {
    iree_hal_allocator_release(device->device_allocator);
  }

#ifndef TT_IREE_ENABLE_MOCK
  if (device->mesh_device) {
    try {
      device->mesh_device->close();
      // shared_ptr will handle deletion
    } catch (...) {}
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
  if (iree_string_view_equal(category, IREE_SV("hal.device")) && device->mesh_device) {
    tt::tt_metal::IDevice* idevice = device->mesh_device->get_device({0, 0});
    if (idevice) {
      auto grid = idevice->compute_with_storage_grid_size();
      if (iree_string_view_equal(key, IREE_SV("core_count_x"))) {
        *out_value = grid.x;
        return iree_ok_status();
      }
      if (iree_string_view_equal(key, IREE_SV("core_count_y"))) {
        *out_value = grid.y;
        return iree_ok_status();
      }
      if (iree_string_view_equal(key, IREE_SV("dram_size"))) {
        *out_value = idevice->num_dram_channels() *
                     idevice->dram_size_per_channel();
        return iree_ok_status();
      }
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

static iree_status_t iree_hal_tt_device_create_event(
    iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_event_flags_t,
    iree_hal_event_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not implemented");
}

static iree_status_t iree_hal_tt_device_create_executable_cache(
    iree_hal_device_t* base_device,
    iree_string_view_t identifier,
    iree_loop_t loop,
    iree_hal_executable_cache_t** out_executable_cache) {
  // Passthrough cache: no caching, every prepare_executable creates fresh.
  // The cache recognizes "tenstorrent-ttex-fb" format and delegates to
  // iree_hal_tt_executable_create().
  return iree_hal_tt_executable_cache_create(
      base_device,
      identifier,
      iree_hal_device_host_allocator(base_device),
      out_executable_cache);
}

static iree_status_t iree_hal_tt_device_import_file(
    iree_hal_device_t*, iree_hal_queue_affinity_t, iree_hal_memory_access_t,
    iree_io_file_handle_t*, iree_hal_external_file_flags_t, iree_hal_file_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "file import not implemented");
}

static iree_status_t iree_hal_tt_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_tt_semaphore_create(
      iree_hal_tt_device_cast(base_device)->host_allocator, initial_value,
      out_semaphore);
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
    iree_hal_device_t* base_device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphores,
    const iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {

  auto* device = iree_hal_tt_device_cast(base_device);

#ifndef TT_IREE_ENABLE_MOCK
  // Get MeshDevice and MeshCommandQueue
  tt::tt_metal::distributed::MeshDevice* mesh_device = device->mesh_device.get();
  tt::tt_metal::distributed::MeshCommandQueue& mesh_cq = *device->mesh_cq;

  // Get recorded commands from command buffer
  iree_host_size_t command_count = 0;
  iree_hal_tt_command_t* commands =
      iree_hal_tt_command_buffer_get_commands(command_buffer, &command_count);

  if (command_count == 0) {
    // No commands to execute, just signal semaphores
    for (iree_host_size_t i = 0; i < signal_semaphores.count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_signal(
          signal_semaphores.semaphores[i],
          signal_semaphores.payload_values[i]));
    }
    return iree_ok_status();
  }

  // Create MeshWorkload for this execution batch
  tt::tt_metal::distributed::MeshWorkload workload;

  // Device range for unit mesh (single device at coordinate {0,0})
  tt::tt_metal::distributed::MeshCoordinateRange device_range =
      tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());

  // Track dispatched params so we can restore programs after execution.
  iree_hal_tt_kernel_params_t* dispatched_params = nullptr;

  // Process each dispatch command
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (commands[i].type == IREE_HAL_TT_COMMAND_TYPE_DISPATCH) {
      auto& cmd = commands[i].dispatch;

      // Lookup kernel params (mutable: we need write access to restore program)
      iree_hal_tt_kernel_params_t* params = nullptr;
      IREE_RETURN_IF_ERROR(iree_hal_tt_executable_lookup_kernel_params_mutable(
          cmd.executable, cmd.export_ordinal, &params));

      if (!params || !params->program) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                               "executable export ordinal %d has no program",
                               cmd.export_ordinal);
      }

      tt::tt_metal::Program* program = static_cast<tt::tt_metal::Program*>(params->program);

      // Set runtime arguments from bindings (v3.9.0 API)
      tt::tt_metal::CoreCoord core = {0, 0};  // Using single core for PoC

      // Extract buffer addresses from bindings
      // Expected layout: bindings = [input0, input1, output]
      uint32_t in0_addr = 0, in1_addr = 0, out_addr = 0;
      uint32_t buffer_size = 0;

      if (cmd.binding_count > 0 && cmd.bindings[0].buffer) {
        in0_addr = iree_hal_tt_buffer_device_address(cmd.bindings[0].buffer);
        buffer_size = iree_hal_buffer_byte_length(cmd.bindings[0].buffer);
      }
      if (cmd.binding_count > 1 && cmd.bindings[1].buffer) {
        in1_addr = iree_hal_tt_buffer_device_address(cmd.bindings[1].buffer);
      }
      if (cmd.binding_count > 2 && cmd.bindings[2].buffer) {
        out_addr = iree_hal_tt_buffer_device_address(cmd.bindings[2].buffer);
      }

      // Calculate number of tiles (assuming float32 data)
      // Each tile is 32x32 floats = 4KB
      uint32_t n_tiles = buffer_size / (32 * 32 * sizeof(float));
      if (n_tiles == 0) n_tiles = 1;  // At least one tile for PoC

      // Set runtime args for reader kernel (in0_addr, in1_addr, n_tiles)
      try {
        std::vector<uint32_t> reader_args = {in0_addr, in1_addr, n_tiles};
        tt::tt_metal::SetRuntimeArgs(*program, params->reader_kernel_id, core, reader_args);

        // Set runtime args for writer kernel (out_addr, n_tiles)
        std::vector<uint32_t> writer_args = {out_addr, n_tiles};
        tt::tt_metal::SetRuntimeArgs(*program, params->writer_kernel_id, core, writer_args);

        // Set runtime args for compute kernel (n_tiles)
        std::vector<uint32_t> compute_args = {n_tiles};
        tt::tt_metal::SetRuntimeArgs(*program, params->compute_kernel_id, core, compute_args);
      } catch (const std::exception& e) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                               "Failed to set runtime args: %s", e.what());
      }

      // Move program into workload (required by MeshWorkload API).
      // We restore it after execution completes.
      try {
        workload.add_program(device_range, std::move(*program));
        dispatched_params = params;
      } catch (const std::exception& e) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                               "Failed to add program to workload: %s", e.what());
      }
    }
  }

  // Execute the workload (blocking for PoC)
  try {
    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_cq, workload, /*blocking=*/true);
  } catch (const std::exception& e) {
    // Attempt to restore program even on execution failure.
    if (dispatched_params) {
      auto& programs = workload.get_programs();
      if (!programs.empty()) {
        *static_cast<tt::tt_metal::Program*>(dispatched_params->program) =
            std::move(programs.begin()->second);
      }
    }
    return iree_make_status(IREE_STATUS_INTERNAL,
                           "Failed to execute workload: %s", e.what());
  }

  // Restore program back from workload so the executable can be re-dispatched.
  if (dispatched_params) {
    auto& programs = workload.get_programs();
    if (!programs.empty()) {
      *static_cast<tt::tt_metal::Program*>(dispatched_params->program) =
          std::move(programs.begin()->second);
    }
  }

  // Signal completion semaphores
  for (iree_host_size_t i = 0; i < signal_semaphores.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_signal(
        signal_semaphores.semaphores[i],
        signal_semaphores.payload_values[i]));
  }
#else
  // Mock mode - just signal completion
  for (iree_host_size_t i = 0; i < signal_semaphores.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_signal(
        signal_semaphores.semaphores[i],
        signal_semaphores.payload_values[i]));
  }
#endif

  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_queue_flush(
    iree_hal_device_t* base, iree_hal_queue_affinity_t) {
#ifndef TT_IREE_ENABLE_MOCK
  auto* device = iree_hal_tt_device_cast(base);
  if (device->mesh_cq) {
    try {
      // Flush the mesh command queue (wait for all pending operations)
      tt::tt_metal::distributed::Finish(*device->mesh_cq);
    } catch (const std::exception& e) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                             "Failed to flush queue: %s", e.what());
    }
  }
#endif
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  if (semaphore_list.count == 0) return iree_ok_status();

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Busy-poll loop over the semaphore list.
  while (true) {
    iree_host_size_t satisfied = 0;
    for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
      uint64_t current_value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(
          semaphore_list.semaphores[i], &current_value));
      if (current_value >= semaphore_list.payload_values[i]) {
        ++satisfied;
        if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
          return iree_ok_status();
        }
      }
    }
    if (satisfied == semaphore_list.count) {
      return iree_ok_status();  // WAIT_ALL satisfied
    }
    if (iree_time_now() >= deadline_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
  }
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
