// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_buffer.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#include "iree/hal/drivers/tenstorrent/tt_buffer_transfer.h"
#include "iree/hal/drivers/tenstorrent/tt_device.h"
#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"

using iree::hal::tenstorrent::RuntimeTransferScope;
using iree::hal::tenstorrent::RuntimeTransferStage;
using iree::hal::tenstorrent::RuntimeTransferStageKind;

#ifndef TT_IREE_ENABLE_MOCK
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tt_metal.hpp"
#endif

//===----------------------------------------------------------------------===//
// Tile Layout Conversion
//===----------------------------------------------------------------------===//

void iree_hal_tt_pack_to_tiles(const float* src, float* dst,
                               int32_t rows, int32_t cols) {
  if (!src || !dst || rows <= 0 || cols <= 0) return;
  
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;
  
  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          int32_t src_idx = (tr * TT_TILE_HEIGHT + r) * cols + (tc * TT_TILE_WIDTH + c);
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t dst_idx = tile_idx * TT_TILE_SIZE + r * TT_TILE_WIDTH + c;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

void iree_hal_tt_unpack_from_tiles(const float* src, float* dst,
                                   int32_t rows, int32_t cols) {
  if (!src || !dst || rows <= 0 || cols <= 0) return;
  
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;
  
  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t src_idx = tile_idx * TT_TILE_SIZE + r * TT_TILE_WIDTH + c;
          int32_t dst_idx = (tr * TT_TILE_HEIGHT + r) * cols + (tc * TT_TILE_WIDTH + c);
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_tt_buffer_t
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK
struct iree_hal_tt_buffer_mapping_state_t {
  iree_hal_tt_buffer_mapping_state_t* next;
  iree_device_size_t byte_offset;
  iree_device_size_t byte_length;
  uint8_t* data;
};
#endif

struct iree_hal_tt_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_tt_device_t* device;
  iree_device_size_t physical_allocation_size;
  
#ifndef TT_IREE_ENABLE_MOCK
  std::shared_ptr<tt::tt_metal::Buffer> tt_buffer;
  std::mutex mapping_mutex;
  iree_hal_tt_buffer_mapping_state_t* active_mappings;
#else
  void* host_ptr;
#endif
};

static void iree_hal_tt_buffer_destroy(iree_hal_buffer_t*);
static iree_status_t iree_hal_tt_buffer_map_range(iree_hal_buffer_t*, iree_hal_mapping_mode_t, iree_hal_memory_access_t, iree_device_size_t, iree_device_size_t, iree_hal_buffer_mapping_t*);
static iree_status_t iree_hal_tt_buffer_unmap_range(iree_hal_buffer_t*, iree_device_size_t, iree_device_size_t, iree_hal_buffer_mapping_t*);
static iree_status_t iree_hal_tt_buffer_invalidate_range(iree_hal_buffer_t*, iree_device_size_t, iree_device_size_t);
static iree_status_t iree_hal_tt_buffer_flush_range(iree_hal_buffer_t*, iree_device_size_t, iree_device_size_t);

static const iree_hal_buffer_vtable_t iree_hal_tt_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_tt_buffer_destroy,
    .map_range = iree_hal_tt_buffer_map_range,
    .unmap_range = iree_hal_tt_buffer_unmap_range,
    .invalidate_range = iree_hal_tt_buffer_invalidate_range,
    .flush_range = iree_hal_tt_buffer_flush_range,
};

static iree_hal_tt_buffer_t* iree_hal_tt_buffer_cast(iree_hal_buffer_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_buffer_vtable);
  return (iree_hal_tt_buffer_t*)base;
}

#ifndef TT_IREE_ENABLE_MOCK
static iree_status_t iree_hal_tt_buffer_read_physical_allocation(
    iree_hal_tt_buffer_t* buffer, uint8_t* target) {
  RuntimeTransferStage stage(RuntimeTransferStageKind::Read);
  if (!buffer->tt_buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "TT-Metal buffer is not initialized");
  }
  try {
    tt::tt_metal::detail::ReadFromBuffer(*buffer->tt_buffer, target);
    RuntimeTransferScope::ReadBytes(buffer->physical_allocation_size);
    return iree_ok_status();
  } catch (const std::exception& e) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "TT-Metal buffer read failed: %s", e.what());
  } catch (...) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "TT-Metal buffer read failed");
  }
}

static iree_status_t iree_hal_tt_buffer_write_physical_allocation(
    iree_hal_tt_buffer_t* buffer, const uint8_t* source) {
  RuntimeTransferStage stage(RuntimeTransferStageKind::Write);
  if (!buffer->tt_buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "TT-Metal buffer is not initialized");
  }
  try {
    ttsl::Span<const uint8_t> span(
        source, static_cast<size_t>(buffer->physical_allocation_size));
    tt::tt_metal::detail::WriteToBuffer(*buffer->tt_buffer, span);
    RuntimeTransferScope::WriteBytes(buffer->physical_allocation_size);
    return iree_ok_status();
  } catch (const std::exception& e) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "TT-Metal buffer write failed: %s", e.what());
  } catch (...) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "TT-Metal buffer write failed");
  }
}

static iree_status_t iree_hal_tt_buffer_read_range(
    iree_hal_tt_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, uint8_t* target) {
  void* physical_data = nullptr;
  iree_status_t status;
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Allocation);
    status = iree_allocator_malloc(
        buffer->host_allocator,
        static_cast<iree_host_size_t>(buffer->physical_allocation_size),
        &physical_data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_tt_buffer_read_physical_allocation(
        buffer, static_cast<uint8_t*>(physical_data));
  }
  if (iree_status_is_ok(status)) {
    RuntimeTransferStage stage(RuntimeTransferStageKind::HostCopy);
    std::memcpy(target, static_cast<uint8_t*>(physical_data) + byte_offset,
                static_cast<size_t>(byte_length));
  }
  if (physical_data) {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Deallocation);
    iree_allocator_free(buffer->host_allocator, physical_data);
  }
  return status;
}

static iree_status_t iree_hal_tt_buffer_write_range(
    iree_hal_tt_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, const uint8_t* source) {
  return iree::hal::tenstorrent::WriteBufferRange(
      buffer->host_allocator, iree_hal_buffer_byte_length(&buffer->base),
      buffer->physical_allocation_size, byte_offset, byte_length, source,
      [&](uint8_t *data) {
        return iree_hal_tt_buffer_read_physical_allocation(buffer, data);
      },
      [&](const uint8_t *data) {
        return iree_hal_tt_buffer_write_physical_allocation(buffer, data);
      });
}

static iree_hal_tt_buffer_mapping_state_t*
iree_hal_tt_buffer_find_mapping(iree_hal_tt_buffer_t* buffer,
                                iree_device_size_t byte_offset,
                                iree_device_size_t byte_length) {
  for (iree_hal_tt_buffer_mapping_state_t* state = buffer->active_mappings;
       state; state = state->next) {
    if (byte_offset < state->byte_offset) continue;
    iree_device_size_t relative_offset = byte_offset - state->byte_offset;
    if (relative_offset <= state->byte_length &&
        byte_length <= state->byte_length - relative_offset) {
      return state;
    }
  }
  return nullptr;
}
#endif

//===----------------------------------------------------------------------===//
// Buffer accessors for kernel runtime arguments
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK
uint32_t iree_hal_tt_buffer_device_address(iree_hal_buffer_t* base_buffer) {
  if (!base_buffer) return 0;
  auto* buffer = iree_hal_tt_buffer_cast(
      iree_hal_buffer_allocated_buffer(base_buffer));
  if (!buffer || !buffer->tt_buffer) return 0;
  uint64_t address = static_cast<uint64_t>(buffer->tt_buffer->address()) +
                     iree_hal_buffer_byte_offset(base_buffer);
  if (address > std::numeric_limits<uint32_t>::max()) return 0;
  return static_cast<uint32_t>(address);
}

tt::tt_metal::Buffer* iree_hal_tt_buffer_handle(iree_hal_buffer_t* base_buffer) {
  if (!base_buffer) return nullptr;
  if (!iree_hal_resource_is(iree_hal_buffer_allocated_buffer(base_buffer),
                            &iree_hal_tt_buffer_vtable))
    return nullptr;
  auto* buffer = iree_hal_tt_buffer_cast(
      iree_hal_buffer_allocated_buffer(base_buffer));
  return buffer ? buffer->tt_buffer.get() : nullptr;
}
#endif

//===----------------------------------------------------------------------===//
// Buffer creation
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_tt_buffer_create_impl(
    iree_hal_tt_device_t *device, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_allocator_t host_allocator,
    iree_hal_buffer_t **out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  iree_hal_tt_buffer_t* buffer = nullptr;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*buffer), (void**)&buffer);

  if (iree_status_is_ok(status)) {
    new (buffer) iree_hal_tt_buffer_t();

    buffer->host_allocator = host_allocator;
    buffer->device = device;
    buffer->physical_allocation_size = allocation_size;

    iree_hal_buffer_placement_t placement = {
        .device = (iree_hal_device_t*)device,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = 0,
        .reserved = 0,
    };
    // init buffer
    iree_hal_buffer_initialize(
        placement,
        &buffer->base,
        allocation_size,
        0,  // byte_offset
        allocation_size,  // byte_length
        params.type,
        params.access,
        params.usage,
        &iree_hal_tt_buffer_vtable,
        &buffer->base);

#ifdef TT_IREE_ENABLE_MOCK
    buffer->host_ptr = std::malloc(allocation_size);
    if (!buffer->host_ptr) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                               "failed to allocate mock buffer");
    }
#else
    tt::tt_metal::IDevice* tt_device = iree_hal_tt_device_idevice(device);
    if (!tt_device) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                               "TT-Metal device not initialized");
    } else {
      try {
        // Page size must match the tile size used by kernels.
        // Using bfloat16 tiles: 32x32 * 2 bytes = 2048 bytes per tile.
        constexpr uint32_t bf16_tile_size = TT_TILE_SIZE * sizeof(uint16_t);
        // Ensure allocation is aligned to tile size
        uint64_t aligned_size = ((static_cast<uint64_t>(allocation_size) + bf16_tile_size - 1)
                                / bf16_tile_size) * bf16_tile_size;
        buffer->physical_allocation_size = aligned_size;
        tt::tt_metal::BufferConfig config{
            .device = tt_device,
            .size = aligned_size,
            .page_size = bf16_tile_size,
            .buffer_type = tt::tt_metal::BufferType::DRAM
        };
        buffer->tt_buffer = tt::tt_metal::CreateBuffer(config);
        if (!buffer->tt_buffer) {
          status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                   "TT-Metal CreateBuffer returned null");
        }
      } catch (const std::exception& e) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                 "TT-Metal buffer creation failed: %s", e.what());
      }
    }
#endif
  }
  
  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    if (buffer) {
#ifdef TT_IREE_ENABLE_MOCK
      if (buffer->host_ptr) std::free(buffer->host_ptr);
#endif
      buffer->~iree_hal_tt_buffer_t();
      iree_allocator_free(host_allocator, buffer);
    }
  }
  
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Buffer vtable
//===----------------------------------------------------------------------===//

static void iree_hal_tt_buffer_destroy_impl(iree_hal_buffer_t *base_buffer) {
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
#ifdef TT_IREE_ENABLE_MOCK
  if (buffer->host_ptr) std::free(buffer->host_ptr);
#endif
  
  buffer->~iree_hal_tt_buffer_t();
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_tt_buffer_map_range_impl(
    iree_hal_buffer_t *base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t *mapping) {
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_all_bits_set(mapping_mode, IREE_HAL_MAPPING_MODE_PERSISTENT)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Tenstorrent device buffers only support scoped mapping");
  }
  if (local_byte_offset > buffer->physical_allocation_size ||
      local_byte_length >
          buffer->physical_allocation_size - local_byte_offset) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "mapping range exceeds the physical TT-Metal allocation");
  }
  if (local_byte_length == 0) {
    mapping->contents = iree_byte_span_empty();
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

#ifdef TT_IREE_ENABLE_MOCK
  uint8_t* data_ptr = (uint8_t*)buffer->host_ptr + local_byte_offset;
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
#else
  iree_hal_tt_buffer_mapping_state_t* state = nullptr;
  iree_status_t status = iree_allocator_malloc(
      buffer->host_allocator, sizeof(*state),
      reinterpret_cast<void**>(&state));
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  state->next = nullptr;
  state->byte_offset = local_byte_offset;
  state->byte_length = local_byte_length;
  state->data = nullptr;
  status = iree_allocator_malloc(
      buffer->host_allocator, static_cast<iree_host_size_t>(local_byte_length),
      reinterpret_cast<void**>(&state->data));
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(buffer->host_allocator, state);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  bool requires_read =
      iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ);
  if (!requires_read) {
    std::memset(state->data, 0, static_cast<size_t>(local_byte_length));
  }

  {
    std::lock_guard<std::mutex> lock(buffer->mapping_mutex);
    if (requires_read) {
      status = iree_hal_tt_buffer_read_range(
          buffer, local_byte_offset, local_byte_length, state->data);
    }
    if (iree_status_is_ok(status)) {
      state->next = buffer->active_mappings;
      buffer->active_mappings = state;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(buffer->host_allocator, state->data);
    iree_allocator_free(buffer->host_allocator, state);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  mapping->impl.reserved[0] =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state));
  mapping->contents = iree_make_byte_span(state->data, local_byte_length);
#endif
  
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_buffer_unmap_range_impl(
    iree_hal_buffer_t *base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t *mapping) {
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  
  IREE_TRACE_ZONE_BEGIN(z0);

#ifdef TT_IREE_ENABLE_MOCK
  // No-op for mock mode
#else
  auto* state = reinterpret_cast<iree_hal_tt_buffer_mapping_state_t*>(
      static_cast<uintptr_t>(mapping->impl.reserved[0]));
  iree_status_t status = iree_ok_status();
  if (!state) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "mapping state is missing during unmap");
  } else {
    {
      std::lock_guard<std::mutex> lock(buffer->mapping_mutex);
      if (mapping->impl.allowed_access & IREE_HAL_MEMORY_ACCESS_WRITE) {
        status = iree_hal_tt_buffer_write_range(
            buffer, local_byte_offset, local_byte_length, state->data);
      }
      iree_hal_tt_buffer_mapping_state_t** link = &buffer->active_mappings;
      while (*link && *link != state) link = &(*link)->next;
      if (*link == state) *link = state->next;
    }
    iree_allocator_free(buffer->host_allocator, state->data);
    iree_allocator_free(buffer->host_allocator, state);
  }
#endif

  mapping->contents = iree_byte_span_empty();
  IREE_TRACE_ZONE_END(z0);
#ifndef TT_IREE_ENABLE_MOCK
  return status;
#else
  return iree_ok_status();
#endif
}

static iree_status_t
iree_hal_tt_buffer_invalidate_range_impl(iree_hal_buffer_t *base_buffer,
                                         iree_device_size_t local_byte_offset,
                                         iree_device_size_t local_byte_length) {
#ifndef TT_IREE_ENABLE_MOCK
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  std::lock_guard<std::mutex> lock(buffer->mapping_mutex);
  iree_hal_tt_buffer_mapping_state_t* state =
      iree_hal_tt_buffer_find_mapping(buffer, local_byte_offset,
                                      local_byte_length);
  if (!state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalidate range has no active mapping");
  }
  return iree_hal_tt_buffer_read_range(
      buffer, local_byte_offset, local_byte_length,
      state->data + (local_byte_offset - state->byte_offset));
#else
  return iree_ok_status();
#endif
}

static iree_status_t
iree_hal_tt_buffer_flush_range_impl(iree_hal_buffer_t *base_buffer,
                                    iree_device_size_t local_byte_offset,
                                    iree_device_size_t local_byte_length) {
#ifndef TT_IREE_ENABLE_MOCK
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  std::lock_guard<std::mutex> lock(buffer->mapping_mutex);
  iree_hal_tt_buffer_mapping_state_t* state =
      iree_hal_tt_buffer_find_mapping(buffer, local_byte_offset,
                                      local_byte_length);
  if (!state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "flush range has no active mapping");
  }
  return iree_hal_tt_buffer_write_range(
      buffer, local_byte_offset, local_byte_length,
      state->data + (local_byte_offset - state->byte_offset));
#else
  return iree_ok_status();
#endif
}

static iree::hal::tenstorrent::RuntimeProfiler *
iree_hal_tt_buffer_profiler(iree_hal_buffer_t *base) {
  return iree_hal_tt_device_runtime_profiler(
      reinterpret_cast<iree_hal_device_t *>(
          iree_hal_tt_buffer_cast(base)->device));
}

iree_status_t iree_hal_tt_buffer_create(iree_hal_tt_device_t *device,
                                        iree_hal_buffer_params_t params,
                                        iree_device_size_t allocation_size,
                                        iree_allocator_t host_allocator,
                                        iree_hal_buffer_t **out_buffer) {
  RuntimeTransferScope transfer(
      iree_hal_tt_device_runtime_profiler(
          reinterpret_cast<iree_hal_device_t *>(device)),
      "ALLOCATE");
  iree_status_t result;
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Allocation);
    result = iree_hal_tt_buffer_create_impl(device, params, allocation_size,
                                            host_allocator, out_buffer);
  }
  if (iree_status_is_ok(result))
    transfer.Target(*out_buffer, 0, allocation_size);
  return transfer.Finish(result);
}

static void iree_hal_tt_buffer_destroy(iree_hal_buffer_t *base) {
  RuntimeTransferScope transfer(iree_hal_tt_buffer_profiler(base),
                                "DEALLOCATE");
  transfer.Target(base, 0, iree_hal_buffer_byte_length(base));
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Deallocation);
    iree_hal_tt_buffer_destroy_impl(base);
  }
  iree_status_ignore(transfer.Finish(iree_ok_status()));
}

static iree_status_t iree_hal_tt_buffer_map_range(
    iree_hal_buffer_t *base, iree_hal_mapping_mode_t mode,
    iree_hal_memory_access_t access, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_buffer_mapping_t *mapping) {
  RuntimeTransferScope transfer(iree_hal_tt_buffer_profiler(base), "MAP");
  if (access & IREE_HAL_MEMORY_ACCESS_READ)
    transfer.Source(base, offset, length);
  if (access & IREE_HAL_MEMORY_ACCESS_WRITE)
    transfer.Target(base, offset, length);
  iree_status_t result;
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Map);
    result = iree_hal_tt_buffer_map_range_impl(base, mode, access, offset,
                                               length, mapping);
  }
  return transfer.Finish(result);
}

static iree_status_t iree_hal_tt_buffer_unmap_range(
    iree_hal_buffer_t *base, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_buffer_mapping_t *mapping) {
  RuntimeTransferScope transfer(iree_hal_tt_buffer_profiler(base), "UNMAP");
  if (mapping->impl.allowed_access & IREE_HAL_MEMORY_ACCESS_WRITE)
    transfer.Target(base, offset, length);
  else
    transfer.Source(base, offset, length);
  iree_status_t result;
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Unmap);
    result = iree_hal_tt_buffer_unmap_range_impl(base, offset, length, mapping);
  }
  return transfer.Finish(result);
}

static iree_status_t
iree_hal_tt_buffer_invalidate_range(iree_hal_buffer_t *base,
                                    iree_device_size_t offset,
                                    iree_device_size_t length) {
  RuntimeTransferScope transfer(iree_hal_tt_buffer_profiler(base),
                                "INVALIDATE");
  transfer.Source(base, offset, length);
  iree_status_t result;
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Invalidate);
    result = iree_hal_tt_buffer_invalidate_range_impl(base, offset, length);
  }
  return transfer.Finish(result);
}

static iree_status_t iree_hal_tt_buffer_flush_range(iree_hal_buffer_t *base,
                                                    iree_device_size_t offset,
                                                    iree_device_size_t length) {
  RuntimeTransferScope transfer(iree_hal_tt_buffer_profiler(base), "FLUSH");
  transfer.Target(base, offset, length);
  iree_status_t result;
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Flush);
    result = iree_hal_tt_buffer_flush_range_impl(base, offset, length);
  }
  return transfer.Finish(result);
}
