// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_buffer.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "iree/hal/drivers/tenstorrent/tt_device.h"

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

struct iree_hal_tt_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_tt_device_t* device;
  
#ifndef TT_IREE_ENABLE_MOCK
  std::shared_ptr<tt::tt_metal::Buffer> tt_buffer;
#else
  void* host_ptr;
#endif
  
  int32_t rows;
  int32_t cols;
  bool uses_tile_layout;
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

//===----------------------------------------------------------------------===//
// Buffer accessors for kernel runtime arguments
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK
uint32_t iree_hal_tt_buffer_device_address(iree_hal_buffer_t* base_buffer) {
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  if (!buffer || !buffer->tt_buffer) return 0;
  return buffer->tt_buffer->address();
}

tt::tt_metal::Buffer* iree_hal_tt_buffer_handle(iree_hal_buffer_t* base_buffer) {
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  return buffer ? buffer->tt_buffer.get() : nullptr;
}
#endif

//===----------------------------------------------------------------------===//
// Buffer creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_buffer_create(
    iree_hal_tt_device_t* device,
    iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
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

    iree_device_size_t num_elements = allocation_size / sizeof(float);
    if (num_elements == 1024) {
      buffer->rows = 32;
      buffer->cols = 32;
    } else {
      buffer->rows = (int32_t)std::sqrt((double)num_elements);
      buffer->cols = buffer->rows;
      buffer->rows = ((buffer->rows + 31) / 32) * 32;
      buffer->cols = ((buffer->cols + 31) / 32) * 32;
    }

    buffer->uses_tile_layout = (buffer->rows % 32 == 0) && (buffer->cols % 32 == 0);

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

static void iree_hal_tt_buffer_destroy(iree_hal_buffer_t* base_buffer) {
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

static iree_status_t iree_hal_tt_buffer_map_range(
    iree_hal_buffer_t* base_buffer,
    iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  
  IREE_TRACE_ZONE_BEGIN(z0);

#ifdef TT_IREE_ENABLE_MOCK
  uint8_t* data_ptr = (uint8_t*)buffer->host_ptr + local_byte_offset;
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
#else
  // Allocate staging buffer
  void* staging = std::malloc(local_byte_length);
  if (!staging) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "failed to allocate staging buffer");
  }
  
  if (memory_access & IREE_HAL_MEMORY_ACCESS_READ) {
    try {
      // ReadFromBuffer fills a host-side vector; copy only what was requested.
      std::vector<uint8_t> tmp;
      tt::tt_metal::detail::ReadFromBuffer(*buffer->tt_buffer, tmp);
      iree_device_size_t copy_len =
          std::min(local_byte_length, (iree_device_size_t)tmp.size());
      std::memcpy(staging, tmp.data(), copy_len);
      if (copy_len < local_byte_length)
        std::memset((uint8_t*)staging + copy_len, 0,
                    local_byte_length - copy_len);
    } catch (...) {
      std::memset(staging, 0, local_byte_length);
    }
  } else {
    std::memset(staging, 0, local_byte_length);
  }
  
  mapping->contents = iree_make_byte_span(staging, local_byte_length);
#endif
  
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  auto* buffer = iree_hal_tt_buffer_cast(base_buffer);
  
  IREE_TRACE_ZONE_BEGIN(z0);

#ifdef TT_IREE_ENABLE_MOCK
  // No-op for mock mode
#else
  // Only write back to device if the mapping allowed writes.
  if (mapping->impl.allowed_access & IREE_HAL_MEMORY_ACCESS_WRITE) {
    try {
      tt::stl::Span<const uint8_t> span(
          mapping->contents.data, local_byte_length);
      tt::tt_metal::detail::WriteToBuffer(*buffer->tt_buffer, span);
    } catch (const std::exception& e) {
      fprintf(stderr, "tt-iree: unmap_range: WriteToBuffer threw: %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "tt-iree: unmap_range: WriteToBuffer threw unknown\n");
    }
  }

  std::free(mapping->contents.data);
#endif

  mapping->contents = iree_byte_span_empty();
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_buffer_invalidate_range(
    iree_hal_buffer_t*, iree_device_size_t, iree_device_size_t) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_buffer_flush_range(
    iree_hal_buffer_t*, iree_device_size_t, iree_device_size_t) {
  return iree_ok_status();
}
