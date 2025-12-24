// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_buffer.h"

#include <math.h>  // FIXED: Added for sqrt()
#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/tenstorrent/tt_device.h"

//===----------------------------------------------------------------------===//
// Tile Layout Conversion Utilities
//===----------------------------------------------------------------------===//

// Pack row-major data into 32x32 tile layout
//
// Algorithm:
//   1. Divide input into grid of 32x32 tiles
//   2. For each tile, copy its 1024 elements contiguously
//
// Example: 64x64 matrix
//   Input (row-major):  [0,1,2,...,63, 64,65,...,4095]
//   Output (tiled):     [Tile(0,0)[1024], Tile(0,1)[1024], Tile(1,0)[1024], Tile(1,1)[1024]]
//
void iree_hal_tt_pack_to_tiles(
    const float* src,
    float* dst,
    int32_t rows,
    int32_t cols) {
  
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(rows > 0 && rows % TT_TILE_HEIGHT == 0);
  IREE_ASSERT_ARGUMENT(cols > 0 && cols % TT_TILE_WIDTH == 0);
  
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;
  
  // For each tile in the grid
  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      
      // For each element within the tile
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          
          // Source: row-major index
          int32_t src_row = tr * TT_TILE_HEIGHT + r;
          int32_t src_col = tc * TT_TILE_WIDTH + c;
          int32_t src_idx = src_row * cols + src_col;
          
          // Destination: tiled index
          // Layout: [Tile0][Tile1]..., each tile has 1024 elements
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t within_tile_idx = r * TT_TILE_WIDTH + c;
          int32_t dst_idx = tile_idx * TT_TILE_SIZE + within_tile_idx;
          
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

// Unpack 32x32 tile layout back to row-major
//
// Inverse of pack_to_tiles()
//
void iree_hal_tt_unpack_from_tiles(
    const float* src,
    float* dst,
    int32_t rows,
    int32_t cols) {
  
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(rows > 0 && rows % TT_TILE_HEIGHT == 0);
  IREE_ASSERT_ARGUMENT(cols > 0 && cols % TT_TILE_WIDTH == 0);
  
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;
  
  // For each tile in the grid
  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      
      // For each element within the tile
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          
          // Source: tiled index
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t within_tile_idx = r * TT_TILE_WIDTH + c;
          int32_t src_idx = tile_idx * TT_TILE_SIZE + within_tile_idx;
          
          // Destination: row-major index
          int32_t dst_row = tr * TT_TILE_HEIGHT + r;
          int32_t dst_col = tc * TT_TILE_WIDTH + c;
          int32_t dst_idx = dst_row * cols + dst_col;
          
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_tt_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  
  // Parent device
  iree_hal_tt_device_t* device;
  
  // TT-Metal buffer (hardware mode)
  // For PoC, we use DRAM with INTERLEAVED layout
#ifndef TT_IREE_ENABLE_MOCK
  // This will be: std::shared_ptr<tt::tt_metal::Buffer>*
  // We store it as void* to avoid C++ in header
  void* tt_buffer;
#else
  // Mock mode: use host memory
  void* host_ptr;
#endif
  
  // For tile conversion: dimensions
  // We need to know the shape for layout conversion
  // Assumption for PoC: 2D tensors only
  int32_t rows;
  int32_t cols;
  
} iree_hal_tt_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_tt_buffer_vtable;

static iree_hal_tt_buffer_t* iree_hal_tt_buffer_cast(
    iree_hal_buffer_t* base_buffer) {
  IREE_HAL_ASSERT_TYPE(base_buffer, &iree_hal_tt_buffer_vtable);
  return (iree_hal_tt_buffer_t*)base_buffer;
}

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
  *out_buffer = NULL;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  // Allocate buffer struct
  iree_hal_tt_buffer_t* buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*buffer), (void**)&buffer);
  
  if (iree_status_is_ok(status)) {
    // For PoC: Assume square tiles (32x32)
    // TODO: Get actual shape from tensor metadata
    // For now, infer from size (assuming float32)
    iree_device_size_t num_elements = allocation_size / sizeof(float);
    
    // Simple heuristic: if size == 1024 → 32x32
    // For general case, we'd need shape metadata from compiler
    if (num_elements == 1024) {
      buffer->rows = 32;
      buffer->cols = 32;
    } else {
      // Try to make square
      buffer->rows = (int32_t)sqrt((double)num_elements);
      buffer->cols = buffer->rows;
      
      // Round up to tile boundaries
      buffer->rows = ((buffer->rows + 31) / 32) * 32;
      buffer->cols = ((buffer->cols + 31) / 32) * 32;
    }
    
    // FIXED: Updated iree_hal_buffer_initialize signature for IREE v3.9.0
    // New signature has 10 parameters
    iree_hal_buffer_initialize(
        // placement (new parameter) - where buffer is allocated
        (iree_hal_buffer_placement_t){
            .device = (iree_hal_device_t*)device,
            .allocator = NULL,  // Will be set by allocator
        },
        // allocated_buffer - pointer to the buffer struct
        &buffer->base,
        // allocation_size
        allocation_size,
        // byte_offset
        0,
        // byte_length
        allocation_size,
        // memory_type
        params.type,
        // allowed_access
        params.access,
        // allowed_usage
        params.usage,
        // vtable
        &iree_hal_tt_buffer_vtable,
        // buffer - output pointer (same as allocated_buffer)
        &buffer->base);
    
    buffer->host_allocator = host_allocator;
    buffer->device = device;
    
#ifdef TT_IREE_ENABLE_MOCK
    // Mock mode: allocate host memory
    buffer->host_ptr = malloc(allocation_size);
    if (!buffer->host_ptr) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                               "failed to allocate mock buffer");
    }
#else
    // Real mode: Create TT-Metal Buffer
    // Get TT-Metal device handle
    void* tt_device = iree_hal_tt_device_get_tt_metal_handle(device);
    if (!tt_device) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                               "TT-Metal device not initialized");
    } else {
      // TODO: Call TT-Metal API to create buffer
      // This requires C++ wrapper:
      //   buffer->tt_buffer = tt_create_buffer(tt_device, allocation_size);
      buffer->tt_buffer = NULL;
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                               "TT-Metal buffer creation not yet implemented");
    }
#endif
  }
  
  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    if (buffer) {
#ifdef TT_IREE_ENABLE_MOCK
      if (buffer->host_ptr) free(buffer->host_ptr);
#endif
      iree_allocator_free(host_allocator, buffer);
    }
  }
  
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Buffer vtable implementation
//===----------------------------------------------------------------------===//

static void iree_hal_tt_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_tt_buffer_t* buffer = iree_hal_tt_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
#ifdef TT_IREE_ENABLE_MOCK
  if (buffer->host_ptr) {
    free(buffer->host_ptr);
  }
#else
  // TODO: Destroy TT-Metal Buffer
  // tt_buffer_destroy(buffer->tt_buffer);
#endif
  
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
  
  iree_hal_tt_buffer_t* buffer = iree_hal_tt_buffer_cast(base_buffer);
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
#ifdef TT_IREE_ENABLE_MOCK
  // Mock mode: direct host pointer access
  uint8_t* data_ptr = (uint8_t*)buffer->host_ptr + local_byte_offset;
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
#else
  // Real mode: Map device memory
  // For PoC, we'll allocate temporary host buffer
  // and handle tile conversion on unmap
  
  void* mapped_ptr = malloc(local_byte_length);
  if (!mapped_ptr) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "failed to allocate mapping buffer");
  }
  
  // If reading, need to unpack tiles
  if (memory_access & IREE_HAL_MEMORY_ACCESS_READ) {
    // TODO: Read from TT-Metal buffer and unpack
    // For now, just zero it
    memset(mapped_ptr, 0, local_byte_length);
  }
  
  mapping->contents = iree_make_byte_span(mapped_ptr, local_byte_length);
#endif
  
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  
  iree_hal_tt_buffer_t* buffer = iree_hal_tt_buffer_cast(base_buffer);
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
#ifdef TT_IREE_ENABLE_MOCK
  // Mock mode: no-op (data is already in host_ptr)
#else
  // Real mode: Pack tiles and write to device
  
  // Allocate tiled buffer
  void* tiled_buffer = malloc(local_byte_length);
  if (!tiled_buffer) {
    free(mapping->contents.data);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "failed to allocate tiled buffer");
  }
  
  // Pack: row-major → tiled
  iree_hal_tt_pack_to_tiles(
      (const float*)mapping->contents.data,
      (float*)tiled_buffer,
      buffer->rows,
      buffer->cols);
  
  // TODO: Write tiled_buffer to TT-Metal device
  // tt::tt_metal::WriteToBuffer(buffer->tt_buffer, tiled_buffer, ...);
  
  free(tiled_buffer);
  free(mapping->contents.data);
#endif
  
  mapping->contents = iree_byte_span_empty();
  
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // No-op: TT-Metal handles cache coherency
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_buffer_flush_range(
    iree_hal_buffer_t* base_buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // No-op: TT-Metal handles cache coherency
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// vtable
//===----------------------------------------------------------------------===//

static const iree_hal_buffer_vtable_t iree_hal_tt_buffer_vtable = {
    .destroy = iree_hal_tt_buffer_destroy,
    .map_range = iree_hal_tt_buffer_map_range,
    .unmap_range = iree_hal_tt_buffer_unmap_range,
    .invalidate_range = iree_hal_tt_buffer_invalidate_range,
    .flush_range = iree_hal_tt_buffer_flush_range,
};