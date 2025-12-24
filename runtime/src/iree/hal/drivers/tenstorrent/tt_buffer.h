// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_BUFFER_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct iree_hal_tt_device_t iree_hal_tt_device_t;

// Tile dimensions for P100A Tensix cores
// All matrix operations MUST use 32x32 tiles
#define TT_TILE_HEIGHT 32
#define TT_TILE_WIDTH 32
#define TT_TILE_SIZE (TT_TILE_HEIGHT * TT_TILE_WIDTH)  // 1024 elements

// Creates a Tenstorrent buffer wrapping a TT-Metal Buffer.
//
// CRITICAL: Data Layout Transformation
// -------------------------------------
// Host memory: Row-major layout
//   [a00, a01, a02, ..., a31, a32, a33, ...]
//
// Device memory: 32x32 Tile layout
//   [Tile(0,0): a00..a31 (row0), a32..a63 (row1), ...]
//   [Tile(0,1): next 32x32 block, ...]
//
// The buffer automatically handles this transformation during:
//   - map_range (Host → Device): pack_to_tiles()
//   - unmap_range (Device → Host): unpack_from_tiles()
//
// For PoC: Only supports float32 (4 bytes per element)
iree_status_t iree_hal_tt_buffer_create(
    iree_hal_tt_device_t* device,
    iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

// Utility: Pack row-major data into 32x32 tile layout
// Used internally during Host → Device transfer
//
// Example: 64x64 matrix → 4 tiles (2x2 grid)
//   src: row-major [4096 floats]
//   dst: tiled     [4096 floats, but reordered]
void iree_hal_tt_pack_to_tiles(
    const float* src,
    float* dst,
    int32_t rows,
    int32_t cols);

// Utility: Unpack 32x32 tile layout back to row-major
// Used internally during Device → Host transfer
void iree_hal_tt_unpack_from_tiles(
    const float* src,
    float* dst,
    int32_t rows,
    int32_t cols);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_BUFFER_H_