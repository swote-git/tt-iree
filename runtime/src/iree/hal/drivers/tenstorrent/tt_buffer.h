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

// Tenstorrent tile dimensions
#define TT_TILE_HEIGHT 32
#define TT_TILE_WIDTH 32
#define TT_TILE_SIZE (TT_TILE_HEIGHT * TT_TILE_WIDTH)

//===----------------------------------------------------------------------===//
// Tile Layout Conversion (Public API)
//===----------------------------------------------------------------------===//

// Pack row-major data into 32x32 tile layout (Host -> Device)
void iree_hal_tt_pack_to_tiles(
    const float* src,
    float* dst,
    int32_t rows,
    int32_t cols);

// Unpack 32x32 tile layout back to row-major (Device -> Host)
void iree_hal_tt_unpack_from_tiles(
    const float* src,
    float* dst,
    int32_t rows,
    int32_t cols);

//===----------------------------------------------------------------------===//
// Buffer creation
//===----------------------------------------------------------------------===//

// Create a Tenstorrent HAL buffer
iree_status_t iree_hal_tt_buffer_create(
    iree_hal_tt_device_t* device,
    iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_BUFFER_H_