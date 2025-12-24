// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
typedef struct iree_hal_tt_device_t iree_hal_tt_device_t;

// Creates a Tenstorrent allocator for the given device.
// POC:Only use DRAM.
//
// Memory hierarchy on P100A:
//   - DRAM: 28GB GDDR6
//   - L1: 1.5MB per core (handled by TT-Metal via Circular Buffers)
iree_status_t iree_hal_tt_allocator_create(
    iree_hal_tt_device_t* device,
    iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_ALLOCATOR_H_
