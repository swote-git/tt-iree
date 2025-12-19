// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward declaration
typedef struct iree_hal_tt_driver_t iree_hal_tt_driver_t;

// Creates a Tenstorrent HAL device.
iree_status_t iree_hal_tt_device_create(
    iree_hal_tt_driver_t* driver,
    iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_
