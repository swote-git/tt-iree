// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_DRIVER_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_DRIVER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a Tenstorrent HAL driver that can enumerate and create devices.
iree_status_t iree_hal_tt_driver_create(
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_DRIVER_H_
