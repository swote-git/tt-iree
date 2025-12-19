// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_API_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_tenstorrent_driver_t
//===----------------------------------------------------------------------===//

// Creates a Tenstorrent HAL driver.
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_tenstorrent_driver_create(
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

//===----------------------------------------------------------------------===//
// iree_hal_tenstorrent_device_t
//===----------------------------------------------------------------------===//

// Creates a Tenstorrent HAL device.
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_tenstorrent_device_create(
    iree_hal_driver_t* driver,
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_API_H_
