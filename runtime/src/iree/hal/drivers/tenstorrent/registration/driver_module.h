// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_REGISTRATION_DRIVER_MODULE_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_REGISTRATION_DRIVER_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the Tenstorrent HAL driver with the given |registry|.
IREE_API_EXPORT iree_status_t iree_hal_tenstorrent_driver_module_register(
    iree_hal_driver_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_REGISTRATION_DRIVER_MODULE_H_
