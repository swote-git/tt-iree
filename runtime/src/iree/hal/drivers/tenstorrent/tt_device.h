// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward declarations
typedef struct iree_hal_tenstorrent_driver_t iree_hal_tenstorrent_driver_t;
typedef struct iree_hal_tt_device_t iree_hal_tt_device_t;

//===----------------------------------------------------------------------===//
// Device creation and management
//===----------------------------------------------------------------------===//

// Creates a Tenstorrent HAL device for the given device ID.
//
// Device lifecycle:
//   1. Create device (this function)
//   2. Create allocator (done internally)
//   3. Allocate buffers, execute commands, etc.
//   4. Destroy device (via iree_hal_device_release)
//
iree_status_t iree_hal_tt_device_create(
    iree_hal_tenstorrent_driver_t* driver,
    iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// Device creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_device_create(
    iree_hal_tenstorrent_driver_t* driver,
    iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"

// C++ only: internal accessors for TT-Metal handles
#ifndef TT_IREE_ENABLE_MOCK
namespace tt::tt_metal {
class Device;
class CommandQueue;
}

tt::tt_metal::Device* iree_hal_tt_device_handle(iree_hal_tt_device_t* device);
tt::tt_metal::CommandQueue* iree_hal_tt_device_queue(iree_hal_tt_device_t* device);
#endif

#else
// C only: opaque accessors
void* iree_hal_tt_device_get_tt_metal_handle(iree_hal_tt_device_t* device);
void* iree_hal_tt_device_get_compute_queue(iree_hal_tt_device_t* device);
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_
