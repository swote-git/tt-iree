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
// For PoC (Week 2-3):
//   - Mock mode: Simulates device using host memory
//   - Hardware mode: Opens P100A via TT-Metal (device_id=0)
//
// Parameters:
//   driver: Parent driver that created this device
//   device_id: Physical device ID (0 for first P100A)
//   host_allocator: Host memory allocator
//   out_device: Receives created device (must be released by caller)
iree_status_t iree_hal_tt_device_create(
    iree_hal_tenstorrent_driver_t* driver,
    iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// Device utilities (internal use by other HAL components)
//===----------------------------------------------------------------------===//

// Gets the TT-Metal device handle (hardware mode only).
// Returns NULL in mock mode or if device not initialized.
//
// INTERNAL USE ONLY - Called by tt_buffer.c to access TT-Metal APIs.
//
// In hardware mode, this returns:
//   tt::tt_metal::Device*
// Cast as void* to avoid C++ in header.
void* iree_hal_tt_device_get_tt_metal_handle(iree_hal_tt_device_t* device);

// Gets the compute command queue (hardware mode only).
// Returns NULL in mock mode.
//
// INTERNAL USE ONLY - Called by tt_command_buffer.c for dispatch.
//
// In hardware mode, this returns:
//   tt::tt_metal::CommandQueue*
// Cast as void* to avoid C++ in header.
void* iree_hal_tt_device_get_compute_queue(iree_hal_tt_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_