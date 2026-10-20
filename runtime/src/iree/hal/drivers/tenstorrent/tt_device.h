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

// Describes where a builtin dispatch performs its tensor computation. This is
// intentionally independent of mock mode so tooling can reason about whether a
// physical-device run is free of host tensor computation.
typedef enum iree_hal_tt_execution_domain_e {
  IREE_HAL_TT_EXECUTION_DOMAIN_UNKNOWN = 0,
  IREE_HAL_TT_EXECUTION_DOMAIN_DEVICE_NATIVE = 1,
  IREE_HAL_TT_EXECUTION_DOMAIN_HOST_ASSISTED_DEVICE = 2,
  IREE_HAL_TT_EXECUTION_DOMAIN_HOST = 3,
} iree_hal_tt_execution_domain_t;

iree_hal_tt_execution_domain_t iree_hal_tt_execution_domain_for_builtin(
    uint32_t builtin_program);

const char* iree_hal_tt_execution_domain_name(
    iree_hal_tt_execution_domain_t domain);

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
} // extern "C"

namespace iree::hal::tenstorrent {
class RuntimeProfiler;
}

iree::hal::tenstorrent::RuntimeProfiler *
iree_hal_tt_device_runtime_profiler(iree_hal_device_t *device);

#ifndef TT_IREE_ENABLE_MOCK
namespace tt::tt_metal {
class IDevice;
namespace distributed {
class MeshDevice;
class MeshCommandQueue;
}
} // namespace tt::tt_metal

tt::tt_metal::distributed::MeshDevice* iree_hal_tt_device_mesh_handle(iree_hal_tt_device_t* device);
tt::tt_metal::distributed::MeshCommandQueue* iree_hal_tt_device_mesh_queue(iree_hal_tt_device_t* device);

tt::tt_metal::IDevice* iree_hal_tt_device_idevice(iree_hal_tt_device_t* device);
#endif

#else
void* iree_hal_tt_device_get_tt_metal_handle(iree_hal_tt_device_t* device);
void* iree_hal_tt_device_get_compute_queue(iree_hal_tt_device_t* device);
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_H_
