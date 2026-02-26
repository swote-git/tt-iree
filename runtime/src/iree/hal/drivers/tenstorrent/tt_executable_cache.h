// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Passthrough executable cache for the Tenstorrent HAL driver.
//
// IREE's vmfb loading path requires:
//   device->create_executable_cache()
//     -> cache->can_prepare_format(format_string)
//     -> cache->prepare_executable(executable_data)
//
// Without this, the runtime cannot route executables to our driver,
// regardless of how well the FlatBuffer schema is designed.
//
// This implementation does no caching — every prepare_executable call
// creates a fresh iree_hal_tt_executable_t. Caching can be added later
// by keying on executable_data hash.

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// The executable format string that identifies TTEX FlatBuffer payloads.
// This must match what the compiler target backend emits in
// hal.executable.binary { format = "tenstorrent-ttex-fb" }.
#define IREE_HAL_TT_EXECUTABLE_FORMAT "tenstorrent-ttex-fb"

// Creates a passthrough executable cache for the Tenstorrent driver.
//
// The cache recognizes IREE_HAL_TT_EXECUTABLE_FORMAT and delegates
// preparation to iree_hal_tt_executable_create().
iree_status_t iree_hal_tt_executable_cache_create(
    iree_hal_device_t* device,
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_CACHE_H_
