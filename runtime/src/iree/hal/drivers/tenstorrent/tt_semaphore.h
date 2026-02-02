// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t iree_hal_tt_semaphore_create(
    iree_allocator_t host_allocator, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_SEMAPHORE_H_
