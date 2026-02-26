// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_executable_cache.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"

//===----------------------------------------------------------------------===//
// Executable Cache Structure
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_device_t* device;
  iree_string_view_t identifier;
} iree_hal_tt_executable_cache_t;

extern const iree_hal_executable_cache_vtable_t
    iree_hal_tt_executable_cache_vtable;

static iree_hal_tt_executable_cache_t* iree_hal_tt_executable_cache_cast(
    iree_hal_executable_cache_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_executable_cache_vtable);
  return (iree_hal_tt_executable_cache_t*)base;
}

//===----------------------------------------------------------------------===//
// Vtable Implementation
//===----------------------------------------------------------------------===//

static void iree_hal_tt_executable_cache_destroy(
    iree_hal_executable_cache_t* base_cache) {
  iree_hal_tt_executable_cache_t* cache =
      iree_hal_tt_executable_cache_cast(base_cache);
  iree_allocator_free(cache->host_allocator, cache);
}

static bool iree_hal_tt_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  // Accept our TTEX format
  return iree_string_view_equal(
      executable_format,
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT));
}

static iree_status_t iree_hal_tt_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_tt_executable_cache_t* cache =
      iree_hal_tt_executable_cache_cast(base_cache);

  // Passthrough: delegate directly to executable creation.
  // No caching — every call creates a fresh executable.
  // Future: hash executable_data for dedup / cache lookup.
  return iree_hal_tt_executable_create(
      cache->device,
      executable_params,
      cache->host_allocator,
      out_executable);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

const iree_hal_executable_cache_vtable_t
    iree_hal_tt_executable_cache_vtable = {
        .destroy = iree_hal_tt_executable_cache_destroy,
        .can_prepare_format =
            iree_hal_tt_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_tt_executable_cache_prepare_executable,
};

//===----------------------------------------------------------------------===//
// Creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_executable_cache_create(
    iree_hal_device_t* device,
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;

  iree_hal_tt_executable_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*cache), (void**)&cache));

  iree_hal_resource_initialize(&iree_hal_tt_executable_cache_vtable,
                               &cache->resource);
  cache->host_allocator = host_allocator;
  cache->device = device;
  cache->identifier = identifier;

  *out_executable_cache = (iree_hal_executable_cache_t*)cache;
  return iree_ok_status();
}
