// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_executable_cache.h"

#include <string.h>

#include <cstdint>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/tenstorrent/tt_device.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"

//===----------------------------------------------------------------------===//
// Executable Cache Structure
//===----------------------------------------------------------------------===//

struct iree_hal_tt_executable_cache_entry_t {
  uint64_t data_hash = 0;
  iree_host_size_t data_length = 0;
  iree_hal_queue_affinity_t queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  iree_hal_executable_caching_mode_t caching_mode = 0;
  std::vector<uint32_t> constants;
  iree_hal_executable_t *executable = nullptr;
};

struct iree_hal_tt_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_device_t *device;
  std::mutex mutex;
  std::vector<iree_hal_tt_executable_cache_entry_t> entries;
};

extern const iree_hal_executable_cache_vtable_t
    iree_hal_tt_executable_cache_vtable;

static iree_hal_tt_executable_cache_t *
iree_hal_tt_executable_cache_cast(iree_hal_executable_cache_t *base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_executable_cache_vtable);
  return (iree_hal_tt_executable_cache_t *)base;
}

static uint64_t
iree_hal_tt_executable_cache_hash_data(iree_const_byte_span_t data) {
  const uint64_t kFnvOffsetBasis = 14695981039346656037ull;
  const uint64_t kFnvPrime = 1099511628211ull;
  uint64_t hash = kFnvOffsetBasis;
  for (iree_host_size_t i = 0; i < data.data_length; ++i) {
    hash ^= data.data[i];
    hash *= kFnvPrime;
  }
  return hash;
}

static bool iree_hal_tt_executable_cache_entry_matches(
    const iree_hal_tt_executable_cache_entry_t &entry,
    const iree_hal_executable_params_t *params, uint64_t data_hash,
    const std::vector<uint32_t> &constants) {
  return entry.data_hash == data_hash &&
         entry.data_length == params->executable_data.data_length &&
         entry.queue_affinity == params->queue_affinity &&
         entry.caching_mode == params->caching_mode &&
         entry.constants == constants &&
         iree_hal_tt_executable_matches_data(entry.executable,
                                             params->executable_data);
}

//===----------------------------------------------------------------------===//
// Vtable Implementation
//===----------------------------------------------------------------------===//

static void
iree_hal_tt_executable_cache_destroy(iree_hal_executable_cache_t *base_cache) {
  iree_hal_tt_executable_cache_t *cache =
      iree_hal_tt_executable_cache_cast(base_cache);
  for (auto &entry : cache->entries) {
    iree_hal_executable_release(entry.executable);
  }
  iree_hal_device_release(cache->device);
  iree_allocator_t host_allocator = cache->host_allocator;
  cache->~iree_hal_tt_executable_cache_t();
  iree_allocator_free(host_allocator, cache);
}

static bool iree_hal_tt_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t *base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  // Accept our TTEX format
  return iree_string_view_equal(
      executable_format, iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT));
}

static iree_status_t iree_hal_tt_executable_cache_prepare_executable_impl(
    iree_hal_executable_cache_t *base_cache,
    const iree_hal_executable_params_t *executable_params,
    iree_hal_executable_t **out_executable,
    iree::hal::tenstorrent::RuntimePreparationRecord *profile) {
  using iree::hal::tenstorrent::RuntimePreparationTimer;
  iree_hal_tt_executable_cache_t *cache =
      iree_hal_tt_executable_cache_cast(base_cache);

  if (executable_params->constant_count > 0 && !executable_params->constants) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable constants are null with count > 0");
  }
  if (!executable_params->executable_data.data ||
      executable_params->executable_data.data_length == 0) {
    return iree_hal_tt_executable_create_with_profile(
        cache->device, executable_params, cache->host_allocator, out_executable,
        profile);
  }

  RuntimePreparationTimer lookup_timer(profile ? &profile->hash_lookup_ns
                                               : nullptr);
  std::vector<uint32_t> constants;
  try {
    if (executable_params->constant_count > 0) {
      constants.assign(executable_params->constants,
                       executable_params->constants +
                           executable_params->constant_count);
    }
  } catch (const std::bad_alloc &) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to allocate executable cache key");
  }
  const uint64_t data_hash = iree_hal_tt_executable_cache_hash_data(
      executable_params->executable_data);

  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    for (const auto &entry : cache->entries) {
      if (iree_hal_tt_executable_cache_entry_matches(entry, executable_params,
                                                     data_hash, constants)) {
        iree_hal_executable_retain(entry.executable);
        *out_executable = entry.executable;
        if (profile)
          profile->cache_hit = true;
        return iree_ok_status();
      }
    }
  }
  lookup_timer.Stop();

  iree_hal_executable_t *executable = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_tt_executable_create_with_profile(
      cache->device, executable_params, cache->host_allocator, &executable,
      profile));

  RuntimePreparationTimer insert_timer(profile ? &profile->cache_insert_ns
                                               : nullptr);
  try {
    std::lock_guard<std::mutex> lock(cache->mutex);
    // Another thread may have populated the same key during preparation.
    for (const auto &entry : cache->entries) {
      if (iree_hal_tt_executable_cache_entry_matches(entry, executable_params,
                                                     data_hash, constants)) {
        iree_hal_executable_retain(entry.executable);
        iree_hal_executable_release(executable);
        *out_executable = entry.executable;
        if (profile)
          profile->cache_hit = true;
        return iree_ok_status();
      }
    }

    cache->entries.push_back(
        {data_hash, executable_params->executable_data.data_length,
         executable_params->queue_affinity, executable_params->caching_mode,
         std::move(constants), executable});
    iree_hal_executable_retain(executable); // Cache-owned reference.
  } catch (const std::bad_alloc &) {
    // Preparation succeeded; an allocation failure only disables insertion.
  }
  *out_executable = executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_executable_cache_prepare_executable(
    iree_hal_executable_cache_t *base_cache,
    const iree_hal_executable_params_t *params,
    iree_hal_executable_t **out_executable) {
  auto *cache = iree_hal_tt_executable_cache_cast(base_cache);
  auto *profiler = iree_hal_tt_device_runtime_profiler(cache->device);
  iree::hal::tenstorrent::RuntimePreparationRecord record;
  if (profiler) {
    record.host_start_ns = profiler->NowNs();
    record.executable_bytes = params->executable_data.data_length;
  }
  iree_status_t status = iree_hal_tt_executable_cache_prepare_executable_impl(
      base_cache, params, out_executable, profiler ? &record : nullptr);
  if (profiler) {
    record.host_end_ns = profiler->NowNs();
    record.host_duration_ns = record.host_end_ns - record.host_start_ns;
    record.status_code = iree_status_code(status);
    iree_status_ignore(profiler->RecordPreparation(std::move(record)));
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

const iree_hal_executable_cache_vtable_t iree_hal_tt_executable_cache_vtable = {
    .destroy = iree_hal_tt_executable_cache_destroy,
    .can_prepare_format = iree_hal_tt_executable_cache_can_prepare_format,
    .prepare_executable = iree_hal_tt_executable_cache_prepare_executable,
};

//===----------------------------------------------------------------------===//
// Creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_executable_cache_create(
    iree_hal_device_t *device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t **out_executable_cache) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;

  iree_hal_tt_executable_cache_t *cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void **)&cache));

  new (cache) iree_hal_tt_executable_cache_t();
  iree_hal_resource_initialize(&iree_hal_tt_executable_cache_vtable,
                               &cache->resource);
  cache->host_allocator = host_allocator;
  cache->device = device;
  iree_hal_device_retain(device);
  (void)identifier;

  *out_executable_cache = (iree_hal_executable_cache_t *)cache;
  return iree_ok_status();
}
