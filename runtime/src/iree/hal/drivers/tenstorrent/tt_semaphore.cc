// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_semaphore.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/utils/semaphore_base.h"

//===----------------------------------------------------------------------===//
// iree_hal_tt_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_semaphore_t {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;
  iree_atomic_int64_t value;
} iree_hal_tt_semaphore_t;

// Forward declare vtable (defined at end of file)
extern const iree_hal_semaphore_vtable_t iree_hal_tt_semaphore_vtable;

static iree_hal_tt_semaphore_t* iree_hal_tt_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_tt_semaphore_vtable);
  return (iree_hal_tt_semaphore_t*)base_value;
}

iree_status_t iree_hal_tt_semaphore_create(
    iree_allocator_t host_allocator, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_tt_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*semaphore), (void**)&semaphore);
  
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_tt_semaphore_vtable,
                                  &semaphore->base);
    semaphore->host_allocator = host_allocator;
    iree_atomic_store(&semaphore->value, initial_value,
                      iree_memory_order_release);
    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_tt_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_tt_semaphore_t* semaphore =
      iree_hal_tt_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_tt_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_tt_semaphore_t* semaphore =
      iree_hal_tt_semaphore_cast(base_semaphore);
  
  *out_value = iree_atomic_load(&semaphore->value, iree_memory_order_acquire);
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_tt_semaphore_t* semaphore =
      iree_hal_tt_semaphore_cast(base_semaphore);
  
  iree_atomic_store(&semaphore->value, new_value, iree_memory_order_release);
  
  iree_hal_semaphore_poll(&semaphore->base);
  
  return iree_ok_status();
}

static void iree_hal_tt_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                       iree_status_t status) {
  iree_hal_tt_semaphore_t* semaphore =
      iree_hal_tt_semaphore_cast(base_semaphore);
  
  // TODO(swote): store the status to return it
  iree_status_ignore(status);
  
  iree_hal_semaphore_poll(&semaphore->base);
}

static iree_status_t iree_hal_tt_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_hal_wait_flags_t flags) {
  iree_hal_tt_semaphore_t* semaphore =
      iree_hal_tt_semaphore_cast(base_semaphore);
  // On Poc, only busy-wait loop
  // Since tt_device execute is synchronous, signals usually happen before wait
  // in single-threaded scenarios, or shortly after in multi-threaded ones
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  
  while (true) {
    uint64_t current_value = iree_atomic_load(&semaphore->value, iree_memory_order_acquire);
    if (current_value >= value) {
      return iree_ok_status();
    }

    if (iree_time_now() >= deadline_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }

    // Yield to other threads (optional for busy-wait)
    // iree_thread_yield() doesn't exist in v3.9.0, so we'll just poll
    iree_hal_semaphore_poll(&semaphore->base);
  }
}

const iree_hal_semaphore_vtable_t iree_hal_tt_semaphore_vtable = {
    .destroy = iree_hal_tt_semaphore_destroy,
    .query = iree_hal_tt_semaphore_query,
    .signal = iree_hal_tt_semaphore_signal,
    .fail = iree_hal_tt_semaphore_fail,
    .wait = iree_hal_tt_semaphore_wait,
};
