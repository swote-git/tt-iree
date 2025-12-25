// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_allocator.h"

#include <cstring>

#include "iree/hal/drivers/tenstorrent/tt_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_device.h"

//===----------------------------------------------------------------------===//
// iree_hal_tt_allocator_t
//===----------------------------------------------------------------------===//

struct iree_hal_tt_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_tt_device_t* device;
  iree_hal_allocator_statistics_t statistics;
};

static const iree_hal_allocator_vtable_t iree_hal_tt_allocator_vtable;

static iree_hal_tt_allocator_t* iree_hal_tt_allocator_cast(
    iree_hal_allocator_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_allocator_vtable);
  return (iree_hal_tt_allocator_t*)base;
}

//===----------------------------------------------------------------------===//
// Creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_allocator_create(
    iree_hal_tt_device_t* device,
    iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = nullptr;
  
  iree_hal_tt_allocator_t* allocator = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*allocator), (void**)&allocator));
  
  iree_hal_resource_initialize(&iree_hal_tt_allocator_vtable, &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->device = device;
  std::memset(&allocator->statistics, 0, sizeof(allocator->statistics));
  
  *out_allocator = (iree_hal_allocator_t*)allocator;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable implementation
//===----------------------------------------------------------------------===//

static void iree_hal_tt_allocator_destroy(iree_hal_allocator_t* base) {
  auto* allocator = iree_hal_tt_allocator_cast(base);
  iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t iree_hal_tt_allocator_host_allocator(
    const iree_hal_allocator_t* base) {
  return ((const iree_hal_tt_allocator_t*)base)->host_allocator;
}

static iree_status_t iree_hal_tt_allocator_trim(iree_hal_allocator_t*) {
  return iree_ok_status();
}

static void iree_hal_tt_allocator_query_statistics(
    iree_hal_allocator_t* base, iree_hal_allocator_statistics_t* out) {
  auto* allocator = iree_hal_tt_allocator_cast(base);
  std::memcpy(out, &allocator->statistics, sizeof(*out));
}

static iree_status_t iree_hal_tt_allocator_query_memory_heaps(
    iree_hal_allocator_t* base,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* heaps,
    iree_host_size_t* out_count) {
  if (out_count) *out_count = 1;
  if (capacity >= 1 && heaps) {
    heaps[0] = (iree_hal_allocator_memory_heap_t){
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        .max_allocation_size = 28ULL * 1024 * 1024 * 1024,
        .min_alignment = 32,
    };
  }
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_tt_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base,
    iree_hal_buffer_params_t* params,
    iree_device_size_t* allocation_size) {
  if (allocation_size && (*allocation_size % 32) != 0) {
    *allocation_size = ((*allocation_size + 31) / 32) * 32;
  }
  if (!(params->type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;
}

static iree_status_t iree_hal_tt_allocator_allocate_buffer(
    iree_hal_allocator_t* base,
    const iree_hal_buffer_params_t* params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  auto* allocator = iree_hal_tt_allocator_cast(base);
  
  allocation_size = ((allocation_size + 31) / 32) * 32;
  
  iree_status_t status = iree_hal_tt_buffer_create(
      allocator->device, *params, allocation_size,
      allocator->host_allocator, out_buffer);
  
  if (iree_status_is_ok(status)) {
    allocator->statistics.device_bytes_allocated += allocation_size;
  }
  return status;
}

static void iree_hal_tt_allocator_deallocate_buffer(
    iree_hal_allocator_t* base, iree_hal_buffer_t* buffer) {
  auto* allocator = iree_hal_tt_allocator_cast(base);
  allocator->statistics.device_bytes_freed += iree_hal_buffer_allocation_size(buffer);
}

static iree_status_t iree_hal_tt_allocator_import_buffer(
    iree_hal_allocator_t*, const iree_hal_buffer_params_t*,
    iree_hal_external_buffer_t*, iree_hal_buffer_release_callback_t,
    iree_hal_buffer_t**) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "import not implemented");
}

static iree_status_t iree_hal_tt_allocator_export_buffer(
    iree_hal_allocator_t*, iree_hal_buffer_t*, iree_hal_external_buffer_type_t,
    iree_hal_external_buffer_flags_t, iree_hal_external_buffer_t*) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "export not implemented");
}

static const iree_hal_allocator_vtable_t iree_hal_tt_allocator_vtable = {
    .destroy = iree_hal_tt_allocator_destroy,
    .host_allocator = iree_hal_tt_allocator_host_allocator,
    .trim = iree_hal_tt_allocator_trim,
    .query_statistics = iree_hal_tt_allocator_query_statistics,
    .query_memory_heaps = iree_hal_tt_allocator_query_memory_heaps,
    .query_buffer_compatibility = iree_hal_tt_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_tt_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_tt_allocator_deallocate_buffer,
    .import_buffer = iree_hal_tt_allocator_import_buffer,
    .export_buffer = iree_hal_tt_allocator_export_buffer,
};
