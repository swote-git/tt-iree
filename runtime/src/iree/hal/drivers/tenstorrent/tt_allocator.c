// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_allocator.h"

#include <stddef.h>

#include "iree/hal/drivers/tenstorrent/tt_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_device.h"

//===----------------------------------------------------------------------===//
// iree_hal_tt_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  
  // Parent device (holds TT-Metal device handle)
  iree_hal_tt_device_t* device;
  
  // Statistics
  iree_hal_allocator_statistics_t statistics;
} iree_hal_tt_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_tt_allocator_vtable;

static iree_hal_tt_allocator_t* iree_hal_tt_allocator_cast(
    iree_hal_allocator_t* base_allocator) {
  IREE_HAL_ASSERT_TYPE(base_allocator, &iree_hal_tt_allocator_vtable);
  return (iree_hal_tt_allocator_t*)base_allocator;
}

//===----------------------------------------------------------------------===//
// Allocator creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_allocator_create(
    iree_hal_tt_device_t* device,
    iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  iree_hal_tt_allocator_t* allocator = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*allocator), (void**)&allocator);
  
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_tt_allocator_vtable,
                                 &allocator->resource);
    allocator->host_allocator = host_allocator;
    allocator->device = device;
    memset(&allocator->statistics, 0, sizeof(allocator->statistics));
    
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }
  
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Allocator vtable implementation
//===----------------------------------------------------------------------===//

static void iree_hal_tt_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_tt_allocator_t* allocator =
      iree_hal_tt_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  iree_allocator_free(host_allocator, allocator);
  
  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_tt_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_tt_allocator_t* allocator =
      (iree_hal_tt_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_tt_allocator_trim(
    iree_hal_allocator_t* base_allocator) {
  // No-op for now; TT-Metal manages its own memory
  return iree_ok_status();
}

static void iree_hal_tt_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  iree_hal_tt_allocator_t* allocator =
      iree_hal_tt_allocator_cast(base_allocator);
  memcpy(out_statistics, &allocator->statistics,
         sizeof(*out_statistics));
}

static iree_status_t iree_hal_tt_allocator_query_memory_heaps(
    iree_hal_allocator_t* base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* heaps,
    iree_host_size_t* out_count) {
  // P100A has a single DRAM heap (28GB)
  // FIXED: Use explicit usage flags instead of IREE_HAL_BUFFER_USAGE_ALL
  const iree_hal_allocator_memory_heap_t heap = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
      .max_allocation_size = 28LL * 1024 * 1024 * 1024,  // 28GB
      .min_alignment = 32,  // 32-byte alignment (tile-friendly)
  };
  
  if (out_count) {
    *out_count = 1;
  }
  if (capacity >= 1 && heaps) {
    heaps[0] = heap;
  }
  
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_tt_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator,
    iree_hal_buffer_params_t* params,
    iree_device_size_t* allocation_size) {
  // For PoC, we only support DEVICE_LOCAL (DRAM) buffers
  
  // Adjust allocation size for tile alignment if needed
  // TT-Metal requires buffers to be aligned to 32 bytes
  if (allocation_size && (*allocation_size % 32) != 0) {
    *allocation_size = ((*allocation_size + 31) / 32) * 32;
  }
  
  // Only support device-local memory for now
  if (!(params->type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  
  return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
         IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE;
}

// FIXED: Add const to params parameter
static iree_status_t iree_hal_tt_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_tt_allocator_t* allocator =
      iree_hal_tt_allocator_cast(base_allocator);
  
  IREE_TRACE_ZONE_BEGIN(z0);
  
  // Align size to 32 bytes (TT-Metal requirement)
  allocation_size = ((allocation_size + 31) / 32) * 32;
  
  // Create buffer through tt_buffer module
  iree_status_t status = iree_hal_tt_buffer_create(
      allocator->device,
      *params,  // Dereference const pointer
      allocation_size,
      allocator->host_allocator,
      out_buffer);
  
  // Update statistics
  if (iree_status_is_ok(status)) {
    allocator->statistics.host_bytes_allocated += allocation_size;
    allocator->statistics.device_bytes_allocated += allocation_size;
  }
  
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_tt_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator,
    iree_hal_buffer_t* buffer) {
  iree_hal_tt_allocator_t* allocator =
      iree_hal_tt_allocator_cast(base_allocator);
  
  // Update statistics
  iree_device_size_t size = iree_hal_buffer_allocation_size(buffer);
  allocator->statistics.host_bytes_freed += size;
  allocator->statistics.device_bytes_freed += size;
  
  // Buffer cleanup is handled by tt_buffer's destroy function
}

// FIXED: Add const to params parameter
static iree_status_t iree_hal_tt_allocator_import_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params,
    iree_hal_external_buffer_t* external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  // Not implemented for PoC
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "buffer import not implemented");
}

static iree_status_t iree_hal_tt_allocator_export_buffer(
    iree_hal_allocator_t* base_allocator,
    iree_hal_buffer_t* buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* out_external_buffer) {
  // Not implemented for PoC
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "buffer export not implemented");
}

//===----------------------------------------------------------------------===//
// vtable
//===----------------------------------------------------------------------===//

static const iree_hal_allocator_vtable_t iree_hal_tt_allocator_vtable = {
    .destroy = iree_hal_tt_allocator_destroy,
    .host_allocator = iree_hal_tt_allocator_host_allocator,
    .trim = iree_hal_tt_allocator_trim,
    .query_statistics = iree_hal_tt_allocator_query_statistics,
    .query_memory_heaps = iree_hal_tt_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_tt_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_tt_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_tt_allocator_deallocate_buffer,
    .import_buffer = iree_hal_tt_allocator_import_buffer,
    .export_buffer = iree_hal_tt_allocator_export_buffer,
};