// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Buffer allocation and data transfer tests

#include <cmath>
#include <cstdio>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/tenstorrent/registration/driver_module.h"

//===----------------------------------------------------------------------===//
// Test utilities
//===----------------------------------------------------------------------===//

#define TEST_ASSERT(cond, msg)                                           \
  do {                                                                   \
    if (!(cond)) {                                                       \
      fprintf(stderr, "FAILED: %s\n  %s:%d\n", msg, __FILE__, __LINE__); \
      return 1;                                                          \
    }                                                                    \
  } while (0)

#define TEST_STATUS_OK(status, msg)                                      \
  do {                                                                   \
    if (!iree_status_is_ok(status)) {                                    \
      fprintf(stderr, "FAILED: %s\n  %s:%d\n  ", msg, __FILE__, __LINE__); \
      iree_status_fprint(stderr, status);                                \
      fprintf(stderr, "\n");                                             \
      iree_status_ignore(status);                                        \
      return 1;                                                          \
    }                                                                    \
  } while (0)

#define TEST_START(name) printf("  %s... ", name); fflush(stdout)
#define TEST_PASS() printf("PASSED\n")

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

static iree_hal_driver_t* g_driver = nullptr;
static iree_hal_device_t* g_device = nullptr;
static iree_hal_allocator_t* g_allocator = nullptr;

static int setup() {
  iree_hal_driver_registry_t* registry = iree_hal_driver_registry_default();
  iree_status_t status = iree_hal_tenstorrent_driver_module_register(registry);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 1;
  }

  status = iree_hal_driver_registry_try_create(
      registry, IREE_SV("tenstorrent"), iree_allocator_system(), &g_driver);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 1;
  }

  status = iree_hal_driver_create_device_by_id(
      g_driver, 0, 0, nullptr, iree_allocator_system(), &g_device);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_hal_driver_release(g_driver);
    return 1;
  }

  g_allocator = iree_hal_device_allocator(g_device);
  if (!g_allocator) {
    iree_hal_device_release(g_device);
    iree_hal_driver_release(g_driver);
    return 1;
  }

  return 0;
}

static void teardown() {
  if (g_device) {
    iree_hal_device_release(g_device);
    g_device = nullptr;
  }
  if (g_driver) {
    iree_hal_driver_release(g_driver);
    g_driver = nullptr;
  }
  g_allocator = nullptr;
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

int test_buffer_allocation_single_tile() {
  TEST_START("Buffer allocation (single tile, 4KB)");

  const iree_device_size_t buffer_size = 32 * 32 * sizeof(float);  // 4KB

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
  };

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      g_allocator, params, buffer_size, &buffer);
  TEST_STATUS_OK(status, "buffer allocation failed");
  TEST_ASSERT(buffer != nullptr, "buffer is NULL");

  iree_device_size_t size = iree_hal_buffer_allocation_size(buffer);
  TEST_ASSERT(size >= buffer_size, "buffer size too small");
  printf("(size=%lu) ", (unsigned long)size);

  iree_hal_buffer_release(buffer);

  TEST_PASS();
  return 0;
}

int test_buffer_allocation_multiple_tiles() {
  TEST_START("Buffer allocation (4 tiles, 16KB)");

  const iree_device_size_t buffer_size = 64 * 64 * sizeof(float);  // 16KB

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
  };

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      g_allocator, params, buffer_size, &buffer);
  TEST_STATUS_OK(status, "buffer allocation failed");
  TEST_ASSERT(buffer != nullptr, "buffer is NULL");

  iree_device_size_t size = iree_hal_buffer_allocation_size(buffer);
  TEST_ASSERT(size >= buffer_size, "buffer size too small");

  iree_hal_buffer_release(buffer);

  TEST_PASS();
  return 0;
}

int test_buffer_map_write() {
  TEST_START("Buffer map for write");

  const size_t num_elements = 32 * 32;
  const iree_device_size_t buffer_size = num_elements * sizeof(float);

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
  };

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      g_allocator, params, buffer_size, &buffer);
  TEST_STATUS_OK(status, "buffer allocation failed");

  iree_hal_buffer_mapping_t mapping;
  status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE,
      0, buffer_size, &mapping);
  TEST_STATUS_OK(status, "map for write failed");

  TEST_ASSERT(mapping.contents.data != nullptr, "mapped data is NULL");
  TEST_ASSERT(mapping.contents.data_length >= buffer_size, "mapped size too small");

  // Write test data
  float* ptr = (float*)mapping.contents.data;
  for (size_t i = 0; i < num_elements; i++) {
    ptr[i] = (float)i;
  }

  status = iree_hal_buffer_unmap_range(&mapping);
  TEST_STATUS_OK(status, "unmap failed");

  iree_hal_buffer_release(buffer);

  TEST_PASS();
  return 0;
}

int test_buffer_roundtrip_single_tile() {
  TEST_START("Buffer roundtrip (single tile)");

  const size_t num_elements = 32 * 32;
  const iree_device_size_t buffer_size = num_elements * sizeof(float);

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
  };

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      g_allocator, params, buffer_size, &buffer);
  TEST_STATUS_OK(status, "buffer allocation failed");

  // Write
  iree_hal_buffer_mapping_t write_mapping;
  status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE,
      0, buffer_size, &write_mapping);
  TEST_STATUS_OK(status, "map for write failed");

  float* write_ptr = (float*)write_mapping.contents.data;
  for (size_t i = 0; i < num_elements; i++) {
    write_ptr[i] = (float)i * 0.1f;
  }

  status = iree_hal_buffer_unmap_range(&write_mapping);
  TEST_STATUS_OK(status, "unmap write failed");

  // Read
  iree_hal_buffer_mapping_t read_mapping;
  status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ,
      0, buffer_size, &read_mapping);
  TEST_STATUS_OK(status, "map for read failed");

  float* read_ptr = (float*)read_mapping.contents.data;
  int errors = 0;
  for (size_t i = 0; i < num_elements; i++) {
    float expected = (float)i * 0.1f;
    if (std::fabs(read_ptr[i] - expected) > 1e-5f) {
      if (errors < 5) {
        fprintf(stderr, "\n    Mismatch at %zu: expected %f, got %f",
                i, expected, read_ptr[i]);
      }
      errors++;
    }
  }

  status = iree_hal_buffer_unmap_range(&read_mapping);
  TEST_STATUS_OK(status, "unmap read failed");

  iree_hal_buffer_release(buffer);

  TEST_ASSERT(errors == 0, "data mismatch");
  TEST_PASS();
  return 0;
}

int test_buffer_roundtrip_multiple_tiles() {
  TEST_START("Buffer roundtrip (2x2 tiles)");

  const size_t num_elements = 64 * 64;
  const iree_device_size_t buffer_size = num_elements * sizeof(float);

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
  };

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      g_allocator, params, buffer_size, &buffer);
  TEST_STATUS_OK(status, "buffer allocation failed");

  // Write
  iree_hal_buffer_mapping_t write_mapping;
  status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE,
      0, buffer_size, &write_mapping);
  TEST_STATUS_OK(status, "map for write failed");

  float* write_ptr = (float*)write_mapping.contents.data;
  for (size_t i = 0; i < num_elements; i++) {
    write_ptr[i] = (float)(i % 1000) * 0.001f;
  }

  status = iree_hal_buffer_unmap_range(&write_mapping);
  TEST_STATUS_OK(status, "unmap write failed");

  // Read
  iree_hal_buffer_mapping_t read_mapping;
  status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ,
      0, buffer_size, &read_mapping);
  TEST_STATUS_OK(status, "map for read failed");

  float* read_ptr = (float*)read_mapping.contents.data;
  int errors = 0;
  for (size_t i = 0; i < num_elements; i++) {
    float expected = (float)(i % 1000) * 0.001f;
    if (std::fabs(read_ptr[i] - expected) > 1e-5f) {
      errors++;
    }
  }

  status = iree_hal_buffer_unmap_range(&read_mapping);
  TEST_STATUS_OK(status, "unmap read failed");

  iree_hal_buffer_release(buffer);

  TEST_ASSERT(errors == 0, "data mismatch");
  TEST_PASS();
  return 0;
}

int test_allocator_statistics() {
  TEST_START("Allocator statistics");

  iree_hal_allocator_statistics_t stats;
  iree_hal_allocator_query_statistics(g_allocator, &stats);

  // Just verify we can query stats without crashing
  printf("(alloc=%lu, freed=%lu) ",
         (unsigned long)stats.device_bytes_allocated,
         (unsigned long)stats.device_bytes_freed);

  TEST_PASS();
  return 0;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  printf("=== Buffer Tests ===\n\n");

  if (setup() != 0) {
    fprintf(stderr, "Setup failed\n");
    return 1;
  }

  int failures = 0;
  failures += test_buffer_allocation_single_tile();
  failures += test_buffer_allocation_multiple_tiles();
  failures += test_buffer_map_write();
  failures += test_buffer_roundtrip_single_tile();
  failures += test_buffer_roundtrip_multiple_tiles();
  failures += test_allocator_statistics();

  teardown();

  printf("\n=== %d test(s) failed ===\n", failures);
  return failures > 0 ? 1 : 0;
}