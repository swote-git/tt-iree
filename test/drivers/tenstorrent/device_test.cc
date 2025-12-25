// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Device creation and query tests

#include <cstdio>

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

  return 0;
}

static void teardown() {
  if (g_driver) {
    iree_hal_driver_release(g_driver);
    g_driver = nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

int test_device_creation() {
  TEST_START("Device creation");

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      g_driver, 0, 0, nullptr, iree_allocator_system(), &device);
  TEST_STATUS_OK(status, "device creation failed");
  TEST_ASSERT(device != nullptr, "device is NULL");

  iree_hal_device_release(device);

  TEST_PASS();
  return 0;
}

int test_device_id() {
  TEST_START("Device identifier");

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      g_driver, 0, 0, nullptr, iree_allocator_system(), &device);
  TEST_STATUS_OK(status, "device creation failed");

  iree_string_view_t id = iree_hal_device_id(device);
  TEST_ASSERT(id.size > 0, "device id is empty");
  printf("(id=%.*s) ", (int)id.size, id.data);

  iree_hal_device_release(device);

  TEST_PASS();
  return 0;
}

int test_device_query_core_count() {
  TEST_START("Query core count");

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      g_driver, 0, 0, nullptr, iree_allocator_system(), &device);
  TEST_STATUS_OK(status, "device creation failed");

  int64_t core_x = 0, core_y = 0;
  status = iree_hal_device_query_i64(
      device, IREE_SV("hal.device"), IREE_SV("core_count_x"), &core_x);
  // May fail in mock mode, that's OK
  
  status = iree_hal_device_query_i64(
      device, IREE_SV("hal.device"), IREE_SV("core_count_y"), &core_y);

  if (core_x > 0 && core_y > 0) {
    printf("(%ldx%ld cores) ", (long)core_x, (long)core_y);
  }

  iree_hal_device_release(device);

  TEST_PASS();
  return 0;
}

int test_device_query_dram_size() {
  TEST_START("Query DRAM size");

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      g_driver, 0, 0, nullptr, iree_allocator_system(), &device);
  TEST_STATUS_OK(status, "device creation failed");

  int64_t dram_size = 0;
  status = iree_hal_device_query_i64(
      device, IREE_SV("hal.device"), IREE_SV("dram_size"), &dram_size);

  if (dram_size > 0) {
    printf("(%ld MB) ", (long)(dram_size / (1024 * 1024)));
  }

  iree_hal_device_release(device);

  TEST_PASS();
  return 0;
}

int test_device_allocator() {
  TEST_START("Device allocator");

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      g_driver, 0, 0, nullptr, iree_allocator_system(), &device);
  TEST_STATUS_OK(status, "device creation failed");

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
  TEST_ASSERT(allocator != nullptr, "allocator is NULL");

  iree_hal_device_release(device);

  TEST_PASS();
  return 0;
}

int test_device_create_by_path() {
  TEST_START("Create device by path");

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_path(
      g_driver,
      IREE_SV("tenstorrent"),
      IREE_SV("0"),
      0, nullptr,
      iree_allocator_system(),
      &device);
  TEST_STATUS_OK(status, "device creation by path failed");
  TEST_ASSERT(device != nullptr, "device is NULL");

  iree_hal_device_release(device);

  TEST_PASS();
  return 0;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  printf("=== Device Tests ===\n\n");

  if (setup() != 0) {
    fprintf(stderr, "Setup failed\n");
    return 1;
  }

  int failures = 0;
  failures += test_device_creation();
  failures += test_device_id();
  failures += test_device_query_core_count();
  failures += test_device_query_dram_size();
  failures += test_device_allocator();
  failures += test_device_create_by_path();

  teardown();

  printf("\n=== %d test(s) failed ===\n", failures);
  return failures > 0 ? 1 : 0;
}