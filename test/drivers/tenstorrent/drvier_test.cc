// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Driver registration and enumeration tests

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
// Tests
//===----------------------------------------------------------------------===//

int test_driver_registration() {
  TEST_START("Driver registration");

  iree_hal_driver_registry_t* registry = iree_hal_driver_registry_default();
  TEST_ASSERT(registry != nullptr, "registry is NULL");

  iree_status_t status = iree_hal_tenstorrent_driver_module_register(registry);
  TEST_STATUS_OK(status, "driver registration failed");

  TEST_PASS();
  return 0;
}

int test_driver_creation() {
  TEST_START("Driver creation");

  iree_hal_driver_t* driver = nullptr;
  iree_status_t status = iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      IREE_SV("tenstorrent"),
      iree_allocator_system(),
      &driver);
  TEST_STATUS_OK(status, "driver creation failed");
  TEST_ASSERT(driver != nullptr, "driver is NULL");

  iree_hal_driver_release(driver);

  TEST_PASS();
  return 0;
}

int test_device_enumeration() {
  TEST_START("Device enumeration");

  iree_hal_driver_t* driver = nullptr;
  iree_status_t status = iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      IREE_SV("tenstorrent"),
      iree_allocator_system(),
      &driver);
  TEST_STATUS_OK(status, "driver creation failed");

  iree_host_size_t device_count = 0;
  iree_hal_device_info_t* device_infos = nullptr;
  status = iree_hal_driver_query_available_devices(
      driver, iree_allocator_system(), &device_count, &device_infos);
  TEST_STATUS_OK(status, "device enumeration failed");

#ifdef TT_IREE_ENABLE_MOCK
  TEST_ASSERT(device_count >= 1, "mock mode should have at least 1 device");
#else
  // Hardware mode: may have 0 or more devices
  printf("(found %zu device(s)) ", device_count);
#endif

  if (device_count > 0) {
    for (iree_host_size_t i = 0; i < device_count; i++) {
      printf("\n    Device %zu: %.*s (id=%lu) ",
             i,
             (int)device_infos[i].name.size,
             device_infos[i].name.data,
             (unsigned long)device_infos[i].device_id);
    }
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
  iree_hal_driver_release(driver);

  TEST_PASS();
  return 0;
}

int test_driver_info_dump() {
  TEST_START("Driver info dump");

  iree_hal_driver_t* driver = nullptr;
  iree_status_t status = iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      IREE_SV("tenstorrent"),
      iree_allocator_system(),
      &driver);
  TEST_STATUS_OK(status, "driver creation failed");

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);

  status = iree_hal_driver_dump_device_info(driver, 0, &builder);
  TEST_STATUS_OK(status, "dump device info failed");

  // Just verify we got some output
  iree_string_view_t info = iree_string_builder_view(&builder);
  TEST_ASSERT(info.size > 0, "no device info returned");

  iree_string_builder_deinitialize(&builder);
  iree_hal_driver_release(driver);

  TEST_PASS();
  return 0;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  printf("=== Driver Tests ===\n\n");

  int failures = 0;
  failures += test_driver_registration();
  failures += test_driver_creation();
  failures += test_device_enumeration();
  failures += test_driver_info_dump();

  printf("\n=== %d test(s) failed ===\n", failures);
  return failures > 0 ? 1 : 0;
}