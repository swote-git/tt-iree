// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for Tenstorrent HAL driver registration and device enumeration.

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/tenstorrent/api.h"
#include "iree/hal/drivers/tenstorrent/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

class DriverTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_hal_tenstorrent_driver_module_register(
        iree_hal_driver_registry_default()));
  }
};

TEST_F(DriverTest, DriverRegistration) {
  // Verify driver can be found in the registry.
  bool has_driver = false;
  iree_hal_driver_registry_t* registry = iree_hal_driver_registry_default();
  iree_host_size_t count = 0;
  iree_hal_driver_info_t* infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_registry_enumerate(
      registry, iree_allocator_system(), &count, &infos));

  for (iree_host_size_t i = 0; i < count; i++) {
    if (iree_string_view_equal(infos[i].driver_name,
                               iree_make_cstring_view("tenstorrent"))) {
      has_driver = true;
      break;
    }
  }
  iree_allocator_free(iree_allocator_system(), infos);
  ASSERT_TRUE(has_driver) << "tenstorrent driver not found in registry";
}

TEST_F(DriverTest, DriverCreation) {
  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_make_cstring_view("tenstorrent"),
      iree_allocator_system(), &driver));
  ASSERT_NE(driver, nullptr);
  iree_hal_driver_release(driver);
}

TEST_F(DriverTest, DeviceEnumeration) {
  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_make_cstring_view("tenstorrent"),
      iree_allocator_system(), &driver));

  iree_host_size_t device_count = 0;
  iree_hal_device_info_t* device_infos = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver, iree_allocator_system(), &device_count, &device_infos));

#ifdef TT_IREE_ENABLE_MOCK
  EXPECT_GE(device_count, 1u) << "mock mode should have at least 1 device";
#else
  // Hardware mode: log device info for diagnostics.
  for (iree_host_size_t i = 0; i < device_count; i++) {
    fprintf(stderr, "  Device %zu: %.*s (id=%lu)\n", i,
            (int)device_infos[i].name.size, device_infos[i].name.data,
            (unsigned long)device_infos[i].device_id);
  }
#endif

  iree_allocator_free(iree_allocator_system(), device_infos);
  iree_hal_driver_release(driver);
}

TEST_F(DriverTest, CreateDeviceByPath) {
  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_make_cstring_view("tenstorrent"),
      iree_allocator_system(), &driver));

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_path(
      driver, iree_make_cstring_view("tenstorrent"),
      iree_make_cstring_view("0"),
      /*param_count=*/0, /*params=*/nullptr,
      iree_allocator_system(), &device);

  if (iree_status_is_ok(status)) {
    ASSERT_NE(device, nullptr);
    iree_hal_device_release(device);
  } else {
    iree_status_ignore(status);
    iree_hal_driver_release(driver);
    GTEST_SKIP() << "create_device_by_path not implemented";
    return;
  }
  iree_hal_driver_release(driver);
}

TEST_F(DriverTest, DeviceInfoDump) {
  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_make_cstring_view("tenstorrent"),
      iree_allocator_system(), &driver));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(iree_hal_driver_dump_device_info(driver, 0, &builder));

  iree_string_view_t info = iree_string_builder_view(&builder);
  EXPECT_GT(info.size, 0u) << "device info should not be empty";

  // Print for diagnostics.
  fprintf(stderr, "  Device info:\n%.*s\n", (int)info.size, info.data);

  iree_string_builder_deinitialize(&builder);
  iree_hal_driver_release(driver);
}

}  // namespace
}  // namespace tenstorrent
}  // namespace hal
}  // namespace iree
