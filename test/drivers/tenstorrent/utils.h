// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared test fixture for Tenstorrent HAL driver tests.
// Follows IREE's gtest-based testing patterns.

#ifndef TT_IREE_TEST_DRIVERS_TENSTORRENT_UTILS_H_
#define TT_IREE_TEST_DRIVERS_TENSTORRENT_UTILS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/tenstorrent/api.h"
#include "iree/hal/drivers/tenstorrent/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/ref_cc.h"

// RAII wrappers: vm::ref<iree_hal_*_t> auto-releases on scope exit.
IREE_VM_DECLARE_CC_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DECLARE_CC_TYPE_ADAPTERS(iree_hal_command_buffer,
                                 iree_hal_command_buffer_t);
IREE_VM_DECLARE_CC_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);

namespace iree {
namespace hal {
namespace tenstorrent {

// Base test fixture that provides driver registration, driver/device creation,
// and cleanup. Modeled after IREE's HAL CTS test_base pattern.
//
// Driver and device are created once per test suite (not per test) to avoid
// TT-Metal segfaults from repeated MeshDevice create/destroy cycles.
class TenstorrentTestBase : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    iree_hal_driver_registry_t* registry = iree_hal_driver_registry_default();
    IREE_ASSERT_OK(iree_hal_tenstorrent_driver_module_register(registry));

    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        registry, iree_make_cstring_view("tenstorrent"),
        iree_allocator_system(), &driver_));

    IREE_ASSERT_OK(iree_hal_driver_create_device_by_id(
        driver_, /*device_id=*/0,
        /*param_count=*/0, /*params=*/NULL,
        iree_allocator_system(), &device_));
  }

  static void TearDownTestSuite() {
    if (device_) iree_hal_device_release(device_);
    if (driver_) iree_hal_driver_release(driver_);
    device_ = nullptr;
    driver_ = nullptr;
  }

  inline static iree_hal_driver_t* driver_ = nullptr;
  inline static iree_hal_device_t* device_ = nullptr;

  iree_allocator_t host_allocator() { return iree_allocator_system(); }

  iree_hal_allocator_t* device_allocator() {
    return iree_hal_device_allocator(device_);
  }
};

// Lightweight fixture for driver-only tests (no device creation).
class TenstorrentDriverTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    iree_hal_driver_registry_t* registry = iree_hal_driver_registry_default();
    IREE_ASSERT_OK(iree_hal_tenstorrent_driver_module_register(registry));
  }
};

}  // namespace tenstorrent
}  // namespace hal
}  // namespace iree

#endif  // TT_IREE_TEST_DRIVERS_TENSTORRENT_UTILS_H_
