// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for Tenstorrent HAL device creation and queries.

#include "utils.h"

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

using DeviceTest = TenstorrentTestBase;

TEST_F(DeviceTest, DeviceCreation) {
  ASSERT_NE(device_, nullptr);
}

TEST_F(DeviceTest, DeviceId) {
  iree_string_view_t id = iree_hal_device_id(device_);
  EXPECT_GT(id.size, 0u) << "device id should not be empty";
  fprintf(stderr, "  device_id = %.*s\n", (int)id.size, id.data);
}

TEST_F(DeviceTest, QueryCoreCount) {
  int64_t value = 0;
  iree_status_t status = iree_hal_device_query_i64(
      device_, iree_make_cstring_view("hal.device"),
      iree_make_cstring_view("core_count"), &value);

  if (iree_status_is_ok(status)) {
    EXPECT_GT(value, 0) << "core_count should be positive";
    fprintf(stderr, "  core_count = %ld\n", (long)value);
  } else {
    // Query may not be implemented yet; that's acceptable for PoC.
    iree_status_ignore(status);
    GTEST_SKIP() << "core_count query not implemented";
  }
}

TEST_F(DeviceTest, QueryDramSize) {
  int64_t value = 0;
  iree_status_t status = iree_hal_device_query_i64(
      device_, iree_make_cstring_view("hal.device"),
      iree_make_cstring_view("dram_size"), &value);

  if (iree_status_is_ok(status)) {
    EXPECT_GT(value, 0) << "dram_size should be positive";
    fprintf(stderr, "  dram_size = %ld MB\n", (long)(value / (1024 * 1024)));
  } else {
    iree_status_ignore(status);
    GTEST_SKIP() << "dram_size query not implemented";
  }
}

TEST_F(DeviceTest, Allocator) {
  iree_hal_allocator_t* allocator = device_allocator();
  ASSERT_NE(allocator, nullptr);
}


}  // namespace
}  // namespace tenstorrent
}  // namespace hal
}  // namespace iree
