// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for Tenstorrent HAL buffer allocation and data transfer.

#include <cmath>
#include <cstring>

#include "utils.h"

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

using BufferTest = TenstorrentTestBase;

TEST_F(BufferTest, AllocateSingleTile) {
  // 32x32 float = 4096 bytes = 1 tile.
  constexpr iree_device_size_t kBufferSize = 32 * 32 * sizeof(float);

  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), kBufferSize);

  iree_hal_buffer_release(buffer);
}

TEST_F(BufferTest, AllocateMultipleTiles) {
  // 4 tiles: 4 x 32x32 x 4 bytes = 16384 bytes.
  constexpr iree_device_size_t kBufferSize = 4 * 32 * 32 * sizeof(float);

  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer));
  ASSERT_NE(buffer, nullptr);

  iree_hal_buffer_release(buffer);
}

TEST_F(BufferTest, MapWrite) {
  constexpr size_t kNumElements = 256;
  constexpr iree_device_size_t kBufferSize = kNumElements * sizeof(float);

  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
  };

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "host-visible buffer allocation not supported";
  }

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE, 0, kBufferSize, &mapping));

  float* ptr = (float*)mapping.contents.data;
  for (size_t i = 0; i < kNumElements; i++) {
    ptr[i] = static_cast<float>(i);
  }

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(buffer);
}

TEST_F(BufferTest, RoundtripSingleTile) {
  constexpr size_t kNumElements = 32 * 32;
  constexpr iree_device_size_t kBufferSize = kNumElements * sizeof(float);

  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
  };

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "host-visible buffer allocation not supported";
  }

  // Write.
  {
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_WRITE, 0, kBufferSize, &mapping));
    float* ptr = (float*)mapping.contents.data;
    for (size_t i = 0; i < kNumElements; i++) {
      ptr[i] = static_cast<float>(i % 1000) * 0.001f;
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  }

  // Read back and verify.
  {
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, 0, kBufferSize, &mapping));
    float* ptr = (float*)mapping.contents.data;
    for (size_t i = 0; i < kNumElements; i++) {
      float expected = static_cast<float>(i % 1000) * 0.001f;
      ASSERT_NEAR(ptr[i], expected, 1e-5f)
          << "mismatch at index " << i;
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  }

  iree_hal_buffer_release(buffer);
}

TEST_F(BufferTest, AllocatorStatistics) {
  iree_hal_allocator_statistics_t stats;
  iree_hal_allocator_query_statistics(device_allocator(), &stats);
  fprintf(stderr, "  device_bytes_allocated = %lu, device_bytes_freed = %lu\n",
          (unsigned long)stats.device_bytes_allocated,
          (unsigned long)stats.device_bytes_freed);
  // Just verify we can query without crashing; values are informational.
}

}  // namespace
}  // namespace tenstorrent
}  // namespace hal
}  // namespace iree
