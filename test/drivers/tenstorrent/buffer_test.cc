// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for Tenstorrent HAL buffer allocation and data transfer.

#include <cmath>
#include <cstring>
#include <vector>

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
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer));

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
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer));

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

TEST_F(BufferTest, PartialMapReadWritePreservesSurroundingBytes) {
  constexpr iree_device_size_t kBufferSize = 4096;
  constexpr iree_device_size_t kPatchOffset = 333;
  constexpr iree_device_size_t kPatchLength = 517;

  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer));

  std::vector<uint8_t> expected(kBufferSize);
  for (iree_device_size_t i = 0; i < kBufferSize; ++i) {
    expected[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
  }
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      buffer, 0, expected.data(), expected.size()));

  std::vector<uint8_t> patch(kPatchLength, 0xA5);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      buffer, kPatchOffset, patch.data(), patch.size()));
  std::memcpy(expected.data() + kPatchOffset, patch.data(), patch.size());

  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      kPatchOffset - 17, kPatchLength + 34, &mapping));
  ASSERT_EQ(mapping.contents.data_length, kPatchLength + 34);
  EXPECT_EQ(std::memcmp(mapping.contents.data,
                        expected.data() + kPatchOffset - 17,
                        mapping.contents.data_length),
            0);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  std::vector<uint8_t> actual(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, 0, actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);

  iree_hal_buffer_release(buffer);
}

TEST_F(BufferTest, MapRangeRejectsOutOfBoundsAccess) {
  constexpr iree_device_size_t kBufferSize = 4096;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer));

  iree_hal_buffer_mapping_t mapping = {};
  iree_status_t status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      kBufferSize - 8, 16, &mapping);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(status);

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
