// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "utils.h"

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

using FileTest = TenstorrentTestBase;

static iree_hal_buffer_params_t TransferBufferParams() {
  return {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
}

static iree_status_t ImportMemoryFile(iree_hal_device_t *device,
                                      std::vector<uint8_t> &contents,
                                      iree_io_file_access_t handle_access,
                                      iree_hal_memory_access_t file_access,
                                      iree_hal_file_t **out_file) {
  iree_io_file_handle_t *handle = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
      handle_access, iree_make_byte_span(contents.data(), contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &handle));
  iree_status_t status =
      iree_hal_file_import(device, IREE_HAL_QUEUE_AFFINITY_ANY, file_access,
                           handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file);
  iree_io_file_handle_release(handle);
  return status;
}

TEST_F(FileTest, ReadsHostAllocationSubrangeAndSignalsSemaphore) {
  constexpr iree_device_size_t kBufferSize = 4096;
  constexpr uint64_t kFileOffset = 73;
  constexpr iree_device_size_t kBufferOffset = 901;
  constexpr iree_device_size_t kLength = 777;

  std::vector<uint8_t> file_contents(kBufferSize);
  for (iree_host_size_t i = 0; i < file_contents.size(); ++i) {
    file_contents[i] = static_cast<uint8_t>((i * 29 + 17) & 0xFF);
  }
  iree_hal_file_t *file = nullptr;
  IREE_ASSERT_OK(ImportMemoryFile(device_, file_contents,
                                  IREE_IO_FILE_ACCESS_READ,
                                  IREE_HAL_MEMORY_ACCESS_READ, &file));

  iree::vm::ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), TransferBufferParams(), kBufferSize, &buffer));
  std::vector<uint8_t> expected(kBufferSize, 0xA5);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer.get(), 0, expected.data(),
                                           expected.size()));
  std::memcpy(expected.data() + kBufferOffset,
              file_contents.data() + kFileOffset, kLength);

  iree_hal_semaphore_t *semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1));
  iree_hal_semaphore_t *semaphores[] = {semaphore};
  uint64_t wait_values[] = {1};
  uint64_t signal_values[] = {2};
  iree_hal_semaphore_list_t wait_list = {1, semaphores, wait_values};
  iree_hal_semaphore_list_t signal_list = {1, semaphores, signal_values};

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list, file,
      kFileOffset, buffer.get(), kBufferOffset, kLength,
      IREE_HAL_READ_FLAG_NONE));
  uint64_t semaphore_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &semaphore_value));
  EXPECT_EQ(semaphore_value, 2u);

  std::vector<uint8_t> actual(kBufferSize);
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(buffer.get(), 0, actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);

  iree_hal_semaphore_release(semaphore);
  iree_hal_file_release(file);
}

TEST_F(FileTest, WritesHostAllocationSubrange) {
  constexpr iree_device_size_t kBufferSize = 4096;
  constexpr iree_device_size_t kBufferOffset = 211;
  constexpr uint64_t kFileOffset = 1303;
  constexpr iree_device_size_t kLength = 633;

  std::vector<uint8_t> file_contents(kBufferSize, 0xCC);
  iree_hal_file_t *file = nullptr;
  IREE_ASSERT_OK(ImportMemoryFile(device_, file_contents,
                                  IREE_IO_FILE_ACCESS_WRITE,
                                  IREE_HAL_MEMORY_ACCESS_WRITE, &file));

  iree::vm::ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), TransferBufferParams(), kBufferSize, &buffer));
  std::vector<uint8_t> source(kBufferSize);
  for (iree_host_size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>((i * 37 + 41) & 0xFF);
  }
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(buffer.get(), 0, source.data(), source.size()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, no_semaphores, no_semaphores,
      buffer.get(), kBufferOffset, file, kFileOffset, kLength,
      IREE_HAL_WRITE_FLAG_NONE));

  std::vector<uint8_t> expected(kBufferSize, 0xCC);
  std::memcpy(expected.data() + kFileOffset, source.data() + kBufferOffset,
              kLength);
  EXPECT_EQ(file_contents, expected);

  iree_hal_file_release(file);
}

TEST_F(FileTest, RejectsAccessRangeAndFlags) {
  constexpr iree_device_size_t kSize = 4096;
  std::vector<uint8_t> file_contents(kSize, 0x5A);

  iree_io_file_handle_t *read_only_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(file_contents.data(), file_contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &read_only_handle));
  iree_hal_file_t *file = nullptr;
  iree_status_t status = iree_hal_file_import(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_WRITE,
      read_only_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_PERMISSION_DENIED);
  iree_status_ignore(status);
  EXPECT_EQ(file, nullptr);
  iree_io_file_handle_release(read_only_handle);

  IREE_ASSERT_OK(
      ImportMemoryFile(device_, file_contents,
                       IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
                       IREE_HAL_MEMORY_ACCESS_WRITE, &file));
  iree::vm::ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), TransferBufferParams(), kSize, &buffer));
  iree_hal_semaphore_list_t no_semaphores = {0};

  status = iree_hal_device_queue_read(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, no_semaphores, no_semaphores, file,
      /*source_offset=*/0, buffer.get(), /*target_offset=*/0, /*length=*/128,
      IREE_HAL_READ_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_PERMISSION_DENIED);
  iree_status_ignore(status);

  status = iree_hal_device_queue_write(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, no_semaphores, no_semaphores,
      buffer.get(), /*source_offset=*/0, file, /*target_offset=*/kSize - 64,
      /*length=*/128, IREE_HAL_WRITE_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(status);

  constexpr iree_hal_write_flags_t kUnsupportedFlag = 1ull << 63;
  status = iree_hal_device_queue_write(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, no_semaphores, no_semaphores,
      buffer.get(), /*source_offset=*/0, file, /*target_offset=*/0,
      /*length=*/128, kUnsupportedFlag);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_ignore(status);

  status = iree_hal_device_queue_write(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, no_semaphores, no_semaphores,
      buffer.get(), /*source_offset=*/kSize - 64, file, /*target_offset=*/0,
      /*length=*/128, IREE_HAL_WRITE_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(status);

  iree_hal_file_release(file);
}

} // namespace
} // namespace tenstorrent
} // namespace hal
} // namespace iree
