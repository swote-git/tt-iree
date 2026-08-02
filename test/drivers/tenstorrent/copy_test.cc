// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"
#include "utils.h"

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

using CopyTest = TenstorrentTestBase;

static iree_hal_buffer_params_t TransferBufferParams() {
  return {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
}

static std::vector<uint8_t> MakePattern(iree_device_size_t length,
                                        uint32_t seed) {
  std::vector<uint8_t> data(length);
  for (iree_device_size_t i = 0; i < length; ++i) {
    data[i] = static_cast<uint8_t>((i * 37 + seed) & 0xFF);
  }
  return data;
}

TEST_F(CopyTest, RejectsMismatchedRecordLengths) {
  iree::vm::ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, /*queue_affinity=*/0,
      /*binding_capacity=*/2, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));

  iree_status_t status = iree_hal_command_buffer_copy_buffer(
      command_buffer.get(),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, 0, 128),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, 0, 127),
      IREE_HAL_COPY_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST_F(CopyTest, RejectsUnsupportedFlags) {
  iree::vm::ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, /*queue_affinity=*/0,
      /*binding_capacity=*/2, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));

  constexpr iree_hal_copy_flags_t kUnsupportedFlag = 1ull << 63;
  iree_status_t status = iree_hal_command_buffer_copy_buffer(
      command_buffer.get(),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, 0, 128),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, 0, 128),
      kUnsupportedFlag);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_ignore(status);
}

#ifndef TT_IREE_ENABLE_MOCK
TEST_F(CopyTest, DirectSubrangesPreserveSurroundingBytes) {
  constexpr iree_device_size_t kBufferSize = 4096;
  constexpr iree_device_size_t kSourceOffset = 137;
  constexpr iree_device_size_t kTargetOffset = 913;
  constexpr iree_device_size_t kCopyLength = 777;

  iree::vm::ref<iree_hal_buffer_t> source_buffer;
  iree::vm::ref<iree_hal_buffer_t> target_buffer;
  iree_hal_buffer_params_t params = TransferBufferParams();
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &source_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &target_buffer));

  std::vector<uint8_t> source = MakePattern(kBufferSize, 11);
  std::vector<uint8_t> expected(kBufferSize, 0xCD);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      source_buffer.get(), 0, source.data(), source.size()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      target_buffer.get(), 0, expected.data(), expected.size()));
  std::memcpy(expected.data() + kTargetOffset,
              source.data() + kSourceOffset, kCopyLength);

  iree::vm::ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, /*queue_affinity=*/0,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer.get(),
      iree_hal_make_buffer_ref(source_buffer.get(), kSourceOffset,
                               kCopyLength),
      iree_hal_make_buffer_ref(target_buffer.get(), kTargetOffset,
                               kCopyLength),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      command_buffer.get(), iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  std::vector<uint8_t> actual(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      target_buffer.get(), 0, actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);
}

TEST_F(CopyTest, IndirectOffsetsComposeWithBindingOffsets) {
  constexpr iree_device_size_t kBufferSize = 4096;
  constexpr iree_device_size_t kSourceBindingOffset = 300;
  constexpr iree_device_size_t kSourceRefOffset = 100;
  constexpr iree_device_size_t kTargetBindingOffset = 700;
  constexpr iree_device_size_t kTargetRefOffset = 200;
  constexpr iree_device_size_t kCopyLength = 512;

  iree::vm::ref<iree_hal_buffer_t> source_buffer;
  iree::vm::ref<iree_hal_buffer_t> target_buffer;
  iree_hal_buffer_params_t params = TransferBufferParams();
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &source_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &target_buffer));

  std::vector<uint8_t> source = MakePattern(kBufferSize, 23);
  std::vector<uint8_t> expected(kBufferSize, 0x5A);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      source_buffer.get(), 0, source.data(), source.size()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      target_buffer.get(), 0, expected.data(), expected.size()));
  std::memcpy(expected.data() + kTargetBindingOffset + kTargetRefOffset,
              source.data() + kSourceBindingOffset + kSourceRefOffset,
              kCopyLength);

  iree::vm::ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, /*queue_affinity=*/0,
      /*binding_capacity=*/2, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer.get(),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, kSourceRefOffset, kCopyLength),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/1, kTargetRefOffset, kCopyLength),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  iree_hal_buffer_binding_t bindings[2] = {
      {source_buffer.get(), kSourceBindingOffset, 1200},
      {target_buffer.get(), kTargetBindingOffset, 1500},
  };
  iree_hal_buffer_binding_table_t binding_table = {2, bindings};
  iree_hal_semaphore_list_t no_semaphores = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      command_buffer.get(), binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  std::vector<uint8_t> actual(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      target_buffer.get(), 0, actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);
}

TEST_F(CopyTest, RejectsResolvedIndirectOverlap) {
  constexpr iree_device_size_t kBufferSize = 4096;
  constexpr iree_device_size_t kCopyLength = 400;

  iree::vm::ref<iree_hal_buffer_t> buffer;
  iree_hal_buffer_params_t params = TransferBufferParams();
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &buffer));

  iree::vm::ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, /*queue_affinity=*/0,
      /*binding_capacity=*/2, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer.get(),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/100, kCopyLength),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/1, /*offset=*/0, kCopyLength),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  iree_hal_buffer_binding_t bindings[2] = {
      {buffer.get(), /*offset=*/100, /*length=*/1000},
      {buffer.get(), /*offset=*/200, /*length=*/1000},
  };
  iree_hal_buffer_binding_table_t binding_table = {2, bindings};
  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_status_t status = iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      command_buffer.get(), binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST_F(CopyTest, QueueCopyHonorsSubranges) {
  constexpr iree_device_size_t kBufferSize = 4096;
  constexpr iree_device_size_t kSourceOffset = 211;
  constexpr iree_device_size_t kTargetOffset = 1703;
  constexpr iree_device_size_t kCopyLength = 633;

  iree::vm::ref<iree_hal_buffer_t> source_buffer;
  iree::vm::ref<iree_hal_buffer_t> target_buffer;
  iree_hal_buffer_params_t params = TransferBufferParams();
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &source_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, kBufferSize, &target_buffer));

  std::vector<uint8_t> source = MakePattern(kBufferSize, 41);
  std::vector<uint8_t> expected(kBufferSize, 0xA7);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      source_buffer.get(), 0, source.data(), source.size()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      target_buffer.get(), 0, expected.data(), expected.size()));
  std::memcpy(expected.data() + kTargetOffset,
              source.data() + kSourceOffset, kCopyLength);

  iree_hal_semaphore_list_t no_semaphores = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      source_buffer.get(), kSourceOffset, target_buffer.get(), kTargetOffset,
      kCopyLength, IREE_HAL_COPY_FLAG_NONE));

  std::vector<uint8_t> actual(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      target_buffer.get(), 0, actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);
}
#endif  // !TT_IREE_ENABLE_MOCK

}  // namespace
}  // namespace tenstorrent
}  // namespace hal
}  // namespace iree
