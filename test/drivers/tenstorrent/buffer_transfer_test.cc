// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <limits>
#include <vector>

#include "iree/hal/drivers/tenstorrent/tt_buffer_transfer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::tenstorrent {
namespace {

struct FakeBuffer {
  std::vector<uint8_t> bytes;
  uint64_t logical_size;
  int reads = 0, writes = 0;
  iree_status_code_t read_error = IREE_STATUS_OK, write_error = IREE_STATUS_OK;
  const uint8_t *write_source = nullptr;

  FakeBuffer(size_t logical, size_t physical)
      : bytes(physical, 0xcc), logical_size(logical) {}
  iree_status_t Write(uint64_t offset, const std::vector<uint8_t> &source) {
    return WriteBufferRange(
        iree_allocator_system(), logical_size, bytes.size(), offset,
        source.size(), source.data(),
        [&](uint8_t *data) {
          ++reads;
          if (read_error)
            return iree_status_from_code(read_error);
          std::copy(bytes.begin(), bytes.end(), data);
          return iree_ok_status();
        },
        [&](const uint8_t *data) {
          ++writes;
          write_source = data;
          if (write_error)
            return iree_status_from_code(write_error);
          std::copy(data, data + bytes.size(), bytes.begin());
          return iree_ok_status();
        });
  }
};

TEST(BufferTransferTest, WholePhysicalOverwriteUsesSourceDirectlyWithoutRead) {
  FakeBuffer buffer(4096, 4096);
  std::vector<uint8_t> source(4096, 0x75);
  IREE_ASSERT_OK(buffer.Write(0, source));
  EXPECT_EQ(buffer.reads, 0);
  EXPECT_EQ(buffer.writes, 1);
  EXPECT_EQ(buffer.write_source, source.data());
  EXPECT_EQ(buffer.bytes, source);
}

TEST(BufferTransferTest, WholeLogicalOverwriteZerosPhysicalTailWithoutRead) {
  FakeBuffer buffer(35, 2048);
  std::vector<uint8_t> source(35, 0x42);
  IREE_ASSERT_OK(buffer.Write(0, source));
  EXPECT_EQ(buffer.reads, 0);
  EXPECT_EQ(buffer.writes, 1);
  EXPECT_TRUE(std::equal(source.begin(), source.end(), buffer.bytes.begin()));
  EXPECT_TRUE(std::all_of(buffer.bytes.begin() + 35, buffer.bytes.end(),
                          [](uint8_t value) { return value == 0; }));
}

TEST(BufferTransferTest, PartialWritesPreserveNeighborsAndPhysicalPadding) {
  for (uint64_t offset : {0u, 7u, 20u}) {
    FakeBuffer buffer(35, 2048);
    std::vector<uint8_t> source(15, 0xff);
    std::vector<uint8_t> expected = buffer.bytes;
    std::copy(source.begin(), source.end(), expected.begin() + offset);
    IREE_ASSERT_OK(buffer.Write(offset, source));
    EXPECT_EQ(buffer.reads, 1);
    EXPECT_EQ(buffer.writes, 1);
    EXPECT_EQ(buffer.bytes, expected);
  }
}

TEST(BufferTransferTest, FailedPreservationReadDoesNotWriteOrModifyBytes) {
  FakeBuffer buffer(35, 2048);
  const auto before = buffer.bytes;
  buffer.read_error = IREE_STATUS_INTERNAL;
  iree_status_t result = buffer.Write(1, {1, 2, 3});
  EXPECT_EQ(iree_status_code(result), IREE_STATUS_INTERNAL);
  iree_status_ignore(result);
  EXPECT_EQ(buffer.reads, 1);
  EXPECT_EQ(buffer.writes, 0);
  EXPECT_EQ(buffer.bytes, before);
}

TEST(BufferTransferTest, WriteFailureIsReturnedOnBothFastPaths) {
  for (size_t physical : {35u, 2048u}) {
    FakeBuffer buffer(35, physical);
    const auto before = buffer.bytes;
    buffer.write_error = IREE_STATUS_UNAVAILABLE;
    iree_status_t result = buffer.Write(0, std::vector<uint8_t>(35, 0));
    EXPECT_EQ(iree_status_code(result), IREE_STATUS_UNAVAILABLE);
    iree_status_ignore(result);
    EXPECT_EQ(buffer.reads, 0);
    EXPECT_EQ(buffer.writes, 1);
    EXPECT_EQ(buffer.bytes, before);
  }
}

TEST(BufferTransferTest, RejectsInvalidAndOverflowingRangesWithoutTransfers) {
  FakeBuffer buffer(35, 2048);
  const std::vector<uint8_t> source(2048, 1);
  for (uint64_t offset : {uint64_t{35}, std::numeric_limits<uint64_t>::max()}) {
    iree_status_t result = buffer.Write(offset, source);
    EXPECT_EQ(iree_status_code(result), IREE_STATUS_OUT_OF_RANGE);
    iree_status_ignore(result);
  }
  buffer.logical_size = 4096;
  iree_status_t result = buffer.Write(0, source);
  EXPECT_EQ(iree_status_code(result), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(result);
  EXPECT_EQ(buffer.reads, 0);
  EXPECT_EQ(buffer.writes, 0);
}

TEST(BufferTransferTest, EmptyWriteDoesNotTransfer) {
  FakeBuffer buffer(35, 2048);
  IREE_ASSERT_OK(buffer.Write(35, {}));
  EXPECT_EQ(buffer.reads, 0);
  EXPECT_EQ(buffer.writes, 0);
}

} // namespace
} // namespace iree::hal::tenstorrent
