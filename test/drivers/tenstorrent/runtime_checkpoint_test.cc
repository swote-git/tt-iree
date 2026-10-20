// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "iree/hal/drivers/tenstorrent/tt_runtime_checkpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::tenstorrent {
namespace {

std::string ReadFile(const std::filesystem::path &path,
                     std::ios::openmode mode = std::ios::in) {
  std::ifstream input(path, mode);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

TEST(RuntimeCheckpointTest, WritesNpyAndManifestInDispatchOrder) {
  const auto unique_id =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path output_directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_runtime_checkpoint_test_" + std::to_string(unique_id));
  std::filesystem::remove_all(output_directory);

  RuntimeCheckpointWriter writer;
  IREE_ASSERT_OK(writer.Initialize(output_directory.string()));

  RuntimeCheckpointRecord first;
  first.op_code = "resnet50/conv,0";
  first.builtin_program = 5;
  first.dispatch_kind = 2;
  first.output_binding = 2;
  first.shape = {2, 2};
  first.values = {1.0f, 2.0f, 3.0f, 4.0f};
  IREE_ASSERT_OK(writer.Record(std::move(first)));

  RuntimeCheckpointRecord second;
  second.op_code = "resnet50_relu";
  second.builtin_program = 4;
  second.dispatch_kind = 1;
  second.output_binding = 1;
  second.shape = {2};
  second.values = {5.0f, 6.0f};
  IREE_ASSERT_OK(writer.Record(std::move(second)));

  const std::filesystem::path first_npy =
      output_directory / "dispatch_0001_resnet50_conv_0.npy";
  const std::filesystem::path second_npy =
      output_directory / "dispatch_0002_resnet50_relu.npy";
  ASSERT_TRUE(std::filesystem::exists(first_npy));
  ASSERT_TRUE(std::filesystem::exists(second_npy));
  const std::string npy = ReadFile(first_npy, std::ios::in | std::ios::binary);
  ASSERT_GE(npy.size(), 10u + 4u * sizeof(float));
  EXPECT_EQ(npy.substr(0, 6), std::string("\x93NUMPY", 6));
  EXPECT_NE(npy.find("'shape': (2, 2)"), std::string::npos);

  const std::string manifest = ReadFile(output_directory / "manifest.json");
  EXPECT_NE(manifest.find("\"checkpoint_count\": 2"), std::string::npos);
  EXPECT_NE(manifest.find("\"dispatch_index\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"op_code\": \"resnet50/conv,0\""),
            std::string::npos);
  EXPECT_NE(manifest.find("\"file\": \"dispatch_0002_resnet50_relu.npy\""),
            std::string::npos);

  std::filesystem::remove_all(output_directory);
}

TEST(RuntimeCheckpointTest, RejectsShapeValueCountMismatch) {
  const auto unique_id =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path output_directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_runtime_checkpoint_invalid_test_" + std::to_string(unique_id));

  RuntimeCheckpointWriter writer;
  IREE_ASSERT_OK(writer.Initialize(output_directory.string()));
  RuntimeCheckpointRecord record;
  record.op_code = "invalid";
  record.shape = {2, 2};
  record.values = {1.0f};
  iree_status_t status = writer.Record(std::move(record));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  std::filesystem::remove_all(output_directory);
}

} // namespace
} // namespace iree::hal::tenstorrent
