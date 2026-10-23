// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::tenstorrent {
namespace {

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

size_t CsvColumnCount(const std::string &line) {
  size_t columns = 1;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (line[i] == ',' && !quoted) {
      ++columns;
    }
  }
  return columns;
}

TEST(RuntimeProfilerTest, WritesComparableOperationAndCoreMetrics) {
  const auto unique_id =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path output_directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_runtime_profiler_test_" + std::to_string(unique_id));
  std::filesystem::remove_all(output_directory);

  RuntimeProfiler profiler;
  IREE_ASSERT_OK(profiler.Initialize(output_directory.string(), /*device_id=*/0,
                                     "Blackhole"));

  RuntimeProfileRecord device_record;
  device_record.op_code = "resnet50_projection";
  device_record.op_type = "MATMUL";
  device_record.execution_domain = "host_assisted_device";
  device_record.storage_datatype = "F32";
  device_record.compute_datatype = "BF16";
  device_record.builtin_program = 4;
  device_record.dispatch_kind = 1;
  device_record.grid_x = 2;
  device_record.grid_y = 2;
  device_record.grid_z = 1;
  device_record.m = 128;
  device_record.n = 64;
  device_record.k = 64;
  device_record.m_tiles = 4;
  device_record.n_tiles = 2;
  device_record.k_tiles = 2;
  device_record.flop_count = 1048576;
  device_record.host_start_ns = 100;
  device_record.host_end_ns = 400;
  device_record.host_duration_ns = 300;
  device_record.stages.input_staging_ns = 11;
  device_record.stages.enqueue_wait_ns = 17;
  device_record.stages.output_staging_ns = 13;
  IREE_ASSERT_OK(profiler.Record(std::move(device_record)));

  RuntimeProfileRecord host_record;
  host_record.op_code = "resnet50,pad";
  host_record.op_type = "PADDED_COPY";
  host_record.execution_domain = "host";
  host_record.storage_datatype = "F32";
  host_record.compute_datatype = "F32";
  host_record.host_start_ns = 500;
  host_record.host_end_ns = 700;
  host_record.host_duration_ns = 200;
  IREE_ASSERT_OK(profiler.Record(std::move(host_record)));

  const std::string operations =
      ReadFile(output_directory / "ops_perf_results.csv");
  EXPECT_NE(operations.find("DEVICE KERNEL DURATION [ns]"), std::string::npos);
  EXPECT_NE(operations.find("IREE ENQUEUE WAIT DURATION [ns]"),
            std::string::npos);
  EXPECT_NE(operations.find("IREE INPUT STAGING DURATION [ns]"),
            std::string::npos);
  EXPECT_NE(operations.find("IREE OUTPUT STAGING DURATION [ns]"),
            std::string::npos);
  EXPECT_NE(operations.find("resnet50_projection,MATMUL,1,0,4"),
            std::string::npos);
  EXPECT_NE(operations.find("\"resnet50,pad\",PADDED_COPY,2,0,0"),
            std::string::npos);
  std::istringstream operation_lines(operations);
  std::vector<std::string> lines;
  for (std::string line; std::getline(operation_lines, line);)
    lines.push_back(std::move(line));
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(CsvColumnCount(lines[0]), CsvColumnCount(lines[1]));
  EXPECT_EQ(CsvColumnCount(lines[0]), CsvColumnCount(lines[2]));

  const std::string mapping = ReadFile(output_directory / "core_mapping.csv");
  EXPECT_NE(mapping.find("1,resnet50_projection,0,0,0,2,0,1,2"),
            std::string::npos);
  EXPECT_NE(mapping.find("1,resnet50_projection,1,1,2,4,1,2,2"),
            std::string::npos);

  const std::string summary = ReadFile(output_directory / "run_results.json");
  EXPECT_NE(summary.find("\"architecture\": \"Blackhole\""), std::string::npos);
  EXPECT_NE(summary.find("\"dispatch_count\": 2"), std::string::npos);
  EXPECT_NE(summary.find("\"total_dispatch_duration_ns\": 500"),
            std::string::npos);
  EXPECT_NE(summary.find("\"total_enqueue_wait_duration_ns\": 17"),
            std::string::npos);
  EXPECT_NE(summary.find("\"device_native\": 0"), std::string::npos);
  EXPECT_NE(summary.find("\"host_assisted_device\": 1"),
            std::string::npos);
  EXPECT_NE(summary.find("\"host\": 1"), std::string::npos);
  EXPECT_NE(summary.find("\"all_dispatches_device_native\": false"),
            std::string::npos);
  EXPECT_NE(summary.find("\"device_trace_available\": false"),
            std::string::npos);

  std::filesystem::remove_all(output_directory);
}

TEST(RuntimeProfilerTest, MarksAnAllDeviceNativeRun) {
  const auto unique_id =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path output_directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_runtime_profiler_native_test_" + std::to_string(unique_id));
  std::filesystem::remove_all(output_directory);

  RuntimeProfiler profiler;
  IREE_ASSERT_OK(profiler.Initialize(output_directory.string(), /*device_id=*/0,
                                     "Blackhole"));
  RuntimeProfileRecord record;
  record.op_code = "matmul";
  record.op_type = "MATMUL";
  record.execution_domain = "device_native";
  record.storage_datatype = "BF16";
  record.compute_datatype = "BF16_FP32_ACC";
  IREE_ASSERT_OK(profiler.Record(std::move(record)));

  const std::string summary = ReadFile(output_directory / "run_results.json");
  EXPECT_NE(summary.find("\"device_native\": 1"), std::string::npos);
  EXPECT_NE(summary.find("\"host_assisted_device\": 0"),
            std::string::npos);
  EXPECT_NE(summary.find("\"all_dispatches_device_native\": true"),
            std::string::npos);
  std::filesystem::remove_all(output_directory);
}

TEST(RuntimeProfilerTest, SeparatesPreparationFromDispatchAndDeviceTiming) {
  const auto output_directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_preparation_test_" + std::to_string(RuntimeProfiler::NowNs()));
  RuntimeProfiler profiler;
  IREE_ASSERT_OK(
      profiler.Initialize(output_directory.string(), 0, "Blackhole"));
  RuntimePreparationRecord create;
  create.event = "DEVICE_CREATE";
  create.host_duration_ns = 100;
  IREE_ASSERT_OK(profiler.RecordPreparation(create));
  RuntimePreparationRecord miss;
  miss.host_start_ns = 100;
  miss.host_end_ns = 400;
  miss.host_duration_ns = 300;
  miss.ttex_version = 15;
  miss.entry_point_count = 1;
  miss.executable_bytes = 4096;
  miss.artifact_count = 3;
  miss.artifact_bytes = 3000;
  miss.materialized_bytes = 1000;
  miss.materialization_ns = 200;
  IREE_ASSERT_OK(profiler.RecordPreparation(miss));
  RuntimePreparationRecord hit;
  hit.cache_hit = true;
  hit.host_duration_ns = 10;
  IREE_ASSERT_OK(profiler.RecordPreparation(hit));
  RuntimePreparationRecord failure;
  failure.status_code = IREE_STATUS_INVALID_ARGUMENT;
  IREE_ASSERT_OK(profiler.RecordPreparation(failure));
  RuntimePreparationRecord destroy;
  destroy.event = "DEVICE_DESTROY";
  destroy.host_duration_ns = 20;
  IREE_ASSERT_OK(profiler.RecordPreparation(destroy));
  const auto summary = ReadFile(output_directory / "run_results.json");
  EXPECT_NE(summary.find("\"dispatch_count\": 0"), std::string::npos);
  EXPECT_NE(summary.find("\"executable_prepare_count\": 3"), std::string::npos);
  EXPECT_NE(summary.find("\"executable_cache_hit_count\": 1"),
            std::string::npos);
  EXPECT_NE(summary.find("\"total_executable_prepare_duration_ns\": 310"),
            std::string::npos);
  EXPECT_NE(summary.find("\"device_create_duration_ns\": 100"),
            std::string::npos);
  EXPECT_NE(summary.find("\"device_destroy_duration_ns\": 20"),
            std::string::npos);
  EXPECT_NE(summary.find("\"total_device_kernel_duration_ns\": null"),
            std::string::npos);
  std::istringstream lines(
      ReadFile(output_directory / "host_setup_results.csv"));
  std::string header, line;
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, header)));
  size_t count = 0;
  while (std::getline(lines, line)) {
    EXPECT_EQ(CsvColumnCount(header), CsvColumnCount(line));
    ++count;
  }
  EXPECT_EQ(count, 5u);
  std::filesystem::remove_all(output_directory);
}

TEST(RuntimeProfilerTest, DisabledPreparationAndTimerAreNoOps) {
  RuntimeProfiler profiler;
  IREE_ASSERT_OK(profiler.RecordPreparation({}));
  RuntimePreparationTimer disabled(nullptr);
  disabled.Stop();
  uint64_t duration = 10;
  {
    RuntimePreparationTimer enabled(&duration);
    enabled.Stop();
    const uint64_t stopped = duration;
    enabled.Stop();
    EXPECT_EQ(stopped, duration);
  }
  EXPECT_GE(duration, 10u);
}

TEST(RuntimeProfilerTest,
     TransferStagesAreExclusiveAndNestedScopesDoNotDuplicate) {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_transfer_test_" + std::to_string(RuntimeProfiler::NowNs()));
  RuntimeProfiler profiler;
  IREE_ASSERT_OK(profiler.Initialize(directory.string(), 0, "mock"));
  profiler.SetInvocationId(7);
  const auto submission = profiler.NextSubmissionId();
  {
    RuntimeProfileCommandScope command(&profiler, submission, 3);
    RuntimeTransferScope fill(&profiler, "FILL");
    {
      RuntimeTransferStage map(RuntimeTransferStageKind::Map);
      RuntimeTransferScope nested(&profiler, "MAP");
      {
        RuntimeTransferStage read(RuntimeTransferStageKind::Read);
        RuntimeTransferScope::ReadBytes(4096);
      }
      IREE_ASSERT_OK(nested.Finish(iree_ok_status()));
    }
    {
      RuntimeTransferStage unmap(RuntimeTransferStageKind::Unmap);
      RuntimeTransferStage write(RuntimeTransferStageKind::Write);
      RuntimeTransferScope::WriteBytes(4096);
    }
    const auto record = *RuntimeTransferScope::CurrentRecord();
    EXPECT_EQ(record.invocation_id, 7u);
    EXPECT_EQ(record.submission_id, submission);
    EXPECT_EQ(record.command_ordinal, 3u);
    EXPECT_FALSE(record.within_dispatch);
    EXPECT_EQ(record.physical_read_bytes, 4096u);
    EXPECT_EQ(record.physical_write_bytes, 4096u);
    const auto stages =
        record.map_ns + record.read_ns + record.unmap_ns + record.write_ns;
    EXPECT_LE(stages, RuntimeProfiler::NowNs() - record.host_start_ns);
    IREE_ASSERT_OK(fill.Finish(iree_ok_status()));
  }
  EXPECT_EQ(RuntimeTransferScope::CurrentRecord(), nullptr);
  IREE_ASSERT_OK(profiler.Flush());
  std::istringstream lines(ReadFile(directory / "transfer_perf_results.csv"));
  std::string header, row, extra;
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, header)));
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, row)));
  EXPECT_FALSE(static_cast<bool>(std::getline(lines, extra)));
  EXPECT_EQ(CsvColumnCount(header), CsvColumnCount(row));
  EXPECT_EQ(row.find("FILL,7,1,3,0,"), 0u);
  const auto summary = ReadFile(directory / "run_results.json");
  EXPECT_NE(summary.find("\"transfer_count\": 1,"), std::string::npos);
  EXPECT_NE(summary.find("\"readfrombuffer_bytes\": 4096,"), std::string::npos);
  EXPECT_NE(summary.find("\"writetobuffer_bytes\": 4096,"), std::string::npos);
  profiler.SetInvocationId(0);
  std::filesystem::remove_all(directory);
}

TEST(RuntimeProfilerTest, RestoresContextAndPreservesTransferFailure) {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_transfer_context_" + std::to_string(RuntimeProfiler::NowNs()));
  RuntimeProfiler profiler;
  IREE_ASSERT_OK(profiler.Initialize(directory.string(), 0, "mock"));
  profiler.SetInvocationId(2);
  {
    RuntimeProfileCommandScope dispatch(&profiler, profiler.NextSubmissionId(),
                                        4, true);
    RuntimeTransferScope read(&profiler, "MAP");
    EXPECT_TRUE(RuntimeTransferScope::CurrentRecord()->within_dispatch);
    iree_status_t result =
        read.Finish(iree_status_from_code(IREE_STATUS_OUT_OF_RANGE));
    EXPECT_EQ(iree_status_code(result), IREE_STATUS_OUT_OF_RANGE);
    iree_status_ignore(result);
    RuntimeProfileRecord operation;
    operation.op_type = "MATMUL";
    IREE_ASSERT_OK(profiler.Record(operation));
  }
  {
    RuntimeTransferScope outside(&profiler, "ALLOCATE");
    EXPECT_EQ(RuntimeTransferScope::CurrentRecord()->submission_id, 0u);
    EXPECT_EQ(RuntimeTransferScope::CurrentRecord()->invocation_id, 2u);
    EXPECT_FALSE(RuntimeTransferScope::CurrentRecord()->within_dispatch);
    IREE_ASSERT_OK(outside.Finish(iree_ok_status()));
  }
  IREE_ASSERT_OK(profiler.Flush());
  const auto operations = ReadFile(directory / "ops_perf_results.csv");
  EXPECT_NE(operations.find(",2,1,4\n"), std::string::npos);
  const auto transfers = ReadFile(directory / "transfer_perf_results.csv");
  EXPECT_NE(transfers.find("MAP,2,1,4,1,"), std::string::npos);
  EXPECT_NE(transfers.find(",11,invoke,0\n"),
            std::string::npos); // OUT_OF_RANGE
  profiler.SetInvocationId(0);
  std::filesystem::remove_all(directory);
}

TEST(RuntimeProfilerTest,
     InvocationContextIsThreadLocalAndSubmissionIdsAreUnique) {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("tt_iree_transfer_threads_" + std::to_string(RuntimeProfiler::NowNs()));
  RuntimeProfiler profiler;
  IREE_ASSERT_OK(profiler.Initialize(directory.string(), 0, "mock"));
  auto record = [&](uint64_t invocation) {
    profiler.SetInvocationId(invocation);
    RuntimeProfileCommandScope command(&profiler, profiler.NextSubmissionId(),
                                       0);
    RuntimeTransferScope transfer(&profiler, "UPDATE");
    EXPECT_EQ(RuntimeTransferScope::CurrentRecord()->invocation_id, invocation);
    IREE_ASSERT_OK(transfer.Finish(iree_ok_status()));
  };
  std::thread first(record, 10), second(record, 20);
  first.join();
  second.join();
  EXPECT_EQ(profiler.NextSubmissionId(), 3u);
  IREE_ASSERT_OK(profiler.Flush());
  const auto summary = ReadFile(directory / "run_results.json");
  EXPECT_NE(summary.find("\"transfer_count\": 2,"), std::string::npos);
  std::filesystem::remove_all(directory);
}

TEST(RuntimeProfilerTest, DisabledTransfersDoNotCreateRecords) {
  RuntimeProfiler profiler;
  profiler.SetInvocationId(1);
  EXPECT_EQ(profiler.NextSubmissionId(), 0u);
  RuntimeTransferScope transfer(&profiler, "FILL");
  RuntimeTransferStage stage(RuntimeTransferStageKind::Read);
  RuntimeTransferScope::ReadBytes(4096);
  EXPECT_EQ(RuntimeTransferScope::CurrentRecord(), nullptr);
  IREE_ASSERT_OK(transfer.Finish(iree_ok_status()));
  IREE_ASSERT_OK(profiler.Flush());
}

} // namespace
} // namespace iree::hal::tenstorrent
