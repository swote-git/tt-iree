// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace iree::hal::tenstorrent {
namespace {

constexpr char kOperationsHeader[] =
    "OP CODE,OP TYPE,GLOBAL CALL COUNT,DEVICE ID,CORE COUNT,HOST START TS "
    "[ns],HOST END TS [ns],HOST DURATION [ns],DEVICE KERNEL DURATION "
    "[ns],DEVICE KERNEL DURATION PER CORE MIN [ns],DEVICE KERNEL DURATION "
    "PER CORE MAX [ns],DEVICE KERNEL DURATION PER CORE AVG [ns],INPUT_0_"
    "DATATYPE,OUTPUT_0_DATATYPE,IREE EXECUTION DOMAIN,IREE COMPUTE DATATYPE,"
    "IREE BUILTIN PROGRAM,IREE DISPATCH KIND,IREE GRID X,IREE GRID Y,IREE "
    "GRID Z,IREE M,IREE N,IREE K,IREE FLOP COUNT,IREE END TO END GFLOP/S,"
    "IREE INPUT STAGING DURATION [ns],IREE BUFFER ALLOCATION DURATION "
    "[ns],IREE PACK DURATION [ns],IREE "
    "UPLOAD DURATION [ns],IREE PROGRAM SETUP DURATION [ns],IREE ENQUEUE "
    "WAIT DURATION [ns],IREE DOWNLOAD DURATION [ns],IREE UNPACK DURATION "
    "[ns],IREE OUTPUT STAGING DURATION [ns]\n";

constexpr char kCoreMappingHeader[] =
    "GLOBAL CALL COUNT,OP CODE,CORE X,CORE Y,M TILE BEGIN,M TILE END,N TILE "
    "BEGIN,N TILE END,K TILE COUNT\n";

constexpr char kPreparationHeader[] =
    "EVENT,HOST START TS [ns],HOST END TS [ns],HOST DURATION [ns],STATUS CODE,"
    "CACHE HIT,TTEX VERSION,ENTRY POINT COUNT,EXECUTABLE BYTES,ARTIFACT COUNT,"
    "ARTIFACT BYTES,MATERIALIZED BYTES,HASH LOOKUP [ns],VALIDATION [ns],"
    "DATA COPY [ns],MATERIALIZATION [ns],PROGRAM CREATION [ns],CACHE INSERT "
    "[ns]\n";

std::string CsvEscape(const std::string &value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos)
    return value;
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"')
      escaped.push_back('"');
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

std::string JsonEscape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(c);
      break;
    }
  }
  return escaped;
}

iree_status_t CheckOutputStream(const std::ofstream &stream,
                                const std::filesystem::path &path) {
  if (stream.good())
    return iree_ok_status();
  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "failed to write profiling output '%s'",
                          path.c_str());
}

} // namespace

uint64_t RuntimeProfiler::NowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t RuntimePreparationTimer::RuntimeProfilerNowNs() {
  return RuntimeProfiler::NowNs();
}

iree_status_t
RuntimeProfiler::InitializeFromEnvironment(int32_t device_id,
                                           std::string architecture) {
  const char *output_directory = std::getenv("TT_IREE_PROFILE_OUTPUT");
  if (!output_directory || output_directory[0] == '\0')
    return iree_ok_status();
  return Initialize(output_directory, device_id, std::move(architecture));
}

iree_status_t RuntimeProfiler::Initialize(std::string output_directory,
                                          int32_t device_id,
                                          std::string architecture) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (output_directory.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profiling output directory is empty");
  }

  std::error_code error;
  std::filesystem::create_directories(output_directory, error);
  if (error) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to create profiling directory '%s': %s",
                            output_directory.c_str(), error.message().c_str());
  }

  output_directory_ = std::move(output_directory);
  device_id_ = device_id;
  architecture_ = std::move(architecture);
  next_call_count_ = 1;
  records_.clear();
  preparation_records_.clear();

  const std::filesystem::path output_path(output_directory_);
  const std::filesystem::path operations_path =
      output_path / "ops_perf_results.csv";
  std::ofstream operations(operations_path, std::ios::out | std::ios::trunc);
  operations << kOperationsHeader;
  IREE_RETURN_IF_ERROR(CheckOutputStream(operations, operations_path));

  const std::filesystem::path mapping_path = output_path / "core_mapping.csv";
  std::ofstream mapping(mapping_path, std::ios::out | std::ios::trunc);
  mapping << kCoreMappingHeader;
  IREE_RETURN_IF_ERROR(CheckOutputStream(mapping, mapping_path));

  const auto preparation_path = output_path / "host_setup_results.csv";
  std::ofstream preparation(preparation_path, std::ios::out | std::ios::trunc);
  preparation << kPreparationHeader;
  IREE_RETURN_IF_ERROR(CheckOutputStream(preparation, preparation_path));

  enabled_ = true;
  IREE_RETURN_IF_ERROR(WriteSummaryLocked());
  std::fprintf(stderr, "tt-iree: profiling output: %s\n",
               output_directory_.c_str());
  return iree_ok_status();
}

iree_status_t RuntimeProfiler::Record(RuntimeProfileRecord record) {
  if (!enabled_)
    return iree_ok_status();
  std::lock_guard<std::mutex> lock(mutex_);
  record.global_call_count = next_call_count_++;
  IREE_RETURN_IF_ERROR(AppendOperationLocked(record));
  IREE_RETURN_IF_ERROR(AppendCoreMappingLocked(record));
  records_.push_back(std::move(record));
  return WriteSummaryLocked();
}

iree_status_t
RuntimeProfiler::RecordPreparation(RuntimePreparationRecord record) {
  if (!enabled_)
    return iree_ok_status();
  std::lock_guard<std::mutex> lock(mutex_);
  const auto path =
      std::filesystem::path(output_directory_) / "host_setup_results.csv";
  std::ofstream output(path, std::ios::out | std::ios::app);
  output << CsvEscape(record.event) << ',' << record.host_start_ns << ','
         << record.host_end_ns << ',' << record.host_duration_ns << ','
         << record.status_code << ',' << (record.cache_hit ? 1 : 0) << ','
         << record.ttex_version << ',' << record.entry_point_count << ','
         << record.executable_bytes << ',' << record.artifact_count << ','
         << record.artifact_bytes << ',' << record.materialized_bytes << ','
         << record.hash_lookup_ns << ',' << record.validation_ns << ','
         << record.data_copy_ns << ',' << record.materialization_ns << ','
         << record.program_creation_ns << ',' << record.cache_insert_ns << '\n';
  IREE_RETURN_IF_ERROR(CheckOutputStream(output, path));
  preparation_records_.push_back(std::move(record));
  return WriteSummaryLocked();
}

iree_status_t
RuntimeProfiler::AppendOperationLocked(const RuntimeProfileRecord &record) {
  const std::filesystem::path path =
      std::filesystem::path(output_directory_) / "ops_perf_results.csv";
  std::ofstream output(path, std::ios::out | std::ios::app);
  if (!output.is_open()) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to open profiling output '%s'",
                            path.c_str());
  }

  const uint64_t core_count =
      static_cast<uint64_t>(record.grid_x) * record.grid_y * record.grid_z;
  const double end_to_end_gflops =
      record.host_duration_ns == 0
          ? 0.0
          : static_cast<double>(record.flop_count) /
                static_cast<double>(record.host_duration_ns);
  output << CsvEscape(record.op_code) << ',' << CsvEscape(record.op_type) << ','
         << record.global_call_count << ',' << device_id_ << ',' << core_count
         << ',' << record.host_start_ns << ',' << record.host_end_ns << ','
         << record.host_duration_ns
         // True device kernel timing is populated by the future TT-Metal
         // profiler path. Do not present blocking enqueue time as kernel time.
         << ",,,,," << CsvEscape(record.storage_datatype) << ','
         << CsvEscape(record.storage_datatype) << ','
         << CsvEscape(record.execution_domain) << ','
         << CsvEscape(record.compute_datatype) << ',' << record.builtin_program
         << ',' << record.dispatch_kind << ',' << record.grid_x << ','
         << record.grid_y << ',' << record.grid_z << ',' << record.m << ','
         << record.n << ',' << record.k << ',' << record.flop_count << ','
         << std::fixed << std::setprecision(6) << end_to_end_gflops << ','
         << record.stages.input_staging_ns << ','
         << record.stages.buffer_allocation_ns << ',' << record.stages.pack_ns
         << ',' << record.stages.upload_ns << ','
         << record.stages.program_setup_ns << ','
         << record.stages.enqueue_wait_ns << ',' << record.stages.download_ns
         << ',' << record.stages.unpack_ns << ','
         << record.stages.output_staging_ns << '\n';
  return CheckOutputStream(output, path);
}

iree_status_t
RuntimeProfiler::AppendCoreMappingLocked(const RuntimeProfileRecord &record) {
  if (record.grid_x == 0 || record.grid_y == 0 || record.grid_z != 1 ||
      record.m_tiles == 0 || record.n_tiles == 0 ||
      record.m_tiles % record.grid_y != 0 ||
      record.n_tiles % record.grid_x != 0) {
    return iree_ok_status();
  }

  const std::filesystem::path path =
      std::filesystem::path(output_directory_) / "core_mapping.csv";
  std::ofstream output(path, std::ios::out | std::ios::app);
  if (!output.is_open()) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to open profiling output '%s'",
                            path.c_str());
  }

  const uint32_t per_core_m_tiles = record.m_tiles / record.grid_y;
  const uint32_t per_core_n_tiles = record.n_tiles / record.grid_x;
  for (uint32_t core_y = 0; core_y < record.grid_y; ++core_y) {
    for (uint32_t core_x = 0; core_x < record.grid_x; ++core_x) {
      const uint32_t m_begin = core_y * per_core_m_tiles;
      const uint32_t n_begin = core_x * per_core_n_tiles;
      output << record.global_call_count << ',' << CsvEscape(record.op_code)
             << ',' << core_x << ',' << core_y << ',' << m_begin << ','
             << m_begin + per_core_m_tiles << ',' << n_begin << ','
             << n_begin + per_core_n_tiles << ',' << record.k_tiles << '\n';
    }
  }
  return CheckOutputStream(output, path);
}

iree_status_t RuntimeProfiler::WriteSummaryLocked() {
  uint64_t total_dispatch_duration_ns = 0;
  uint64_t total_enqueue_wait_duration_ns = 0;
  uint64_t device_native_dispatch_count = 0;
  uint64_t host_assisted_dispatch_count = 0;
  uint64_t host_dispatch_count = 0;
  uint64_t host_mock_dispatch_count = 0;
  uint64_t unknown_dispatch_count = 0;
  for (const RuntimeProfileRecord &record : records_) {
    total_dispatch_duration_ns += record.host_duration_ns;
    total_enqueue_wait_duration_ns += record.stages.enqueue_wait_ns;
    if (record.execution_domain == "device_native") {
      ++device_native_dispatch_count;
    } else if (record.execution_domain == "host_assisted_device") {
      ++host_assisted_dispatch_count;
    } else if (record.execution_domain == "host") {
      ++host_dispatch_count;
    } else if (record.execution_domain == "host_mock") {
      ++host_mock_dispatch_count;
    } else {
      ++unknown_dispatch_count;
    }
  }
  const bool all_dispatches_device_native =
      !records_.empty() && device_native_dispatch_count == records_.size();
  uint64_t preparation_count = 0, preparation_ns = 0, cache_hit_count = 0;
  uint64_t device_create_ns = 0, device_destroy_ns = 0;
  for (const auto &record : preparation_records_) {
    if (record.event == "EXECUTABLE_PREPARE") {
      ++preparation_count;
      preparation_ns += record.host_duration_ns;
      cache_hit_count += record.cache_hit ? 1 : 0;
    } else if (record.event == "DEVICE_CREATE") {
      device_create_ns += record.host_duration_ns;
    } else if (record.event == "DEVICE_DESTROY") {
      device_destroy_ns += record.host_duration_ns;
    }
  }

  const std::filesystem::path directory(output_directory_);
  const std::filesystem::path temporary_path =
      directory / "run_results.json.tmp";
  const std::filesystem::path final_path = directory / "run_results.json";
  std::ofstream output(temporary_path, std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to open profiling summary '%s'",
                            temporary_path.c_str());
  }
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"device_id\": " << device_id_ << ",\n"
         << "  \"architecture\": \"" << JsonEscape(architecture_) << "\",\n"
         << "  \"clock\": \"steady_clock\",\n"
         << "  \"dispatch_count\": " << records_.size() << ",\n"
         << "  \"total_dispatch_duration_ns\": " << total_dispatch_duration_ns
         << ",\n"
         << "  \"total_enqueue_wait_duration_ns\": "
         << total_enqueue_wait_duration_ns << ",\n"
         << "  \"execution_domain_counts\": {\n"
         << "    \"device_native\": " << device_native_dispatch_count << ",\n"
         << "    \"host_assisted_device\": " << host_assisted_dispatch_count
         << ",\n"
         << "    \"host\": " << host_dispatch_count << ",\n"
         << "    \"host_mock\": " << host_mock_dispatch_count << ",\n"
         << "    \"unknown\": " << unknown_dispatch_count << "\n"
         << "  },\n"
         << "  \"all_dispatches_device_native\": "
         << (all_dispatches_device_native ? "true" : "false") << ",\n"
         << "  \"total_device_kernel_duration_ns\": null,\n"
         << "  \"device_trace_available\": false,\n"
         << "  \"executable_prepare_count\": " << preparation_count << ",\n"
         << "  \"executable_cache_hit_count\": " << cache_hit_count << ",\n"
         << "  \"total_executable_prepare_duration_ns\": " << preparation_ns
         << ",\n"
         << "  \"device_create_duration_ns\": " << device_create_ns << ",\n"
         << "  \"device_destroy_duration_ns\": " << device_destroy_ns << ",\n"
         << "  \"host_setup\": \"host_setup_results.csv\",\n"
         << "  \"operations\": \"ops_perf_results.csv\",\n"
         << "  \"core_mapping\": \"core_mapping.csv\"\n"
         << "}\n";
  IREE_RETURN_IF_ERROR(CheckOutputStream(output, temporary_path));
  output.close();

  std::error_code error;
  std::filesystem::rename(temporary_path, final_path, error);
  if (error) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to publish profiling summary '%s': %s",
                            final_path.c_str(), error.message().c_str());
  }
  return iree_ok_status();
}

} // namespace iree::hal::tenstorrent
