// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"
#include "iree/hal/drivers/tenstorrent/tt_native_partition.h"

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
    "[ns],IREE OUTPUT STAGING DURATION [ns],IREE INVOCATION ID,"
    "IREE SUBMISSION ID,IREE COMMAND ORDINAL\n";

constexpr char kTransfersHeader[] =
    "OPERATION,INVOCATION ID,SUBMISSION ID,COMMAND ORDINAL,WITHIN DISPATCH,"
    "SOURCE BUFFER ID,TARGET BUFFER ID,SOURCE OFFSET,TARGET OFFSET,"
    "REQUESTED BYTES,READFROMBUFFER BYTES,WRITETOBUFFER BYTES,"
    "HOST START TS [ns],HOST END TS [ns],HOST DURATION [ns],"
    "MAP EXCLUSIVE [ns],UNMAP EXCLUSIVE [ns],ALLOCATION EXCLUSIVE [ns],"
    "READ EXCLUSIVE [ns],WRITE EXCLUSIVE [ns],HOST COPY EXCLUSIVE [ns],"
    "FLUSH EXCLUSIVE [ns],INVALIDATE EXCLUSIVE [ns],"
    "DEALLOCATION EXCLUSIVE [ns],DEVICE SETUP EXCLUSIVE [ns],"
    "DEVICE ENQUEUE WAIT EXCLUSIVE [ns],OTHER [ns],STATUS CODE,INVOCATION "
    "PHASE,DEVICE FILL BYTES\n";

thread_local RuntimeProfileContext current_context;
thread_local RuntimeTransferScope *current_transfer = nullptr;
thread_local RuntimeTransferStage *current_stage = nullptr;

constexpr char kCoreMappingHeader[] =
    "GLOBAL CALL COUNT,OP CODE,CORE X,CORE Y,M TILE BEGIN,M TILE END,N TILE "
    "BEGIN,N TILE END,K TILE COUNT,ELEMENT BEGIN,ELEMENT END,PARTITION UNIT\n";

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
  dispatch_count_ = dispatch_ns_ = enqueue_ns_ = 0;
  for (auto &count : domain_counts_)
    count = 0;
  preparation_count_ = preparation_ns_ = cache_hits_ = 0;
  device_create_ns_ = device_destroy_ns_ = 0;
  transfer_count_ = transfer_ns_ = transfer_record_ns_ = 0;
  transfer_read_bytes_ = transfer_write_bytes_ = 0;
  next_submission_id_ = 1;

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

  if (transfers_.is_open())
    transfers_.close();
  const auto transfer_path = output_path / "transfer_perf_results.csv";
  transfers_.open(transfer_path, std::ios::out | std::ios::trunc);
  transfers_ << kTransfersHeader;
  transfers_.flush();
  IREE_RETURN_IF_ERROR(CheckOutputStream(transfers_, transfer_path));

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
  if (current_context.profiler == this) {
    record.invocation_id = current_context.invocation;
    record.submission_id = current_context.submission;
    record.command_ordinal = current_context.command;
  }
  IREE_RETURN_IF_ERROR(AppendOperationLocked(record));
  IREE_RETURN_IF_ERROR(AppendCoreMappingLocked(record));
  ++dispatch_count_;
  dispatch_ns_ += record.host_duration_ns;
  enqueue_ns_ += record.stages.enqueue_wait_ns;
  size_t domain = record.execution_domain == "device_native"          ? 0
                  : record.execution_domain == "host_assisted_device" ? 1
                  : record.execution_domain == "host"                 ? 2
                  : record.execution_domain == "host_mock"            ? 3
                                                                      : 4;
  ++domain_counts_[domain];
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
  if (record.event == "EXECUTABLE_PREPARE") {
    ++preparation_count_;
    preparation_ns_ += record.host_duration_ns;
    cache_hits_ += record.cache_hit ? 1 : 0;
  } else if (record.event == "DEVICE_CREATE") {
    device_create_ns_ += record.host_duration_ns;
  } else if (record.event == "DEVICE_DESTROY") {
    device_destroy_ns_ += record.host_duration_ns;
  }
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
         << record.stages.output_staging_ns << ',' << record.invocation_id
         << ',' << record.submission_id << ',' << record.command_ordinal
         << '\n';
  return CheckOutputStream(output, path);
}

iree_status_t
RuntimeProfiler::AppendCoreMappingLocked(const RuntimeProfileRecord &record) {
  if (record.grid_x == 0 || record.grid_y == 0 || record.grid_z != 1 ||
      (!record.scalar_elements && (record.m_tiles == 0 || record.n_tiles == 0 ||
                                   record.grid_y > record.m_tiles ||
                                   record.grid_x > record.n_tiles))) {
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

  if (record.scalar_elements) {
    for (uint32_t y = 0; y < record.grid_y; ++y)
      for (uint32_t x = 0; x < record.grid_x; ++x) {
        NativeElementRange range;
        if (!PartitionNativeScalarRange(
                record.scalar_elements, record.scalar_row_width,
                y * record.grid_x + x, record.grid_x * record.grid_y, &range))
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid scalar core mapping");
        output << record.global_call_count << ',' << CsvEscape(record.op_code)
               << ',' << x << ',' << y << ",,,,,," << range.begin << ','
               << uint64_t(range.begin) + range.count << ",rows\n";
      }
    return CheckOutputStream(output, path);
  }

  for (uint32_t core_y = 0; core_y < record.grid_y; ++core_y) {
    for (uint32_t core_x = 0; core_x < record.grid_x; ++core_x) {
      NativeTileRectangle partition;
      if (!PartitionNativeMatrixTiles(record.m_tiles, record.n_tiles, core_y,
                                       core_x, record.grid_y, record.grid_x,
                                       &partition))
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid matrix core mapping");
      output << record.global_call_count << ',' << CsvEscape(record.op_code)
             << ',' << core_x << ',' << core_y << ',' << partition.m_begin << ','
             << uint64_t(partition.m_begin) + partition.m_count << ','
             << partition.n_begin << ','
             << uint64_t(partition.n_begin) + partition.n_count << ','
             << record.k_tiles
             << ",,,tiles\n";
    }
  }
  return CheckOutputStream(output, path);
}

iree_status_t RuntimeProfiler::WriteSummaryLocked() {
  const bool all_dispatches_device_native =
      dispatch_count_ != 0 && domain_counts_[0] == dispatch_count_;

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
         << "  \"dispatch_count\": " << dispatch_count_ << ",\n"
         << "  \"total_dispatch_duration_ns\": " << dispatch_ns_ << ",\n"
         << "  \"total_enqueue_wait_duration_ns\": " << enqueue_ns_ << ",\n"
         << "  \"execution_domain_counts\": {\n"
         << "    \"device_native\": " << domain_counts_[0] << ",\n"
         << "    \"host_assisted_device\": " << domain_counts_[1] << ",\n"
         << "    \"host\": " << domain_counts_[2] << ",\n"
         << "    \"host_mock\": " << domain_counts_[3] << ",\n"
         << "    \"unknown\": " << domain_counts_[4] << "\n"
         << "  },\n"
         << "  \"all_dispatches_device_native\": "
         << (all_dispatches_device_native ? "true" : "false") << ",\n"
         << "  \"total_device_kernel_duration_ns\": null,\n"
         << "  \"device_trace_available\": false,\n"
         << "  \"executable_prepare_count\": " << preparation_count_ << ",\n"
         << "  \"executable_cache_hit_count\": " << cache_hits_ << ",\n"
         << "  \"total_executable_prepare_duration_ns\": " << preparation_ns_
         << ",\n"
         << "  \"device_create_duration_ns\": " << device_create_ns_ << ",\n"
         << "  \"device_destroy_duration_ns\": " << device_destroy_ns_ << ",\n"
         << "  \"transfer_count\": " << transfer_count_ << ",\n"
         << "  \"total_transfer_duration_ns\": " << transfer_ns_ << ",\n"
         << "  \"readfrombuffer_bytes\": " << transfer_read_bytes_ << ",\n"
         << "  \"writetobuffer_bytes\": " << transfer_write_bytes_ << ",\n"
         << "  \"transfer_record_overhead_ns\": " << transfer_record_ns_
         << ",\n"
         << "  \"transfer_stages_exclusive\": true,\n"
         << "  \"transfer_perf\": \"transfer_perf_results.csv\",\n"
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

iree_status_t RuntimeProfiler::Flush() {
  if (!enabled_)
    return iree_ok_status();
  std::lock_guard<std::mutex> lock(mutex_);
  transfers_.flush();
  IREE_RETURN_IF_ERROR(
      CheckOutputStream(transfers_, std::filesystem::path(output_directory_) /
                                        "transfer_perf_results.csv"));
  return WriteSummaryLocked();
}

iree_status_t RuntimeProfiler::RecordTransfer(RuntimeTransferRecord record) {
  if (!enabled_)
    return iree_ok_status();
  const uint64_t start = NowNs();
  std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t stages =
      record.map_ns + record.unmap_ns + record.allocation_ns + record.read_ns +
      record.write_ns + record.host_copy_ns + record.flush_ns +
      record.invalidate_ns + record.deallocation_ns + record.device_setup_ns +
      record.device_enqueue_wait_ns;
  transfers_ << CsvEscape(record.operation) << ',' << record.invocation_id
             << ',' << record.submission_id << ',' << record.command_ordinal
             << ',' << record.within_dispatch << ',' << record.source_buffer_id
             << ',' << record.target_buffer_id << ',' << record.source_offset
             << ',' << record.target_offset << ',' << record.requested_bytes
             << ',' << record.physical_read_bytes << ','
             << record.physical_write_bytes << ',' << record.host_start_ns
             << ',' << record.host_end_ns << ',' << record.host_duration_ns
             << ',' << record.map_ns << ',' << record.unmap_ns << ','
             << record.allocation_ns << ',' << record.read_ns << ','
             << record.write_ns << ',' << record.host_copy_ns << ','
             << record.flush_ns << ',' << record.invalidate_ns << ','
             << record.deallocation_ns << ',' << record.device_setup_ns << ','
             << record.device_enqueue_wait_ns << ','
             << (record.host_duration_ns >= stages
                     ? record.host_duration_ns - stages
                     : 0)
             << ',' << record.status_code << ','
             << (record.phase == RuntimeInvocationPhase::Input    ? "input"
                 : record.phase == RuntimeInvocationPhase::Invoke ? "invoke"
                 : record.phase == RuntimeInvocationPhase::Output ? "output"
                                                                  : "unscoped")
             << ',' << record.device_fill_bytes << '\n';
  IREE_RETURN_IF_ERROR(
      CheckOutputStream(transfers_, std::filesystem::path(output_directory_) /
                                        "transfer_perf_results.csv"));
  ++transfer_count_;
  transfer_ns_ += record.host_duration_ns;
  transfer_read_bytes_ += record.physical_read_bytes;
  transfer_write_bytes_ += record.physical_write_bytes;
  transfer_record_ns_ += NowNs() - start;
  return iree_ok_status();
}

void RuntimeProfiler::SetInvocationId(uint64_t id,
                                      RuntimeInvocationPhase phase) {
  if (enabled_)
    current_context = id ? RuntimeProfileContext{this, id, 0, 0, false, phase}
                         : RuntimeProfileContext{};
}

RuntimeProfileCommandScope::RuntimeProfileCommandScope(
    RuntimeProfiler *profiler, uint64_t submission, uint64_t command,
    bool within_dispatch)
    : previous_(current_context) {
  current_context = {profiler,
                     previous_.profiler == profiler ? previous_.invocation : 0,
                     submission,
                     command,
                     within_dispatch,
                     previous_.profiler == profiler
                         ? previous_.phase
                         : RuntimeInvocationPhase::Unscoped};
}
RuntimeProfileCommandScope::~RuntimeProfileCommandScope() {
  current_context = previous_;
}

RuntimeTransferScope::RuntimeTransferScope(RuntimeProfiler *profiler,
                                           const char *operation)
    : profiler_(profiler), previous_(current_transfer) {
  if (!profiler || !profiler->enabled() || current_transfer)
    return;
  owns_record_ = true;
  record_.host_start_ns = RuntimeProfiler::NowNs();
  record_.operation = operation;
  if (current_context.profiler == profiler) {
    record_.invocation_id = current_context.invocation;
    record_.submission_id = current_context.submission;
    record_.command_ordinal = current_context.command;
    record_.within_dispatch = current_context.within_dispatch;
    record_.phase = current_context.phase;
  }
  current_transfer = this;
}
RuntimeTransferScope::~RuntimeTransferScope() {
  if (owns_record_) {
    if (!finished_)
      iree_status_ignore(Finish(iree_status_from_code(IREE_STATUS_UNKNOWN)));
    current_transfer = previous_;
  }
}
void RuntimeTransferScope::Source(iree_hal_buffer_t *buffer, uint64_t offset,
                                  uint64_t length) {
  if (!owns_record_ || !buffer)
    return;
  record_.source_buffer_id =
      reinterpret_cast<uintptr_t>(iree_hal_buffer_allocated_buffer(buffer));
  record_.source_offset = iree_hal_buffer_byte_offset(buffer) + offset;
  record_.requested_bytes = length;
}
void RuntimeTransferScope::Target(iree_hal_buffer_t *buffer, uint64_t offset,
                                  uint64_t length) {
  if (!owns_record_ || !buffer)
    return;
  record_.target_buffer_id =
      reinterpret_cast<uintptr_t>(iree_hal_buffer_allocated_buffer(buffer));
  record_.target_offset = iree_hal_buffer_byte_offset(buffer) + offset;
  record_.requested_bytes = length;
}
iree_status_t RuntimeTransferScope::Finish(iree_status_t status) {
  if (!owns_record_ || finished_)
    return status;
  finished_ = true;
  record_.host_end_ns = RuntimeProfiler::NowNs();
  record_.host_duration_ns = record_.host_end_ns - record_.host_start_ns;
  record_.status_code = iree_status_code(status);
  // A successful map/allocation transfers ownership to the caller. Do not turn
  // it into a failure (and leak that ownership) because CSV output failed.
  // Flush reports sticky stream errors at submission/invocation boundaries.
  iree_status_ignore(profiler_->RecordTransfer(std::move(record_)));
  return status;
}
RuntimeTransferRecord *RuntimeTransferScope::CurrentRecord() {
  return current_transfer && !current_transfer->finished_
             ? &current_transfer->record_
             : nullptr;
}
void RuntimeTransferScope::ReadBytes(uint64_t bytes) {
  if (auto *record = CurrentRecord())
    record->physical_read_bytes += bytes;
}
void RuntimeTransferScope::WriteBytes(uint64_t bytes) {
  if (auto *record = CurrentRecord())
    record->physical_write_bytes += bytes;
}
RuntimeTransferStage::RuntimeTransferStage(RuntimeTransferStageKind kind)
    : record_(RuntimeTransferScope::CurrentRecord()) {
  if (!record_)
    return;
  uint64_t *durations[] = {&record_->map_ns,
                           &record_->unmap_ns,
                           &record_->allocation_ns,
                           &record_->read_ns,
                           &record_->write_ns,
                           &record_->host_copy_ns,
                           &record_->flush_ns,
                           &record_->invalidate_ns,
                           &record_->deallocation_ns,
                           &record_->device_setup_ns,
                           &record_->device_enqueue_wait_ns};
  duration_ = durations[static_cast<size_t>(kind)];
  previous_ = current_stage;
  current_stage = this;
  start_ = RuntimeProfiler::NowNs();
}
RuntimeTransferStage::~RuntimeTransferStage() {
  if (!record_)
    return;
  const uint64_t elapsed = RuntimeProfiler::NowNs() - start_;
  *duration_ += elapsed - children_ns_;
  if (previous_ && previous_->record_ == record_)
    previous_->children_ns_ += elapsed;
  current_stage = previous_;
}

} // namespace iree::hal::tenstorrent
