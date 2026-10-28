// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_PROFILER_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_PROFILER_H_

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"

namespace iree::hal::tenstorrent {

enum class RuntimeInvocationPhase { Unscoped, Input, Invoke, Output };

struct RuntimeProfileStages {
  uint64_t input_staging_ns = 0;
  uint64_t buffer_allocation_ns = 0;
  uint64_t pack_ns = 0;
  uint64_t upload_ns = 0;
  uint64_t program_setup_ns = 0;
  uint64_t enqueue_wait_ns = 0;
  uint64_t download_ns = 0;
  uint64_t unpack_ns = 0;
  uint64_t output_staging_ns = 0;
};

struct RuntimeProfileRecord {
  std::string op_code;
  std::string op_type;
  std::string execution_domain;
  std::string storage_datatype;
  std::string compute_datatype;
  uint64_t global_call_count = 0;
  uint64_t invocation_id = 0;
  uint64_t submission_id = 0;
  uint64_t command_ordinal = 0;
  uint32_t builtin_program = 0;
  uint32_t dispatch_kind = 0;
  uint32_t grid_x = 0;
  uint32_t grid_y = 0;
  uint32_t grid_z = 0;
  uint32_t m = 0;
  uint32_t n = 0;
  uint32_t k = 0;
  uint32_t m_tiles = 0;
  uint32_t n_tiles = 0;
  uint32_t k_tiles = 0;
  uint32_t scalar_elements = 0, scalar_row_width = 1;
  uint64_t flop_count = 0;
  uint64_t host_start_ns = 0;
  uint64_t host_end_ns = 0;
  uint64_t host_duration_ns = 0;
  RuntimeProfileStages stages;
};

// One root operation per thread. Nested map/read/write stages contribute
// exclusive time to this row instead of producing double-counted child rows.
struct RuntimeTransferRecord {
  std::string operation;
  uint64_t invocation_id = 0, submission_id = 0, command_ordinal = 0;
  uint64_t source_buffer_id = 0, target_buffer_id = 0;
  uint64_t source_offset = 0, target_offset = 0, requested_bytes = 0;
  uint64_t physical_read_bytes = 0, physical_write_bytes = 0;
  uint64_t device_fill_bytes = 0;
  uint64_t host_start_ns = 0, host_end_ns = 0, host_duration_ns = 0;
  uint64_t map_ns = 0, unmap_ns = 0, allocation_ns = 0;
  uint64_t read_ns = 0, write_ns = 0, host_copy_ns = 0;
  uint64_t flush_ns = 0, invalidate_ns = 0, deallocation_ns = 0;
  uint64_t device_setup_ns = 0, device_enqueue_wait_ns = 0;
  int status_code = 0;
  bool within_dispatch = false;
  RuntimeInvocationPhase phase = RuntimeInvocationPhase::Unscoped;
};

// Host setup events are disjoint from dispatch records and device timing.
struct RuntimePreparationRecord {
  std::string event = "EXECUTABLE_PREPARE";
  uint64_t host_start_ns = 0;
  uint64_t host_end_ns = 0;
  uint64_t host_duration_ns = 0;
  uint64_t executable_bytes = 0;
  uint64_t artifact_count = 0;
  uint64_t artifact_bytes = 0;
  uint64_t materialized_bytes = 0;
  uint64_t hash_lookup_ns = 0;
  uint64_t validation_ns = 0;
  uint64_t data_copy_ns = 0;
  uint64_t materialization_ns = 0;
  uint64_t program_creation_ns = 0;
  uint64_t cache_insert_ns = 0;
  uint32_t ttex_version = 0;
  uint32_t entry_point_count = 0;
  bool cache_hit = false;
  int status_code = 0;
};

class RuntimePreparationTimer {
public:
  explicit RuntimePreparationTimer(uint64_t *duration)
      : duration_(duration), start_(duration ? RuntimeProfilerNowNs() : 0) {}
  ~RuntimePreparationTimer() { Stop(); }
  void Stop() {
    if (duration_) {
      *duration_ += RuntimeProfilerNowNs() - start_;
      duration_ = nullptr;
    }
  }
  RuntimePreparationTimer(const RuntimePreparationTimer &) = delete;
  RuntimePreparationTimer &operator=(const RuntimePreparationTimer &) = delete;

private:
  static uint64_t RuntimeProfilerNowNs();
  uint64_t *duration_;
  uint64_t start_;
};

// Writes a ttrt-compatible subset plus tt-iree-specific timing columns.
// Profiling is disabled unless TT_IREE_PROFILE_OUTPUT names an output
// directory.
class RuntimeProfiler {
public:
  RuntimeProfiler() = default;
  RuntimeProfiler(const RuntimeProfiler &) = delete;
  RuntimeProfiler &operator=(const RuntimeProfiler &) = delete;

  iree_status_t InitializeFromEnvironment(int32_t device_id,
                                          std::string architecture);
  iree_status_t Initialize(std::string output_directory, int32_t device_id,
                           std::string architecture);

  bool enabled() const { return enabled_; }
  iree_status_t Record(RuntimeProfileRecord record);
  iree_status_t RecordPreparation(RuntimePreparationRecord record);
  iree_status_t RecordTransfer(RuntimeTransferRecord record);
  iree_status_t Flush();
  void SetInvocationId(uint64_t id, RuntimeInvocationPhase phase =
                                        RuntimeInvocationPhase::Invoke);
  uint64_t NextSubmissionId() { return enabled_ ? next_submission_id_++ : 0; }

  static uint64_t NowNs();

private:
  iree_status_t WriteSummaryLocked();
  iree_status_t AppendOperationLocked(const RuntimeProfileRecord &record);
  iree_status_t AppendCoreMappingLocked(const RuntimeProfileRecord &record);

  bool enabled_ = false;
  int32_t device_id_ = 0;
  std::string architecture_;
  std::string output_directory_;
  uint64_t next_call_count_ = 1;
  uint64_t dispatch_count_ = 0, dispatch_ns_ = 0, enqueue_ns_ = 0;
  uint64_t domain_counts_[5] = {};
  uint64_t preparation_count_ = 0, preparation_ns_ = 0, cache_hits_ = 0;
  uint64_t device_create_ns_ = 0, device_destroy_ns_ = 0;
  uint64_t transfer_count_ = 0, transfer_ns_ = 0;
  uint64_t transfer_read_bytes_ = 0, transfer_write_bytes_ = 0;
  uint64_t transfer_record_ns_ = 0;
  std::atomic<uint64_t> next_submission_id_{1};
  std::ofstream transfers_;
  mutable std::mutex mutex_;
};

struct RuntimeProfileContext {
  RuntimeProfiler *profiler = nullptr;
  uint64_t invocation = 0, submission = 0, command = 0;
  bool within_dispatch = false;
  RuntimeInvocationPhase phase = RuntimeInvocationPhase::Unscoped;
};

class RuntimeProfileCommandScope {
public:
  RuntimeProfileCommandScope(RuntimeProfiler *profiler, uint64_t submission,
                             uint64_t command, bool within_dispatch = false);
  ~RuntimeProfileCommandScope();
  RuntimeProfileCommandScope(const RuntimeProfileCommandScope &) = delete;
  RuntimeProfileCommandScope &
  operator=(const RuntimeProfileCommandScope &) = delete;

private:
  RuntimeProfileContext previous_;
};

enum class RuntimeTransferStageKind {
  Map,
  Unmap,
  Allocation,
  Read,
  Write,
  HostCopy,
  Flush,
  Invalidate,
  Deallocation,
  DeviceSetup,
  DeviceEnqueueWait
};

class RuntimeTransferScope {
public:
  RuntimeTransferScope(RuntimeProfiler *profiler, const char *operation);
  ~RuntimeTransferScope();
  void Source(iree_hal_buffer_t *buffer, uint64_t offset, uint64_t length);
  void Target(iree_hal_buffer_t *buffer, uint64_t offset, uint64_t length);
  iree_status_t Finish(iree_status_t status);
  static void ReadBytes(uint64_t bytes);
  static void WriteBytes(uint64_t bytes);
  static RuntimeTransferRecord *CurrentRecord();
  RuntimeTransferScope(const RuntimeTransferScope &) = delete;
  RuntimeTransferScope &operator=(const RuntimeTransferScope &) = delete;

private:
  RuntimeProfiler *profiler_;
  RuntimeTransferScope *previous_;
  RuntimeTransferRecord record_;
  bool owns_record_ = false;
  bool finished_ = false;
};

class RuntimeTransferStage {
public:
  explicit RuntimeTransferStage(RuntimeTransferStageKind kind);
  ~RuntimeTransferStage();
  RuntimeTransferStage(const RuntimeTransferStage &) = delete;
  RuntimeTransferStage &operator=(const RuntimeTransferStage &) = delete;

private:
  RuntimeTransferRecord *record_ = nullptr;
  RuntimeTransferStage *previous_ = nullptr;
  uint64_t *duration_ = nullptr;
  uint64_t start_ = 0, children_ns_ = 0;
};

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_PROFILER_H_
