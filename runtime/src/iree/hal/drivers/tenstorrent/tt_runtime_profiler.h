// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_PROFILER_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_PROFILER_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "iree/base/api.h"

namespace iree::hal::tenstorrent {

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
  uint64_t flop_count = 0;
  uint64_t host_start_ns = 0;
  uint64_t host_end_ns = 0;
  uint64_t host_duration_ns = 0;
  RuntimeProfileStages stages;
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
  std::vector<RuntimeProfileRecord> records_;
  std::vector<RuntimePreparationRecord> preparation_records_;
  mutable std::mutex mutex_;
};

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_PROFILER_H_
