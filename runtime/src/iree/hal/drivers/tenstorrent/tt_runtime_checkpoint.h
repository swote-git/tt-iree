// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_CHECKPOINT_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_CHECKPOINT_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "iree/base/api.h"

namespace iree::hal::tenstorrent {

struct RuntimeCheckpointRecord {
  std::string op_code;
  uint64_t dispatch_index = 0;
  uint32_t builtin_program = 0;
  uint32_t dispatch_kind = 0;
  uint32_t output_binding = 0;
  std::vector<uint32_t> shape;
  std::vector<float> values;
  std::string file;
};

// Writes logical dispatch outputs as F32 NPY files. Checkpointing is disabled
// unless TT_IREE_CHECKPOINT_OUTPUT names an output directory.
class RuntimeCheckpointWriter {
public:
  RuntimeCheckpointWriter() = default;
  RuntimeCheckpointWriter(const RuntimeCheckpointWriter &) = delete;
  RuntimeCheckpointWriter &operator=(const RuntimeCheckpointWriter &) = delete;

  iree_status_t InitializeFromEnvironment();
  iree_status_t Initialize(std::string output_directory);

  bool enabled() const { return enabled_; }
  iree_status_t Record(RuntimeCheckpointRecord record);

private:
  iree_status_t WriteNpyLocked(const RuntimeCheckpointRecord &record);
  iree_status_t WriteManifestLocked();

  bool enabled_ = false;
  std::string output_directory_;
  uint64_t next_dispatch_index_ = 1;
  std::vector<RuntimeCheckpointRecord> records_;
  mutable std::mutex mutex_;
};

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_RUNTIME_CHECKPOINT_H_
