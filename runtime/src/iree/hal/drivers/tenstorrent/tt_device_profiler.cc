// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_device_profiler.h"

#include <atomic>

#include "impl/profiler/profiler_state.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_device.hpp"

namespace iree::hal::tenstorrent {

bool DeviceProfilerEnabled() { return tt::tt_metal::getDeviceProfilerState(); }

bool UsePrecompiledKernels() { return !DeviceProfilerEnabled(); }

uint64_t NextProgramRuntimeId() {
  static std::atomic<uint64_t> next{1};
  return next.fetch_add(1);
}

void ReadDeviceProfiler(tt::tt_metal::distributed::MeshDevice *mesh_device) {
  if (mesh_device && DeviceProfilerEnabled())
    tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_device);
}

} // namespace iree::hal::tenstorrent
