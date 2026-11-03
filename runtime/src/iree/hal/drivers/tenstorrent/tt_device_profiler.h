// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_PROFILER_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_PROFILER_H_

#include <cstdint>

namespace tt::tt_metal::distributed {
class MeshDevice;
} // namespace tt::tt_metal::distributed

namespace iree::hal::tenstorrent {

// tt-metal's device profiler (TT_METAL_DEVICE_PROFILER=1 with a Tracy-enabled
// tt-metal build). Results go to tt-metal's profile_log_device.csv under
// TT_METAL_PROFILER_DIR.

// True when tt-metal records device zones for the opened device.
bool DeviceProfilerEnabled();

// AOT kernels link against firmware symbols, so profiler firmware may only run
// kernels built with the profiler for the live device. Offline device configs
// cannot build those, so while profiling the embedded kernel source is
// JIT-compiled instead of loading the precompiled ELFs.
bool UsePrecompiledKernels();

// Unique per-program id so profiler rows ("run host ID") separate dispatches;
// tt-metal otherwise reports 0 for every freshly created program.
uint64_t NextProgramRuntimeId();

// Drains device profiler buffers. Call after each blocking program so the
// per-core buffers cannot overflow between reads.
void ReadDeviceProfiler(tt::tt_metal::distributed::MeshDevice *mesh_device);

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_DEVICE_PROFILER_H_
