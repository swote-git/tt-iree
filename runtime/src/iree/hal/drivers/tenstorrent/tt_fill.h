// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_H_

#include <memory>

#include "iree/base/api.h"

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCommandQueue;
} // namespace tt::tt_metal::distributed

namespace iree::hal::tenstorrent {

// One cached AOT workload per device, serialized by the HAL queue mutex.
class DeviceFill {
public:
  DeviceFill();
  ~DeviceFill();
  iree_status_t Execute(tt::tt_metal::distributed::MeshDevice *device,
                        tt::tt_metal::distributed::MeshCommandQueue *queue,
                        uint32_t address, uint32_t offset, uint32_t length,
                        uint32_t pattern);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_H_
