// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_AOT_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_AOT_H_

#include <cstddef>
#include <cstdint>

namespace iree::hal::tenstorrent {

struct FillAOTArtifact {
  const char *path;
  const uint8_t *data;
  size_t size;
};
struct FillAOTBundle {
  const char *source;
  const FillAOTArtifact *artifacts;
  size_t artifact_count;
};

// Generated at build time. Architecture values follow TTEX target arch IDs.
const FillAOTBundle *GetFillAOTBundle(uint8_t architecture);

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_AOT_H_
