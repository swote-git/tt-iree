// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_BUFFER_TRANSFER_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_BUFFER_TRANSFER_H_

#include <cstring>

#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"

namespace iree::hal::tenstorrent {

// The callbacks transfer the entire physical allocation. Keeping the byte-range
// algorithm separate permits host tests of preservation and padding semantics.
template <typename Read, typename Write>
iree_status_t WriteBufferRange(iree_allocator_t allocator,
                               uint64_t logical_size, uint64_t physical_size,
                               uint64_t offset, uint64_t length,
                               const uint8_t *source, Read read, Write write) {
  if (logical_size > physical_size || offset > logical_size ||
      length > logical_size - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid TT buffer write range");
  }
  if (!length)
    return iree_ok_status();
  const bool full_overwrite = offset == 0 && length == logical_size;
  if (full_overwrite && logical_size == physical_size)
    return write(source);

  void *staging = nullptr;
  iree_status_t result;
  {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Allocation);
    result = iree_allocator_malloc(allocator, physical_size, &staging);
  }
  if (iree_status_is_ok(result) && !full_overwrite) {
    result = read(static_cast<uint8_t *>(staging));
  }
  if (iree_status_is_ok(result)) {
    {
      RuntimeTransferStage stage(RuntimeTransferStageKind::HostCopy);
      if (full_overwrite) {
        // Physical tile padding is not part of the logical HAL buffer.
        std::memset(static_cast<uint8_t *>(staging) + length, 0,
                    physical_size - length);
      }
      std::memcpy(static_cast<uint8_t *>(staging) + offset, source, length);
    }
    result = write(static_cast<const uint8_t *>(staging));
  }
  if (staging) {
    RuntimeTransferStage stage(RuntimeTransferStageKind::Deallocation);
    iree_allocator_free(allocator, staging);
  }
  return result;
}

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_BUFFER_TRANSFER_H_
