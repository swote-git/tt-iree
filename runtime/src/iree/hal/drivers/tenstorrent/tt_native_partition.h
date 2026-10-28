// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_NATIVE_PARTITION_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_NATIVE_PARTITION_H_

#include <cstdint>

namespace iree::hal::tenstorrent {

struct NativeElementRange {
  uint32_t begin = 0, count = 0;
};

struct NativeTileRectangle {
  uint32_t m_begin = 0, n_begin = 0, m_count = 0, n_count = 0;
};

// Preserve the native m/n start/count ABI. Each axis differs by at most one
// tile, and rejecting oversized grids guarantees that no matrix core is idle.
inline bool PartitionNativeMatrixTiles(uint32_t m_tiles, uint32_t n_tiles,
                                       uint32_t row, uint32_t col,
                                       uint32_t rows, uint32_t cols,
                                       NativeTileRectangle *out) {
  if (!out || !rows || !cols || row >= rows || col >= cols || rows > m_tiles ||
      cols > n_tiles)
    return false;
  uint64_t m_begin = uint64_t(m_tiles) * row / rows;
  uint64_t m_end = uint64_t(m_tiles) * (uint64_t(row) + 1) / rows;
  uint64_t n_begin = uint64_t(n_tiles) * col / cols;
  uint64_t n_end = uint64_t(n_tiles) * (uint64_t(col) + 1) / cols;
  *out = {static_cast<uint32_t>(m_begin), static_cast<uint32_t>(n_begin),
          static_cast<uint32_t>(m_end - m_begin),
          static_cast<uint32_t>(n_end - n_begin)};
  return true;
}

// Range args stay in elements. Only the placement policy changes to whole
// logical rows, including uneven partitions and idle cores for tiny tensors.
inline bool PartitionNativeScalarRange(uint32_t elements, uint32_t row_width,
                                       uint32_t ordinal, uint32_t cores,
                                       NativeElementRange *out) {
  if (!out || !row_width || elements % row_width || !cores || ordinal >= cores)
    return false;
  uint32_t rows = elements / row_width;
  uint64_t begin = uint64_t(rows) * ordinal / cores;
  uint64_t end = uint64_t(rows) * (uint64_t(ordinal) + 1) / cores;
  *out = {static_cast<uint32_t>(begin * row_width),
          static_cast<uint32_t>((end - begin) * row_width)};
  return true;
}

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_NATIVE_PARTITION_H_
