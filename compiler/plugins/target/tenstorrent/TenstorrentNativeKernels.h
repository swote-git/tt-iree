// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef TT_IREE_TENSTORRENT_NATIVE_KERNELS_H_
#define TT_IREE_TENSTORRENT_NATIVE_KERNELS_H_
#include <cstdint>
#include <string>
struct tt_iree_ttex_entry_point_desc_t;
struct tt_iree_ttex_tensor_desc_t;
namespace mlir::iree_compiler::IREE::HAL {
struct TenstorrentNativeSources {
  std::string reader, writer, compute;
};
struct TenstorrentScalarGrid {
  uint32_t x = 1, y = 1;
};
// Default limits retain the portable/Wormhole policy for existing callers.
inline TenstorrentScalarGrid selectNativeScalarGrid(uint64_t output_rows,
                                                   uint32_t max_x = 8,
                                                   uint32_t max_y = 8) {
  if (!output_rows || !max_x || !max_y)
    return {};
  uint32_t x = static_cast<uint32_t>(output_rows < max_x ? output_rows : max_x);
  uint64_t rows_per_column = output_rows / x;
  return {x, static_cast<uint32_t>(rows_per_column < max_y ? rows_per_column
                                                        : max_y)};
}
// Exact BF16 copy or 3x3/stride-2 max-pool with bounded stack windows and the
// existing 128-byte CB/element-range ABI. Empty means keep scalar codegen.
std::string
generateBF16NativeSpatial(uint8_t operation_kind,
                          const tt_iree_ttex_tensor_desc_t *const *tensors);
TenstorrentNativeSources
generateNativeSources(const tt_iree_ttex_entry_point_desc_t &entry,
                      bool use_tensix = false);
std::string generateBF16NativeCommon(
    const tt_iree_ttex_tensor_desc_t *const *tensors);
TenstorrentNativeSources generateBF16NativeMatrix(
    const tt_iree_ttex_entry_point_desc_t &geometry,
    const tt_iree_ttex_tensor_desc_t *const *tensors,
    const std::string &epilogue, bool use_tensix = false,
    // Disable only to compare against the original device reader in tests.
    bool prefetch_bf16_inputs = true,
    // Pair output tiles using the existing two-page input/output CB contract.
    bool reuse_output_tiles = false);
} // namespace mlir::iree_compiler::IREE::HAL
#endif
