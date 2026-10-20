// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef TT_IREE_TENSTORRENT_NATIVE_KERNELS_H_
#define TT_IREE_TENSTORRENT_NATIVE_KERNELS_H_
#include <string>
struct tt_iree_ttex_entry_point_desc_t;
struct tt_iree_ttex_tensor_desc_t;
namespace mlir::iree_compiler::IREE::HAL {
struct TenstorrentNativeSources {
  std::string reader, writer, compute;
};
TenstorrentNativeSources
generateNativeSources(const tt_iree_ttex_entry_point_desc_t &entry,
                      bool use_tensix = false);
std::string generateBF16NativeCommon(
    const tt_iree_ttex_tensor_desc_t *const *tensors);
TenstorrentNativeSources generateBF16NativeMatrix(
    const tt_iree_ttex_entry_point_desc_t &geometry,
    const tt_iree_ttex_tensor_desc_t *const *tensors,
    const std::string &epilogue, bool use_tensix = false);
} // namespace mlir::iree_compiler::IREE::HAL
#endif
