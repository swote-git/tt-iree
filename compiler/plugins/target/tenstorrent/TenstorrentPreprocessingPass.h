// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_PREPROCESSING_PASS_H_
#define TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_PREPROCESSING_PASS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

// Function-level attributes emitted by the preprocessing pass. These are a
// Phase 1 contract only: serializer/runtime still use the existing fallback
// inference path until later phases consume them directly.
inline constexpr const char *kTenstorrentBuiltinProgramAttr =
    "iree.tenstorrent.builtin_program";
inline constexpr const char *kTenstorrentShapeModeAttr =
    "iree.tenstorrent.shape_mode";
inline constexpr const char *kTenstorrentMTilesAttr =
    "iree.tenstorrent.m_tiles";
inline constexpr const char *kTenstorrentNTilesAttr =
    "iree.tenstorrent.n_tiles";
inline constexpr const char *kTenstorrentKTilesAttr =
    "iree.tenstorrent.k_tiles";
inline constexpr const char *kTenstorrentPhaseAttr =
    "iree.tenstorrent.phase";

std::unique_ptr<Pass> createTenstorrentAnnotateMatmulPreprocessingPass();

}  // namespace mlir::iree_compiler::IREE::HAL

#endif  // TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_PREPROCESSING_PASS_H_
