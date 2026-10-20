// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_PREPROCESSING_PASS_H_
#define TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_PREPROCESSING_PASS_H_

#include <memory>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::HAL {

// Attributes emitted by the preprocessing pass on both the source function
// and the selected linalg.matmul. The operation-level attributes survive
// dispatch formation and are consumed by workgroup lowering and serialization.
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
inline constexpr const char *kTenstorrentPhaseAttr = "iree.tenstorrent.phase";

struct TenstorrentMatmulSelection {
  uint32_t builtinProgram = 0;
  bool hasStaticTileCounts = false;
  uint32_t mTiles = 0;
  uint32_t nTiles = 0;
  uint32_t kTiles = 0;
  bool isDynamic = false;
};

// Applies the canonical shape/type rules for a supported BF16 matmul.
FailureOr<TenstorrentMatmulSelection>
matchTenstorrentBf16Matmul(linalg::MatmulOp matmulOp);

// Reads and validates the preprocessing selection attached to |matmulOp|.
// Fails when the annotation is missing, malformed, or disagrees with the
// canonical matcher, preventing silent drift between compiler phases.
FailureOr<TenstorrentMatmulSelection>
readTenstorrentMatmulSelection(linalg::MatmulOp matmulOp);

std::unique_ptr<Pass> createTenstorrentAnnotateMatmulPreprocessingPass();

} // namespace mlir::iree_compiler::IREE::HAL

#endif // TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_PREPROCESSING_PASS_H_
