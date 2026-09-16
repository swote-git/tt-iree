// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_AOT_COMPILER_H_
#define TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_AOT_COMPILER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::iree_compiler::IREE::HAL {

enum class TenstorrentAOTKernelRole : uint8_t {
  Reader = 1,
  Writer = 2,
  Compute = 3,
};

enum class TenstorrentAOTTargetArch : uint8_t {
  WormholeB0 = 1,
  Blackhole = 2,
};

struct TenstorrentAOTKernelSource {
  uint8_t role = 0;
  std::string source;
};

struct TenstorrentAOTArtifact {
  std::string path;
  std::vector<uint8_t> data;
};

struct TenstorrentAOTKernelBundle {
  TenstorrentAOTTargetArch targetArch = TenstorrentAOTTargetArch::Blackhole;
  std::vector<TenstorrentAOTKernelSource> kernels;
  std::vector<TenstorrentAOTArtifact> artifacts;
};

// Generates and offline-compiles the elementwise add reader, writer, and
// compute kernels. Returns false and sets |error_message| on failure.
bool compileAddAOT(TenstorrentAOTTargetArch target_arch,
                   TenstorrentAOTKernelBundle& out_bundle,
                   std::string& error_message);

// Generates and offline-compiles shape-independent tiled matmul kernels.
// Matrix dimensions and per-core tile counts are supplied as runtime args.
bool compileMatmulAOT(TenstorrentAOTTargetArch target_arch,
                      TenstorrentAOTKernelBundle& out_bundle,
                      std::string& error_message);

}  // namespace mlir::iree_compiler::IREE::HAL

#endif  // TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_AOT_COMPILER_H_
