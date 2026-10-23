// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_AOT_COMPILER_H_
#define TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_AOT_COMPILER_H_

#include <cstdint>
#include <string>
#include <vector>

struct tt_iree_ttex_entry_point_desc_t;

namespace mlir::iree_compiler::IREE::HAL {
struct TenstorrentNativeSources;

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

// Removes only non-runtime debug information from an ELF copy. Runtime symbols,
// relocations, load segments and the upstream cache entry remain intact.
bool stripTenstorrentAOTArtifactDebugInfo(const std::string &input_path,
                                        const std::string &output_path,
                                        std::string &error_message);

// Generates and offline-compiles the elementwise add reader, writer, and
// compute kernels. Returns false and sets |error_message| on failure.
bool compileAddAOT(TenstorrentAOTTargetArch target_arch,
                   TenstorrentAOTKernelBundle &out_bundle,
                   std::string &error_message);

// Generates and offline-compiles shape-independent tiled matmul kernels.
// Matrix dimensions and per-core tile counts are supplied as runtime args.
bool compileMatmulAOT(TenstorrentAOTTargetArch target_arch,
                      TenstorrentAOTKernelBundle &out_bundle,
                      std::string &error_message);

// Generates and offline-compiles an F32-storage matmul bundle. The reader
// converts row-major F32 operands into BF16 tiles on device and the writer
// converts BF16 result tiles back to F32 storage without host staging.
bool compileF32MatmulAOT(TenstorrentAOTTargetArch target_arch,
                         TenstorrentAOTKernelBundle &out_bundle,
                         std::string &error_message);

// Generates and offline-compiles the page-aware F32 padded-copy data-movement
// kernel. The single reader-role kernel performs both the DRAM read and write.
bool compilePaddedCopyAOT(TenstorrentAOTTargetArch target_arch,
                          TenstorrentAOTKernelBundle &out_bundle,
                          std::string &error_message);

// Shape-independent HAL fill. Compiled at build time and embedded in the HAL;
// this needs no executable metadata or HAL linkage to the offline compiler.
bool compileFillAOT(TenstorrentAOTTargetArch target_arch,
                    TenstorrentAOTKernelBundle &out_bundle,
                    std::string &error_message);

// Descriptor-specialized native kernels without host tensor staging: v15
// device-RISC-V FP32 by default, or experimental v16 INT8-digit Tensix math.
bool compileNativeDispatchAOT(const tt_iree_ttex_entry_point_desc_t &entry,
                              TenstorrentAOTTargetArch target_arch,
                              TenstorrentAOTKernelBundle &out_bundle,
                              std::string &error_message,
                              bool use_tensix = false);
bool compileBF16NativeAOT(const tt_iree_ttex_entry_point_desc_t &entry,
                          const std::string &source,
                          const TenstorrentNativeSources &sources,
                          TenstorrentAOTTargetArch target_arch,
                          TenstorrentAOTKernelBundle &out_bundle,
                          std::string &error_message);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_AOT_COMPILER_H_
