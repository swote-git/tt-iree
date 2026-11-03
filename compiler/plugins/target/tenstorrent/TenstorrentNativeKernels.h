// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef TT_IREE_TENSTORRENT_NATIVE_KERNELS_H_
#define TT_IREE_TENSTORRENT_NATIVE_KERNELS_H_
#include <cstdint>
#include <string>
#include <vector>
struct tt_iree_ttex_entry_point_desc_t;
struct tt_iree_ttex_tensor_desc_t;
namespace mlir::iree_compiler::IREE::HAL {
struct TenstorrentNativeSources {
  std::string reader, writer, compute;
  // v22 BF16 Tensix: SFPU epilogue, operand CBs 26-31.
  bool sfpuEpilogue = false;
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

// v22 SFPU epilogue operand. Per output tile the writer streams operand tiles
// into CBs c_26.., unpacked straight to the FP32 Dst and oriented like the
// accumulator. An FP32 operand fills a tile; two BF16 operands share one as its
// high and low halves (read back with raw INT32 loads), so six tiles carry up
// to twelve operands.
struct TenstorrentEpilogueOperand {
  // Per-output-channel vector broadcast over the tile, else a full tensor.
  bool channel = false;
  // Tensor slot; -1 for a compile-time constant channel vector.
  int slot = -1;
  // FP32 bits of a constant channel vector, indexed by channel.
  std::vector<uint32_t> constant;
  // FP32 source tensor elements; BF16 otherwise.
  bool f32 = false;
};
inline constexpr unsigned kTenstorrentMaxEpilogueTiles = 6;
// Dst tile of each operand and its half: 0 the whole FP32 word, 1 the high and
// 2 the low BF16 half.
struct TenstorrentEpiloguePlacement {
  unsigned tile = 0;
  unsigned half = 0;
};
std::vector<TenstorrentEpiloguePlacement>
placeEpilogueOperands(const std::vector<TenstorrentEpilogueOperand> &operands,
                      unsigned &tiles);
// SFPI statements over vFloat `value` (FP32 accumulator) and operand vectors
// `TT_X<i>`; the BF16-rounded result is left in `value`.
struct TenstorrentSfpuEpilogue {
  std::string body;
  std::vector<TenstorrentEpilogueOperand> operands;
};

// SFPI helpers (tt_rne, tt_ugt, tt_select, tt_div) for code inside
// `namespace ckernel::sfpu` of a TRISC_MATH compute kernel.
std::string tenstorrentSfpuHelperSource();

TenstorrentNativeSources generateBF16NativeMatrix(
    const tt_iree_ttex_entry_point_desc_t &geometry,
    const tt_iree_ttex_tensor_desc_t *const *tensors,
    const std::string &epilogue, bool use_tensix = false,
    // Disable only to compare against the original device reader in tests.
    bool prefetch_bf16_inputs = true,
    // Pair output tiles using the existing two-page input/output CB contract.
    bool reuse_output_tiles = false,
    // v22 Tensix: run the epilogue on the SFPU; the writer stores BF16.
    const TenstorrentSfpuEpilogue *sfpu = nullptr);
// v21 tilize: row-major BF16 slot 0 into the face-tiled slot 2, whole tiles
// per core (element range in units of 1024).
std::string
generateBF16NativeTilize(const tt_iree_ttex_tensor_desc_t *const *tensors);
} // namespace mlir::iree_compiler::IREE::HAL
#endif
