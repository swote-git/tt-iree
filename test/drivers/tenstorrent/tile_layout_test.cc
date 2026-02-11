// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tile layout conversion standalone tests.
// Verifies row-major ↔ 32x32 tile layout round-trip correctness.

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"

//===----------------------------------------------------------------------===//
// Tile constants and conversion functions
//===----------------------------------------------------------------------===//

constexpr int32_t TT_TILE_HEIGHT = 32;
constexpr int32_t TT_TILE_WIDTH = 32;
constexpr int32_t TT_TILE_SIZE = TT_TILE_HEIGHT * TT_TILE_WIDTH;

static void pack_to_tiles(const float* src, float* dst, int32_t rows,
                          int32_t cols) {
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;
  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          int32_t src_idx =
              (tr * TT_TILE_HEIGHT + r) * cols + (tc * TT_TILE_WIDTH + c);
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t dst_idx = tile_idx * TT_TILE_SIZE + r * TT_TILE_WIDTH + c;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

static void unpack_from_tiles(const float* src, float* dst, int32_t rows,
                              int32_t cols) {
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;
  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t src_idx = tile_idx * TT_TILE_SIZE + r * TT_TILE_WIDTH + c;
          int32_t dst_idx =
              (tr * TT_TILE_HEIGHT + r) * cols + (tc * TT_TILE_WIDTH + c);
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

namespace {

class TileLayoutTest : public ::testing::Test {};

TEST_F(TileLayoutTest, SingleTileRoundTrip) {
  constexpr int32_t kRows = 32, kCols = 32;
  constexpr size_t kN = kRows * kCols;

  std::vector<float> src(kN);
  std::vector<float> tiled(kN);
  std::vector<float> dst(kN);

  for (size_t i = 0; i < kN; i++) src[i] = static_cast<float>(i);

  pack_to_tiles(src.data(), tiled.data(), kRows, kCols);
  unpack_from_tiles(tiled.data(), dst.data(), kRows, kCols);

  for (size_t i = 0; i < kN; i++) {
    ASSERT_EQ(src[i], dst[i]) << "round-trip mismatch at index " << i;
  }
}

TEST_F(TileLayoutTest, TwoByTwoTilesRoundTrip) {
  constexpr int32_t kRows = 64, kCols = 64;
  constexpr size_t kN = kRows * kCols;

  std::vector<float> src(kN);
  std::vector<float> tiled(kN);
  std::vector<float> dst(kN);

  for (size_t i = 0; i < kN; i++) src[i] = static_cast<float>(i);

  pack_to_tiles(src.data(), tiled.data(), kRows, kCols);
  unpack_from_tiles(tiled.data(), dst.data(), kRows, kCols);

  for (size_t i = 0; i < kN; i++) {
    ASSERT_EQ(src[i], dst[i]) << "round-trip mismatch at index " << i;
  }
}

TEST_F(TileLayoutTest, TileOrdering) {
  constexpr int32_t kRows = 64, kCols = 64;
  constexpr size_t kN = kRows * kCols;

  std::vector<float> src(kN);
  std::vector<float> tiled(kN);

  for (size_t i = 0; i < kN; i++) src[i] = static_cast<float>(i);

  pack_to_tiles(src.data(), tiled.data(), kRows, kCols);

  // Tile (0,0) starts at row 0, col 0 → src[0].
  EXPECT_EQ(tiled[0], 0.0f);
  // Tile (0,1) starts at row 0, col 32 → src[32].
  EXPECT_EQ(tiled[TT_TILE_SIZE], 32.0f);
  // Tile (1,0) starts at row 32, col 0 → src[32*64].
  EXPECT_EQ(tiled[2 * TT_TILE_SIZE], 32.0f * 64.0f);
  // Tile (1,1) starts at row 32, col 32 → src[32*64 + 32].
  EXPECT_EQ(tiled[3 * TT_TILE_SIZE], 32.0f * 64.0f + 32.0f);
}

TEST_F(TileLayoutTest, LargeMatrixRoundTrip) {
  constexpr int32_t kRows = 128, kCols = 256;
  constexpr size_t kN = kRows * kCols;

  std::vector<float> src(kN);
  std::vector<float> tiled(kN);
  std::vector<float> dst(kN);

  for (size_t i = 0; i < kN; i++) {
    src[i] = static_cast<float>(i % 1000) * 0.001f;
  }

  pack_to_tiles(src.data(), tiled.data(), kRows, kCols);
  unpack_from_tiles(tiled.data(), dst.data(), kRows, kCols);

  for (size_t i = 0; i < kN; i++) {
    ASSERT_EQ(src[i], dst[i]) << "round-trip mismatch at index " << i;
  }
}

TEST_F(TileLayoutTest, IntraTileElementLayout) {
  constexpr int32_t kRows = 32, kCols = 32;

  std::vector<float> src(TT_TILE_SIZE);
  std::vector<float> tiled(TT_TILE_SIZE);

  // Fill with row-major indices: value = row*100 + col.
  for (int32_t r = 0; r < kRows; r++) {
    for (int32_t c = 0; c < kCols; c++) {
      src[r * kCols + c] = static_cast<float>(r * 100 + c);
    }
  }

  pack_to_tiles(src.data(), tiled.data(), kRows, kCols);

  // Within a single tile, element [r,c] maps to index r*32+c.
  EXPECT_EQ(tiled[0], 0.0f);      // [0,0]
  EXPECT_EQ(tiled[1], 1.0f);      // [0,1]
  EXPECT_EQ(tiled[32], 100.0f);   // [1,0]
  EXPECT_EQ(tiled[33], 101.0f);   // [1,1]
}

}  // namespace
