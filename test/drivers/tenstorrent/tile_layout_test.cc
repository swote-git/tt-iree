// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tile layout conversion test (standalone, no IREE/TT-Metal dependency)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

//===----------------------------------------------------------------------===//
// Tile constants and conversion functions
//===----------------------------------------------------------------------===//

constexpr int32_t TT_TILE_HEIGHT = 32;
constexpr int32_t TT_TILE_WIDTH = 32;
constexpr int32_t TT_TILE_SIZE = TT_TILE_HEIGHT * TT_TILE_WIDTH;

static void pack_to_tiles(const float* src, float* dst,
                          int32_t rows, int32_t cols) {
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;

  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          int32_t src_idx = (tr * TT_TILE_HEIGHT + r) * cols + (tc * TT_TILE_WIDTH + c);
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t dst_idx = tile_idx * TT_TILE_SIZE + r * TT_TILE_WIDTH + c;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

static void unpack_from_tiles(const float* src, float* dst,
                              int32_t rows, int32_t cols) {
  const int32_t num_tile_rows = rows / TT_TILE_HEIGHT;
  const int32_t num_tile_cols = cols / TT_TILE_WIDTH;

  for (int32_t tr = 0; tr < num_tile_rows; tr++) {
    for (int32_t tc = 0; tc < num_tile_cols; tc++) {
      for (int32_t r = 0; r < TT_TILE_HEIGHT; r++) {
        for (int32_t c = 0; c < TT_TILE_WIDTH; c++) {
          int32_t tile_idx = tr * num_tile_cols + tc;
          int32_t src_idx = tile_idx * TT_TILE_SIZE + r * TT_TILE_WIDTH + c;
          int32_t dst_idx = (tr * TT_TILE_HEIGHT + r) * cols + (tc * TT_TILE_WIDTH + c);
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Test utilities
//===----------------------------------------------------------------------===//

#define TEST_ASSERT(cond, msg)                                     \
  do {                                                             \
    if (!(cond)) {                                                 \
      fprintf(stderr, "FAILED: %s\n  %s:%d\n", msg, __FILE__, __LINE__); \
      return 1;                                                    \
    }                                                              \
  } while (0)

#define TEST_START(name) printf("  %s... ", name); fflush(stdout)
#define TEST_PASS() printf("PASSED\n")

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

int test_single_tile() {
  TEST_START("Single 32x32 tile");

  const int32_t rows = 32, cols = 32;
  const size_t n = rows * cols;

  float* src = (float*)malloc(n * sizeof(float));
  float* tiled = (float*)malloc(n * sizeof(float));
  float* dst = (float*)malloc(n * sizeof(float));

  for (size_t i = 0; i < n; i++) src[i] = (float)i;

  pack_to_tiles(src, tiled, rows, cols);
  unpack_from_tiles(tiled, dst, rows, cols);

  int errors = 0;
  for (size_t i = 0; i < n; i++) {
    if (src[i] != dst[i]) errors++;
  }

  free(src);
  free(tiled);
  free(dst);

  TEST_ASSERT(errors == 0, "round-trip mismatch");
  TEST_PASS();
  return 0;
}

int test_2x2_tiles() {
  TEST_START("2x2 tiles (64x64)");

  const int32_t rows = 64, cols = 64;
  const size_t n = rows * cols;

  float* src = (float*)malloc(n * sizeof(float));
  float* tiled = (float*)malloc(n * sizeof(float));
  float* dst = (float*)malloc(n * sizeof(float));

  for (size_t i = 0; i < n; i++) src[i] = (float)i;

  pack_to_tiles(src, tiled, rows, cols);
  unpack_from_tiles(tiled, dst, rows, cols);

  int errors = 0;
  for (size_t i = 0; i < n; i++) {
    if (src[i] != dst[i]) errors++;
  }

  free(src);
  free(tiled);
  free(dst);

  TEST_ASSERT(errors == 0, "round-trip mismatch");
  TEST_PASS();
  return 0;
}

int test_tile_ordering() {
  TEST_START("Tile ordering (first element of each tile)");

  const int32_t rows = 64, cols = 64;
  const size_t n = rows * cols;

  float* src = (float*)malloc(n * sizeof(float));
  float* tiled = (float*)malloc(n * sizeof(float));

  for (size_t i = 0; i < n; i++) src[i] = (float)i;

  pack_to_tiles(src, tiled, rows, cols);

  // Tile (0,0) starts at row 0, col 0 -> src[0]
  TEST_ASSERT(tiled[0] == 0.0f, "tile (0,0)[0,0] wrong");

  // Tile (0,1) starts at row 0, col 32 -> src[32]
  TEST_ASSERT(tiled[TT_TILE_SIZE] == 32.0f, "tile (0,1)[0,0] wrong");

  // Tile (1,0) starts at row 32, col 0 -> src[32*64]
  TEST_ASSERT(tiled[2 * TT_TILE_SIZE] == 32.0f * 64.0f, "tile (1,0)[0,0] wrong");

  // Tile (1,1) starts at row 32, col 32 -> src[32*64 + 32]
  TEST_ASSERT(tiled[3 * TT_TILE_SIZE] == 32.0f * 64.0f + 32.0f, "tile (1,1)[0,0] wrong");

  free(src);
  free(tiled);

  TEST_PASS();
  return 0;
}

int test_large_matrix() {
  TEST_START("Large matrix (128x256, 32 tiles)");

  const int32_t rows = 128, cols = 256;
  const size_t n = rows * cols;

  float* src = (float*)malloc(n * sizeof(float));
  float* tiled = (float*)malloc(n * sizeof(float));
  float* dst = (float*)malloc(n * sizeof(float));

  for (size_t i = 0; i < n; i++) src[i] = (float)(i % 1000) * 0.001f;

  pack_to_tiles(src, tiled, rows, cols);
  unpack_from_tiles(tiled, dst, rows, cols);

  int errors = 0;
  for (size_t i = 0; i < n; i++) {
    if (src[i] != dst[i]) errors++;
  }

  free(src);
  free(tiled);
  free(dst);

  TEST_ASSERT(errors == 0, "round-trip mismatch");
  TEST_PASS();
  return 0;
}

int test_intra_tile_layout() {
  TEST_START("Intra-tile element layout");

  const int32_t rows = 32, cols = 32;
  float* src = (float*)malloc(TT_TILE_SIZE * sizeof(float));
  float* tiled = (float*)malloc(TT_TILE_SIZE * sizeof(float));

  // Fill with row-major indices
  for (int32_t r = 0; r < rows; r++) {
    for (int32_t c = 0; c < cols; c++) {
      src[r * cols + c] = (float)(r * 100 + c);  // e.g., 0, 1, 2, ..., 100, 101, ...
    }
  }

  pack_to_tiles(src, tiled, rows, cols);

  // In tile layout, element at row r, col c should be at index r*32+c
  // Check a few specific positions
  TEST_ASSERT(tiled[0] == 0.0f, "element [0,0] wrong");
  TEST_ASSERT(tiled[1] == 1.0f, "element [0,1] wrong");
  TEST_ASSERT(tiled[32] == 100.0f, "element [1,0] wrong");
  TEST_ASSERT(tiled[33] == 101.0f, "element [1,1] wrong");

  free(src);
  free(tiled);

  TEST_PASS();
  return 0;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  printf("=== Tile Layout Tests ===\n\n");

  int failures = 0;
  failures += test_single_tile();
  failures += test_2x2_tiles();
  failures += test_tile_ordering();
  failures += test_large_matrix();
  failures += test_intra_tile_layout();

  printf("\n=== %d test(s) failed ===\n", failures);
  return failures > 0 ? 1 : 0;
}
