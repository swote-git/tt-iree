// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The exact tile-aligned, no-epilogue channel-major projection used by
// ResNet50 async_dispatch_5. The compiler selects the direct
// F32-storage/BF16-compute/F32-storage device-native path.
func.func @device_native_f32_matmul(
    %input: tensor<64x3136xf32>,
    %weight: tensor<256x64xf32>) -> tensor<256x3136xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<256x3136xf32>
  %initial = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<256x3136xf32>) -> tensor<256x3136xf32>
  %projection = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input, %weight : tensor<64x3136xf32>, tensor<256x64xf32>)
      outs(%initial : tensor<256x3136xf32>) {
    ^bb0(%activation: f32, %filter: f32, %acc: f32):
      %product = arith.mulf %activation, %filter : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<256x3136xf32>
  return %projection : tensor<256x3136xf32>
}
