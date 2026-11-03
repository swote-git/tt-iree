// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// M=49, N=33, K=35: unaligned channel starts, interleaved 2KB page
// crossings, and all three masked tile tails in the native F32 path.
func.func @device_native_f32_matmul_page_tail(
    %input: tensor<35x49xf32>,
    %weight: tensor<33x35xf32>) -> tensor<33x49xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x49xf32>
  %initial = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<33x49xf32>) -> tensor<33x49xf32>
  %projection = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input, %weight : tensor<35x49xf32>, tensor<33x35xf32>)
      outs(%initial : tensor<33x49xf32>) {
    ^bb0(%activation: f32, %filter: f32, %acc: f32):
      %product = arith.mulf %activation, %filter : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<33x49xf32>
  return %projection : tensor<33x49xf32>
}
