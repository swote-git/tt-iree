// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Real BF16 storage, FP32 contraction, then explicit BF16 rounding.
func.func @bf16_projection(%input: tensor<35x49xbf16>,
                           %weight: tensor<33x35xbf16>) -> tensor<33x49xbf16> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x49xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x49xf32>) -> tensor<33x49xf32>
  %dot = linalg.generic {
      indexing_maps = [affine_map<(n,m,k)->(k,m)>, affine_map<(n,m,k)->(n,k)>, affine_map<(n,m,k)->(n,m)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input, %weight : tensor<35x49xbf16>, tensor<33x35xbf16>)
      outs(%init : tensor<33x49xf32>) {
    ^bb0(%a: bf16, %b: bf16, %c: f32):
      %af = arith.extf %a : bf16 to f32
      %bf = arith.extf %b : bf16 to f32
      %product = arith.mulf %af, %bf : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<33x49xf32>
  %out = tensor.empty() : tensor<33x49xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(n,m)->(n,m)>, affine_map<(n,m)->(n,m)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%dot : tensor<33x49xf32>) outs(%out : tensor<33x49xbf16>) {
    ^bb0(%v: f32, %unused: bf16):
      %rounded = arith.truncf %v : f32 to bf16
      linalg.yield %rounded : bf16
  } -> tensor<33x49xbf16>
  return %result : tensor<33x49xbf16>
}
