// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A ResNet-style channel-major projection with runtime batch-normalization
// vectors. TTEX v12 applies the per-channel affine and ReLU in the device
// writer without host staging.
func.func @device_native_f32_runtime_affine_matmul(
    %input: tensor<64x49xf32>,
    %weight: tensor<64x64xf32>,
    %mean: tensor<64xf32>,
    %scale: tensor<64xf32>,
    %bias: tensor<64xf32>) -> tensor<64x49xf32> {
  %zero = arith.constant 0.0 : f32
  %common_scale = arith.constant 0.5 : f32
  %empty = tensor.empty() : tensor<64x49xf32>
  %initial = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<64x49xf32>) -> tensor<64x49xf32>
  %projection = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input, %weight : tensor<64x49xf32>, tensor<64x64xf32>)
      outs(%initial : tensor<64x49xf32>) {
    ^bb0(%activation: f32, %filter: f32, %acc: f32):
      %product = arith.mulf %activation, %filter : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<64x49xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%projection, %mean, %scale, %bias
          : tensor<64x49xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
      outs(%empty : tensor<64x49xf32>) {
    ^bb0(%activation: f32, %mean_value: f32, %scale_value: f32,
         %bias_value: f32, %out: f32):
      %centered = arith.subf %activation, %mean_value : f32
      %normalized = arith.mulf %centered, %common_scale : f32
      %scaled = arith.mulf %normalized, %scale_value : f32
      %shifted = arith.addf %scaled, %bias_value : f32
      %positive = arith.cmpf ugt, %shifted, %zero : f32
      %relu = arith.select %positive, %shifted, %zero : f32
      linalg.yield %relu : f32
  } -> tensor<64x49xf32>
  return %result : tensor<64x49xf32>
}
