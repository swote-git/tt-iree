// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shape-exact first-stage ResNet50 projection with the fused per-channel
// batch-normalization affine, ReLU, and one-element output border.
func.func @resnet50_per_channel_projection(
    %input: tensor<64x56x56xf32>,
    %weight: tensor<64x64xf32>) -> tensor<64x58x58xf32> {
  %zero = arith.constant 0.0 : f32
  %common_scale = arith.constant 0.5 : f32
  %mean = arith.constant dense<[
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0
    ]> : tensor<64xf32>
  %scale = arith.constant dense<[
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125,
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125,
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125,
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125,
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125,
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125,
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125,
      0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125
    ]> : tensor<64xf32>
  %bias = arith.constant dense<[
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125,
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125,
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125,
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125,
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125,
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125,
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125,
      0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25, 0.125
    ]> : tensor<64xf32>
  %empty = tensor.empty() : tensor<64x56x56xf32>
  %initial = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<64x56x56xf32>) -> tensor<64x56x56xf32>

  %projection = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%input, %weight : tensor<64x56x56xf32>, tensor<64x64xf32>)
      outs(%initial : tensor<64x56x56xf32>) {
    ^bb0(%activation: f32, %filter: f32, %acc: f32):
      %product = arith.mulf %activation, %filter : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<64x56x56xf32>

  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%projection, %mean, %scale, %bias
          : tensor<64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
      outs(%empty : tensor<64x56x56xf32>) {
    ^bb0(%activation: f32, %mean_value: f32, %scale_value: f32,
         %bias_value: f32, %out: f32):
      %centered = arith.subf %activation, %mean_value : f32
      %normalized = arith.mulf %centered, %common_scale : f32
      %scaled = arith.mulf %normalized, %scale_value : f32
      %shifted = arith.addf %scaled, %bias_value : f32
      %positive = arith.cmpf ugt, %shifted, %zero : f32
      %relu = arith.select %positive, %shifted, %zero : f32
      linalg.yield %relu : f32
  } -> tensor<64x56x56xf32>

  %padded = tensor.pad %result low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%c: index, %h: index, %w: index):
      tensor.yield %zero : f32
  } : tensor<64x56x56xf32> to tensor<64x58x58xf32>
  return %padded : tensor<64x58x58xf32>
}
