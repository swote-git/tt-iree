// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shape-exact ResNet50 stem convolution. Input padding is a separate dispatch;
// this operation consumes the already-padded CHW tensor and applies the fused
// per-channel batch-normalization affine plus ReLU.
func.func @resnet50_stem_conv(
    %input: tensor<3x230x230xf32>,
    %weight: tensor<64x3x7x7xf32>) -> tensor<64x112x112xf32> {
  %zero = arith.constant 0.0 : f32
  %epsilon_scale = arith.constant 0.999994993 : f32
  %mean = arith.constant dense<[
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    ]> : tensor<64xf32>
  %scale = arith.constant dense<[
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25,
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25,
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25,
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25,
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25,
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25,
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25,
      0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25
    ]> : tensor<64xf32>
  %bias = arith.constant dense<[
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125,
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125,
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125,
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125,
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125,
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125,
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125,
      0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125
    ]> : tensor<64xf32>

  %empty = tensor.empty() : tensor<64x112x112xf32>
  %initial = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<64x112x112xf32>) -> tensor<64x112x112xf32>
  %convolution = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) ->
                   (d3, d1 * 2 + d4, d2 * 2 + d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction",
                        "reduction", "reduction"]}
      ins(%input, %weight
          : tensor<3x230x230xf32>, tensor<64x3x7x7xf32>)
      outs(%initial : tensor<64x112x112xf32>) {
    ^bb0(%activation: f32, %filter: f32, %acc: f32):
      %product = arith.mulf %activation, %filter : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<64x112x112xf32>

  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%convolution, %mean, %scale, %bias
          : tensor<64x112x112xf32>, tensor<64xf32>, tensor<64xf32>,
            tensor<64xf32>)
      outs(%empty : tensor<64x112x112xf32>) {
    ^bb0(%activation: f32, %mean_value: f32, %scale_value: f32,
         %bias_value: f32, %out: f32):
      %centered = arith.subf %activation, %mean_value : f32
      %epsilon_scaled = arith.mulf %centered, %epsilon_scale : f32
      %scaled = arith.mulf %epsilon_scaled, %scale_value : f32
      %shifted = arith.addf %scaled, %bias_value : f32
      %positive = arith.cmpf ugt, %shifted, %zero : f32
      %relu = arith.select %positive, %shifted, %zero : f32
      linalg.yield %relu : f32
  } -> tensor<64x112x112xf32>
  return %result : tensor<64x112x112xf32>
}
