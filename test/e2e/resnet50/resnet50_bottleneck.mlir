// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shape-reduced ResNet50 stage-2 projection bottleneck. Channel dimensions and
// operation topology match the model while 56x56 is reduced to 8x8 for a
// practical correctness-first device smoke test.
func.func @resnet50_bottleneck(
    %input: tensor<64x8x8xf32>,
    %conv1_weight: tensor<64x64xf32>,
    %conv2_weight: tensor<64x64x3x3xf32>,
    %conv3_weight: tensor<256x64xf32>,
    %shortcut_weight: tensor<256x64xf32>) -> tensor<256x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %mean_64 = arith.constant dense<1.250000e-01> : tensor<64xf32>
  %scale_64 = arith.constant dense<5.000000e-01> : tensor<64xf32>
  %bias_64 = arith.constant dense<2.500000e-01> : tensor<64xf32>
  %mean_256 = arith.constant dense<1.250000e-01> : tensor<256xf32>
  %scale_256 = arith.constant dense<5.000000e-01> : tensor<256xf32>
  %bias_256 = arith.constant dense<2.500000e-01> : tensor<256xf32>

  %empty_64 = tensor.empty() : tensor<64x8x8xf32>
  %zero_64 = linalg.fill ins(%zero : f32)
      outs(%empty_64 : tensor<64x8x8xf32>) -> tensor<64x8x8xf32>

  // Main branch: 1x1 convolution, batch normalization, and ReLU.
  %conv1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%input, %conv1_weight : tensor<64x8x8xf32>, tensor<64x64xf32>)
      outs(%zero_64 : tensor<64x8x8xf32>) {
    ^bb0(%activation: f32, %weight: f32, %acc: f32):
      %product = arith.mulf %activation, %weight : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<64x8x8xf32>
  %conv1_relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%conv1, %mean_64, %scale_64, %bias_64
          : tensor<64x8x8xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
      outs(%empty_64 : tensor<64x8x8xf32>) {
    ^bb0(%activation: f32, %mean: f32, %scale: f32, %bias: f32, %out: f32):
      %centered = arith.subf %activation, %mean : f32
      %scaled = arith.mulf %centered, %scale : f32
      %shifted = arith.addf %scaled, %bias : f32
      %positive = arith.cmpf ugt, %shifted, %zero : f32
      %relu = arith.select %positive, %shifted, %zero : f32
      linalg.yield %relu : f32
  } -> tensor<64x8x8xf32>

  // Main branch: padded 3x3 convolution, batch normalization, and ReLU.
  %padded = tensor.pad %conv1_relu low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%c: index, %h: index, %w: index):
      tensor.yield %zero : f32
  } : tensor<64x8x8xf32> to tensor<64x10x10xf32>
  %conv2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d2 + d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]}
      ins(%padded, %conv2_weight
          : tensor<64x10x10xf32>, tensor<64x64x3x3xf32>)
      outs(%zero_64 : tensor<64x8x8xf32>) {
    ^bb0(%activation: f32, %weight: f32, %acc: f32):
      %product = arith.mulf %activation, %weight : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<64x8x8xf32>
  %conv2_relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%conv2, %mean_64, %scale_64, %bias_64
          : tensor<64x8x8xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
      outs(%empty_64 : tensor<64x8x8xf32>) {
    ^bb0(%activation: f32, %mean: f32, %scale: f32, %bias: f32, %out: f32):
      %centered = arith.subf %activation, %mean : f32
      %scaled = arith.mulf %centered, %scale : f32
      %shifted = arith.addf %scaled, %bias : f32
      %positive = arith.cmpf ugt, %shifted, %zero : f32
      %relu = arith.select %positive, %shifted, %zero : f32
      linalg.yield %relu : f32
  } -> tensor<64x8x8xf32>

  %empty_256 = tensor.empty() : tensor<256x8x8xf32>
  %zero_256 = linalg.fill ins(%zero : f32)
      outs(%empty_256 : tensor<256x8x8xf32>) -> tensor<256x8x8xf32>

  // Main branch expansion to 256 channels and batch normalization.
  %conv3 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%conv2_relu, %conv3_weight
          : tensor<64x8x8xf32>, tensor<256x64xf32>)
      outs(%zero_256 : tensor<256x8x8xf32>) {
    ^bb0(%activation: f32, %weight: f32, %acc: f32):
      %product = arith.mulf %activation, %weight : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<256x8x8xf32>
  %main = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%conv3, %mean_256, %scale_256, %bias_256
          : tensor<256x8x8xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>)
      outs(%empty_256 : tensor<256x8x8xf32>) {
    ^bb0(%activation: f32, %mean: f32, %scale: f32, %bias: f32, %out: f32):
      %centered = arith.subf %activation, %mean : f32
      %scaled = arith.mulf %centered, %scale : f32
      %shifted = arith.addf %scaled, %bias : f32
      linalg.yield %shifted : f32
  } -> tensor<256x8x8xf32>

  // Projection shortcut from 64 to 256 channels and batch normalization.
  %shortcut_conv = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%input, %shortcut_weight
          : tensor<64x8x8xf32>, tensor<256x64xf32>)
      outs(%zero_256 : tensor<256x8x8xf32>) {
    ^bb0(%activation: f32, %weight: f32, %acc: f32):
      %product = arith.mulf %activation, %weight : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<256x8x8xf32>
  %shortcut = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%shortcut_conv, %mean_256, %scale_256, %bias_256
          : tensor<256x8x8xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>)
      outs(%empty_256 : tensor<256x8x8xf32>) {
    ^bb0(%activation: f32, %mean: f32, %scale: f32, %bias: f32, %out: f32):
      %centered = arith.subf %activation, %mean : f32
      %scaled = arith.mulf %centered, %scale : f32
      %shifted = arith.addf %scaled, %bias : f32
      linalg.yield %shifted : f32
  } -> tensor<256x8x8xf32>

  // Residual add and final ReLU.
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%main, %shortcut : tensor<256x8x8xf32>, tensor<256x8x8xf32>)
      outs(%empty_256 : tensor<256x8x8xf32>) {
    ^bb0(%main_value: f32, %shortcut_value: f32, %out: f32):
      %sum = arith.addf %main_value, %shortcut_value : f32
      %positive = arith.cmpf ugt, %sum, %zero : f32
      %relu = arith.select %positive, %sum, %zero : f32
      linalg.yield %relu : f32
  } -> tensor<256x8x8xf32>

  return %result : tensor<256x8x8xf32>
}
