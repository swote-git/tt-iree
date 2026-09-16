// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shape-exact ResNet50 stem max-pool. The stem convolution stores its result
// with a one-element halo, so this dispatch itself uses valid pooling.
func.func @resnet50_max_pool(
    %input: tensor<64x114x114xf32>) -> tensor<64x56x56xf32> {
  %negative_infinity = arith.constant 0xFF800000 : f32
  %output_empty = tensor.empty() : tensor<64x56x56xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %initial = linalg.fill ins(%negative_infinity : f32)
      outs(%output_empty : tensor<64x56x56xf32>) -> tensor<64x56x56xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) ->
                   (d0, d1 * 2 + d3, d2 * 2 + d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction",
                        "reduction"]}
      ins(%input, %window : tensor<64x114x114xf32>, tensor<3x3xf32>)
      outs(%initial : tensor<64x56x56xf32>) {
    ^bb0(%activation: f32, %unused_window: f32, %accumulator: f32):
      %maximum = arith.maximumf %accumulator, %activation : f32
      linalg.yield %maximum : f32
  } -> tensor<64x56x56xf32>
  return %result : tensor<64x56x56xf32>
}
