// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The first dispatch in the device-agnostic optimized ResNet50 dump.
// It pads the CHW input by three pixels on each spatial edge before conv1.
func.func @resnet50_input_pad(
    %input: tensor<3x224x224xf32>) -> tensor<3x230x230xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %input low[0, 3, 3] high[0, 3, 3] {
    ^bb0(%c: index, %h: index, %w: index):
      tensor.yield %zero : f32
  } : tensor<3x224x224xf32> to tensor<3x230x230xf32>
  return %padded : tensor<3x230x230xf32>
}
