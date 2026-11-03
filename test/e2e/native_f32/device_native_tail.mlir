// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Same 7x7 -> sum/divisor -> divide structure as the ResNet50 classifier head.
func.func @device_native_average(%input: tensor<4x7x7xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %divisor = arith.constant 49.0 : f32
  %c7 = arith.constant 7 : index
  %dummy = tensor.empty() : tensor<7x7xi1>
  %padded = tensor.pad %input low[0, 0, 0] high[0, 1, 1] {
    ^bb0(%c: index, %h: index, %w: index):
      tensor.yield %zero : f32
  } : tensor<4x7x7xf32> to tensor<4x8x8xf32>
  %empty = tensor.empty() : tensor<4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %scalar = tensor.empty() : tensor<f32>
  %sum:2 = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0)>,
                       affine_map<(d0,d1,d2)->()>],
      iterator_types = ["parallel","reduction","reduction"]}
      ins(%dummy : tensor<7x7xi1>) outs(%init, %scalar : tensor<4xf32>, tensor<f32>) {
    ^bb0(%unused: i1, %acc: f32, %unused_scalar: f32):
      %c = linalg.index 0 : index
      %h = linalg.index 1 : index
      %w = linalg.index 2 : index
      %x = tensor.extract %padded[%c, %h, %w] : tensor<4x8x8xf32>
      %valid_h = arith.cmpi ult, %h, %c7 : index
      %v_h = arith.select %valid_h, %x, %zero : f32
      %valid_w = arith.cmpi ult, %w, %c7 : index
      %v = arith.select %valid_w, %v_h, %zero : f32
      %add = arith.addf %acc, %v : f32
      linalg.yield %add, %divisor : f32, f32
  } -> (tensor<4xf32>, tensor<f32>)
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0)->()>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]}
      ins(%sum#1, %sum#0 : tensor<f32>, tensor<4xf32>) outs(%empty : tensor<4xf32>) {
    ^bb0(%denominator: f32, %numerator: f32, %unused: f32):
      %v = arith.divf %numerator, %denominator : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// M, N and K all exercise masked tile tails, with a separate bias view.
func.func @device_native_linear(%input: tensor<1x33xf32>,
                              %weights: tensor<17x33xf32>,
                              %bias: tensor<1x17xf32>) -> tensor<1x17xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x17xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x17xf32>) -> tensor<1x17xf32>
  %mm = linalg.matmul indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>,
      affine_map<(d0,d1,d2)->(d1,d2)>,affine_map<(d0,d1,d2)->(d0,d1)>]
      ins(%input, %weights : tensor<1x33xf32>, tensor<17x33xf32>)
      outs(%init : tensor<1x17xf32>) -> tensor<1x17xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]}
      ins(%mm, %bias : tensor<1x17xf32>, tensor<1x17xf32>) outs(%empty : tensor<1x17xf32>) {
    ^bb0(%v: f32, %b: f32, %unused: f32):
      %add = arith.addf %v, %b : f32
      linalg.yield %add : f32
  } -> tensor<1x17xf32>
  return %result : tensor<1x17xf32>
}
