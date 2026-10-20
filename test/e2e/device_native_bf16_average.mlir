// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Same 7x7 -> sum/divisor -> divide structure as the ResNet50 classifier head.
func.func @bf16_average(%input: tensor<4x7x7xbf16>) -> tensor<4xbf16> {
  %zero = arith.constant 0.0 : bf16
  %divisor = arith.constant 49.0 : bf16
  %c7 = arith.constant 7 : index
  %dummy = tensor.empty() : tensor<7x7xi1>
  %padded = tensor.pad %input low[0, 0, 0] high[0, 1, 1] {
    ^bb0(%c: index, %h: index, %w: index):
      tensor.yield %zero : bf16
  } : tensor<4x7x7xbf16> to tensor<4x8x8xbf16>
  %empty = tensor.empty() : tensor<4xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<4xbf16>) -> tensor<4xbf16>
  %scalar = tensor.empty() : tensor<bf16>
  %sum:2 = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0)>,
                       affine_map<(d0,d1,d2)->()>],
      iterator_types = ["parallel","reduction","reduction"]}
      ins(%dummy : tensor<7x7xi1>) outs(%init, %scalar : tensor<4xbf16>, tensor<bf16>) {
    ^bb0(%unused: i1, %acc: bf16, %unused_scalar: bf16):
      %c = linalg.index 0 : index
      %h = linalg.index 1 : index
      %w = linalg.index 2 : index
      %x = tensor.extract %padded[%c, %h, %w] : tensor<4x8x8xbf16>
      %valid_h = arith.cmpi ult, %h, %c7 : index
      %v_h = arith.select %valid_h, %x, %zero : bf16
      %valid_w = arith.cmpi ult, %w, %c7 : index
      %v = arith.select %valid_w, %v_h, %zero : bf16
      %add = arith.addf %acc, %v : bf16
      linalg.yield %add, %divisor : bf16, bf16
  } -> (tensor<4xbf16>, tensor<bf16>)
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0)->()>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]}
      ins(%sum#1, %sum#0 : tensor<bf16>, tensor<4xbf16>) outs(%empty : tensor<4xbf16>) {
    ^bb0(%denominator: bf16, %numerator: bf16, %unused: bf16):
      %v = arith.divf %numerator, %denominator : bf16
      linalg.yield %v : bf16
  } -> tensor<4xbf16>
  return %result : tensor<4xbf16>
}

