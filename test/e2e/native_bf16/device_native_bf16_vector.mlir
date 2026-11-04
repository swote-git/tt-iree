// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Multi-tile, multi-core shapes for the v22 SFPU vector programs: 2500
// channels give 40 GLOBAL_SUM tiles (64 channels, a 4-channel tail) and three
// SCALAR_DIV tiles (1024 elements, a 452-element tail); the pool output has
// 84 rows of 57 (17 rows per tile, a 16-row tail).
func.func @bf16_average_wide(%input: tensor<2500x7x7xbf16>) -> tensor<2500xbf16> {
  %zero = arith.constant 0.0 : bf16
  %divisor = arith.constant 49.0 : bf16
  %c7 = arith.constant 7 : index
  %dummy = tensor.empty() : tensor<7x7xi1>
  %padded = tensor.pad %input low[0, 0, 0] high[0, 1, 1] {
    ^bb0(%c: index, %h: index, %w: index):
      tensor.yield %zero : bf16
  } : tensor<2500x7x7xbf16> to tensor<2500x8x8xbf16>
  %empty = tensor.empty() : tensor<2500xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<2500xbf16>) -> tensor<2500xbf16>
  %scalar = tensor.empty() : tensor<bf16>
  %sum:2 = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0)>,
                       affine_map<(d0,d1,d2)->()>],
      iterator_types = ["parallel","reduction","reduction"]}
      ins(%dummy : tensor<7x7xi1>) outs(%init, %scalar : tensor<2500xbf16>, tensor<bf16>) {
    ^bb0(%unused: i1, %acc: bf16, %unused_scalar: bf16):
      %c = linalg.index 0 : index
      %h = linalg.index 1 : index
      %w = linalg.index 2 : index
      %x = tensor.extract %padded[%c, %h, %w] : tensor<2500x8x8xbf16>
      %valid_h = arith.cmpi ult, %h, %c7 : index
      %v_h = arith.select %valid_h, %x, %zero : bf16
      %valid_w = arith.cmpi ult, %w, %c7 : index
      %v = arith.select %valid_w, %v_h, %zero : bf16
      %add = arith.addf %acc, %v : bf16
      linalg.yield %add, %divisor : bf16, bf16
  } -> (tensor<2500xbf16>, tensor<bf16>)
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0)->()>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]}
      ins(%sum#1, %sum#0 : tensor<bf16>, tensor<2500xbf16>) outs(%empty : tensor<2500xbf16>) {
    ^bb0(%denominator: bf16, %numerator: bf16, %unused: bf16):
      %v = arith.divf %numerator, %denominator : bf16
      linalg.yield %v : bf16
  } -> tensor<2500xbf16>
  return %result : tensor<2500xbf16>
}

func.func @bf16_pool_wide(%input: tensor<6x29x115xbf16>) -> tensor<6x14x57xbf16> {
  %inf = arith.constant 0xFF80 : bf16
  %empty = tensor.empty() : tensor<6x14x57xbf16>
  %window = tensor.empty() : tensor<3x3xbf16>
  %init = linalg.fill ins(%inf : bf16) outs(%empty : tensor<6x14x57xbf16>) -> tensor<6x14x57xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(c,h,w,kh,kw)->(c,h*2+kh,w*2+kw)>,
                       affine_map<(c,h,w,kh,kw)->(kh,kw)>,
                       affine_map<(c,h,w,kh,kw)->(c,h,w)>],
      iterator_types = ["parallel","parallel","parallel","reduction","reduction"]}
      ins(%input, %window : tensor<6x29x115xbf16>, tensor<3x3xbf16>)
      outs(%init : tensor<6x14x57xbf16>) {
    ^bb0(%v: bf16, %unused: bf16, %acc: bf16):
      %max = arith.maximumf %acc, %v : bf16
      linalg.yield %max : bf16
  } -> tensor<6x14x57xbf16>
  return %result : tensor<6x14x57xbf16>
}
