// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func.func @spatial_pool(%input: tensor<5x7x39xbf16>) -> tensor<5x3x19xbf16> {
  %inf = arith.constant 0xFF80 : bf16
  %empty = tensor.empty() : tensor<5x3x19xbf16>
  %window = tensor.empty() : tensor<3x3xbf16>
  %init = linalg.fill ins(%inf : bf16) outs(%empty : tensor<5x3x19xbf16>) -> tensor<5x3x19xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(c,h,w,kh,kw)->(c,h*2+kh,w*2+kw)>,
                       affine_map<(c,h,w,kh,kw)->(kh,kw)>,
                       affine_map<(c,h,w,kh,kw)->(c,h,w)>],
      iterator_types = ["parallel","parallel","parallel","reduction","reduction"]}
      ins(%input, %window : tensor<5x7x39xbf16>, tensor<3x3xbf16>)
      outs(%init : tensor<5x3x19xbf16>) {
    ^bb0(%v: bf16, %unused: bf16, %acc: bf16):
      %max = arith.maximumf %acc, %v : bf16
      linalg.yield %max : bf16
  } -> tensor<5x3x19xbf16>
  return %result : tensor<5x3x19xbf16>
}

func.func @spatial_copy(%input: tensor<5x7x39xbf16>) -> tensor<5x9x43xbf16> {
  %zero = arith.constant 0.0 : bf16
  %result = tensor.pad %input low[0,1,2] high[0,1,2] {
    ^bb0(%c: index, %h: index, %w: index):
      tensor.yield %zero : bf16
  } : tensor<5x7x39xbf16> to tensor<5x9x43xbf16>
  return %result : tensor<5x9x43xbf16>
}
