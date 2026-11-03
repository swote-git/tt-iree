// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// ResNet projection-shortcut epilogue: two batch norms (BF16 mean/gamma/beta,
// FP32 inverse std), a BF16 residual add and ReLU. Nine operands need two BF16
// operands per SFPU Dst tile; the result must match the RISC-V reference bits.
func.func @bf16_shortcut(%input: tensor<35x49xbf16>, %weight: tensor<33x35xbf16>,
                         %mean0: tensor<33xbf16>, %inv0: tensor<33xf32>,
                         %gamma0: tensor<33xbf16>, %beta0: tensor<33xbf16>,
                         %other: tensor<33x49xbf16>, %mean1: tensor<33xbf16>,
                         %inv1: tensor<33xf32>, %gamma1: tensor<33xbf16>,
                         %beta1: tensor<33xbf16>) -> tensor<33x49xbf16> {
  %zero = arith.constant 0.0 : f32
  %zero_bf16 = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<33x49xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x49xf32>) -> tensor<33x49xf32>
  %dot = linalg.generic {
      indexing_maps = [affine_map<(n,m,k)->(k,m)>, affine_map<(n,m,k)->(n,k)>, affine_map<(n,m,k)->(n,m)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input, %weight : tensor<35x49xbf16>, tensor<33x35xbf16>)
      outs(%init : tensor<33x49xf32>) {
    ^bb0(%a: bf16, %b: bf16, %c: f32):
      %af = arith.extf %a : bf16 to f32
      %bf = arith.extf %b : bf16 to f32
      %product = arith.mulf %af, %bf : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<33x49xf32>
  %out = tensor.empty() : tensor<33x49xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(n,m)->(n,m)>, affine_map<(n,m)->(n)>,
                       affine_map<(n,m)->(n)>, affine_map<(n,m)->(n)>,
                       affine_map<(n,m)->(n)>, affine_map<(n,m)->(n,m)>,
                       affine_map<(n,m)->(n)>, affine_map<(n,m)->(n)>,
                       affine_map<(n,m)->(n)>, affine_map<(n,m)->(n)>,
                       affine_map<(n,m)->(n,m)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%dot, %mean0, %inv0, %gamma0, %beta0, %other, %mean1, %inv1, %gamma1, %beta1
          : tensor<33x49xf32>, tensor<33xbf16>, tensor<33xf32>, tensor<33xbf16>,
            tensor<33xbf16>, tensor<33x49xbf16>, tensor<33xbf16>, tensor<33xf32>,
            tensor<33xbf16>, tensor<33xbf16>)
      outs(%out : tensor<33x49xbf16>) {
    ^bb0(%v: f32, %m0: bf16, %i0: f32, %g0: bf16, %b0: bf16, %o: bf16, %m1: bf16,
         %i1: f32, %g1: bf16, %b1: bf16, %unused: bf16):
      %t = arith.truncf %v : f32 to bf16
      %m0f = arith.extf %m0 : bf16 to f32
      %tf = arith.extf %t : bf16 to f32
      %d0 = arith.subf %tf, %m0f : f32
      %n0 = arith.mulf %d0, %i0 : f32
      %g0f = arith.extf %g0 : bf16 to f32
      %s0 = arith.mulf %n0, %g0f : f32
      %b0f = arith.extf %b0 : bf16 to f32
      %y0f = arith.addf %s0, %b0f : f32
      %y0 = arith.truncf %y0f : f32 to bf16
      %m1f = arith.extf %m1 : bf16 to f32
      %of = arith.extf %o : bf16 to f32
      %d1 = arith.subf %of, %m1f : f32
      %n1 = arith.mulf %d1, %i1 : f32
      %g1f = arith.extf %g1 : bf16 to f32
      %s1 = arith.mulf %n1, %g1f : f32
      %b1f = arith.extf %b1 : bf16 to f32
      %y1f = arith.addf %s1, %b1f : f32
      %y1 = arith.truncf %y1f : f32 to bf16
      %sum = arith.addf %y0, %y1 : bf16
      %positive = arith.cmpf ugt, %sum, %zero_bf16 : bf16
      %relu = arith.select %positive, %sum, %zero_bf16 : bf16
      linalg.yield %relu : bf16
  } -> tensor<33x49xbf16>
  return %result : tensor<33x49xbf16>
}
