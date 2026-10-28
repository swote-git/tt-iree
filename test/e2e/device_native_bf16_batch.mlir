// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// High-level static B2 tensors, not a host loop over two B1 invocations.
// Compile through global optimization. Keep FC named with
// --iree-opt-generalize-matmul=false (this does not disable global optimization).

// NCHW: M=2*5*9=90, N=33, K=3*3*3=27. Tile 1 crosses batch at M=45.
// The returned tensor.pad also checks the zero halo in physical output storage.
func.func @batch_conv_s1(%input: tensor<2x3x7x11xbf16>,
                         %weight: tensor<33x3x3x3xbf16>) -> tensor<2x33x7x11xbf16> {
  %zero = arith.constant 0.0 : f32
  %zero_bf16 = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<2x33x5x9xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x33x5x9xf32>) -> tensor<2x33x5x9xf32>
  %dot = linalg.generic {
      indexing_maps = [affine_map<(b,n,h,w,k,kh,kw)->(b,k,h+kh,w+kw)>,
                       affine_map<(b,n,h,w,k,kh,kw)->(n,k,kh,kw)>,
                       affine_map<(b,n,h,w,k,kh,kw)->(b,n,h,w)>],
      iterator_types = ["parallel","parallel","parallel","parallel",
                        "reduction","reduction","reduction"]}
      ins(%input, %weight : tensor<2x3x7x11xbf16>, tensor<33x3x3x3xbf16>)
      outs(%init : tensor<2x33x5x9xf32>) {
    ^bb0(%a: bf16, %b: bf16, %acc: f32):
      %af = arith.extf %a : bf16 to f32
      %bf = arith.extf %b : bf16 to f32
      %product = arith.mulf %af, %bf : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<2x33x5x9xf32>
  %out = tensor.empty() : tensor<2x33x5x9xbf16>
  %rounded = linalg.generic {
      indexing_maps = [affine_map<(b,n,h,w)->(b,n,h,w)>,
                       affine_map<(b,n,h,w)->(b,n,h,w)>],
      iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%dot : tensor<2x33x5x9xf32>) outs(%out : tensor<2x33x5x9xbf16>) {
    ^bb0(%v: f32, %unused: bf16):
      %r = arith.truncf %v : f32 to bf16
      linalg.yield %r : bf16
  } -> tensor<2x33x5x9xbf16>
  %result = tensor.pad %rounded low[0,0,1,1] high[0,0,1,1] {
    ^bb0(%b: index, %n: index, %h: index, %w: index):
      tensor.yield %zero_bf16 : bf16
  } : tensor<2x33x5x9xbf16> to tensor<2x33x7x11xbf16>
  return %result : tensor<2x33x7x11xbf16>
}

// NCHW: M=2*4*7=56, N=35, K=5*3*3=45. Batch boundary is inside tile 0.
func.func @batch_conv_s2(%input: tensor<2x5x9x15xbf16>,
                         %weight: tensor<35x5x3x3xbf16>) -> tensor<2x35x4x7xbf16> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x35x4x7xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x35x4x7xf32>) -> tensor<2x35x4x7xf32>
  %dot = linalg.generic {
      indexing_maps = [affine_map<(b,n,h,w,k,kh,kw)->(b,k,h*2+kh,w*2+kw)>,
                       affine_map<(b,n,h,w,k,kh,kw)->(n,k,kh,kw)>,
                       affine_map<(b,n,h,w,k,kh,kw)->(b,n,h,w)>],
      iterator_types = ["parallel","parallel","parallel","parallel",
                        "reduction","reduction","reduction"]}
      ins(%input, %weight : tensor<2x5x9x15xbf16>, tensor<35x5x3x3xbf16>)
      outs(%init : tensor<2x35x4x7xf32>) {
    ^bb0(%a: bf16, %b: bf16, %acc: f32):
      %af = arith.extf %a : bf16 to f32
      %bf = arith.extf %b : bf16 to f32
      %product = arith.mulf %af, %bf : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<2x35x4x7xf32>
  %out = tensor.empty() : tensor<2x35x4x7xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(b,n,h,w)->(b,n,h,w)>,
                       affine_map<(b,n,h,w)->(b,n,h,w)>],
      iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%dot : tensor<2x35x4x7xf32>) outs(%out : tensor<2x35x4x7xbf16>) {
    ^bb0(%v: f32, %unused: bf16):
      %r = arith.truncf %v : f32 to bf16
      linalg.yield %r : bf16
  } -> tensor<2x35x4x7xbf16>
  return %result : tensor<2x35x4x7xbf16>
}

// BATCH_CHANNEL_MAJOR: M=98, N=33, K=35, batch boundary at M=49.
// Each BF16 operation is a separate RNE boundary, including the contraction
// result before BN. Channel parameters and residual are runtime tensor inputs.
func.func @batch_projection(%input: tensor<2x35x49xbf16>,
                            %weight: tensor<33x35xbf16>,
                            %mean: tensor<33xbf16>, %scale: tensor<33xbf16>,
                            %bias: tensor<33xbf16>,
                            %residual: tensor<2x33x49xbf16>) -> tensor<2x33x49xbf16> {
  %zero = arith.constant 0.0 : f32
  %zero_bf16 = arith.constant 0.0 : bf16
  %half = arith.constant 0.5 : bf16
  %empty = tensor.empty() : tensor<2x33x49xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x33x49xf32>) -> tensor<2x33x49xf32>
  %dot = linalg.generic {
      indexing_maps = [affine_map<(b,n,m,k)->(b,k,m)>,
                       affine_map<(b,n,m,k)->(n,k)>,
                       affine_map<(b,n,m,k)->(b,n,m)>],
      iterator_types = ["parallel","parallel","parallel","reduction"]}
      ins(%input, %weight : tensor<2x35x49xbf16>, tensor<33x35xbf16>)
      outs(%init : tensor<2x33x49xf32>) {
    ^bb0(%a: bf16, %b: bf16, %acc: f32):
      %af = arith.extf %a : bf16 to f32
      %bf = arith.extf %b : bf16 to f32
      %product = arith.mulf %af, %bf : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<2x33x49xf32>
  %out = tensor.empty() : tensor<2x33x49xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(b,n,m)->(b,n,m)>,
                       affine_map<(b,n,m)->(n)>, affine_map<(b,n,m)->(n)>,
                       affine_map<(b,n,m)->(n)>, affine_map<(b,n,m)->(b,n,m)>,
                       affine_map<(b,n,m)->(b,n,m)>],
      iterator_types = ["parallel","parallel","parallel"]}
      ins(%dot, %mean, %scale, %bias, %residual
          : tensor<2x33x49xf32>, tensor<33xbf16>, tensor<33xbf16>,
            tensor<33xbf16>, tensor<2x33x49xbf16>)
      outs(%out : tensor<2x33x49xbf16>) {
    ^bb0(%v: f32, %mu: bf16, %s: bf16, %bias_value: bf16,
         %skip: bf16, %unused: bf16):
      %r = arith.truncf %v : f32 to bf16
      %centered = arith.subf %r, %mu : bf16
      %normalized = arith.mulf %centered, %half : bf16
      %scaled = arith.mulf %normalized, %s : bf16
      %shifted = arith.addf %scaled, %bias_value : bf16
      %added = arith.addf %shifted, %skip : bf16
      %positive = arith.cmpf ugt, %added, %zero_bf16 : bf16
      %relu = arith.select %positive, %added, %zero_bf16 : bf16
      linalg.yield %relu : bf16
  } -> tensor<2x33x49xbf16>
  return %result : tensor<2x33x49xbf16>
}

// Nonuniform, asymmetric padding catches both storage offsets and batch stride.
func.func @batch_input_pad(%input: tensor<2x5x7x39xbf16>) -> tensor<2x5x10x42xbf16> {
  %zero = arith.constant 0.0 : bf16
  %result = tensor.pad %input low[0,0,1,2] high[0,0,2,1] {
    ^bb0(%b: index, %c: index, %h: index, %w: index):
      tensor.yield %zero : bf16
  } : tensor<2x5x7x39xbf16> to tensor<2x5x10x42xbf16>
  return %result : tensor<2x5x10x42xbf16>
}

func.func @batch_maxpool(%input: tensor<2x5x7x39xbf16>) -> tensor<2x5x3x19xbf16> {
  %inf = arith.constant 0xFF80 : bf16
  %empty = tensor.empty() : tensor<2x5x3x19xbf16>
  %window = tensor.empty() : tensor<3x3xbf16>
  %init = linalg.fill ins(%inf : bf16) outs(%empty : tensor<2x5x3x19xbf16>) -> tensor<2x5x3x19xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(b,c,h,w,kh,kw)->(b,c,h*2+kh,w*2+kw)>,
                       affine_map<(b,c,h,w,kh,kw)->(kh,kw)>,
                       affine_map<(b,c,h,w,kh,kw)->(b,c,h,w)>],
      iterator_types = ["parallel","parallel","parallel","parallel","reduction","reduction"]}
      ins(%input, %window : tensor<2x5x7x39xbf16>, tensor<3x3xbf16>)
      outs(%init : tensor<2x5x3x19xbf16>) {
    ^bb0(%v: bf16, %unused: bf16, %acc: bf16):
      %max = arith.maximumf %acc, %v : bf16
      linalg.yield %max : bf16
  } -> tensor<2x5x3x19xbf16>
  return %result : tensor<2x5x3x19xbf16>
}

// Match B16 ResNet's GLOBAL_SUM exactly: flatten batch*channel into a 1D
// output, extract NCHW via divsi/remsi, divide in 1D, then reshape for FC.
// The 8x8 input's last row/column are nonzero sentinels; only 7x7 is averaged.
// FC uses M=2, N=33, K=35 with a BF16 contraction-round then channel bias.
func.func @batch_average_fc(%input: tensor<2x35x8x8xbf16>,
                            %weight: tensor<33x35xbf16>,
                            %bias: tensor<33xbf16>) -> tensor<2x33xbf16> {
  %zero = arith.constant 0.0 : bf16
  %zero_f32 = arith.constant 0.0 : f32
  %divisor = arith.constant 49.0 : bf16
  %c7 = arith.constant 7 : index
  %c35 = arith.constant 35 : index
  %dummy = tensor.empty() : tensor<49xi1>
  %empty_sum = tensor.empty() : tensor<70xbf16>
  %init_sum = linalg.fill ins(%zero : bf16) outs(%empty_sum : tensor<70xbf16>) -> tensor<70xbf16>
  %scalar = tensor.empty() : tensor<bf16>
  %sum:2 = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(j)>, affine_map<(i,j)->(i)>,
                       affine_map<(i,j)->()>],
      iterator_types = ["parallel","reduction"]}
      ins(%dummy : tensor<49xi1>) outs(%init_sum, %scalar : tensor<70xbf16>, tensor<bf16>) {
    ^bb0(%unused: i1, %acc: bf16, %unused_scalar: bf16):
      %i = linalg.index 0 : index
      %b = arith.divsi %i, %c35 : index
      %c = arith.remsi %i, %c35 : index
      %j = linalg.index 1 : index
      %h = arith.divsi %j, %c7 : index
      %w = arith.remsi %j, %c7 : index
      %x = tensor.extract %input[%b, %c, %h, %w] : tensor<2x35x8x8xbf16>
      %valid_h = arith.cmpi ult, %h, %c7 : index
      %v_h = arith.select %valid_h, %x, %zero : bf16
      %valid_w = arith.cmpi ult, %w, %c7 : index
      %v = arith.select %valid_w, %v_h, %zero : bf16
      %add = arith.addf %acc, %v : bf16
      linalg.yield %add, %divisor : bf16, bf16
  } -> (tensor<70xbf16>, tensor<bf16>)
  %average = linalg.generic {
      indexing_maps = [affine_map<(i)->()>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
      iterator_types = ["parallel"]}
      ins(%sum#1, %sum#0 : tensor<bf16>, tensor<70xbf16>) outs(%empty_sum : tensor<70xbf16>) {
    ^bb0(%denominator: bf16, %numerator: bf16, %unused: bf16):
      %v = arith.divf %numerator, %denominator : bf16
      linalg.yield %v : bf16
  } -> tensor<70xbf16>
  %matrix = tensor.expand_shape %average [[0,1]] output_shape [2,35] : tensor<70xbf16> into tensor<2x35xbf16>
  %empty_dot = tensor.empty() : tensor<2x33xf32>
  %init_dot = linalg.fill ins(%zero_f32 : f32) outs(%empty_dot : tensor<2x33xf32>) -> tensor<2x33xf32>
  %dot = linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                                      affine_map<(m,n,k)->(n,k)>,
                                      affine_map<(m,n,k)->(m,n)>]
      ins(%matrix, %weight : tensor<2x35xbf16>, tensor<33x35xbf16>)
      outs(%init_dot : tensor<2x33xf32>) -> tensor<2x33xf32>
  %out = tensor.empty() : tensor<2x33xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(n)>,
                       affine_map<(m,n)->(m,n)>],
      iterator_types = ["parallel","parallel"]}
      ins(%dot, %bias : tensor<2x33xf32>, tensor<33xbf16>) outs(%out : tensor<2x33xbf16>) {
    ^bb0(%v: f32, %bias_value: bf16, %unused: bf16):
      %r = arith.truncf %v : f32 to bf16
      %added = arith.addf %r, %bias_value : bf16
      linalg.yield %added : bf16
  } -> tensor<2x33xbf16>
  return %result : tensor<2x33xbf16>
}
