// RUN: iree-compile \
// RUN:   --iree-hal-target-backends=tenstorrent \
// RUN:   %s -o %t.vmfb
// RUN: iree-run-module \
// RUN:   --device=tenstorrent \
// RUN:   --module=%t.vmfb \
// RUN:   --function=simple_matmul \
// RUN:   --input="32x32xbf16=1.0" \
// RUN:   --input="32x32xbf16=2.0" \
// RUN: | FileCheck %s

// CHECK: 32x32xbf16=
// CHECK: 64

// Single-tile (32x32x32) bfloat16 matmul.
// Maps to BF16_MATMUL_32X32X32 builtin in TTEX.
func.func @simple_matmul(
    %lhs: tensor<32x32xbf16>,
    %rhs: tensor<32x32xbf16>)
    -> tensor<32x32xbf16> {
  %init = tensor.empty() : tensor<32x32xbf16>
  %zero = arith.constant 0.0 : bf16
  %filled = linalg.fill ins(%zero : bf16) outs(%init : tensor<32x32xbf16>)
      -> tensor<32x32xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<32x32xbf16>, tensor<32x32xbf16>)
      outs(%filled : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %result : tensor<32x32xbf16>
}
