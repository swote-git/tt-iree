// RUN: iree-compile \
// RUN:   --iree-hal-target-backends=tenstorrent \
// RUN:   %s -o %t.vmfb

// Static tile-aligned matmul that should route to the tiled builtin path.
func.func @simple_matmul_64(
    %lhs: tensor<64x64xbf16>,
    %rhs: tensor<64x64xbf16>)
    -> tensor<64x64xbf16> {
  %init = tensor.empty() : tensor<64x64xbf16>
  %zero = arith.constant 0.0 : bf16
  %filled = linalg.fill ins(%zero : bf16) outs(%init : tensor<64x64xbf16>)
      -> tensor<64x64xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%filled : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %result : tensor<64x64xbf16>
}
