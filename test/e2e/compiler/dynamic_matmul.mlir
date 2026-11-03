// RUN: iree-compile \
// RUN:   --iree-hal-target-backends=tenstorrent \
// RUN:   %s -o %t.vmfb

// Dynamic rank-2 bf16 matmul. Runtime dims are carried in dispatch constants.
func.func @dynamic_matmul(
    %lhs: tensor<?x?xbf16>,
    %rhs: tensor<?x?xbf16>)
    -> tensor<?x?xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x?xbf16>
  %n = tensor.dim %rhs, %c1 : tensor<?x?xbf16>
  %init = tensor.empty(%m, %n) : tensor<?x?xbf16>
  %zero = arith.constant 0.0 : bf16
  %filled = linalg.fill ins(%zero : bf16) outs(%init : tensor<?x?xbf16>)
      -> tensor<?x?xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%filled : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %result : tensor<?x?xbf16>
}
