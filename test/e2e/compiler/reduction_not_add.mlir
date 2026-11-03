// A reduction containing one arith.addf must not be classified as the
// elementwise add builtin.
func.func @reduction_not_add(%arg0: tensor<1x32xbf16>) -> tensor<1xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<1xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<1xbf16>)
      -> tensor<1xbf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<1x32xbf16>)
      outs(%init : tensor<1xbf16>) {
    ^bb0(%input: bf16, %acc: bf16):
      %sum = arith.addf %input, %acc : bf16
      linalg.yield %sum : bf16
  } -> tensor<1xbf16>
  return %result : tensor<1xbf16>
}
