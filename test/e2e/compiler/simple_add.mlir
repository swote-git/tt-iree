// RUN: iree-compile \
// RUN:   --iree-hal-target-backends=tenstorrent \
// RUN:   %s -o %t.vmfb
// RUN: iree-run-module \
// RUN:   --device=tenstorrent \
// RUN:   --module=%t.vmfb \
// RUN:   --function=simple_add \
// RUN:   --input="1x1x32x32xbf16=1.0" \
// RUN:   --input="1x1x32x32xbf16=2.0" \
// RUN: | FileCheck %s

// CHECK: 1x1x32x32xbf16=
// CHECK: 3

// Single tile (32x32) bfloat16 elementwise addition.
// Maps to CUSTOM_SFPI_ADD builtin in TTEX.
func.func @simple_add(
    %arg0: tensor<1x1x32x32xbf16>,
    %arg1: tensor<1x1x32x32xbf16>)
    -> tensor<1x1x32x32xbf16> {
  %0 = arith.addf %arg0, %arg1 : tensor<1x1x32x32xbf16>
  return %0 : tensor<1x1x32x32xbf16>
}
