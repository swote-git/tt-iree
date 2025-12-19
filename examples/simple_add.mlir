// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple elementwise add for PoC testing
// RUN: iree-compile --iree-hal-target-backends=tenstorrent %s -o %t.vmfb
// RUN: iree-run-module --device=tenstorrent --module=%t.vmfb --function=add_one --input=4xf32=1,2,3,4

module @simple_add {
  func.func @add_one(%input: tensor<4xf32>) -> tensor<4xf32> {
    %cst = arith.constant dense<1.0> : tensor<4xf32>
    %result = arith.addf %input, %cst : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
