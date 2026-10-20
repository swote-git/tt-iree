#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Two-layer MNIST-shape BF16 transport gate; no dataset or training required.

One-hot weights make the expected arithmetic exact. This is a compiler/runtime
regression fixture, not an MNIST classification-accuracy measurement.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from device_native_bf16_test import bf16, bf16_bits, write_bf16

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_mnist_mlp import require_mnist_profile


def source():
    return '''#lhs = affine_map<(m,n,k)->(m,k)>
#rhs = affine_map<(m,n,k)->(n,k)>
#out = affine_map<(m,n,k)->(m,n)>
#id = affine_map<(m,n)->(m,n)>
func.func @main(%images: tensor<1x1x28x28xbf16>, %w1: tensor<128x784xbf16>,
                %b1: tensor<1x128xbf16>, %w2: tensor<10x128xbf16>,
                %b2: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
  %input = tensor.collapse_shape %images [[0], [1,2,3]] :
    tensor<1x1x28x28xbf16> into tensor<1x784xbf16>
  %zero = arith.constant 0.0 : f32
  %empty1 = tensor.empty() : tensor<1x128xf32>
  %init1 = linalg.fill ins(%zero : f32) outs(%empty1 : tensor<1x128xf32>) -> tensor<1x128xf32>
  %dot1 = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%input, %w1 : tensor<1x784xbf16>, tensor<128x784xbf16>)
    outs(%init1 : tensor<1x128xf32>) -> tensor<1x128xf32>
  %emptyh = tensor.empty() : tensor<1x128xbf16>
  %hidden = linalg.generic {indexing_maps=[#id,#id,#id], iterator_types=["parallel","parallel"]}
    ins(%dot1, %b1 : tensor<1x128xf32>, tensor<1x128xbf16>) outs(%emptyh : tensor<1x128xbf16>) {
    ^bb0(%value: f32, %bias: bf16, %unused: bf16):
      %rounded = arith.truncf %value : f32 to bf16
      %sum = arith.addf %rounded, %bias : bf16
      %z = arith.constant 0.0 : bf16
      %positive = arith.cmpf ugt, %sum, %z : bf16
      %relu = arith.select %positive, %sum, %z : bf16
      linalg.yield %relu : bf16
  } -> tensor<1x128xbf16>
  %empty2 = tensor.empty() : tensor<1x10xf32>
  %init2 = linalg.fill ins(%zero : f32) outs(%empty2 : tensor<1x10xf32>) -> tensor<1x10xf32>
  %dot2 = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%hidden, %w2 : tensor<1x128xbf16>, tensor<10x128xbf16>)
    outs(%init2 : tensor<1x10xf32>) -> tensor<1x10xf32>
  %emptyo = tensor.empty() : tensor<1x10xbf16>
  %result = linalg.generic {indexing_maps=[#id,#id,#id], iterator_types=["parallel","parallel"]}
    ins(%dot2, %b2 : tensor<1x10xf32>, tensor<1x10xbf16>) outs(%emptyo : tensor<1x10xbf16>) {
    ^bb0(%value: f32, %bias: bf16, %unused: bf16):
      %rounded = arith.truncf %value : f32 to bf16
      %sum = arith.addf %rounded, %bias : bf16
      linalg.yield %sum : bf16
  } -> tensor<1x10xbf16>
  return %result : tensor<1x10xbf16>
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--iree-run-module", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    for name in ("TT_METAL_SIMULATOR", "TT_METAL_DEVICE_PROFILER", "TT_IREE_CHECKPOINT_OUTPUT"):
        env.pop(name, None)
    env.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1", TT_METAL_RUNTIME_ROOT=str(root / "third_party/tt-metal"))
    (out / "model.mlir").write_text(source())
    with (out / "compile.log").open("w") as stream:
        subprocess.run([args.iree_compile, str(out / "model.mlir"), "--iree-hal-target-backends=tenstorrent",
                        "--iree-tenstorrent-native-bf16-use-tensix=true", "-o", str(out / "model.vmfb")],
                       env=env, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    write_bf16(out / "w1.bin", [float(k == 783 - (n % 16)) for n in range(128) for k in range(784)])
    write_bf16(out / "w2.bin", [float(k == 127 - n) for n in range(10) for k in range(128)])
    bias1 = [(n % 5 - 2) / 8 for n in range(128)]
    bias2 = [(n % 3 - 1) / 4 for n in range(10)]
    write_bf16(out / "b1.bin", bias1)
    write_bf16(out / "b2.bin", bias2)
    results = []
    for ordinal, offset in enumerate((0, 3, 0)):
        sample = out / str(ordinal)
        sample.mkdir(exist_ok=True)
        values = [((i * 13 + offset) % 17 - 8) / 8 for i in range(784)]
        write_bf16(sample / "input.bin", values)
        hidden = [max(bf16(values[783 - (n % 16)] + bias1[n]), 0) for n in range(128)]
        expected = [bf16_bits(hidden[127 - n] + bias2[n]) for n in range(10)]
        env["TT_IREE_PROFILE_OUTPUT"] = str(sample / "profile")
        command = [args.iree_run_module, "--device=tenstorrent", f"--module={out / 'model.vmfb'}", "--function=main",
                   f"--input=1x1x28x28xbf16=@{sample / 'input.bin'}",
                   f"--input=128x784xbf16=@{out / 'w1.bin'}", f"--input=1x128xbf16=@{out / 'b1.bin'}",
                   f"--input=10x128xbf16=@{out / 'w2.bin'}", f"--input=1x10xbf16=@{out / 'b2.bin'}",
                   f"--output=@{sample / 'output.bin'}"]
        with (sample / "run.log").open("w") as stream:
            subprocess.run(command, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        actual = struct.unpack("<10H", (sample / "output.bin").read_bytes())
        assert list(actual) == expected, (actual, expected)
        profile = require_mnist_profile(sample / "profile", 1, "tensix")
        results.append(dict(case=ordinal, native_dispatches=profile["dispatch_count"], bitwise_mismatches=0))
    assert (out / "0/output.bin").read_bytes() == (out / "2/output.bin").read_bytes()
    assert (out / "0/output.bin").read_bytes() != (out / "1/output.bin").read_bytes()
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PASS: two native FCs + bias/ReLU, exact BF16, M/N/K tails, A/B/A inputs")


if __name__ == "__main__":
    main()
