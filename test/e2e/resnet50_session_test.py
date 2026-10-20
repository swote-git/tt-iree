#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""CPU transport gate for the persistent runner; does not claim ResNet hardware accuracy."""
import argparse
import csv
import pathlib
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--session-runner", required=True)
    parser.add_argument("--mnist", action="store_true")
    args = parser.parse_args()
    input_shape = "1x1x28x28" if args.mnist else "1x3x224x224"
    input_elements = 784 if args.mnist else 3 * 224 * 224
    output_classes = 10 if args.mnist else 1000
    flat_shape = f"1x{input_elements}"
    with tempfile.TemporaryDirectory(prefix="tt-iree-session-test-") as directory:
        root = pathlib.Path(directory)
        source = root / "transport.mlir"
        source.write_text('''module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x1000xbf16> {
    %flat = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] :
      tensor<1x3x224x224xbf16> into tensor<1x150528xbf16>
    %result = tensor.extract_slice %flat[0, 0] [1, 1000] [1, 1] :
      tensor<1x150528xbf16> to tensor<1x1000xbf16>
    return %result : tensor<1x1000xbf16>
  }
}
'''.replace("1x3x224x224", input_shape).replace("1x150528", flat_shape)
                 .replace("1x1000", f"1x{output_classes}").replace("[1, 1000]", f"[1, {output_classes}]"))
        module = root / "transport.vmfb"
        subprocess.run([args.iree_compile, str(source), "--iree-hal-target-backends=llvm-cpu",
                        "--iree-llvmcpu-target-cpu=host", "-o", str(module)], check=True)
        inputs = []
        for i in range(2):
            filename = root / f"input-{i}.bin"
            payload = b"".join(struct.pack("<H", 0x3f00 + (j * 13 + i * 7) % 128)
                               for j in range(input_elements))
            filename.write_bytes(payload)
            inputs.append(filename)
        manifest = root / "inputs.tsv"
        outputs = [root / f"output-{i}.bin" for i in range(3)]
        manifest.write_text("".join(f"{inputs[i % 2]}\t{output}\n" for i, output in enumerate(outputs)))
        timings = root / "timings.csv"
        command = [args.session_runner, str(module), str(manifest), str(timings), "local-sync"]
        subprocess.run(command, check=True)
        for i, output in enumerate(outputs):
            assert output.read_bytes() == inputs[i % 2].read_bytes()[:output_classes * 2]
        assert outputs[0].read_bytes() != outputs[1].read_bytes()
        assert [r["sample"] for r in csv.DictReader(timings.open())] == ["0", "1", "2"]
        # Existing results and malformed/short inputs must fail without mutation.
        before = outputs[0].read_bytes()
        assert subprocess.run(command, capture_output=True).returncode != 0
        assert outputs[0].read_bytes() == before
        short = root / "short.bin"
        short.write_bytes(b"\0\0")
        manifest.write_text(f"{short}\t{root / 'new.bin'}\n")
        command[3] = str(root / "new-timings.csv")
        assert subprocess.run(command, capture_output=True).returncode != 0
        assert not (root / "new.bin").exists()
        # Keep the native tool's I/O ABI strict: F32 output is not BF16 output.
        bad_source = root / "wrong-output.mlir"
        bad_source.write_text('''module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x1000xf32> {
    %zero = arith.constant dense<0.0> : tensor<1x1000xf32>
    return %zero : tensor<1x1000xf32>
  }
}
'''.replace("1x3x224x224", input_shape).replace("1x1000", f"1x{output_classes}"))
        bad_module = root / "wrong-output.vmfb"
        subprocess.run([args.iree_compile, str(bad_source), "--iree-hal-target-backends=llvm-cpu",
                        "--iree-llvmcpu-target-cpu=host", "-o", str(bad_module)], check=True)
        manifest.write_text(f"{inputs[0]}\t{root / 'wrong-output.bin'}\n")
        command[1] = str(bad_module)
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0 and f"1x{output_classes}xbf16" in failed.stderr
        assert not (root / "wrong-output.bin").exists()
        print("PASS: raw BF16 transport A/B/A, session reuse, immutable results, short input rejection")


if __name__ == "__main__":
    main()
