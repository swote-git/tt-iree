#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exact BF16 Tensix transport/identity and bounded-integer contraction gates.

These are separate from the unchanged RISC-V FP32 bitwise fixtures. General
nonuniform/model accuracy is reported against PyTorch, never tolerance-fudged.
"""
import argparse
import csv
import json
import os
import pathlib
import struct
import subprocess

from device_native_bf16_test import bf16_bits, write_bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--iree-run-module", required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("TT_METAL_SIMULATOR", "TT_IREE_CHECKPOINT_OUTPUT"):
        environment.pop(name, None)
    environment.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1",
                       TT_METAL_RUNTIME_ROOT=str(root / "third_party/tt-metal"))
    command = [args.iree_compile, str(root / "test/e2e/device_native_bf16_projection.mlir"),
               "--iree-hal-target-backends=tenstorrent",
               "--iree-tenstorrent-native-bf16-use-tensix=true", "-o", str(out / "module.vmfb")]
    with (out / "compile.log").open("w") as stream:
        subprocess.run(command, env=environment, check=True, stdout=stream,
                       stderr=subprocess.STDOUT, timeout=300)
    results = []
    for name in ("identity", "bounded_integer"):
        target = out / name
        target.mkdir(exist_ok=True)
        if name == "identity":
            bits = [((i % 2) << 15) | ((121 + (i * 3) % 13) << 7) | ((i * 19) % 128)
                    for i in range(35 * 49)]
            a = [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in bits]
            b = [float(n == k) for n in range(33) for k in range(35)]
            expected = bits[:33 * 49]
        else:
            a = [float((i * 13) % 5 - 2) for i in range(35 * 49)]
            b = [float((i * 17) % 5 - 2) for i in range(33 * 35)]
            expected = [bf16_bits(sum(a[k * 49 + m] * b[n * 35 + k] for k in range(35)))
                        for n in range(33) for m in range(49)]
        write_bf16(target / "a.bin", a)
        write_bf16(target / "b.bin", b)
        environment["TT_IREE_PROFILE_OUTPUT"] = str(target / "profile")
        command = [args.iree_run_module, "--device=tenstorrent", f"--module={out / 'module.vmfb'}",
                   "--function=bf16_projection", f"--input=35x49xbf16=@{target / 'a.bin'}",
                   f"--input=33x35xbf16=@{target / 'b.bin'}", f"--output=@{target / 'output.bin'}"]
        with (target / "run.log").open("w") as stream:
            subprocess.run(command, env=environment, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=120)
        data = (target / "output.bin").read_bytes()
        assert len(data) == len(expected) * 2, "BF16 output byte count"
        actual = struct.unpack("<" + "H" * len(expected), data)
        mismatches = sum(a != e for a, e in zip(actual, expected))
        with (target / "profile/ops_perf_results.csv").open() as stream:
            records = list(csv.DictReader(stream))
        assert len(records) == 1
        record = records[0]
        assert record["IREE EXECUTION DOMAIN"] == "device_native"
        assert record["IREE BUILTIN PROGRAM"] == "12"
        assert record["IREE COMPUTE DATATYPE"] == "BF16_HIFI4_FP32_DST_F32_EPILOGUE"
        assert int(record["CORE COUNT"]) == 4
        for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
            assert int(record[f"IREE {stage} DURATION [ns]"]) == 0
        results.append({"fixture": name, "elements": len(expected),
                        "bitwise_mismatches": mismatches, "cores": 4})
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        assert mismatches == 0, f"{name}: {mismatches} BF16 bitwise mismatches"
        print(f"PASS: {name}, {len(expected)} BF16 values, 4 cores, M/N/K tails", flush=True)


if __name__ == "__main__":
    main()
