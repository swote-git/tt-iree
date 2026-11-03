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

from device_native_bf16_test import bf16, bf16_bits, write_bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--iree-run-module", required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("TT_METAL_SIMULATOR", "TT_IREE_CHECKPOINT_OUTPUT"):
        environment.pop(name, None)
    environment.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1",
                       TT_METAL_RUNTIME_ROOT=str(root / "third_party/tt-metal"))
    command = [args.iree_compile, str(root / "test/e2e/native_bf16/device_native_bf16_projection.mlir"),
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

    # SFPU epilogue FP32 chain with division versus the device-RISC-V reference.
    # Bounded integer contractions are exact in both engines, so the epilogue
    # alone decides the bits. Channel 0 divides by zero, channel 1 by NaN.
    target = out / "divide"
    target.mkdir(exist_ok=True)
    a = [float((i * 13) % 5 - 2) for i in range(35 * 49)]
    b = [float((i * 17) % 5 - 2) for i in range(33 * 35)]
    scale = [bf16(0.37 + 0.731 * ((n * 7) % 11)) * (-1 if n % 3 == 0 else 1) for n in range(33)]
    divisor = [bf16(0.13 + 1.917 * ((n * 5) % 9)) for n in range(33)]
    divisor[0] = 0.0
    divisor[1] = float("nan")
    residual = [bf16(((i * 29) % 97 - 48) / 7.0) for i in range(33 * 49)]
    for name, values in (("a", a), ("b", b), ("scale", scale), ("divisor", divisor),
                         ("residual", residual)):
        write_bf16(target / f"{name}.bin", values)
    outputs = {}
    for engine in ("tensix", "riscv"):
        vmfb = target / f"{engine}.vmfb"
        command = [args.iree_compile, str(root / "test/e2e/native_bf16/device_native_bf16_divide.mlir"),
                   "--iree-hal-target-backends=tenstorrent",
                   f"--iree-tenstorrent-native-bf16-use-tensix={'true' if engine == 'tensix' else 'false'}",
                   "-o", str(vmfb)]
        with (target / f"{engine}.compile.log").open("w") as stream:
            subprocess.run(command, env=environment, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=300)
        environment["TT_IREE_PROFILE_OUTPUT"] = str(target / f"{engine}.profile")
        command = [args.iree_run_module, "--device=tenstorrent", f"--module={vmfb}",
                   "--function=bf16_divide", f"--input=35x49xbf16=@{target / 'a.bin'}",
                   f"--input=33x35xbf16=@{target / 'b.bin'}",
                   f"--input=33xbf16=@{target / 'scale.bin'}",
                   f"--input=33xbf16=@{target / 'divisor.bin'}",
                   f"--input=33x49xbf16=@{target / 'residual.bin'}",
                   f"--output=@{target / (engine + '.bin')}"]
        with (target / f"{engine}.run.log").open("w") as stream:
            subprocess.run(command, env=environment, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=120)
        with (target / f"{engine}.profile/ops_perf_results.csv").open() as stream:
            records = list(csv.DictReader(stream))
        assert [r["IREE BUILTIN PROGRAM"] for r in records] == \
            (["12"] if engine == "tensix" else ["11"]), records
        outputs[engine] = (target / f"{engine}.bin").read_bytes()
    count = 33 * 49
    tensix = struct.unpack("<" + "H" * count, outputs["tensix"])
    riscv = struct.unpack("<" + "H" * count, outputs["riscv"])
    mismatches = sum(x != y for x, y in zip(tensix, riscv))
    results.append({"fixture": "divide", "elements": count,
                    "bitwise_mismatches_vs_riscv": mismatches,
                    "special_bits": [tensix[0], tensix[49]]})
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert mismatches == 0, f"divide: {mismatches} BF16 mismatches versus RISC-V"
    assert tensix[49] == 0x7fc0, "NaN divisor must give the canonical NaN"
    print(f"PASS: divide, {count} BF16 values equal to the RISC-V reference", flush=True)


if __name__ == "__main__":
    main()
