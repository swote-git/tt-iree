#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile BF16 spatial kernels, checking exact bits and per-core row coverage."""
import argparse
import csv
import json
import os
from pathlib import Path
import struct
import subprocess


def maximum(a, b):
    if a & 0x7fff > 0x7f80 or b & 0x7fff > 0x7f80:
        return 0x7fc0
    af = struct.unpack("<f", struct.pack("<I", a << 16))[0]
    bf = struct.unpack("<f", struct.pack("<I", b << 16))[0]
    if bf > af:
        return b
    return a & b if af == bf == 0 else a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--iree-run-module", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    module = out / "spatial.vmfb"
    source = root / "test/e2e/device_native_bf16_spatial.mlir"
    environment = dict(os.environ)
    for name in ("TT_METAL_SIMULATOR", "TT_METAL_DEVICE_PROFILER", "TT_IREE_CHECKPOINT_OUTPUT"):
        environment.pop(name, None)
    environment.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1",
                       TT_METAL_RUNTIME_ROOT=str(root / "third_party/tt-metal"))
    with (out / "compile.log").open("w") as stream:
        subprocess.run([args.iree_compile, str(source),
                        "--iree-hal-target-backends=tenstorrent", "-o", str(module)],
                       env=environment, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    cases = [0, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc5, 0x7f81]
    values = [cases[i % len(cases)] for i in range(5*7*39)]
    (out / "input.bin").write_bytes(struct.pack("<" + "H"*len(values), *values))
    results = {}
    for op, shape in [("pool", (5, 3, 19)), ("copy", (5, 9, 43))]:
        expected = []
        for c in range(shape[0]):
            for h in range(shape[1]):
                for w in range(shape[2]):
                    if op == "pool":
                        value = 0xff80
                        for kh in range(3):
                            for kw in range(3):
                                value = maximum(value, values[(c*7+h*2+kh)*39+w*2+kw])
                    else:
                        value = values[(c*7+h-1)*39+w-2] if 1 <= h < 8 and 2 <= w < 41 else 0
                    expected.append(value)
        profile = out / f"{op}-profile"
        env = dict(environment, TT_IREE_PROFILE_OUTPUT=str(profile))
        with (out / f"{op}.log").open("w") as stream:
            subprocess.run([args.iree_run_module, "--device=tenstorrent", f"--module={module}",
                            f"--function=spatial_{op}", f"--input=5x7x39xbf16=@{out/'input.bin'}",
                            f"--output=@{out/f'{op}.bin'}"], env=env, check=True,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        raw = (out / f"{op}.bin").read_bytes()
        assert len(raw) == len(expected)*2
        actual = struct.unpack("<" + "H"*len(expected), raw)
        assert list(actual) == expected, f"{op}: output bits or halo differ"
        with (profile / "ops_perf_results.csv").open() as stream:
            ops = list(csv.DictReader(stream))
        assert len(ops) == 1 and int(ops[0]["CORE COUNT"]) > 1, ops
        with (profile / "core_mapping.csv").open() as stream:
            ranges = list(csv.DictReader(stream))
        previous = 0
        width = 39 if op == "copy" else 19
        for row in ranges:
            begin, end = int(row["ELEMENT BEGIN"]), int(row["ELEMENT END"])
            assert begin == previous and begin % width == end % width == 0
            previous = end
        expected_count = 5*7*39 if op == "copy" else 5*3*19
        assert previous == expected_count
        summary = json.loads((profile / "run_results.json").read_text())
        assert summary["all_dispatches_device_native"]
        results[op] = {"values": len(expected), "bitwise_mismatches": 0,
                       "cores": int(ops[0]["CORE COUNT"]), "covered_elements": previous}
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
