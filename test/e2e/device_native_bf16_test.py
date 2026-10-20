#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Bitwise BF16 compiler-to-hardware checks, with no NumPy dependency."""
import argparse
import csv
import json
import os
import pathlib
import struct
import subprocess


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_bits(value):
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    if raw & 0x7fffffff > 0x7f800000:
        return (raw >> 16) | 0x40
    return ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16) & 0xffff


def bf16(value):
    return struct.unpack("<f", struct.pack("<I", bf16_bits(value) << 16))[0]


def write_bf16(filename, values):
    pathlib.Path(filename).write_bytes(b"".join(struct.pack("<H", bf16_bits(v)) for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--iree-run-module", required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[2]
    a = [bf16(((i * 13) % 71 - 35) / 32) for i in range(35 * 49)]
    b = [bf16(((i * 17) % 67 - 33) / 256) for i in range(33 * 35)]
    # The first two spatial positions separately probe even/odd BF16 ties.
    for k in range(35):
        a[k * 49] = a[k * 49 + 1] = 0.0
    a[0] = a[1] = 1.0
    a[49] = 1.0
    a[50] = 3.0
    b[:35] = [1.0, 1 / 256] + [0.0] * 33
    expected = []
    for n in range(33):
        for m in range(49):
            acc = 0.0
            for k in range(35):
                acc = f32(acc + f32(a[k * 49 + m] * b[n * 35 + k]))
            expected.append(bf16_bits(acc))
    assert expected[:2] == [0x3f80, 0x3f82], "RNE tie reference"
    write_bf16(out / "a.bin", a)
    write_bf16(out / "b.bin", b)
    env = os.environ.copy()
    env.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1", TT_IREE_PROFILE_OUTPUT=str(out / "profile"),
               TT_METAL_RUNTIME_ROOT=str(root / "third_party/tt-metal"))
    env.pop("TT_METAL_SIMULATOR", None)
    commands = [
        [args.iree_compile, str(root / "test/e2e/device_native_bf16_projection.mlir"),
         "--iree-hal-target-backends=tenstorrent", "--iree-tenstorrent-device-id=p100a",
         "-o", str(out / "module.vmfb")],
        [args.iree_run_module, "--device=tenstorrent", f"--module={out / 'module.vmfb'}",
         "--function=bf16_projection", f"--input=35x49xbf16=@{out / 'a.bin'}",
         f"--input=33x35xbf16=@{out / 'b.bin'}", f"--output=@{out / 'output.bin'}"],
    ]
    for index, command in enumerate(commands):
        with (out / f"stage-{index}.log").open("w") as stream:
            subprocess.run(command, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    data = (out / "output.bin").read_bytes()
    assert len(data) == len(expected) * 2, "BF16 output byte count"
    actual = struct.unpack("<" + "H" * len(expected), data)
    mismatches = sum(x != y for x, y in zip(actual, expected))
    profile = json.loads((out / "profile/run_results.json").read_text())
    assert profile["all_dispatches_device_native"] and profile["dispatch_count"] == 1, profile
    with (out / "profile/ops_perf_results.csv").open() as stream:
        for record in csv.DictReader(stream):
            for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
                assert int(record[f"IREE {stage} DURATION [ns]"]) == 0, record
    (out / "results.json").write_text(json.dumps({"elements": len(expected),
        "bf16_bitwise_mismatches": mismatches, "even_odd_tie_bits": list(actual[:2]),
        "profile": profile}, indent=2) + "\n")
    assert mismatches == 0, f"{mismatches} BF16 mismatches"
    print(f"PASS: {len(expected)} BF16 values, page/tile tails and RNE ties")
    average = out / "average"
    average.mkdir(exist_ok=True)
    values = [bf16(((i * 11) % 43 - 21) / 32) for i in range(4 * 49)]
    values[:49] = [1.0] + [1 / 512] * 48
    sums = []
    for channel in range(4):
        acc = 0.0
        for value in values[channel * 49:(channel + 1) * 49]:
            acc = bf16(acc + value)
        sums.append(bf16_bits(acc / 49))
    assert sums[0] != bf16_bits(sum(values[:49]) / 49), "must exercise BF16 reduction rounding"
    write_bf16(average / "input.bin", values)
    env["TT_IREE_PROFILE_OUTPUT"] = str(average / "profile")
    source = root / "test/e2e/device_native_bf16_average.mlir"
    for index, command in enumerate((
        [args.iree_compile, str(source), "--iree-hal-target-backends=tenstorrent",
         "-o", str(average / "module.vmfb")],
        [args.iree_run_module, "--device=tenstorrent", f"--module={average / 'module.vmfb'}",
         "--function=bf16_average", f"--input=4x7x7xbf16=@{average / 'input.bin'}",
         f"--output=@{average / 'output.bin'}"],
    )):
        with (average / f"stage-{index}.log").open("w") as stream:
            subprocess.run(command, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    actual = struct.unpack("<4H", (average / "output.bin").read_bytes())
    assert list(actual) == sums, (actual, sums)
    profile = json.loads((average / "profile/run_results.json").read_text())
    assert profile["all_dispatches_device_native"], profile
    (average / "results.json").write_text(json.dumps({
        "elements": 4, "bf16_bitwise_mismatches": 0, "actual_bits": list(actual),
        "expected_bits": sums, "profile": profile}, indent=2) + "\n")
    print("PASS: BF16 padding, per-add reduction rounding and scalar division")
    negative = average / "wrong-divisor.mlir"
    negative.write_text(source.read_text().replace("49.0 : bf16", "48.0 : bf16"))
    result = subprocess.run([args.iree_compile, str(negative), "--iree-hal-target-backends=tenstorrent",
                             "-o", str(average / "invalid.vmfb")], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    (average / "negative.log").write_text(result.stdout)
    assert result.returncode != 0 and "unsupported Tenstorrent dispatch" in result.stdout
    result = subprocess.run([args.iree_compile, str(root / "test/e2e/device_native_bf16_projection.mlir"),
                             "--iree-hal-target-backends=tenstorrent", "--iree-tenstorrent-native-f32-use-tensix=true",
                             "-o", str(out / "invalid-f32-mode.vmfb")], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    (out / "invalid-f32-mode.log").write_text(result.stdout)
    assert result.returncode != 0 and "BF16-native v17 cannot use" in result.stdout
    print("PASS: reject changed reduction semantics and F32 Tensix mode")


if __name__ == "__main__":
    main()
