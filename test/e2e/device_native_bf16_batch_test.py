#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Static B2 BF16 hardware gates, using only the Python standard library.

Compile once for all six entry points. --compile-only allows AOT to overlap
other work; --skip-compile later runs that module. --prepare-only needs no
compiler or device. No NumPy, PyTorch, B1 host loop, or accuracy tolerance.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("device_native_bf16_batch.mlir")
STAGING = ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING")
COMPUTE = {11: "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING",
           12: "BF16_HIFI4_FP32_DST_F32_EPILOGUE"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def bf16_bits(value):
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    if raw & 0x7fffffff > 0x7f800000:
        return (raw >> 16) | 0x40
    return ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16) & 0xffff


def from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def bf16(value):
    return from_bits(bf16_bits(value))


def write_bits(path, values):
    path.write_bytes(struct.pack("<" + "H" * len(values), *values))


@dataclass
class Tensor:
    name: str
    shape: tuple
    values: list

    def __post_init__(self):
        require(len(self.values) == math.prod(self.shape), f"{self.name}: shape mismatch")
        require(all(bf16(v) == v for v in self.values), f"{self.name}: non-BF16 input")

    @property
    def type(self):
        return "x".join(map(str, self.shape)) + "xbf16"


@dataclass
class Case:
    name: str
    inputs: list
    shape: tuple
    expected: list
    # (profile OP TYPE, M, N, K); scalar geometry is zero.
    dispatches: tuple
    # Logical elements and row width for padded-copy/maxpool core partitions.
    scalar_range: tuple = ()

    def __post_init__(self):
        require(len(self.expected) == math.prod(self.shape), f"{self.name}: output shape mismatch")
        half = len(self.expected) // 2
        require(self.expected[:half] != self.expected[half:], f"{self.name}: identical batch references")


def convolution(stride):
    channels, height, width, outputs = (3, 7, 11, 33) if stride == 1 else (5, 9, 15, 35)
    oh, ow = (height - 3) // stride + 1, (width - 3) // stride + 1
    a = [sign * (1 + (c * 7 + h * 3 + w) % 3)
         for sign in (1, -2) for c in range(channels)
         for h in range(height) for w in range(width)]
    weight = [(n * 11 + c * 7 + kh * 3 + kw) % 5 - 2
              for n in range(outputs) for c in range(channels)
              for kh in range(3) for kw in range(3)]
    halo = int(stride == 1)
    expected = []
    for b in range(2):
        for n in range(outputs):
            for h in range(oh + 2 * halo):
                for w in range(ow + 2 * halo):
                    if not (halo <= h < oh + halo and halo <= w < ow + halo):
                        expected.append(0)
                        continue
                    # All products and partial sums are small exact integers,
                    # independent of Tensix reduction order. Round only output.
                    total = sum(
                        a[((b * channels + c) * height + (h - halo) * stride + kh) * width
                          + (w - halo) * stride + kw]
                        * weight[((n * channels + c) * 3 + kh) * 3 + kw]
                        for c in range(channels) for kh in range(3) for kw in range(3))
                    expected.append(bf16_bits(total))
    return Case(f"conv_s{stride}", [Tensor("input", (2, channels, height, width), a),
                                   Tensor("weight", (outputs, channels, 3, 3), weight)],
                (2, outputs, oh + 2 * halo, ow + 2 * halo), expected,
                (("CONV2D", 2 * oh * ow, outputs, channels * 9),))


def projection_epilogue(value, mean, scale, bias, residual):
    value = bf16(value)
    value = bf16(value - mean)
    value = bf16(value * 0.5)
    value = bf16(value * scale)
    value = bf16(value + bias)
    value = bf16(value + residual)
    return bf16_bits(value if value > 0 else 0.0)


def projection():
    a = [sign * (1 + (k + m * 3) % 3 / 8)
         for sign in (1, -2) for k in range(35) for m in range(49)]
    weight = [((n * 17 + k * 7) % 5 - 2) / 8 for n in range(33) for k in range(35)]
    for b, sign in enumerate((1, -2)):
        for k in range(35):
            a[(b * 35 + k) * 49] = sign if k < 2 else 0.0
            a[(b * 35 + k) * 49 + 1] = sign * (1 if k == 0 else 3 if k == 1 else 0)
    weight[:35] = [1.0, 1 / 256] + [0.0] * 33
    mean = [(n % 3 - 1) / 128 for n in range(33)]
    scale = [1 + (n % 3 + 1) / 128 for n in range(33)]
    bias = [(n % 5 - 2) / 64 for n in range(33)]
    mean[0], bias[0] = 1 / 256, 1 / 256
    residual = [sign * ((n * 5 + m * 3) % 7 - 3) / 64
                for sign in (1, -2) for n in range(33) for m in range(49)]
    expected = []
    for b in range(2):
        for n in range(33):
            for m in range(49):
                # Dyadic products/partial sums fit exactly in FP32.
                total = sum(a[(b * 35 + k) * 49 + m] * weight[n * 35 + k] for k in range(35))
                expected.append(projection_epilogue(
                    total, mean[n], scale[n], bias[n], residual[(b * 33 + n) * 49 + m]))
    require([bf16_bits(1 + v / 256) for v in (1, 3)] == [0x3f80, 0x3f82], "RNE tie reference")
    return Case("projection", [Tensor("input", (2, 35, 49), a),
                               Tensor("weight", (33, 35), weight),
                               Tensor("mean", (33,), mean), Tensor("scale", (33,), scale),
                               Tensor("bias", (33,), bias),
                               Tensor("residual", (2, 33, 49), residual)],
                (2, 33, 49), expected, (("MATMUL", 98, 33, 35),))


def spatial(pool):
    a = [sign * (1 + (c * 17 + h * 5 + w * 3) % 31) / 8
         for sign in (1, -2) for c in range(5) for h in range(7) for w in range(39)]
    oh, ow = (3, 19) if pool else (10, 42)
    expected = []
    for b in range(2):
        for c in range(5):
            for h in range(oh):
                for w in range(ow):
                    if pool:
                        value = max(a[((b * 5 + c) * 7 + h * 2 + kh) * 39 + w * 2 + kw]
                                    for kh in range(3) for kw in range(3))
                    else:
                        value = a[((b * 5 + c) * 7 + h - 1) * 39 + w - 2] \
                            if 1 <= h < 8 and 2 <= w < 41 else 0.0
                    expected.append(bf16_bits(value))
    return Case("maxpool" if pool else "input_pad", [Tensor("input", (2, 5, 7, 39), a)],
                (2, 5, oh, ow), expected, (("MAX_POOL_2D" if pool else "PADDED_COPY", 0, 0, 0),),
                (2 * 5 * 3 * 19, 19) if pool else (2 * 5 * 7 * 39, 39))


def ordered_average(values):
    total = 0.0
    for value in values:
        total = bf16(total + value)
    return bf16(total / 49)


def average_fc():
    a = []
    averages = []
    for sign in (1, -2):
        for c in range(35):
            # Channel zero is deliberately NOT an FP32 sum: every tiny add
            # ties back to 1/-2. Other channels vary with position/channel.
            valid = [sign * (1 if j == 0 else 1 / 512) if c == 0
                     else sign * ((c * 7 + j * 3) % 9 - 4) / 32 for j in range(49)]
            averages.append(ordered_average(valid))
            a.extend(valid[h * 7 + w] if h < 7 and w < 7 else sign * 64
                     for h in range(8) for w in range(8))
    # One exact product per output makes this a transport/layout gate, not an
    # IEEE-vs-Tensix multi-product accuracy test. The permutation reaches both
    # K-tail channels 33/34; n=0 still exposes the ordered GAP probe directly.
    weight = [float(k == (n * 17) % 35) for n in range(33) for k in range(35)]
    bias = [(n % 5 - 2) / 128 for n in range(33)]
    bias[0] = 0.0
    expected = [bf16_bits(averages[b * 35 + (n * 17) % 35] + bias[n])
                for b in range(2) for n in range(33)]
    require(expected[0] == bf16_bits(1 / 49), "ordered positive GAP probe")
    require(expected[33] == bf16_bits(-2 / 49), "ordered negative GAP probe")
    require(expected[0] != bf16_bits((1 + 48 / 512) / 49), "GAP must distinguish FP32 sum")
    return Case("average_fc", [Tensor("input", (2, 35, 8, 8), a),
                               Tensor("weight", (33, 35), weight), Tensor("bias", (33,), bias)],
                (2, 33), expected, (("GLOBAL_SUM", 0, 0, 0),
                                    ("SCALAR_DIV", 0, 0, 0), ("LINEAR", 2, 33, 35)))


def build_cases():
    return [convolution(1), convolution(2), projection(), spatial(False), spatial(True), average_fc()]


def validate_mapping(record, rows, scalar_range=()):
    gx, gy, gz = (int(record[f"IREE GRID {axis}"]) for axis in "XYZ")
    require(1 <= gx <= 11 and 1 <= gy <= 10 and gz == 1, f"invalid native grid: {(gx, gy, gz)}")
    require(int(record["CORE COUNT"]) == gx * gy, "core count does not match grid")
    matrix = record["OP TYPE"] in ("CONV2D", "MATMUL", "LINEAR")
    if not matrix and not scalar_range:
        require((gx, gy) == (1, 1), "unpartitioned scalar kernel must use one core")
        require(not rows, "unexpected scalar tile mapping")
        return
    require(len(rows) == gx * gy, "missing or duplicate core partitions")
    coordinates = {(int(row["CORE X"]), int(row["CORE Y"])) for row in rows}
    require(coordinates == {(x, y) for y in range(gy) for x in range(gx)}, "incomplete core coordinates")
    if scalar_range:
        count, width = scalar_range
        previous = 0
        for row in sorted(rows, key=lambda r: int(r["ELEMENT BEGIN"])):
            begin, end = int(row["ELEMENT BEGIN"]), int(row["ELEMENT END"])
            require(row["PARTITION UNIT"] == "rows", "scalar partition must use rows")
            require(begin == previous and begin < end <= count, "scalar coverage gap/overlap")
            require(begin % width == end % width == 0, "partition split a spatial row")
            previous = end
        require(previous == count, "incomplete scalar coverage")
        require(gx * gy > 1, "spatial case did not use multiple cores")
        return
    mt, nt, kt = ((int(record[f"IREE {dim}"]) + 31) // 32 for dim in "MNK")
    covered = set()
    for row in rows:
        mb, me = int(row["M TILE BEGIN"]), int(row["M TILE END"])
        nb, ne = int(row["N TILE BEGIN"]), int(row["N TILE END"])
        require(row["PARTITION UNIT"] == "tiles", "matrix partition must use tiles")
        require(0 <= mb < me <= mt and 0 <= nb < ne <= nt, "invalid tile rectangle")
        require(int(row["K TILE COUNT"]) == kt, "K tail was not covered")
        tiles = {(m, n) for m in range(mb, me) for n in range(nb, ne)}
        require(not covered.intersection(tiles), "overlapping matrix rectangles")
        covered.update(tiles)
    require(len(covered) == mt * nt, "incomplete matrix tile coverage")
    require(gx * gy > 1, "matrix case did not use multiple cores")


def validate_profile(directory, case):
    summary = json.loads((directory / "run_results.json").read_text())
    require(summary["all_dispatches_device_native"] is True, "non-native dispatch in summary")
    with (directory / "ops_perf_results.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    observed = [(row["OP TYPE"], *(int(row[f"IREE {dim}"]) for dim in "MNK")) for row in records]
    require(Counter(observed) == Counter(case.dispatches),
            f"{case.name}: expected true-batch dispatches {case.dispatches}, got {observed}")
    require(summary["dispatch_count"] == len(records), "summary dispatch count mismatch")
    with (directory / "core_mapping.csv").open(newline="") as stream:
        mappings = list(csv.DictReader(stream))
    calls = [row["GLOBAL CALL COUNT"] for row in records]
    require(len(set(calls)) == len(calls), "duplicate dispatch call IDs")
    require(all(row["GLOBAL CALL COUNT"] in calls for row in mappings), "orphan core mapping")
    for row in records:
        matrix = row["OP TYPE"] in ("CONV2D", "MATMUL", "LINEAR")
        program = 12 if matrix else 11
        require(row["IREE EXECUTION DOMAIN"] == "device_native", "host-assisted/mock dispatch")
        require(int(row["IREE BUILTIN PROGRAM"]) == program, "wrong BF16 compute program")
        require(row["IREE COMPUTE DATATYPE"] == COMPUTE[program], "wrong BF16 compute datatype")
        require(row["INPUT_0_DATATYPE"] == row["OUTPUT_0_DATATYPE"] == "BF16", "non-BF16 storage")
        require(int(row["IREE DISPATCH KIND"]) == 8, "not a BF16-native dispatch")
        for stage in STAGING:
            require(int(row[f"IREE {stage} DURATION [ns]"]) == 0, f"unexpected {stage.lower()}")
        rows = [mapping for mapping in mappings if mapping["GLOBAL CALL COUNT"] == row["GLOBAL CALL COUNT"]]
        require(all(mapping["OP CODE"] == row["OP CODE"] for mapping in rows), "mapping opcode mismatch")
        validate_mapping(row, rows, case.scalar_range)
    return {"dispatches": observed, "cores": [int(row["CORE COUNT"]) for row in records],
            "all_dispatches_device_native": True}


def compare_output(path, case):
    raw = path.read_bytes()
    require(len(raw) == len(case.expected) * 2,
            f"{case.name}: expected {len(case.expected) * 2} BF16 bytes, got {len(raw)}")
    actual = struct.unpack("<" + "H" * len(case.expected), raw)
    mismatches = [(i, a, e) for i, (a, e) in enumerate(zip(actual, case.expected)) if a != e]
    require(not mismatches, f"{case.name}: {len(mismatches)} bit mismatches; first (index, actual, expected): "
            + ", ".join(f"({i}, 0x{a:04x}, 0x{e:04x})" for i, a, e in mismatches[:8]))


def environment():
    env = dict(os.environ)
    for key in ("TT_METAL_SIMULATOR", "TT_METAL_DEVICE_PROFILER", "TT_IREE_CHECKPOINT_OUTPUT",
                "TT_IREE_PROFILE_OUTPUT"):
        env.pop(key, None)
    env.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1")
    env.setdefault("TT_METAL_RUNTIME_ROOT", str(ROOT / "third_party/tt-metal"))
    return env


def compile_command(compiler, output, reuse_output_tiles=False):
    command = [compiler, str(SOURCE), "--iree-hal-target-backends=tenstorrent",
            "--iree-tenstorrent-device-id=p100a", "--iree-opt-level=O2",
            "--iree-opt-generalize-matmul=false",
            "--iree-tenstorrent-native-bf16-use-tensix=true", "-o", str(output)]
    if reuse_output_tiles:
        command.insert(-2, "--iree-tenstorrent-native-bf16-reuse-output-tiles=true")
    return command


def run_logged(command, log, env, timeout):
    with log.open("w") as stream:
        try:
            subprocess.run(command, env=env, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"command failed; see {log}: {command}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile")
    parser.add_argument("--iree-run-module")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-output-tiles", action="store_true")
    parser.add_argument("--case", action="append", choices=[case.name for case in build_cases()])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--compile-timeout", type=int, default=900)
    parser.add_argument("--run-timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if not (args.prepare_only or args.skip_compile) and not args.iree_compile:
        parser.error("--iree-compile is required for compilation")
    if not (args.prepare_only or args.compile_only) and not args.iree_run_module:
        parser.error("--iree-run-module is required for hardware execution")
    if args.compile_timeout <= 0 or args.run_timeout <= 0:
        parser.error("timeouts must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = [case for case in build_cases() if not args.case or case.name in args.case]
    manifest = {}
    for case in cases:
        target = out / case.name
        target.mkdir(exist_ok=True)
        for tensor in case.inputs:
            write_bits(target / f"{tensor.name}.bin", [bf16_bits(v) for v in tensor.values])
        write_bits(target / "expected.bin", case.expected)
        manifest[case.name] = {"function": f"batch_{case.name}", "output_shape": case.shape,
                               "elements": len(case.expected), "dispatches": case.dispatches,
                               "inputs": {tensor.name: tensor.type for tensor in case.inputs}}
    (out / "expectations.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    module = out / "batch.vmfb"
    env = environment()
    if args.skip_compile:
        require(module.is_file(), f"missing compiled module: {module}")
    else:
        run_logged(compile_command(args.iree_compile, module, args.reuse_output_tiles),
                   out / "compile.log", env, args.compile_timeout)
    if args.compile_only:
        print(f"Compiled six B2 entry points: {module}")
        return
    results = {}
    # Do not let a previous successful run leave a stale aggregate result.
    (out / "results.json").unlink(missing_ok=True)
    for case in cases:
        target = out / case.name
        profile = target / "profile"
        require(not profile.exists(), f"profile already exists: {profile}; use a fresh output directory")
        output = target / "output.bin"
        output.unlink(missing_ok=True)
        command = [args.iree_run_module, "--device=tenstorrent", f"--module={module}",
                   f"--function=batch_{case.name}"]
        command += [f"--input={tensor.type}=@{target / (tensor.name + '.bin')}" for tensor in case.inputs]
        command += [f"--output=@{output}"]
        run_logged(command, target / "run.log", dict(env, TT_IREE_PROFILE_OUTPUT=str(profile)), args.run_timeout)
        compare_output(output, case)
        routing = validate_profile(profile, case)
        results[case.name] = {"elements": len(case.expected), "bitwise_mismatches": 0, **routing}
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"PASS: {case.name}: {len(case.expected)} exact BF16 values; cores={routing['cores']}", flush=True)


if __name__ == "__main__":
    main()
