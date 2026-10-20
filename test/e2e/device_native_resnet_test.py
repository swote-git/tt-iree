#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compiler-to-hardware native gate, with nonzero CPU-reference comparison."""
import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import struct
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--iree-run-module", required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--case", help="run only the named regression case")
    parser.add_argument("--reference-dir", type=pathlib.Path,
                        help="also require bitwise equality to a previous native run")
    parser.add_argument("--use-tensix", action="store_true",
                        help="test experimental v16 INT8-digit matrix kernels")
    parser.add_argument("--stress-dynamic-range", action="store_true",
                        help="include the cancellation stress diagnostic (not a supported v16 accuracy gate)")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("correctness", root / "scripts/check_resnet50_correctness.py")
    correctness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = correctness
    spec.loader.exec_module(correctness)
    inputs_dir = args.output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    precision_input = inputs_dir / "precision-input.npy"
    precision_weights = inputs_dir / "precision-weights.npy"
    correctness.write_f32_npy(precision_input, (64, 49),
                             [math.sin(i * 0.13) * 0.73 + 0.14 for i in range(64 * 49)])
    correctness.write_f32_npy(precision_weights, (64, 64),
                             [math.cos(i * 0.07) * 0.021 + 0.003 for i in range(64 * 64)])
    cases = [
        ("device_native_f32_matmul_tail", "device_native_f32_matmul_tail.mlir",
         [f"@{precision_input}", f"@{precision_weights}"]),
        ("device_native_f32_runtime_affine_matmul", "device_native_f32_runtime_affine_matmul.mlir", ["64x49xf32=1", "64x64xf32=0.015625", "64xf32=1", "64xf32=0.5", "64xf32=0.25"]),
        ("resnet50_conv2d", "resnet50_conv2d.mlir", ["64x10x10xf32=1", "64x64x3x3xf32=0.015625"]),
        ("resnet50_stem_conv", "resnet50_stem_conv.mlir", ["3x230x230xf32=1", "64x3x7x7xf32=0.015625"]),
        ("resnet50_max_pool", "resnet50_max_pool.mlir", ["64x114x114xf32=1.25"]),
        ("resnet50_per_channel_projection", "resnet50_per_channel_projection.mlir", ["64x56x56xf32=1", "64x64xf32=0.015625"]),
        ("resnet50_bottleneck", "resnet50_bottleneck.mlir", ["64x8x8xf32=1", "64x64xf32=0.015625", "64x64x3x3xf32=0.015625", "256x64xf32=0.015625", "256x64xf32=0.015625"]),
        ("device_native_average", "device_native_tail.mlir", ["4x7x7xf32=1.25"]),
        ("device_native_linear", "device_native_tail.mlir", ["1x33xf32=1.25", "17x33xf32=0.03125", "1x17xf32=0.5"]),
    ]
    for name, shape_a, shape_b, source in (
            ("device_native_f32_matmul_page_tail", (35, 49), (33, 35),
             "device_native_f32_matmul_page_tail.mlir"),
            ("nonuniform_conv2d", (64, 10, 10), (64, 64, 3, 3),
             "resnet50_conv2d.mlir")):
        a_path, b_path = inputs_dir / f"{name}-a.npy", inputs_dir / f"{name}-b.npy"
        correctness.write_f32_npy(a_path, shape_a,
                                 [math.sin(i * 0.13) * 0.73 + 0.14 for i in range(math.prod(shape_a))])
        correctness.write_f32_npy(b_path, shape_b,
                                 [math.cos(i * 0.07) * 0.021 + 0.003 for i in range(math.prod(shape_b))])
        cases.append((name, source, [f"@{a_path}", f"@{b_path}"]))
    for name, channels, a_path, b_path, source in (
            ("nonuniform_runtime_affine", 64, precision_input, precision_weights,
             "device_native_f32_runtime_affine_matmul.mlir"),
            ("nonuniform_runtime_affine_tail", 33,
             inputs_dir / "device_native_f32_matmul_page_tail-a.npy",
             inputs_dir / "device_native_f32_matmul_page_tail-b.npy",
             "device_native_f32_runtime_affine_tail.mlir")):
        # Dyadic coefficients exercise distinct channels, zero/negative scale,
        # and ReLU without introducing a new CPU reassociation accuracy gate.
        vectors = (
            [((i % 7) - 3) / 32.0 for i in range(channels)],
            [(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)[i % 7]
             for i in range(channels)],
            [(((i * 3) % 11) - 5) / 128.0 for i in range(channels)])
        inputs = [f"@{a_path}", f"@{b_path}"]
        for role, values in zip(("mean", "scale", "bias"), vectors):
            path = inputs_dir / f"{name}-{role}.npy"
            correctness.write_f32_npy(path, (channels,), values)
            inputs.append(f"@{path}")
        cases.append((name, source, inputs))
    # The BF16 float-math precision regression also covers signed integer dots.
    for name, a, b in (
            ("precision_signed_integer",
             [float((i * 37) % 255 - 127) for i in range(64 * 49)],
             [float((i * 59) % 255 - 127) for i in range(64 * 64)]),
            ("precision_basis",
             [1.0 if i // 49 == i % 49 else 0.0 for i in range(64 * 49)],
             [math.cos(i * 0.07) * 0.021 + 0.003 for i in range(64 * 64)])):
        a_path, b_path = inputs_dir / f"{name}-a.npy", inputs_dir / f"{name}-b.npy"
        correctness.write_f32_npy(a_path, (64, 49), a)
        correctness.write_f32_npy(b_path, (64, 64), b)
        cases.append((name, "device_native_f32_matmul_tail.mlir", [f"@{a_path}", f"@{b_path}"]))
    if args.stress_dynamic_range:
        a_path, b_path = inputs_dir / "stress-a.npy", inputs_dir / "stress-b.npy"
        correctness.write_f32_npy(a_path, (64, 49),
                                 [math.sin(i * 0.13) * (2.0 ** ((i % 17) - 8)) for i in range(64 * 49)])
        correctness.write_f32_npy(b_path, (64, 64),
                                 [math.cos(i * 0.07) * (2.0 ** ((i % 13) - 6)) for i in range(64 * 64)])
        cases.append(("precision_dynamic_range", "device_native_f32_matmul_tail.mlir", [f"@{a_path}", f"@{b_path}"]))
    results = {}
    if args.case:
        cases = [case for case in cases if case[0] == args.case]
        if not cases:
            parser.error(f"unknown case: {args.case}")
    for name, source, inputs in cases:
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        def run(command, log, env=None):
            with (out / log).open("w") as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
        source = root / "test/e2e" / source
        for backend, device in (("llvm-cpu", "local-task"), ("tenstorrent", "tenstorrent")):
            vmfb = out / f"{backend}.vmfb"
            command = [args.iree_compile, str(source), f"--iree-hal-target-backends={backend}", "-o", str(vmfb)]
            if backend == "llvm-cpu":
                command += ["--iree-llvmcpu-target-cpu=host"]
            else:
                command += ["--iree-tenstorrent-device-id=p100a"]
                if args.use_tensix:
                    command += ["--iree-tenstorrent-native-f32-use-tensix=true"]
            run(command, f"{backend}-compile.log")
            env = os.environ.copy()
            if device == "tenstorrent":
                env.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1", TT_IREE_PROFILE_OUTPUT=str(out / "profile"))
                env.setdefault("TT_METAL_RUNTIME_ROOT", str(root / "third_party/tt-metal"))
            functions = {
                "nonuniform_conv2d": "resnet50_conv2d",
                "nonuniform_runtime_affine": "device_native_f32_runtime_affine_matmul",
                "nonuniform_runtime_affine_tail": "device_native_f32_runtime_affine_tail",
            }
            function = ("device_native_f32_matmul_tail" if name.startswith("precision_")
                        else functions.get(name, name))
            command = [args.iree_run_module, f"--device={device}", f"--module={vmfb}", f"--function={function}", f"--output=@{out / (backend + '.npy')}"]
            command += [f"--input={value}" for value in inputs]
            run(command, f"{backend}-run.log", env)
        metrics = correctness.compare_arrays(correctness.read_npy(out / "llvm-cpu.npy"), correctness.read_npy(out / "tenstorrent.npy"), 1e-5, 1e-5)
        (out / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
        assert metrics["passed"], f"{name}: CPU reference mismatch: {metrics}"
        reference_comparison = None
        if args.reference_dir:
            reference = correctness.read_npy(args.reference_dir / name / "tenstorrent.npy")
            actual = correctness.read_npy(out / "tenstorrent.npy")
            assert reference.shape == actual.shape and reference.descr == actual.descr, name
            mismatches = sum(struct.pack("<f", a) != struct.pack("<f", b)
                             for a, b in zip(reference.values, actual.values))
            reference_comparison = {"elements": len(actual.values), "bitwise_mismatches": mismatches}
            assert mismatches == 0, f"{name}: native baseline bit mismatch: {reference_comparison}"
        profile = json.loads((out / "profile/run_results.json").read_text())
        assert profile["dispatch_count"] > 0 and profile["all_dispatches_device_native"], profile
        with (out / "profile/ops_perf_results.csv").open() as stream:
            for record in csv.DictReader(stream):
                for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
                    assert int(record[f"IREE {stage} DURATION [ns]"]) == 0, record
        results[name] = {"comparison": metrics, "profile": profile,
                         "native_reference_comparison": reference_comparison}
        print(f"PASS: {name}, {profile['dispatch_count']} native dispatches", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
