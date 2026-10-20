#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run real batch-one BF16 tensors on tt-iree and retain diagnostic comparisons.

No precision fallback or tolerance relaxation. Full-model metric comparisons
are diagnostics; deterministic kernel fixtures are the bitwise correctness gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import pathlib
import subprocess
import time

from build_resnet50_precision_golden import (
    error_metrics, file_record, require_new_directory, write_json,
)
from export_resnet50_bf16 import verify_record


def decode_bf16(filename, shape):
    import numpy as np
    raw = np.fromfile(filename, dtype="<u2")
    if raw.size != np.prod(shape):
        raise ValueError(f"BF16 output length does not match {shape}: {filename}")
    values = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite BF16 output: {filename}")
    return values


def require_native_profile(directory, use_tensix=False, sample_count=1):
    profile = json.loads((directory / "run_results.json").read_text())
    if sample_count < 1 or not profile["all_dispatches_device_native"] or profile["dispatch_count"] != 58 * sample_count:
        raise ValueError(f"expected {58 * sample_count} device-native ResNet dispatches: {profile}")
    with (directory / "ops_perf_results.csv").open() as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 58 * sample_count:
        raise ValueError("expected complete 58-dispatch profile")
    expected_ops = Counter(MATMUL=36, CONV2D=17, LINEAR=1, PADDED_COPY=1,
                           MAX_POOL_2D=1, GLOBAL_SUM=1, SCALAR_DIV=1)
    for begin in range(0, len(records), 58):
        if Counter(r["OP TYPE"] for r in records[begin:begin + 58]) != expected_ops:
            raise ValueError("unexpected per-image ResNet operation coverage")
    for record in records:
        if record["INPUT_0_DATATYPE"] != "BF16" or record["OUTPUT_0_DATATYPE"] != "BF16":
            raise ValueError("unexpected non-BF16 dispatch storage")
        if record["IREE EXECUTION DOMAIN"] != "device_native":
            raise ValueError("non-native execution domain in BF16 profile")
        matrix = record["OP TYPE"] in ("MATMUL", "CONV2D", "LINEAR")
        builtin = "12" if use_tensix and matrix else "11"
        if record["IREE BUILTIN PROGRAM"] != builtin:
            raise ValueError(f"unexpected BF16 compute routing: expected builtin {builtin}")
        expected_compute = ("BF16_HIFI4_FP32_DST_F32_EPILOGUE" if use_tensix and matrix
                            else "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING")
        if record["IREE COMPUTE DATATYPE"] != expected_compute:
            raise ValueError("BF16 compute profile does not match requested engine")
        for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
            if int(record[f"IREE {stage} DURATION [ns]"]) != 0:
                raise ValueError(f"host tensor staging detected: {stage}")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iree-compile", required=True, type=pathlib.Path)
    parser.add_argument("--iree-run-module", required=True, type=pathlib.Path)
    parser.add_argument("--session-runner", type=pathlib.Path,
                        help="reuse tt-iree-resnet-eval session across distinct inputs; final logits only")
    parser.add_argument("--tt-metal-root", required=True, type=pathlib.Path)
    parser.add_argument("--indices", default="0", help="distinct source image indices; no batch padding")
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--use-tensix", action="store_true",
                        help="opt into v18 BF16 HiFi4 matrix contractions; scalar dispatches/epilogues remain device RISC-V")
    parser.add_argument("--cpu-reference", action="store_true",
                        help="also compile/run identical BF16 IR on LLVM CPU; fail if that backend rejects it")
    args = parser.parse_args()
    if args.session_runner and (args.checkpoints or args.cpu_reference):
        parser.error("session runner currently supports final-logits-only hardware evaluation")
    import numpy as np
    import torch

    source = json.loads((args.export_dir / "manifest.json").read_text())
    if source["precision_policy"] != "bf16_module" or source["batch"] != 1:
        parser.error("requires the static batch-one BF16 export")
    golden = pathlib.Path(source["golden_dir"])
    golden_manifest = json.loads((golden / "manifest.json").read_text())
    reference_record = source.get("bf16_reference", golden_manifest["logits"]["bf16_module"])
    verify_record(golden / "bf16_module/logits.npy", reference_record)
    for key in ("state_dict", "input"):
        verify_record(source[key]["path"], source[key])
    for record in source["artifacts"].values():
        verify_record(record["path"], record)
    inputs = np.load(source["input"]["path"])
    indices = [int(i) for i in args.indices.split(",")]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(inputs):
        parser.error("indices must be distinct and within the golden dataset")
    require_new_directory(args.output_dir)
    out = args.output_dir.resolve()
    results = {"status": "running", "precision_policy": "bf16_module",
               "batch": 1, "matrix_engine": "tensix_hifi4" if args.use_tensix else "riscv_fp32",
               "indices": indices, "export": file_record(args.export_dir / "manifest.json"),
               "golden_dir": str(golden), "samples": [],
               "comparison_scope": "diagnostics, not a new BF16 tolerance gate"}
    write_json(out / "results.json", results)
    env = os.environ.copy()
    for variable in ("TT_METAL_SIMULATOR", "TT_IREE_REQUIRE_DEVICE_NATIVE",
                     "TT_IREE_PROFILE_OUTPUT", "TT_IREE_CHECKPOINT_OUTPUT"):
        env.pop(variable, None)
    env["TT_METAL_RUNTIME_ROOT"] = str(args.tt_metal_root.resolve())

    def run(command, filename, run_env=env, timeout=900):
        with filename.open("w") as stream:
            start = time.perf_counter()
            subprocess.run(command, env=run_env, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=timeout)
            return time.perf_counter() - start

    try:
        for backend in (("tenstorrent", "llvm-cpu") if args.cpu_reference else ("tenstorrent",)):
            command = [str(args.iree_compile.resolve()),
                       str((args.export_dir / "resnet50_input.mlirbc").resolve()),
                       "--compile-from=input", f"--iree-hal-target-backends={backend}",
                       "-o", str(out / f"{backend}.vmfb")]
            command += (["--iree-tenstorrent-device-id=p100a"] if backend == "tenstorrent"
                        else ["--iree-llvmcpu-target-cpu=host"])
            if backend == "tenstorrent" and args.use_tensix:
                command.append("--iree-tenstorrent-native-bf16-use-tensix=true")
            run(command, out / f"{backend}-compile.log")
        expected = np.load(golden / "bf16_module/logits.npy")
        actual_outputs, cpu_outputs = [], []
        session_timings, session_records = [], []
        if args.session_runner:
            rows = []
            for index in indices:
                sample = out / f"sample-{index:02d}"
                sample.mkdir()
                raw = torch.from_numpy(inputs[index:index + 1].copy()).to(torch.bfloat16).view(torch.uint16).numpy()
                raw.astype("<u2").tofile(sample / "input.bf16.bin")
                rows.append(f"{sample / 'input.bf16.bin'}\t{sample / 'tenstorrent.bf16.bin'}\n")
            (out / "inputs.tsv").write_text("".join(rows))
            run_env = env.copy()
            run_env.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1", TT_IREE_PROFILE_OUTPUT=str(out / "profile"))
            results["session_runner"] = file_record(args.session_runner)
            results["session_process_wall_seconds"] = run([
                str(args.session_runner.resolve()), str(out / "tenstorrent.vmfb"),
                str(out / "inputs.tsv"), str(out / "call_timings.csv"),
            ], out / "session-run.log", run_env, timeout=600 + 120 * len(indices))
            results["session_profile"] = require_native_profile(out / "profile", args.use_tensix, len(indices))
            with (out / "call_timings.csv").open() as stream:
                session_timings = list(csv.DictReader(stream))
            with (out / "profile/ops_perf_results.csv").open() as stream:
                session_records = list(csv.DictReader(stream))
            if len(session_timings) != len(indices) or [int(t["sample"]) for t in session_timings] != list(range(len(indices))):
                raise ValueError("session timings do not cover all distinct inputs in order")
            first_ops = [r["OP CODE"] for r in session_records[:58]]
            for begin in range(0, len(session_records), 58):
                if [r["OP CODE"] for r in session_records[begin:begin + 58]] != first_ops:
                    raise ValueError("ResNet dispatch sequence changed between session calls")
            results["timing_scope"] = "single session; call wall includes I/O; first call may include lazy initialization"
        for position, index in enumerate(indices):
            sample = out / f"sample-{index:02d}"
            if not args.session_runner:
                sample.mkdir()
                raw = torch.from_numpy(inputs[index:index + 1].copy()).to(torch.bfloat16).view(torch.uint16).numpy()
                raw.astype("<u2").tofile(sample / "input.bf16.bin")
            durations = {}
            backends = [("tenstorrent", "tenstorrent")]
            if args.cpu_reference:
                backends.insert(0, ("llvm-cpu", "local-task"))
            for backend, device in ([] if args.session_runner else backends):
                run_env = env.copy()
                if device == "tenstorrent":
                    run_env.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1",
                                   TT_IREE_PROFILE_OUTPUT=str(sample / "profile"))
                    if args.checkpoints:
                        run_env["TT_IREE_CHECKPOINT_OUTPUT"] = str(sample / "checkpoints")
                durations[backend] = run([
                    str(args.iree_run_module.resolve()), f"--device={device}",
                    f"--module={out / (backend + '.vmfb')}", "--function=main",
                    f"--input=1x3x224x224xbf16=@{sample / 'input.bf16.bin'}",
                    f"--output=@{sample / (backend + '.bf16.bin')}",
                ], sample / f"{backend}-run.log", run_env, timeout=600)
            actual = decode_bf16(sample / "tenstorrent.bf16.bin", (1, 1000))
            cpu = (decode_bf16(sample / "llvm-cpu.bf16.bin", (1, 1000))
                   if args.cpu_reference else None)
            np.save(sample / "output.npy", actual)
            if cpu is not None:
                np.save(sample / "cpu.npy", cpu)
            actual_outputs.append(actual)
            if cpu is not None:
                cpu_outputs.append(cpu)
            if args.session_runner:
                records = session_records[position * 58:(position + 1) * 58]
                profile = {"dispatch_count": 58, "all_dispatches_device_native": True,
                           "total_dispatch_duration_ns": sum(int(r["HOST DURATION [ns]"]) for r in records),
                           "total_enqueue_wait_duration_ns": sum(int(r["IREE ENQUEUE WAIT DURATION [ns]"]) for r in records),
                           "scope": "derived from 58 consecutive records in the shared session profile"}
            else:
                profile = require_native_profile(sample / "profile", args.use_tensix)
            results["samples"].append({
                "index": index, "host_process_wall_seconds": durations,
                "device_profile": profile,
                "vs_pytorch_bf16": error_metrics(expected[index:index + 1], actual),
                "vs_iree_cpu_same_ir": error_metrics(cpu, actual) if cpu is not None else None,
                "cpu_bf16_bitwise_mismatches": int(np.count_nonzero(cpu.view(np.uint32) != actual.view(np.uint32))) if cpu is not None else None,
                "pytorch_top1": int(expected[index].argmax()), "ttiree_top1": int(actual[0].argmax()),
                "input": file_record(sample / "input.bf16.bin"),
                "call_timings": session_timings[position] if args.session_runner else None,
            })
            write_json(out / "results.json", results)
            print(f"sample {index}: 58 native BF16 dispatches", flush=True)
        actual = np.concatenate(actual_outputs)
        cpu = np.concatenate(cpu_outputs) if cpu_outputs else None
        np.save(out / "logits.npy", actual)
        if cpu is not None:
            np.save(out / "cpu-logits.npy", cpu)
        results.update(status="completed", vs_pytorch_bf16=error_metrics(expected[indices], actual),
                       vs_iree_cpu_same_ir=error_metrics(cpu, actual) if cpu is not None else None,
                       top1_agreement=int(np.count_nonzero(expected[indices].argmax(1) == actual.argmax(1))),
                       sample_count=len(indices))
    except Exception as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "results.json", results)


if __name__ == "__main__":
    main()
