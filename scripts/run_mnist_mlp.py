#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the trained, static-B1 BF16 MNIST MLP on Tenstorrent and IREE CPU.

Numerical comparisons are diagnostics, not a new tolerance policy. Strict
native profiling, tensor shape/encoding and completed sample coverage are gates.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from build_resnet50_precision_golden import (
    classifier_comparison, file_record, require_new_directory, write_json,
)
from export_resnet50_bf16 import verify_record
from run_ttiree_resnet50_bf16 import decode_bf16


def require_mnist_profile(directory, samples, engine):
    profile = json.loads((directory / "run_results.json").read_text())
    with (directory / "ops_perf_results.csv").open() as stream:
        records = list(csv.DictReader(stream))
    if samples < 1 or not profile["all_dispatches_device_native"]:
        raise ValueError("MNIST requires device-native numerical dispatches")
    if profile["dispatch_count"] != samples * 2 or len(records) != samples * 2:
        raise ValueError("MNIST requires exactly two complete FC dispatches per image")
    compute = ("BF16_HIFI4_FP32_DST_F32_EPILOGUE" if engine == "tensix" else
               "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING")
    for index, record in enumerate(records):
        geometry = tuple(int(record[f"IREE {key}"]) for key in ("M", "N", "K"))
        if geometry != ((1, 128, 784), (1, 10, 128))[index % 2]:
            raise ValueError("unexpected MNIST FC geometry")
        if (record["OP TYPE"] != "LINEAR" or
                record["IREE EXECUTION DOMAIN"] != "device_native" or
                record["INPUT_0_DATATYPE"] != "BF16" or record["OUTPUT_0_DATATYPE"] != "BF16" or
                record["IREE BUILTIN PROGRAM"] != ("12" if engine == "tensix" else "11") or
                record["IREE COMPUTE DATATYPE"] != compute):
            raise ValueError("MNIST precision/engine profile mismatch")
        for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
            if int(record[f"IREE {stage} DURATION [ns]"]) != 0:
                raise ValueError("host tensor staging within MNIST numerical dispatch")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iree-compile", type=Path, required=True)
    parser.add_argument("--session-runner", type=Path, required=True)
    parser.add_argument("--tt-metal-root", type=Path, required=True)
    parser.add_argument("--engine", choices=("tensix", "riscv"), default="tensix")
    args = parser.parse_args()
    import numpy as np

    source = args.prepared_dir.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["export_status"] != "completed" or manifest["batch"] != 1 or
            manifest["input_shape"] != [1, 1, 28, 28] or manifest["output_shape"] != [1, 10]):
        raise ValueError("requires a completed BF16 MNIST export")
    for artifact in (manifest["state_dict"], manifest["input"],
                     *manifest["reference"].values(), *manifest["artifacts"].values()):
        verify_record(artifact["path"], artifact)
    inputs = np.load(source / "input.npy")
    count = len(manifest["indices"])
    if inputs.shape != (count, 1, 28, 28) or len(set(manifest["indices"])) != count:
        raise ValueError("unexpected or duplicate held-out MNIST inputs")
    for index, value in enumerate(inputs):
        bits = np.ascontiguousarray(value, dtype="<f4").view("<u4")
        rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("<u2")
        if (source / "inputs" / f"{index:03d}.bf16.bin").read_bytes() != rounded.tobytes():
            raise ValueError("raw BF16 input differs from recorded preprocessing")
    require_new_directory(args.output_dir)
    out = args.output_dir.resolve()
    environment = os.environ.copy()
    for name in ("TT_METAL_SIMULATOR", "TT_METAL_DEVICE_PROFILER", "TT_IREE_CHECKPOINT_OUTPUT",
                 "TT_IREE_REQUIRE_DEVICE_NATIVE", "TT_IREE_PROFILE_OUTPUT"):
        environment.pop(name, None)
    environment["TT_METAL_RUNTIME_ROOT"] = str(args.tt_metal_root.resolve())
    result = dict(status="running", engine=args.engine, samples=count, batch=1,
                  prepared_manifest=file_record(source / "manifest.json"),
                  compiler=file_record(args.iree_compile), runner=file_record(args.session_runner),
                  script=file_record(__file__), commands=[], backends={})
    write_json(out / "results.json", result)

    def run(command, log, env=environment):
        result["commands"].append([str(value) for value in command])
        with log.open("w") as stream:
            before = time.perf_counter()
            subprocess.run(command, env=env, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=600)
            return time.perf_counter() - before

    try:
        labels = manifest["labels"]
        categories = [str(digit) for digit in range(10)]
        outputs = {name: np.load(source / f"pytorch_{name}_logits.npy") for name in ("fp32", "bf16")}
        for backend in ("llvm-cpu", "tenstorrent"):
            target = out / backend
            target.mkdir()
            command = [args.iree_compile, source / "mnist_input.mlirbc", "--compile-from=input",
                       f"--iree-hal-target-backends={backend}", "-o", target / "mnist.vmfb"]
            if backend == "tenstorrent":
                command.append("--iree-tenstorrent-device-id=p100a")
                if args.engine == "tensix":
                    command.append("--iree-tenstorrent-native-bf16-use-tensix=true")
            else:
                command.append("--iree-llvmcpu-target-cpu=host")
            compile_seconds = run(command, target / "compile.log")
            rows = [f"{source / 'inputs' / f'{index:03d}.bf16.bin'}\t{target / f'{index:03d}.bf16.bin'}\n"
                    for index in range(count)]
            (target / "inputs.tsv").write_text("".join(rows))
            run_env = environment.copy()
            if backend == "tenstorrent":
                run_env.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1", TT_IREE_PROFILE_OUTPUT=str(target / "profile"))
            process_seconds = run([args.session_runner, target / "mnist.vmfb", target / "inputs.tsv",
                                   target / "timings.csv", "tenstorrent" if backend == "tenstorrent" else "local-sync"],
                                  target / "run.log", run_env)
            actual = np.concatenate([decode_bf16(target / f"{index:03d}.bf16.bin", (1, 10))
                                     for index in range(count)])
            outputs[backend] = actual
            np.save(target / "logits.npy", actual)
            with (target / "timings.csv").open() as stream:
                timings = list(csv.DictReader(stream))
            if [int(row["sample"]) for row in timings] != list(range(count)):
                raise ValueError("incomplete or reordered session outputs")
            invocations = [float(row["invoke_seconds"]) for row in timings]
            result["backends"][backend] = dict(
                compile_seconds=compile_seconds, session_process_seconds=process_seconds,
                invoke_mean_seconds=statistics.mean(invocations),
                invoke_p50_seconds=statistics.median(invocations),
                invoke_after_first_mean_seconds=statistics.mean(invocations[1:]) if count > 1 else None,
                correct=int(np.sum(actual.argmax(1) == labels)), samples=count,
                logits=file_record(target / "logits.npy"))
            if backend == "tenstorrent":
                result["backends"][backend]["profile"] = require_mnist_profile(target / "profile", count, args.engine)
            write_json(out / "results.json", result)
        comparisons = {}
        for expected, actual in (("fp32", "tenstorrent"), ("bf16", "tenstorrent"),
                                 ("llvm-cpu", "tenstorrent"), ("bf16", "llvm-cpu")):
            name = f"{actual}_vs_{expected}"
            comparisons[name] = classifier_comparison(outputs[expected], outputs[actual], categories, labels)
        write_json(out / "comparisons.json", comparisons)
        predictions = [dict(sample=i, test_index=manifest["indices"][i], label=labels[i],
                            **{name: int(value[i].argmax()) for name, value in outputs.items()})
                       for i in range(count)]
        with (out / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        result.update(status="completed", comparisons={name: value["aggregate"] for name, value in comparisons.items()},
                      pytorch_correct={name: int(np.sum(outputs[name].argmax(1) == labels)) for name in ("fp32", "bf16")},
                      acceptance_scope="completed native B1 BF16 execution; numerical/accuracy values are diagnostics, not tolerance changes")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
