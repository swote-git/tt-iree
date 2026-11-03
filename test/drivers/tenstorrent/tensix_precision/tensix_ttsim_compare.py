#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Replay archived precision probes without modifying their host/device code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np


ARTIFACTS = (
    "lhs.f32", "rhs.f32", "lhs-packed.bin", "rhs-packed.bin",
    "reader.cpp", "writer.cpp", "compute.cpp",
)
CONTRACT_FIELDS = (
    "format", "operation", "math_fidelity", "repeats", "k_slice_width",
    "fp32_dest_acc_en", "math_approx_mode", "f32_unpack_to_dst",
    "dst_check_enabled", "dst_check_bits", "dst_prefill_enabled",
    "dst_prefill_bits", "fill_bits", "architecture", "core", "tile_shape",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_command(probe, reference, output):
    meta = json.loads((reference / "summary.json").read_text())
    if meta.get("rhs_component", "full") != "full":
        raise ValueError("Replay requires an unmodified RHS component")
    command = [
        str(probe), "--case", "custom", "--format", meta["format"],
        "--op", meta["operation"], "--lhs-file", str(reference / "lhs.f32"),
        "--rhs-file", str(reference / "rhs.f32"), "--fidelity",
        str(meta["math_fidelity"]), "--repeats", str(meta["repeats"]),
        "--k", str(meta["logical_k"]), "--k-slice-width",
        str(meta["k_slice_width"]), "--fill-bits", f"{meta['fill_bits']:08x}",
        "--output-dir", str(output),
    ]
    if meta.get("f32_unpack_to_dst"):
        command.append("--f32-unpack-to-dst")
    for enabled, option, value in (
        ("dst_check_enabled", "--dst-check-bits", "dst_check_bits"),
        ("dst_prefill_enabled", "--dst-prefill-bits", "dst_prefill_bits"),
    ):
        if meta.get(enabled):
            command.extend((option, f"{meta[value]:08x}"))
    return command


def compare(reference, output):
    original = json.loads((reference / "summary.json").read_text())
    replayed = json.loads((output / "summary.json").read_text())
    changed_fields = [
        key for key in CONTRACT_FIELDS
        if original.get(key, False if key.endswith("enabled") else 0)
        != replayed.get(key, False if key.endswith("enabled") else 0)
    ]
    changed_artifacts = [
        name for name in ARTIFACTS
        if digest(reference / name) != digest(output / name)
    ]
    expected = np.fromfile(reference / "actual.f32", dtype=np.float32)
    actual = np.fromfile(output / "actual.f32", dtype=np.float32)
    if actual.shape != expected.shape or actual.size != original["repeats"] * 1024:
        raise ValueError("Output shape does not match the archived repeated tiles")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    mismatches = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    unstable = int(np.count_nonzero(
        actual.view(np.uint32).reshape(-1, 1024)
        != actual.view(np.uint32)[:1024]
    ))
    status = "match"
    if changed_fields or changed_artifacts:
        status = "contract-mismatch"
    elif not finite or unstable:
        status = "nonfinite-or-unstable"
    elif mismatches:
        status = "numerical-mismatch"
    return {
        "status": status,
        "changed_contract_fields": changed_fields,
        "changed_input_or_kernel_artifacts": changed_artifacts,
        "compared_elements": int(actual.size),
        "bitwise_mismatches": mismatches,
        "max_abs_difference": float(difference.max()) if finite else None,
        "nonfinite": not finite,
        "different_from_first_repeat": unstable,
        "mathematical_reference_mismatches": replayed["bitwise_mismatches"],
        "note": "Match means equality to archived device output, not IEEE FP32 correctness.",
    }


def environment(metal_root, cache, simulator, dispatch):
    env = os.environ.copy()
    # Backend selection belongs to the launcher; the probe has no simulator branch.
    for key in (
        "TT_METAL_MOCK_DEVICE", "TT_METAL_SIMULATOR", "TT_UMD_SIMULATOR",
        "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_DISABLE_SFPLOADMACRO",
        "TT_METAL_SIMULATOR_DIRECT_TENSOR_WRITES", "TT_SIMULATOR_DRAM_TELEPORT",
    ):
        env.pop(key, None)
    env["TT_METAL_HOME"] = env["TT_METAL_RUNTIME_ROOT"] = str(metal_root)
    env["TT_METAL_CACHE"] = str(cache)
    if simulator:
        env["TT_METAL_SIMULATOR"] = str(simulator)
    if dispatch == "slow":
        env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    return env


def execute(command, env, log, timeout):
    with log.open("x") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return {"returncode": process.wait(timeout=timeout), "timed_out": False}
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            return {"returncode": process.returncode, "timed_out": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--metal-root", type=Path, required=True)
    parser.add_argument("--simulator", type=Path)
    parser.add_argument("--dispatch", choices=("slow", "fast"), default="slow")
    parser.add_argument("--measurement", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    probe, reference_probe = args.probe.resolve(), args.reference_probe.resolve()
    if digest(probe) != digest(reference_probe):
        parser.error("The replay and reference host binaries must be identical")
    metal_root = args.metal_root.resolve()
    simulator = args.simulator.resolve() if args.simulator else None
    if simulator:
        if not simulator.is_file() or not (simulator.parent / "soc_descriptor.yaml").is_file():
            parser.error("Simulator and adjacent soc_descriptor.yaml are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = environment(metal_root, output / "jit-cache", simulator, args.dispatch)
    metadata = {
        "probe_sha256": digest(probe),
        "reference_probe_sha256": digest(reference_probe),
        "simulator": str(simulator) if simulator else None,
        "simulator_sha256": digest(simulator) if simulator else None,
        "soc_descriptor_sha256": digest(simulator.parent / "soc_descriptor.yaml") if simulator else None,
        "metal_root": str(metal_root), "dispatch": args.dispatch,
        "environment_overrides": {key: value for key, value in env.items() if key.startswith("TT_")},
        "measurements": [],
    }
    manifest = output / "comparison.json"
    for index, reference in enumerate(args.measurement):
        reference = reference.resolve()
        destination = output / f"{index:03d}-{reference.name}"
        command = replay_command(probe, reference, destination)
        record = {
            "reference": str(reference), "output": str(destination), "command": command,
            "reference_actual_sha256": digest(reference / "actual.f32"),
        }
        metadata["measurements"].append(record)
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        start = time.monotonic()
        execution = execute(command, env, Path(str(destination) + ".log"), args.timeout)
        record.update(execution)
        record["elapsed_seconds"] = time.monotonic() - start
        if execution["timed_out"]:
            record["status"] = "timeout"
        elif execution["returncode"] != 0:
            record["status"] = "execution-failed"
        else:
            try:
                record.update(compare(reference, destination))
            except (OSError, ValueError, KeyError) as error:
                record.update(status="artifact-error", error=str(error))
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{index + 1}/{len(args.measurement)} {reference.name}: {record['status']}", flush=True)
        if record["status"] in ("timeout", "execution-failed", "contract-mismatch", "artifact-error"):
            break
    failures = sum(row.get("status") != "match" for row in metadata["measurements"])
    metadata["requested_measurements"] = len(args.measurement)
    metadata["completed_measurements"] = len(metadata["measurements"])
    metadata["all_match"] = not failures and len(metadata["measurements"]) == len(args.measurement)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return 0 if metadata["all_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
