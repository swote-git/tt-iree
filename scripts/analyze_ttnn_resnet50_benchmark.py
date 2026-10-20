#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Summarize benchmark walls and same-core device intervals, without hardware.

Never subtract counters belonging to different cores. The sum of longest
same-core kernel intervals is a diagnostic proxy, not graph wall latency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import re

from benchmark_ttnn_resnet50 import timing_summary, write_timings
from build_resnet50_precision_golden import file_record, write_json


def read_kernel_intervals(filename):
    pending = defaultdict(deque)
    per_operation = defaultdict(list)
    risc_cycles = defaultdict(lambda: defaultdict(list))
    with filename.open(newline="") as stream:
        metadata = stream.readline()
        match = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+(?:\.\d+)?)", metadata)
        if not match or float(match[1]) <= 0:
            raise ValueError("device log has no positive clock frequency")
        ns_per_cycle = 1000 / float(match[1])
        for row in csv.DictReader(stream, skipinitialspace=True):
            zone = row["zone name"]
            if not zone.endswith("-KERNEL"):
                continue
            # Include timer and execution identity; repeated zones must not alias.
            key = tuple(row.get(field, "") for field in (
                "PCIe slot", "core_x", "core_y", "RISC processor type", "timer_id",
                "run host ID", "trace id", "trace id counter", "zone name"))
            cycle = int(row["time[cycles since reset]"])
            if row["type"] == "ZONE_START":
                pending[key].append(cycle)
            elif row["type"] == "ZONE_END":
                if not pending[key]:
                    raise ValueError(f"unmatched kernel end: {key}")
                start = pending[key].popleft()
                if cycle < start:
                    raise ValueError("negative same-core interval")
                operation = int(row["run host ID"])
                per_operation[operation].append((cycle - start) * ns_per_cycle)
                risc_cycles[operation][row["RISC processor type"]].append(cycle - start)
        if any(pending.values()):
            raise ValueError("unmatched kernel starts; device log may be truncated")
    if not per_operation:
        raise ValueError("no paired device kernel intervals")
    return ({op: max(values) for op, values in per_operation.items()},
            {op: {risc: max(values) * ns_per_cycle for risc, values in riscs.items()}
             for op, riscs in risc_cycles.items()}, float(match[1]))


def marked_batches(rows):
    active, batches = None, {}
    for row in rows:
        marker = re.fullmatch(r"benchmark_batch_(\d+)_(start|stop)", row["OP CODE"])
        if marker and row["OP TYPE"] == "signpost":
            index = int(marker[1])
            if marker[2] == "start":
                if active is not None or index in batches:
                    raise ValueError("nested or duplicate benchmark start")
                batches[index], active = [], index
            else:
                if active != index:
                    raise ValueError("unmatched benchmark stop")
                active = None
        elif active is not None and row["OP TYPE"] == "tt_dnn_device":
            batches[active].append(row)
    if active is not None or not batches or any(not rows for rows in batches.values()):
        raise ValueError("missing or empty marked benchmark batches")
    if sorted(batches) != list(range(len(batches))):
        raise ValueError("noncontiguous benchmark batch IDs")
    return batches


def analyze_device(artifacts, destination, expected_batches, batch_size):
    candidates = list(artifacts.glob("reports/**/ops_perf_results*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"expected one operation CSV in {artifacts}: {candidates}")
    ops = candidates[0]
    raw = ops.parent / "profile_log_device.csv"
    if not raw.exists():
        raw = artifacts / ".logs/profile_log_device.csv"
    durations, riscs, frequency = read_kernel_intervals(raw)
    with ops.open(newline="", encoding="utf-8-sig") as stream:
        batches = marked_batches(list(csv.DictReader(stream)))
    if len(batches) != expected_batches:
        raise ValueError("profile batch count differs from completed runner")
    rows, totals, breakdown = [], [], defaultdict(list)
    signatures = []
    for index, batch in sorted(batches.items()):
        signatures.append([row["OP CODE"] for row in batch])
        total = 0
        for ordinal, row in enumerate(batch):
            call = int(row["GLOBAL CALL COUNT"])
            if call not in durations:
                raise ValueError(f"no raw interval for operation {call}")
            nanoseconds = durations[call]
            total += nanoseconds
            breakdown[row["OP CODE"]].append(nanoseconds)
            rows.append(dict(batch=index, ordinal=ordinal, operation=row["OP CODE"],
                             global_call_count=call, core_count=row["CORE COUNT"],
                             longest_same_core_kernel_ns=nanoseconds,
                             upstream_raw_device_kernel_ns=row.get("DEVICE KERNEL DURATION [ns]", ""),
                             risc_maxima_ns=json.dumps(riscs[call], sort_keys=True)))
        totals.append(total / 1e9)
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError("profile operation inventory changed between batches")
    destination.mkdir(exist_ok=True)
    write_timings(destination / "operations.csv", rows)
    table = [dict(operation=name, calls_per_batch=len(values) // len(batches),
                  mean_sum_ms_per_batch=sum(values) / len(batches) / 1e6)
             for name, values in breakdown.items()]
    write_timings(destination / "operation_breakdown.csv", table)
    metric = timing_summary(totals, batch_size)
    metric["kernel_interval_reciprocal_images_per_second_not_wall_throughput"] = metric.pop("images_per_second")
    result = dict(
        scope="sum of per-operation longest same-core/RISC kernel intervals; excludes launch gaps and cross-core skew",
        graph_wall_latency=False, clock_mhz=frequency,
        operations_per_batch=len(signatures[0]), summary=metric,
        sources=[file_record(ops), file_record(raw)], operation_breakdown=table)
    write_json(destination / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    results, table = {}, []
    for precision in ("bf8_lofi", "bf16_hifi4"):
        hashes = set()
        for mode in ("plain", "trace1", "trace2"):
            name = f"{precision}-{mode}"
            result = json.loads((root / name / "results.json").read_text())
            if result["status"] != "completed" or result["device_profiler_enabled"]:
                raise ValueError(f"invalid timing result: {name}")
            if result["accuracy"]["bf16_module"]["samples"] != 112:
                raise ValueError("incomplete accuracy collection")
            if file_record(root / name / "logits.npy")["sha256"] != result["logits"]["sha256"]:
                raise ValueError(f"recorded logits changed: {name}")
            hashes.add(result["logits"]["sha256"])
            results[name] = result
            for scope, metric in result["timings"].items():
                table.append(dict(case=name, scope=scope, batch_size=result["batch"],
                                  mean_batch_ms=metric["mean_batch_ms"],
                                  p50_batch_ms=metric["p50_batch_ms"],
                                  p95_batch_ms=metric["p95_batch_ms"],
                                  images_per_second=metric["images_per_second"]))
        if len(hashes) != 1:
            raise ValueError(f"trace/CQ logits are not bitwise identical: {precision}")
    for field in ("golden_manifest", "input_sha256", "state_dict_sha256", "tt_metal_commit", "device_grid"):
        if len({json.dumps(result[field], sort_keys=True) for result in results.values()}) != 1:
            raise ValueError(f"benchmark cases disagree on {field}")
    profiles = {}
    for precision in ("bf8_lofi", "bf16_hifi4"):
        name = f"profile-{precision}"
        profile = json.loads((root / name / "results.json").read_text())
        if profile["status"] != "completed" or not profile["device_profiler_enabled"]:
            raise ValueError(f"invalid device profile: {name}")
        if profile["logits"]["sha256"] != results[f"{precision}-plain"]["logits"]["sha256"]:
            raise ValueError(f"profiling changed numerical results: {precision}")
        profiles[precision] = analyze_device(
            root / f"{name}-artifacts", root / f"{name}-analysis",
            profile["profile_measurement_batches"], profile["batch"])
    summary = dict(
        analysis_date="2026-10-20", timing_table=table, device_profiles=profiles,
        logits_bitwise_identical_across_trace_and_cq=True,
        accuracy={name: result["accuracy"]["bf16_module"] for name, result in results.items()},
        provenance=[file_record(root / name / "results.json") for name in results],
        benchmark_source_as_run=file_record(root / "benchmark-as-run.py"),
        analyzer=file_record(__file__))
    write_timings(root / "timing-summary.csv", table)
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
