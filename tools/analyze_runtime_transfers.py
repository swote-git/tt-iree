#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Attribute blocking HAL transfer time without adding nested stages twice."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("interval ends before it starts")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in union_intervals(intervals))


def intersection_duration(left, right):
    left, right = union_intervals(left), union_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def read_csv(path, required):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")
        return list(reader)


def analyze(profile_dir, call_timings=None):
    transfers = read_csv(profile_dir / "transfer_perf_results.csv", [
        "OPERATION", "INVOCATION ID", "WITHIN DISPATCH", "INVOCATION PHASE",
        "HOST START TS [ns]", "HOST END TS [ns]", "HOST DURATION [ns]",
        "READFROMBUFFER BYTES", "WRITETOBUFFER BYTES", "STATUS CODE", "OTHER [ns]",
    ])
    operations = read_csv(profile_dir / "ops_perf_results.csv", [
        "IREE INVOCATION ID", "HOST START TS [ns]", "HOST END TS [ns]",
    ])
    timings = {}
    if call_timings:
        for row in read_csv(call_timings, ["sample", "invoke_seconds"]):
            invocation = int(row["sample"]) + 1
            if invocation in timings:
                raise ValueError(f"duplicate timing invocation {invocation}")
            timings[invocation] = round(float(row["invoke_seconds"]) * 1e9)

    by_invocation = defaultdict(list)
    dispatches = defaultdict(list)
    for row in transfers:
        start, end = int(row["HOST START TS [ns]"]), int(row["HOST END TS [ns]"])
        if end - start != int(row["HOST DURATION [ns]"]):
            raise ValueError("transfer duration does not match timestamps")
        exclusive = sum(int(value) for key, value in row.items()
                        if key and ("EXCLUSIVE [ns]" in key or key == "OTHER [ns]"))
        if exclusive != end - start:
            raise ValueError("exclusive transfer stages do not sum to root duration")
        if row["WITHIN DISPATCH"] not in ("0", "1"):
            raise ValueError("invalid WITHIN DISPATCH flag")
        if row["INVOCATION PHASE"] not in ("input", "invoke", "output", "unscoped"):
            raise ValueError("invalid invocation phase")
        by_invocation[int(row["INVOCATION ID"])].append(row)
    for row in operations:
        dispatches[int(row["IREE INVOCATION ID"])].append(
            (int(row["HOST START TS [ns]"]), int(row["HOST END TS [ns]"])))

    invocations = []
    for invocation in sorted(set(by_invocation) | set(dispatches) | set(timings)):
        rows = by_invocation[invocation]
        invoke_rows = [row for row in rows if row["INVOCATION PHASE"] == "invoke"]
        outside = [(int(row["HOST START TS [ns]"]), int(row["HOST END TS [ns]"]))
                   for row in invoke_rows if row["WITHIN DISPATCH"] == "0"]
        dispatch = dispatches[invocation]
        outside_ns = duration(outside)
        overlap_ns = intersection_duration(outside, dispatch)
        dispatch_ns = duration(dispatch)
        accounted_ns = duration(outside + dispatch)
        operation_totals = defaultdict(lambda: {"calls": 0, "host_duration_ns": 0})
        for row in invoke_rows:
            totals = operation_totals[row["OPERATION"]]
            totals["calls"] += 1
            totals["host_duration_ns"] += int(row["HOST DURATION [ns]"])
        result = {
            "invocation_id": invocation,
            "transfer_count_all_phases": len(rows),
            "transfer_count_invoke": len(invoke_rows),
            "transfer_failures_all_phases": sum(int(row["STATUS CODE"]) != 0 for row in rows),
            "dispatch_count": len(dispatch),
            "dispatch_wall_union_ns": dispatch_ns,
            "invoke_transfer_outside_dispatch_wall_union_ns": outside_ns,
            "outside_transfer_dispatch_overlap_ns": overlap_ns,
            "accounted_invoke_wall_union_ns": accounted_ns,
            "readfrombuffer_bytes_invoke": sum(int(row["READFROMBUFFER BYTES"]) for row in invoke_rows),
            "writetobuffer_bytes_invoke": sum(int(row["WRITETOBUFFER BYTES"]) for row in invoke_rows),
            "device_fill_bytes_invoke": sum(int(row.get("DEVICE FILL BYTES", 0)) for row in invoke_rows),
            "invoke_operations": dict(operation_totals),
        }
        if dispatch:
            span = [(min(start for start, _ in dispatch), max(end for _, end in dispatch))]
            result["internal_dispatch_gap_ns"] = duration(span) - dispatch_ns
            result["internal_gap_transfer_coverage_ns"] = (
                intersection_duration(outside, span) - overlap_ns)
        if invocation in timings:
            result["invoke_ns"] = timings[invocation]
            # Do not clamp a negative residual: it exposes clock/attribution errors.
            result["unattributed_invoke_ns"] = timings[invocation] - accounted_ns
        invocations.append(result)
    return {
        "schema_version": 1,
        "timing": "host steady_clock; interval unions, not device kernel timing",
        "byte_semantics": "completed ReadFromBuffer/WriteToBuffer payloads in tt_buffer.cc; not PCIe counters or all device traffic",
        "invocation_semantics": "eval sample+1; 0 is setup/unmarked; input/output excluded from invoke attribution",
        "transfer_count": len(transfers),
        "operation_counts_all_phases": dict(Counter(row["OPERATION"] for row in transfers)),
        "invocations": invocations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--call-timings", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.profile_dir, args.call_timings)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2) + "\n"
    if args.output_json:
        args.output_json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
