#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalizes and summarizes TT-NN, TT-MLIR, and tt-iree perf CSVs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


NORMALIZED_FIELDS = [
    "implementation",
    "op_code",
    "op_type",
    "global_call_count",
    "core_count",
    "host_duration_ns",
    "device_kernel_duration_ns",
    "iree_enqueue_wait_duration_ns",
    "m",
    "n",
    "k",
    "flop_count",
    "iree_input_staging_duration_ns",
    "iree_buffer_allocation_duration_ns",
    "iree_pack_duration_ns",
    "iree_upload_duration_ns",
    "iree_program_setup_duration_ns",
    "iree_download_duration_ns",
    "iree_unpack_duration_ns",
    "iree_output_staging_duration_ns",
    "iree_unattributed_host_duration_ns",
]

IREE_STAGE_FIELDS = [
    ("input staging", "iree_input_staging_duration_ns"),
    ("buffer allocation", "iree_buffer_allocation_duration_ns"),
    ("pack", "iree_pack_duration_ns"),
    ("upload", "iree_upload_duration_ns"),
    ("program setup", "iree_program_setup_duration_ns"),
    ("enqueue and wait", "iree_enqueue_wait_duration_ns"),
    ("download", "iree_download_duration_ns"),
    ("unpack", "iree_unpack_duration_ns"),
    ("output staging", "iree_output_staging_duration_ns"),
]


def parse_integer(value: str) -> int:
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def parse_optional_integer(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def normalize_rows(label: str, path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or "OP CODE" not in reader.fieldnames:
            raise ValueError(f"{path}: missing required 'OP CODE' column")
        rows = []
        for source in reader:
            row = {
                "implementation": label,
                "op_code": source.get("OP CODE", ""),
                "op_type": source.get("OP TYPE", "") or "UNKNOWN",
                "global_call_count": parse_integer(
                    source.get("GLOBAL CALL COUNT", "")
                ),
                "core_count": parse_integer(source.get("CORE COUNT", "")),
                "host_duration_ns": parse_optional_integer(
                    source.get("HOST DURATION [ns]", "")
                ),
                "device_kernel_duration_ns": parse_optional_integer(
                    source.get("DEVICE KERNEL DURATION [ns]", "")
                ),
                "iree_enqueue_wait_duration_ns": parse_optional_integer(
                    source.get("IREE ENQUEUE WAIT DURATION [ns]", "")
                ),
                "m": parse_integer(source.get("IREE M", "")),
                "n": parse_integer(source.get("IREE N", "")),
                "k": parse_integer(source.get("IREE K", "")),
                "flop_count": parse_integer(source.get("IREE FLOP COUNT", "")),
                "iree_input_staging_duration_ns": parse_optional_integer(
                    source.get("IREE INPUT STAGING DURATION [ns]", "")
                ),
                "iree_buffer_allocation_duration_ns": parse_optional_integer(
                    source.get("IREE BUFFER ALLOCATION DURATION [ns]", "")
                ),
                "iree_pack_duration_ns": parse_optional_integer(
                    source.get("IREE PACK DURATION [ns]", "")
                ),
                "iree_upload_duration_ns": parse_optional_integer(
                    source.get("IREE UPLOAD DURATION [ns]", "")
                ),
                "iree_program_setup_duration_ns": parse_optional_integer(
                    source.get("IREE PROGRAM SETUP DURATION [ns]", "")
                ),
                "iree_download_duration_ns": parse_optional_integer(
                    source.get("IREE DOWNLOAD DURATION [ns]", "")
                ),
                "iree_unpack_duration_ns": parse_optional_integer(
                    source.get("IREE UNPACK DURATION [ns]", "")
                ),
                "iree_output_staging_duration_ns": parse_optional_integer(
                    source.get("IREE OUTPUT STAGING DURATION [ns]", "")
                ),
            }
            measured_stage_ns = [
                int(row[field])
                for _, field in IREE_STAGE_FIELDS
                if row[field] is not None
            ]
            host_duration_ns = row["host_duration_ns"]
            row["iree_unattributed_host_duration_ns"] = (
                max(0, int(host_duration_ns) - sum(measured_stage_ns))
                if host_duration_ns is not None and measured_stage_ns
                else None
            )
            rows.append(row)
    return rows


def format_ms(duration_ns: int) -> str:
    return f"{duration_ns / 1_000_000.0:.3f}"


def sum_measurements(
    rows: Iterable[Dict[str, object]], field: str
) -> Optional[int]:
    measurements = [
        int(row[field]) for row in rows if row.get(field) is not None
    ]
    return sum(measurements) if measurements else None


def format_optional_ms(duration_ns: Optional[int]) -> str:
    return "n/a" if duration_ns is None else format_ms(duration_ns)


def print_summary(
    rows: Iterable[Dict[str, object]], top_count: int
) -> None:
    rows = list(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["implementation"])].append(row)

    print(
        "IMPLEMENTATION           CALLS  HOST TOTAL(ms)  DEVICE KERNEL(ms)  "
        "IREE ENQUEUE(ms)"
    )
    for implementation, implementation_rows in grouped.items():
        host_ns = sum_measurements(implementation_rows, "host_duration_ns")
        device_ns = sum_measurements(
            implementation_rows, "device_kernel_duration_ns"
        )
        enqueue_ns = sum_measurements(
            implementation_rows, "iree_enqueue_wait_duration_ns"
        )
        print(
            f"{implementation:<24} {len(implementation_rows):>5}  "
            f"{format_optional_ms(host_ns):>14}  "
            f"{format_optional_ms(device_ns):>17}  "
            f"{format_optional_ms(enqueue_ns):>16}"
        )

    print("\nPer operation type")
    print("IMPLEMENTATION           OP TYPE          CALLS  HOST TOTAL(ms)")
    by_type = defaultdict(list)
    for row in rows:
        by_type[(str(row["implementation"]), str(row["op_type"]))].append(row)
    for (implementation, op_type), op_rows in sorted(by_type.items()):
        host_ns = sum_measurements(op_rows, "host_duration_ns")
        print(
            f"{implementation:<24} {op_type:<16} {len(op_rows):>5}  "
            f"{format_optional_ms(host_ns):>14}"
        )

    print("\nIREE stage totals")
    print("IMPLEMENTATION           STAGE                 TOTAL(ms)  % HOST")
    for implementation, implementation_rows in grouped.items():
        host_ns = sum_measurements(implementation_rows, "host_duration_ns")
        for stage_name, field in IREE_STAGE_FIELDS:
            stage_ns = sum_measurements(implementation_rows, field)
            if stage_ns is None:
                continue
            host_percent = (
                "n/a"
                if not host_ns
                else f"{100.0 * stage_ns / host_ns:.3f}"
            )
            print(
                f"{implementation:<24} {stage_name:<20} "
                f"{format_ms(stage_ns):>10}  {host_percent:>6}"
            )
        unattributed_ns = sum_measurements(
            implementation_rows, "iree_unattributed_host_duration_ns"
        )
        if unattributed_ns is not None:
            host_percent = (
                "n/a"
                if not host_ns
                else f"{100.0 * unattributed_ns / host_ns:.3f}"
            )
            print(
                f"{implementation:<24} {'unattributed host':<20} "
                f"{format_ms(unattributed_ns):>10}  {host_percent:>6}"
            )

    if top_count <= 0:
        return
    print(f"\nTop {top_count} operations by host duration")
    print("IMPLEMENTATION           CALL  OP TYPE          HOST(ms)  OP CODE")
    timed_rows = [row for row in rows if row["host_duration_ns"] is not None]
    timed_rows.sort(key=lambda row: int(row["host_duration_ns"]), reverse=True)
    for row in timed_rows[:top_count]:
        print(
            f"{str(row['implementation']):<24} "
            f"{int(row['global_call_count']):>4}  "
            f"{str(row['op_type']):<16} "
            f"{format_ms(int(row['host_duration_ns'])):>9}  "
            f"{row['op_code']}"
        )


def parse_profile(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("profile must use LABEL=/path/to/file.csv")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare TT-NN, TT-MLIR, and tt-iree ops perf CSV files"
    )
    parser.add_argument(
        "profiles",
        metavar="LABEL=CSV",
        nargs="+",
        type=parse_profile,
        help="implementation label and ops_perf_results.csv path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for normalized operation rows",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of operations to rank by host duration (default: 10)",
    )
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for label, path in args.profiles:
        rows.extend(normalize_rows(label, path))
    print_summary(rows, max(args.top, 0))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=NORMALIZED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
