#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds and checks a CPU golden for the external ResNet50 artifact."""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
import pathlib
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


NPY_MAGIC = b"\x93NUMPY"
DEFAULT_SHAPE = (1, 3, 224, 224)


@dataclass(frozen=True)
class NpyArray:
    shape: tuple[int, ...]
    descr: str
    values: tuple[int | float, ...]


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError(f"negative NPY dimension: {dimension}")
        count *= dimension
    return count


def _dtype_format(descr: str) -> tuple[str, int]:
    match = re.fullmatch(r"([<>=|])([fiu])(\d+)", descr)
    if not match:
        raise ValueError(f"unsupported NPY dtype: {descr}")
    byte_order, kind, width_text = match.groups()
    width = int(width_text)
    formats = {
        ("f", 2): "e",
        ("f", 4): "f",
        ("f", 8): "d",
        ("i", 1): "b",
        ("i", 2): "h",
        ("i", 4): "i",
        ("i", 8): "q",
        ("u", 1): "B",
        ("u", 2): "H",
        ("u", 4): "I",
        ("u", 8): "Q",
    }
    format_code = formats.get((kind, width))
    if format_code is None:
        raise ValueError(f"unsupported NPY dtype: {descr}")
    if byte_order == "|" and width != 1:
        raise ValueError(f"invalid byte-order-independent dtype: {descr}")
    struct_order = "=" if byte_order in ("=", "|") else byte_order
    return struct_order + format_code, width


def read_npy(path: pathlib.Path) -> NpyArray:
    with path.open("rb") as file:
        if file.read(len(NPY_MAGIC)) != NPY_MAGIC:
            raise ValueError(f"not an NPY file: {path}")
        version = file.read(2)
        if len(version) != 2:
            raise ValueError(f"truncated NPY version: {path}")
        major, _minor = version
        if major == 1:
            header_size_data = file.read(2)
            header_size_format = "<H"
            header_encoding = "latin1"
        elif major in (2, 3):
            header_size_data = file.read(4)
            header_size_format = "<I"
            header_encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise ValueError(f"unsupported NPY version {major}: {path}")
        if len(header_size_data) != struct.calcsize(header_size_format):
            raise ValueError(f"truncated NPY header size: {path}")
        header_size = struct.unpack(header_size_format, header_size_data)[0]
        header_data = file.read(header_size)
        if len(header_data) != header_size:
            raise ValueError(f"truncated NPY header: {path}")
        try:
            header = ast.literal_eval(header_data.decode(header_encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"invalid NPY header in {path}: {error}") from error
        if not isinstance(header, dict):
            raise ValueError(f"invalid NPY header dictionary: {path}")
        if header.get("fortran_order") is not False:
            raise ValueError(f"Fortran-order NPY arrays are unsupported: {path}")
        shape = header.get("shape")
        descr = header.get("descr")
        if not isinstance(shape, tuple) or not all(
            isinstance(value, int) for value in shape
        ):
            raise ValueError(f"invalid NPY shape: {path}")
        if not isinstance(descr, str):
            raise ValueError(f"invalid NPY dtype: {path}")
        element_format, element_size = _dtype_format(descr)
        element_count = _element_count(shape)
        payload = file.read()
        expected_size = element_count * element_size
        if len(payload) != expected_size:
            raise ValueError(
                f"NPY payload size mismatch in {path}: "
                f"expected {expected_size}, got {len(payload)}"
            )
        values = struct.unpack(f"{element_format[0]}{element_count}{element_format[1]}", payload)
        return NpyArray(shape=shape, descr=descr, values=values)


def write_f32_npy(
    path: pathlib.Path, shape: Sequence[int], values: Sequence[float] | None = None
) -> None:
    shape = tuple(shape)
    element_count = _element_count(shape)
    if values is not None and len(values) != element_count:
        raise ValueError(
            f"value count mismatch: expected {element_count}, got {len(values)}"
        )
    shape_text = repr(shape)
    if len(shape) == 1:
        shape_text = f"({shape[0]},)"
    header = (
        "{'descr': '<f4', 'fortran_order': False, "
        f"'shape': {shape_text}, }}"
    ).encode("latin1")
    prefix_size = len(NPY_MAGIC) + 2 + 2
    padding_size = (64 - ((prefix_size + len(header) + 1) % 64)) % 64
    header += b" " * padding_size + b"\n"
    if len(header) > 0xFFFF:
        raise ValueError("NPY v1 header is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(NPY_MAGIC)
        file.write(bytes((1, 0)))
        file.write(struct.pack("<H", len(header)))
        file.write(header)
        if values is None:
            file.write(b"\x00" * (element_count * 4))
        else:
            file.write(struct.pack(f"<{element_count}f", *values))


def compare_arrays(
    expected: NpyArray, actual: NpyArray, atol: float, rtol: float
) -> dict[str, object]:
    result: dict[str, object] = {
        "passed": False,
        "atol": atol,
        "rtol": rtol,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_dtype": expected.descr,
        "actual_dtype": actual.descr,
    }
    if expected.shape != actual.shape:
        result["failure"] = "shape mismatch"
        return result
    if expected.descr != actual.descr:
        result["failure"] = "dtype mismatch"
        return result

    expected_values = tuple(float(value) for value in expected.values)
    actual_values = tuple(float(value) for value in actual.values)
    expected_nonfinite = sum(not math.isfinite(value) for value in expected_values)
    actual_nonfinite = sum(not math.isfinite(value) for value in actual_values)
    result["element_count"] = len(expected_values)
    result["expected_nonfinite_count"] = expected_nonfinite
    result["actual_nonfinite_count"] = actual_nonfinite
    if expected_nonfinite or actual_nonfinite:
        result["failure"] = "non-finite values"
        return result

    differences = [
        abs(expected_value - actual_value)
        for expected_value, actual_value in zip(expected_values, actual_values)
    ]
    mismatches = [
        index
        for index, (expected_value, difference) in enumerate(
            zip(expected_values, differences)
        )
        if difference > atol + rtol * abs(expected_value)
    ]
    max_index = max(range(len(differences)), key=differences.__getitem__, default=0)
    max_difference = differences[max_index] if differences else 0.0
    mean_difference = sum(differences) / len(differences) if differences else 0.0
    rmse = (
        math.sqrt(sum(value * value for value in differences) / len(differences))
        if differences
        else 0.0
    )
    relative_errors = [
        difference / abs(expected_value)
        for expected_value, difference in zip(expected_values, differences)
        if expected_value != 0.0
    ]
    has_infinite_relative_error = any(
        expected_value == 0.0 and difference != 0.0
        for expected_value, difference in zip(expected_values, differences)
    )

    dot_product = sum(
        expected_value * actual_value
        for expected_value, actual_value in zip(expected_values, actual_values)
    )
    expected_norm = math.sqrt(sum(value * value for value in expected_values))
    actual_norm = math.sqrt(sum(value * value for value in actual_values))
    if expected_norm == 0.0 or actual_norm == 0.0:
        cosine_similarity = 1.0 if expected_values == actual_values else 0.0
    else:
        cosine_similarity = dot_product / (expected_norm * actual_norm)

    top_count = min(5, len(expected_values))
    expected_top = sorted(
        range(len(expected_values)), key=expected_values.__getitem__, reverse=True
    )[:top_count]
    actual_top = sorted(
        range(len(actual_values)), key=actual_values.__getitem__, reverse=True
    )[:top_count]
    result.update(
        {
            "passed": not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch_index": mismatches[0] if mismatches else None,
            "max_absolute_error": max_difference,
            "max_absolute_error_index": max_index,
            "expected_at_max_error": (
                expected_values[max_index] if expected_values else None
            ),
            "actual_at_max_error": actual_values[max_index] if actual_values else None,
            "mean_absolute_error": mean_difference,
            "rmse": rmse,
            "max_relative_error": max(relative_errors, default=0.0),
            "has_infinite_relative_error": has_infinite_relative_error,
            "cosine_similarity": cosine_similarity,
            "expected_top1": expected_top[0] if expected_top else None,
            "actual_top1": actual_top[0] if actual_top else None,
            "expected_top5": expected_top,
            "actual_top5": actual_top,
            "top5_overlap": len(set(expected_top).intersection(actual_top)),
        }
    )
    if mismatches:
        result["failure"] = "values outside tolerance"
    return result


def compare_files(
    expected_path: pathlib.Path,
    actual_path: pathlib.Path,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    result = compare_arrays(
        read_npy(expected_path), read_npy(actual_path), atol=atol, rtol=rtol
    )
    result["expected"] = str(expected_path.resolve())
    result["actual"] = str(actual_path.resolve())
    return result


def _load_checkpoint_manifest(directory: pathlib.Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read checkpoint manifest {manifest_path}: {error}")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported checkpoint manifest: {manifest_path}")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"checkpoint records are missing: {manifest_path}")
    if manifest.get("checkpoint_count") != len(records):
        raise ValueError(f"checkpoint count is inconsistent: {manifest_path}")
    return manifest


def _checkpoint_file(directory: pathlib.Path, record: dict[str, object]) -> pathlib.Path:
    file_name = record.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("checkpoint record has no file")
    directory = directory.resolve()
    path = (directory / file_name).resolve()
    if directory != path.parent:
        raise ValueError(f"checkpoint file escapes its directory: {file_name}")
    return path


def compare_checkpoint_directories(
    expected_directory: pathlib.Path,
    actual_directory: pathlib.Path,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    expected_directory = expected_directory.resolve()
    actual_directory = actual_directory.resolve()
    expected_manifest = _load_checkpoint_manifest(expected_directory)
    actual_manifest = _load_checkpoint_manifest(actual_directory)
    expected_records = expected_manifest["records"]
    actual_records = actual_manifest["records"]
    result: dict[str, object] = {
        "passed": False,
        "atol": atol,
        "rtol": rtol,
        "expected_directory": str(expected_directory),
        "actual_directory": str(actual_directory),
        "expected_checkpoint_count": len(expected_records),
        "actual_checkpoint_count": len(actual_records),
        "compared_checkpoint_count": 0,
        "max_absolute_error": 0.0,
        "max_absolute_error_dispatch_index": None,
        "first_divergence": None,
    }

    metadata_keys = (
        "dispatch_index",
        "op_code",
        "builtin_program",
        "dispatch_kind",
        "output_binding",
        "shape",
    )
    for ordinal, (expected_record, actual_record) in enumerate(
        zip(expected_records, actual_records), start=1
    ):
        if not isinstance(expected_record, dict) or not isinstance(actual_record, dict):
            raise ValueError(f"checkpoint record {ordinal} is not an object")
        metadata_mismatches = {
            key: {
                "expected": expected_record.get(key),
                "actual": actual_record.get(key),
            }
            for key in metadata_keys
            if expected_record.get(key) != actual_record.get(key)
        }
        if metadata_mismatches:
            result["first_divergence"] = {
                "dispatch_index": ordinal,
                "op_code": expected_record.get("op_code"),
                "failure": "checkpoint metadata mismatch",
                "metadata_mismatches": metadata_mismatches,
            }
            return result

        comparison = compare_files(
            _checkpoint_file(expected_directory, expected_record),
            _checkpoint_file(actual_directory, actual_record),
            atol=atol,
            rtol=rtol,
        )
        result["compared_checkpoint_count"] = ordinal
        if comparison["max_absolute_error"] > result["max_absolute_error"]:
            result["max_absolute_error"] = comparison["max_absolute_error"]
            result["max_absolute_error_dispatch_index"] = ordinal
        if not comparison["passed"]:
            result["first_divergence"] = {
                "dispatch_index": ordinal,
                "op_code": expected_record.get("op_code"),
                "failure": comparison.get("failure"),
                "comparison": comparison,
            }
            return result

    if len(expected_records) != len(actual_records):
        ordinal = min(len(expected_records), len(actual_records)) + 1
        result["first_divergence"] = {
            "dispatch_index": ordinal,
            "failure": "checkpoint count mismatch",
            "expected_record": (
                expected_records[ordinal - 1]
                if ordinal <= len(expected_records)
                else None
            ),
            "actual_record": (
                actual_records[ordinal - 1] if ordinal <= len(actual_records) else None
            ),
        }
        return result

    result["passed"] = True
    return result


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: pathlib.Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _run_command(command: Sequence[str], log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def _tool_version(path: pathlib.Path) -> str:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return (result.stdout or result.stderr).strip()


def _validate_resnet_input(path: pathlib.Path) -> None:
    array = read_npy(path)
    if array.shape != DEFAULT_SHAPE or array.descr != "<f4":
        raise ValueError(
            f"expected input shape {DEFAULT_SHAPE} and dtype <f4, "
            f"got {array.shape} and {array.descr}"
        )


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def run_correctness(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise ValueError(f"source IR does not exist: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)

    iree_compile = args.iree_compile.resolve()
    iree_run_module = args.iree_run_module.resolve()
    for tool in (iree_compile, iree_run_module):
        if not tool.is_file():
            raise ValueError(f"IREE tool does not exist: {tool}")

    input_path = (
        args.input.resolve() if args.input else output_dir / "input_zero_nchw_f32.npy"
    )
    if args.input:
        _validate_resnet_input(input_path)
        preprocessing = "caller-provided NCHW f32 tensor; external preprocessing"
    else:
        write_f32_npy(input_path, DEFAULT_SHAPE)
        preprocessing = "deterministic all-zero NCHW f32 tensor; no image preprocessing"

    commands: list[list[str]] = []
    if args.cpu_vmfb:
        cpu_vmfb = args.cpu_vmfb.resolve()
        if not cpu_vmfb.is_file():
            raise ValueError(f"CPU VMFB does not exist: {cpu_vmfb}")
    else:
        cpu_vmfb = output_dir / "resnet50_cpu.vmfb"
        compile_command = [
            str(iree_compile),
            str(source),
            "--compile-from=global-optimization",
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
            "-o",
            str(cpu_vmfb),
        ]
        commands.append(compile_command)
        _run_command(compile_command, output_dir / "cpu_compile.log")

    cpu_golden = output_dir / "cpu_golden.npy"
    cpu_run_command = [
        str(iree_run_module),
        "--device=local-task",
        f"--module={cpu_vmfb}",
        "--function=main",
        f"--input=@{input_path}",
        f"--output=@{cpu_golden}",
    ]
    commands.append(cpu_run_command)
    _run_command(cpu_run_command, output_dir / "cpu_run.log")

    tt_output: pathlib.Path | None = None
    if args.run_tenstorrent:
        if args.tt_output:
            raise ValueError("--run-tenstorrent and --tt-output are mutually exclusive")
        if args.tt_vmfb:
            tt_vmfb = args.tt_vmfb.resolve()
            if not tt_vmfb.is_file():
                raise ValueError(f"Tenstorrent VMFB does not exist: {tt_vmfb}")
        else:
            tt_vmfb = output_dir / "resnet50_tenstorrent.vmfb"
            tt_compile_command = [
                str(iree_compile),
                str(source),
                "--compile-from=global-optimization",
                "--iree-hal-target-device=tenstorrent",
                "-o",
                str(tt_vmfb),
            ]
            commands.append(tt_compile_command)
            _run_command(tt_compile_command, output_dir / "tt_compile.log")
        tt_output = output_dir / "tt_output.npy"
        tt_run_command = [
            str(iree_run_module),
            "--device=tenstorrent",
            f"--module={tt_vmfb}",
            "--function=main",
            f"--input=@{input_path}",
            f"--output=@{tt_output}",
        ]
        commands.append(tt_run_command)
        _run_command(tt_run_command, output_dir / "tt_run.log")
    elif args.tt_output:
        tt_output = args.tt_output.resolve()
        if not tt_output.is_file():
            raise ValueError(f"Tenstorrent output does not exist: {tt_output}")

    comparison = None
    if tt_output:
        comparison = compare_files(
            cpu_golden, tt_output, atol=args.atol, rtol=args.rtol
        )
        _write_json(output_dir / "comparison.json", comparison)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": (
            "PASS"
            if comparison and comparison["passed"]
            else "FAIL" if comparison else "GOLDEN_ONLY"
        ),
        "model": {
            "architecture": "ResNet50 (external artifact naming; not independently verified)",
            "weight_provenance": "unknown; weights are embedded in the source IR",
            "frontend": "torch-mlir inferred from dense resource names; version unknown",
        },
        "preprocessing": preprocessing,
        "source_ir": _artifact(source),
        "input": _artifact(input_path),
        "cpu_vmfb": _artifact(cpu_vmfb),
        "cpu_golden": _artifact(cpu_golden),
        "tenstorrent_output": _artifact(tt_output) if tt_output else None,
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "comparison": comparison,
        "tools": {
            "iree_compile": str(iree_compile),
            "iree_compile_version": _tool_version(iree_compile),
            "iree_run_module": str(iree_run_module),
            "iree_run_module_version": _tool_version(iree_run_module),
        },
        "commands": commands,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(comparison or {"status": "GOLDEN_ONLY"}, indent=2))
    print(f"manifest={manifest_path}")
    return 0 if comparison is None or comparison["passed"] else 1


def run_compare(args: argparse.Namespace) -> int:
    result = compare_files(args.expected, args.actual, args.atol, args.rtol)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.json:
        _write_json(args.json, result)
    return 0 if result["passed"] else 1


def run_compare_checkpoints(args: argparse.Namespace) -> int:
    result = compare_checkpoint_directories(
        args.expected_directory,
        args.actual_directory,
        args.atol,
        args.rtol,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.json:
        _write_json(args.json, result)
    return 0 if result["passed"] else 1


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    default_tool_dir = root_dir / "build-top" / "third_party" / "iree" / "tools"
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="generate a CPU golden and optionally compare Tenstorrent output"
    )
    run_parser.add_argument("source", type=pathlib.Path)
    run_parser.add_argument("output_dir", type=pathlib.Path)
    run_parser.add_argument("--input", type=pathlib.Path)
    run_parser.add_argument("--cpu-vmfb", type=pathlib.Path)
    run_parser.add_argument("--tt-output", type=pathlib.Path)
    run_parser.add_argument("--run-tenstorrent", action="store_true")
    run_parser.add_argument("--tt-vmfb", type=pathlib.Path)
    run_parser.add_argument("--atol", type=float, default=1e-5)
    run_parser.add_argument("--rtol", type=float, default=1e-5)
    run_parser.add_argument(
        "--iree-compile",
        type=pathlib.Path,
        default=default_tool_dir / "iree-compile",
    )
    run_parser.add_argument(
        "--iree-run-module",
        type=pathlib.Path,
        default=default_tool_dir / "iree-run-module",
    )
    run_parser.set_defaults(handler=run_correctness)

    compare_parser = subparsers.add_parser(
        "compare", help="compare an NPY result with a golden NPY"
    )
    compare_parser.add_argument("expected", type=pathlib.Path)
    compare_parser.add_argument("actual", type=pathlib.Path)
    compare_parser.add_argument("--atol", type=float, default=1e-5)
    compare_parser.add_argument("--rtol", type=float, default=1e-5)
    compare_parser.add_argument("--json", type=pathlib.Path)
    compare_parser.set_defaults(handler=run_compare)

    checkpoint_parser = subparsers.add_parser(
        "compare-checkpoints",
        help="find the first divergent dispatch in two checkpoint directories",
    )
    checkpoint_parser.add_argument("expected_directory", type=pathlib.Path)
    checkpoint_parser.add_argument("actual_directory", type=pathlib.Path)
    checkpoint_parser.add_argument("--atol", type=float, default=1e-5)
    checkpoint_parser.add_argument("--rtol", type=float, default=1e-5)
    checkpoint_parser.add_argument("--json", type=pathlib.Path)
    checkpoint_parser.set_defaults(handler=run_compare_checkpoints)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])
    if args.atol < 0.0 or args.rtol < 0.0:
        print("error: tolerances must be non-negative", file=sys.stderr)
        return 2
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
