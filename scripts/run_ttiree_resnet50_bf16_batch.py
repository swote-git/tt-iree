#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Prepare, run, and analyze a real static-batch BF16 ResNet50 dataset.

Example (no compilation is performed by this tool):
  prepare --source-run B1_RUN --export-dir B16_EXPORT --output-dir NEW_RUN
  run --run-dir NEW_RUN --module B16.vmfb --session-runner tt-iree-resnet-eval \
      --tt-metal-root TT_METAL_ROOT
  analyze --run-dir NEW_RUN

Preparation concatenates saved B1 BF16 bytes in dataset order, without padding
or regenerating weights, inputs, or PyTorch references. Run invokes one C++
process/session for the entire TSV and then analyzes all logits. Comparisons
are diagnostics, not a new tolerance gate or a full ImageNet accuracy claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
import pathlib
import subprocess
import time

from analyze_resnet50_bf16 import require_bf16_values
from build_resnet50_precision_golden import (
    error_metrics, file_record, require_new_directory, write_json,
)
from export_resnet50_bf16 import batch_size, verify_record
from run_ttiree_resnet50_bf16 import decode_bf16


INPUT_SAMPLE_BYTES = 3 * 224 * 224 * 2
STAGING_STAGES = ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING")


def read_json(filename):
    return json.loads(pathlib.Path(filename).read_text())


def load_logits(filename, count):
    import numpy as np
    values = np.load(filename, allow_pickle=False)
    if values.shape != (count, 1000):
        raise ValueError(f"expected {count}x1000 logits: {filename}")
    require_bf16_values(values, filename)
    return values


def prepare(source_run, export_dir, output_dir, size=16):
    import numpy as np
    size = batch_size(str(size))
    source_run, export_dir = pathlib.Path(source_run).resolve(), pathlib.Path(export_dir).resolve()
    source = read_json(source_run / "results.json")
    export = read_json(export_dir / "manifest.json")
    if (export.get("precision_policy") != "bf16_module" or export.get("batch") != size
            or export.get("input_shape") != [size, 3, 224, 224]
            or export.get("output_shape") != [size, 1000]
            or export.get("input_dtype") != "bf16" or export.get("output_dtype") != "bf16"):
        raise ValueError("export must describe the requested static BF16 batch shape")
    golden = pathlib.Path(export["golden_dir"]).resolve()
    out = pathlib.Path(output_dir).resolve()
    if any(out.is_relative_to(folder) for folder in (golden, source_run, export_dir)):
        raise ValueError("batch output directory must be separate from golden, source run, and export")
    manifest = read_json(golden / "manifest.json")
    count = len(manifest["labels"])
    indices = list(range(count))
    if (not count or count % size or source.get("status") != "completed"
            or source.get("batch") != 1 or source.get("precision_policy") != "bf16_module"
            or source.get("indices") != indices
            or pathlib.Path(source["golden_dir"]).resolve() != golden
            or [row["index"] for row in source["samples"]] != indices):
        raise ValueError("requires a complete ordered B1 dataset divisible by batch size; no padding")
    old_export_path = pathlib.Path(source["export"]["path"])
    verify_record(old_export_path, source["export"])
    old_export = read_json(old_export_path)
    for key in ("state_dict", "input"):
        if not (export[key]["sha256"] == old_export[key]["sha256"] == manifest[key]["sha256"]):
            raise ValueError(f"different immutable golden {key}")
        filename = golden / ("state_dict.pt" if key == "state_dict" else "input.npy")
        verify_record(filename, manifest[key])
    reference = golden / "bf16_module/logits.npy"
    verify_record(reference, manifest["logits"]["bf16_module"])
    verify_record(reference, export["bf16_reference"])
    load_logits(reference, count)
    previous = load_logits(source_run / "logits.npy", count)
    for record in export["artifacts"].values():
        verify_record(export_dir / pathlib.Path(record["path"]).name, record)

    # Check saved raw inputs against the immutable FP32 dataset using BF16 RNE,
    # but write the original raw bytes, never newly quantized replacements.
    inputs = np.load(golden / "input.npy", mmap_mode="r", allow_pickle=False)
    if inputs.shape != (count, 3, 224, 224) or inputs.dtype != np.dtype("float32"):
        raise ValueError("expected immutable NCHW FP32 source dataset")
    input_records = []
    for index, sample in enumerate(source["samples"]):
        filename = source_run / f"sample-{index:02d}/input.bf16.bin"
        record = verify_record(filename, sample["input"])
        bits = inputs[index].view(np.uint32)
        if not np.isfinite(inputs[index]).all():
            raise ValueError("non-finite source input")
        expected = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
        raw = filename.read_bytes()
        if len(raw) != INPUT_SAMPLE_BYTES or raw != expected.tobytes(order="C"):
            raise ValueError(f"saved BF16 input differs from golden at sample {index}")
        saved_output = decode_bf16(source_run / f"sample-{index:02d}/tenstorrent.bf16.bin", (1, 1000))
        if not np.array_equal(saved_output.view(np.uint32), previous[index:index + 1].view(np.uint32)):
            raise ValueError(f"B1 raw output differs from saved logits at sample {index}")
        input_records.append(record)

    require_new_directory(output_dir)
    batches, rows = [], []
    for number, begin in enumerate(range(0, count, size)):
        directory = out / f"batch-{number:03d}"
        directory.mkdir()
        filename, output = directory / "input.bf16.bin", directory / "tenstorrent.bf16.bin"
        with filename.open("xb") as stream:
            for record in input_records[begin:begin + size]:
                stream.write(pathlib.Path(record["path"]).read_bytes())
        batches.append({"batch_index": number, "indices": indices[begin:begin + size],
                        "input": file_record(filename), "output_path": str(output)})
        rows.append(f"{filename}\t{output}\n")
    (out / "inputs.tsv").write_text("".join(rows))
    prepared = {
        "schema_version": 1, "precision_policy": "bf16_module", "batch": size,
        "input_shape": [size, 3, 224, 224], "output_shape": [size, 1000],
        "sample_count": count, "indices": indices, "batch_count": len(batches),
        "golden_dir": str(golden), "golden_manifest": file_record(golden / "manifest.json"),
        "export": file_record(export_dir / "manifest.json"),
        "state_dict": file_record(golden / "state_dict.pt"), "golden_input": file_record(golden / "input.npy"),
        "pytorch_bf16": file_record(reference), "previous_b1_logits": file_record(source_run / "logits.npy"),
        "previous_b1_results": file_record(source_run / "results.json"),
        "source_inputs": input_records, "batches": batches, "tsv": file_record(out / "inputs.tsv"),
        "comparison_scope": "saved B1 PyTorch BF16 and runtime logits; diagnostics, not a new tolerance gate",
    }
    write_json(out / "prepared.json", prepared)
    return prepared


def validate_prepared(directory):
    directory = pathlib.Path(directory).resolve()
    prepared = read_json(directory / "prepared.json")
    size = batch_size(str(prepared["batch"]))
    count = prepared["sample_count"]
    batches = prepared["batches"]
    if (prepared["schema_version"] != 1 or prepared["precision_policy"] != "bf16_module"
            or count < 1 or count % size or prepared["indices"] != list(range(count))
            or prepared["input_shape"] != [size, 3, 224, 224]
            or prepared["output_shape"] != [size, 1000]
            or len(batches) != count // size or prepared["batch_count"] != len(batches)):
        raise ValueError("invalid prepared batch coverage or shapes")
    for key in ("golden_manifest", "export", "state_dict", "golden_input", "pytorch_bf16",
                "previous_b1_logits", "previous_b1_results"):
        verify_record(prepared[key]["path"], prepared[key])
    for key in ("pytorch_bf16", "previous_b1_logits"):
        load_logits(prepared[key]["path"], count)
    tsv_rows = []
    for number, batch in enumerate(batches):
        folder = directory / f"batch-{number:03d}"
        filename, output = folder / "input.bf16.bin", folder / "tenstorrent.bf16.bin"
        recorded_input = pathlib.Path(batch["input"]["path"])
        recorded_output = pathlib.Path(batch["output_path"])
        if (batch["batch_index"] != number or batch["indices"] != list(range(number * size, (number + 1) * size))
                or not recorded_input.is_absolute() or not recorded_output.is_absolute()
                or recorded_input.resolve() != filename
                or recorded_output.resolve() != output
                or folder.is_symlink() or filename.is_symlink() or output.is_symlink()):
            raise ValueError("invalid batch order or output path")
        verify_record(filename, batch["input"])
        if filename.stat().st_size != size * INPUT_SAMPLE_BYTES:
            raise ValueError("incorrect static batch input byte length")
        # Keep immutable TSV spelling when an archived run retains a parent alias.
        tsv_rows.append(f"{recorded_input}\t{recorded_output}\n")
    verify_record(directory / "inputs.tsv", prepared["tsv"])
    if (directory / "inputs.tsv").read_text() != "".join(tsv_rows):
        raise ValueError("TSV differs from prepared batches")
    return prepared


def require_native_profile(directory, batch_count):
    """Audit observed dispatches by invocation, independent of compiler fusion."""
    directory = pathlib.Path(directory)
    profile = read_json(directory / "run_results.json")
    with (directory / "ops_perf_results.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if (batch_count < 1 or profile.get("all_dispatches_device_native") is not True
            or not records or profile.get("dispatch_count") != len(records)):
        raise ValueError("incomplete or non-native batch profile")
    counts = Counter()
    for record in records:
        invocation = int(record["IREE INVOCATION ID"])
        counts[invocation] += 1
        if record["IREE EXECUTION DOMAIN"] != "device_native":
            raise ValueError("non-native execution domain")
        if record["INPUT_0_DATATYPE"] != "BF16" or record["OUTPUT_0_DATATYPE"] != "BF16":
            raise ValueError("non-BF16 dispatch storage")
        builtin = record["IREE BUILTIN PROGRAM"]
        compute = {"11": "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING",
                   "12": "BF16_HIFI4_FP32_DST_F32_EPILOGUE"}
        if builtin not in compute or record["IREE COMPUTE DATATYPE"] != compute[builtin]:
            raise ValueError("unexpected BF16 compute routing")
        for stage in STAGING_STAGES:
            if int(record[f"IREE {stage} DURATION [ns]"]) != 0:
                raise ValueError(f"host tensor staging detected: {stage}")
    if set(counts) != set(range(1, batch_count + 1)):
        raise ValueError("profile invocation IDs do not cover every batch")
    domains = profile.get("execution_domain_counts", {})
    if domains and (domains.get("device_native") != len(records)
                    or any(value for key, value in domains.items() if key != "device_native")):
        raise ValueError("non-native summary execution domains")
    return {"summary": profile, "dispatches_per_batch": [counts[i] for i in range(1, batch_count + 1)],
            "coverage": "all observed dispatches grouped by invocation ID; no fixed dispatch count"}


def timing_summary(filename, size, count):
    import numpy as np
    with pathlib.Path(filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count or [int(row["sample"]) for row in rows] != list(range(count)):
        raise ValueError("timings must cover every batch in order")
    columns = ("input_read_upload_seconds", "invoke_seconds", "output_download_write_seconds", "total_seconds")
    for row in rows:
        for column in columns:
            value = float(row[column])
            if not math.isfinite(value) or value < 0 or (column in ("invoke_seconds", "total_seconds") and value == 0):
                raise ValueError("invalid batch timing")

    def summarize(selected):
        result = {"batch_count": len(selected), "image_count": len(selected) * size}
        for column in ("invoke_seconds", "total_seconds"):
            values = [float(row[column]) for row in selected]
            result[column] = {"per_batch": values, "mean_per_batch": float(np.mean(values)),
                              "median_per_batch": float(np.median(values)),
                              "p95_per_batch": float(np.percentile(values, 95)),
                              "images_per_second": len(values) * size / sum(values)}
        return result

    return {"scope": "persistent session host wall time, not device kernel time; batch time is not B1 latency",
            "all_batches": summarize(rows),
            "excluding_first_batch": summarize(rows[1:]) if len(rows) > 1 else None,
            "first_batch_may_include_lazy_initialization": True}


def comparison(expected, actual):
    import numpy as np
    rows = [{"sample": i, **error_metrics(e, a), "expected_top1": int(e.argmax()),
             "actual_top1": int(a.argmax()), "top1_agreement": bool(e.argmax() == a.argmax()),
             "bitwise_mismatches": int(np.count_nonzero(e.view(np.uint32) != a.view(np.uint32)))}
            for i, (e, a) in enumerate(zip(expected, actual))]
    agreement = sum(row["top1_agreement"] for row in rows)
    return {"aggregate": {**error_metrics(expected, actual), "sample_count": len(rows),
                          "top1_agreement": agreement, "top1_agreement_fraction": agreement / len(rows),
                          "bitwise_mismatches": sum(row["bitwise_mismatches"] for row in rows)},
            "per_image": rows}


def analyze(directory):
    import numpy as np
    directory = pathlib.Path(directory).resolve()
    for name in ("analysis.json", "logits.npy", "per_image_metrics.csv"):
        path = directory / name
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite analysis: {path}")
    prepared = validate_prepared(directory)
    size, count = prepared["batch"], prepared["sample_count"]
    native = require_native_profile(directory / "profile", prepared["batch_count"])
    timing = timing_summary(directory / "call_timings.csv", size, prepared["batch_count"])
    actual = np.concatenate([decode_bf16(batch["output_path"], (size, 1000)) for batch in prepared["batches"]])
    expected = load_logits(prepared["pytorch_bf16"]["path"], count)
    previous = load_logits(prepared["previous_b1_logits"]["path"], count)
    comparisons = {"vs_pytorch_bf16": comparison(expected, actual), "vs_previous_b1": comparison(previous, actual)}
    labels = np.asarray(read_json(prepared["golden_manifest"]["path"])["labels"])
    result = {"status": "completed", "batch": size, "sample_count": count,
              "batch_count": prepared["batch_count"], "comparisons": comparisons,
              "top1_correct": int(np.count_nonzero(actual.argmax(1) == labels)),
              "top1_accuracy": float(np.mean(actual.argmax(1) == labels)),
              "native_profile": native, "timing": timing,
              "comparison_scope": prepared["comparison_scope"],
              "prepared": file_record(directory / "prepared.json"),
              "outputs": [file_record(batch["output_path"]) for batch in prepared["batches"]],
              "profile_artifacts": [file_record(directory / "profile" / name) for name in
                                    ("run_results.json", "ops_perf_results.csv")],
              "call_timings": file_record(directory / "call_timings.csv")}
    np.save(directory / "logits.npy", actual)
    result["logits"] = file_record(directory / "logits.npy")
    rows = [{"comparison": name, "batch_index": row["sample"] // size,
             "batch_row": row["sample"] % size, **row}
            for name, values in comparisons.items() for row in values["per_image"]]
    with (directory / "per_image_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "analysis.json", result)
    return result


def run(directory, module, session_runner, tt_metal_root, timeout=3600):
    directory = pathlib.Path(directory).resolve()
    prepared = validate_prepared(directory)
    for path in [directory / name for name in ("results.json", "session-run.log", "call_timings.csv", "profile",
                                               "analysis.json", "logits.npy", "per_image_metrics.csv")] + [
            pathlib.Path(batch["output_path"]) for batch in prepared["batches"]]:
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite runtime output: {path}")
    command = [str(pathlib.Path(session_runner).resolve()), str(pathlib.Path(module).resolve()),
               str(directory / "inputs.tsv"), str(directory / "call_timings.csv"),
               "tenstorrent", "--batch-size", str(prepared["batch"])]
    env = os.environ.copy()
    for variable in ("TT_METAL_SIMULATOR", "TT_IREE_CHECKPOINT_OUTPUT"):
        env.pop(variable, None)
    env.update(TT_METAL_RUNTIME_ROOT=str(pathlib.Path(tt_metal_root).resolve()),
               TT_IREE_REQUIRE_DEVICE_NATIVE="1", TT_IREE_PROFILE_OUTPUT=str(directory / "profile"))
    result = {"status": "running", "batch": prepared["batch"], "sample_count": prepared["sample_count"],
              "batch_count": prepared["batch_count"], "command": command,
              "module": file_record(module), "session_runner": file_record(session_runner),
              "prepared": file_record(directory / "prepared.json"),
              "environment": {key: env[key] for key in ("TT_METAL_RUNTIME_ROOT", "TT_IREE_REQUIRE_DEVICE_NATIVE", "TT_IREE_PROFILE_OUTPUT")}}
    write_json(directory / "results.json", result)
    try:
        start = time.perf_counter()
        with (directory / "session-run.log").open("x") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        elapsed = time.perf_counter() - start
        result.update(session_process_wall_seconds=elapsed,
                      session_process_images_per_second=prepared["sample_count"] / elapsed)
        analyze(directory)
        result.update(status="completed", analysis=file_record(directory / "analysis.json"))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(directory / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    preparation = modes.add_parser("prepare", help="copy saved B1 inputs into contiguous static batches")
    for name in ("source-run", "export-dir", "output-dir"):
        preparation.add_argument(f"--{name}", required=True, type=pathlib.Path)
    preparation.add_argument("--batch-size", type=batch_size, default=16,
                             help="static batch size matching the export (default: 16)")
    runtime = modes.add_parser("run", help="invoke one persistent C++ session and analyze all batches")
    for name in ("run-dir", "module", "session-runner", "tt-metal-root"):
        runtime.add_argument(f"--{name}", required=True, type=pathlib.Path)
    runtime.add_argument("--timeout", type=int, default=3600, help="whole dataset process timeout in seconds")
    analysis = modes.add_parser("analyze", help="analyze existing outputs/profile without execution")
    analysis.add_argument("--run-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.source_run, args.export_dir, args.output_dir, args.batch_size)
        print(args.output_dir.resolve() / "prepared.json")
    elif args.mode == "run":
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
        run(args.run_dir, args.module, args.session_runner, args.tt_metal_root, args.timeout)
        print(args.run_dir.resolve() / "analysis.json")
    else:
        analyze(args.run_dir)
        print(args.run_dir.resolve() / "analysis.json")


if __name__ == "__main__":
    main()
