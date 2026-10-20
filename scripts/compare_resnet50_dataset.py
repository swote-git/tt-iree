#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare labeled BF16 dataset logits without requiring activation checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

from analyze_resnet50_bf16 import validate_run_metadata, require_bf16_values
from build_resnet50_precision_golden import (
    classifier_comparison, file_record, predictions, require_new_directory, write_json,
)
from export_resnet50_bf16 import verify_record
from run_ttiree_resnet50_bf16 import require_native_profile


REPORT_TEMPLATE = pathlib.Path(__file__).resolve().parent / "templates/resnet50-dataset-report-template-ko.md"


def write_csv(filename, rows):
    with filename.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classifier_accuracy(logits, categories, labels):
    rows = predictions(logits, categories, labels)
    n = len(rows)
    return {"samples": n, "top1_correct": sum(r["top1_correct"] for r in rows),
            "top5_correct": sum(r["top5_correct"] for r in rows),
            "top1_accuracy": sum(r["top1_correct"] for r in rows) / n,
            "top5_accuracy": sum(r["top5_correct"] for r in rows) / n}


def comparison(expected, actual, categories, labels):
    import numpy as np
    result = classifier_comparison(expected, actual, categories, labels)
    values = [r["pcc"] for r in result["per_image"]]
    if any(v is None for v in values):
        raise ValueError("undefined PCC on a constant image-logit vector")
    result["per_image_pcc"] = {
        "min": min(values), "p05": float(np.percentile(values, 5)),
        "median": statistics.median(values), "max": max(values),
    }
    result["worst_pcc_samples"] = sorted(result["per_image"], key=lambda r: r["pcc"])[:5]
    return result


def label_boundary(logits, label):
    import numpy as np
    order = np.argsort(-logits, kind="stable")
    score, cutoff = float(logits[label]), float(logits[order[4]])
    return {"stable_label_rank": int(np.flatnonzero(order == label)[0]) + 1,
            "label_logit": score, "fifth_logit": cutoff,
            "sixth_logit": float(logits[order[5]]),
            "fifth_sixth_margin": cutoff - float(logits[order[5]]),
            "label_ties_fifth": score == cutoff,
            "count_strictly_above_label": int(np.count_nonzero(logits > score)),
            "count_tied_with_label": int(np.count_nonzero(logits == score))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("golden-dir", "ttiree-run", "ttnn-run", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=pathlib.Path)
    args = parser.parse_args()
    import numpy as np

    golden, run, nn = (p.resolve() for p in (args.golden_dir, args.ttiree_run, args.ttnn_run))
    manifest = json.loads((golden / "manifest.json").read_text())
    native_meta = json.loads((run / "results.json").read_text())
    nn_meta = json.loads((nn / "results.json").read_text())
    provenance = [file_record(p) for p in (
        golden / "manifest.json", run / "results.json", nn / "results.json", __file__)]
    for record in (manifest["state_dict"], manifest["input"], *manifest["logits"].values()):
        provenance.append(verify_record(record["path"], record))
    provenance.append(verify_record(native_meta["export"]["path"], native_meta["export"]))
    export = json.loads(pathlib.Path(native_meta["export"]["path"]).read_text())
    if (export["state_dict"]["sha256"] != manifest["state_dict"]["sha256"]
            or export["input"]["sha256"] != manifest["input"]["sha256"]):
        raise ValueError("tt-iree export uses different model weights or inputs")
    provenance.extend(file_record(p) for p in (run / "logits.npy", nn / "logits.npy"))
    verify_record(nn / "logits.npy", nn_meta["output"])
    native = np.load(run / "logits.npy", allow_pickle=False)
    ttnn_full = np.load(nn / "logits.npy", allow_pickle=False)
    indices = validate_run_metadata(golden, manifest, native_meta, nn_meta, native, ttnn_full)
    if indices != list(range(len(manifest["labels"]))):
        raise ValueError("dataset comparison requires all images once, in manifest order")
    require_bf16_values(native, "tt-iree logits")
    require_bf16_values(ttnn_full, "TTNN logits")
    batch_cpu = np.load(nn / "pytorch_bf16_batch_reference.npy", allow_pickle=False)
    if batch_cpu.shape != native.shape:
        raise ValueError("hardware-batch PyTorch reference has different row coverage")
    require_bf16_values(batch_cpu, "PyTorch hardware-batch reference")
    provenance.append(file_record(nn / "pytorch_bf16_batch_reference.npy"))
    logits = {"pytorch_fp32": np.load(golden / "fp32/logits.npy", allow_pickle=False),
              "pytorch_bf16": np.load(golden / "bf16_module/logits.npy", allow_pickle=False),
              "pytorch_bf16_hardware_batch": batch_cpu, "ttiree_bf16": native, "ttnn_bf16": ttnn_full}
    for name, values in logits.items():
        if values.shape != (len(indices), 1000) or not np.isfinite(values).all():
            raise ValueError(f"incomplete/nonfinite classifier logits: {name}")
    pairs = (("ttiree_bf16", "pytorch_bf16"), ("ttnn_bf16", "pytorch_bf16"),
             ("ttnn_bf16", "pytorch_bf16_hardware_batch"), ("pytorch_bf16_hardware_batch", "pytorch_bf16"),
             ("ttiree_bf16", "ttnn_bf16"), ("pytorch_bf16", "pytorch_fp32"),
             ("ttiree_bf16", "pytorch_fp32"), ("ttnn_bf16", "pytorch_fp32"))
    labels, categories = manifest["labels"], manifest["categories"]
    results = {f"{actual}_vs_{expected}": comparison(logits[expected], logits[actual], categories, labels)
               for actual, expected in pairs}
    accuracy = {name: classifier_accuracy(value, categories, labels) for name, value in logits.items()}
    timings = None
    profile = None
    if native_meta.get("matrix_engine") not in ("tensix_hifi4", "riscv_fp32"):
        raise ValueError("unknown tt-iree matrix engine")
    if "session_runner" in native_meta:
        profile = require_native_profile(run / "profile", native_meta["matrix_engine"] == "tensix_hifi4", len(indices))
        with (run / "call_timings.csv").open() as stream:
            timing_rows = list(csv.DictReader(stream))
        if len(timing_rows) != len(indices):
            raise ValueError("incomplete session call timings")
        calls = [float(r["invoke_seconds"]) for r in timing_rows]
        dispatch = [s["device_profile"]["total_dispatch_duration_ns"] / 1e9 for s in native_meta["samples"]]
        timings = {
            "ttiree_session_process_wall_seconds": native_meta["session_process_wall_seconds"],
            "ttiree_call_invoke_mean_seconds": statistics.mean(calls),
            "ttiree_call_invoke_first_seconds": calls[0],
            "ttiree_call_invoke_after_first_mean_seconds": statistics.mean(calls[1:]) if len(calls) > 1 else None,
            "ttiree_dispatch_mean_seconds": statistics.mean(dispatch),
            "ttnn_batch_size": nn_meta["batch"],
            "ttnn_batch_host_wall_mean_seconds": statistics.mean(nn_meta["batch_host_wall_seconds"]),
            "ttnn_pipeline_compile_wall_seconds": nn_meta["pipeline_compile_wall_seconds"],
            "scope": "host wall, not device kernel-only latency; B1 IREE and B16 TTNN are not equal-batch latency",
        }
    else:
        per_sample_profiles = [require_native_profile(run / f"sample-{i:02d}" / "profile",
                               native_meta["matrix_engine"] == "tensix_hifi4") for i in indices]
        profile = {"dispatch_count": 58 * len(indices), "all_dispatches_device_native": True,
                   "scope": "validated individual sample profiles",
                   "total_dispatch_duration_ns": sum(p["total_dispatch_duration_ns"] for p in per_sample_profiles)}
    require_new_directory(args.output_dir)
    out = args.output_dir.resolve()
    write_json(out / "summary.json", {
        "sample_count": len(indices), "dataset": manifest["source"].get("dataset"),
        "weight_source": manifest["source"]["weights"], "classification": accuracy,
        "comparisons": results, "timings": timings, "native_profile": profile,
        "provenance": provenance,
        "scope": f"{len(indices)}-image validation diagnostic; no new numerical acceptance tolerance",
        "activation_checkpoints_collected": False,
    })
    write_csv(out / "logits_metrics.csv", [{"comparison": name, **value["aggregate"]}
                                            for name, value in results.items()])
    write_csv(out / "classification.csv", [{"model": name, **value} for name, value in accuracy.items()])
    rows = []
    predicted = {name: predictions(value, categories, labels) for name, value in logits.items()}
    for name, value in predicted.items():
        write_json(out / f"{name}_predictions.json", value)
    for name, value in results.items():
        for row in value["per_image"]:
            i = row["sample"]
            rows.append({"comparison": name, "label": labels[i], "category": categories[labels[i]], **row})
    write_csv(out / "per_image_metrics.csv", rows)
    changes = []
    for i in indices:
        ref = predicted["pytorch_bf16"][i]
        iree = predicted["ttiree_bf16"][i]
        metal = predicted["ttnn_bf16"][i]
        if iree["top1"] != ref["top1"] or metal["top1"] != ref["top1"]:
            top = np.argsort(-logits["pytorch_bf16"][i], kind="stable")[:2]
            changes.append({"sample": i, "label": labels[i], "category": categories[labels[i]],
                            "pytorch_top1": ref["top1"], "ttiree_top1": iree["top1"],
                            "ttnn_top1": metal["top1"],
                            "pytorch_top2_margin": float(logits["pytorch_bf16"][i, top[0]] - logits["pytorch_bf16"][i, top[1]]),
                            "pytorch_correct": ref["top1_correct"], "ttiree_correct": iree["top1_correct"],
                            "ttnn_correct": metal["top1_correct"]})
    write_json(out / "top1_disagreements.json", changes)
    top5_changes = []
    for i in indices:
        flags = [predicted[name][i]["top5_correct"] for name in
                 ("pytorch_bf16", "ttiree_bf16", "ttnn_bf16")]
        if len(set(flags)) != 1:
            top5_changes.append({"sample": i, "label": labels[i], "category": categories[labels[i]],
                                 **{name: {"top5_correct": predicted[name][i]["top5_correct"],
                                           **label_boundary(logits[name][i], labels[i])}
                                    for name in ("pytorch_bf16", "ttiree_bf16", "ttnn_bf16")}})
    write_json(out / "top5_disagreements.json", top5_changes)
    template = REPORT_TEMPLATE
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |",
                          "| " + " | ".join("---" for _ in headers) + " |",
                          *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)])
    accuracy_table = table(["Model", "Top-1", "Top-5"], [
        [name, f"{a['top1_correct']}/{a['samples']} ({100*a['top1_accuracy']:.2f}%)",
         f"{a['top5_correct']}/{a['samples']} ({100*a['top5_accuracy']:.2f}%)"]
        for name, a in accuracy.items()])
    metric_table = table(["Comparison", "PCC", "Relative-L2", "RMSE", "Max abs", "Scale", "Top-1 agreement"], [
        [name, f"{v['aggregate']['pcc']:.9f}", f"{100*v['aggregate']['relative_l2']:.4f}%",
         f"{v['aggregate']['rmse']:.6f}", f"{v['aggregate']['max_absolute_error']:.6f}",
         f"{v['aggregate']['scale']:.6f}", f"{v['aggregate']['top1_agreement']}/{len(indices)}"]
        for name, v in results.items()])
    pcc_table = table(["Comparison", "Min PCC", "p05 PCC", "Median PCC"], [
        [name, *(f"{v['per_image_pcc'][key]:.9f}" for key in ("min", "p05", "median"))]
        for name, v in results.items()])
    worst_table = table(["Comparison", "Sample", "Label", "PCC", "Relative-L2", "Max abs"], [
        [name, row["sample"], categories[labels[row["sample"]]], f"{row['pcc']:.9f}",
         f"{100*row['relative_l2']:.4f}%", f"{row['max_absolute_error']:.6f}"]
        for name in ("ttiree_bf16_vs_pytorch_bf16", "ttnn_bf16_vs_pytorch_bf16")
        for row in results[name]["worst_pcc_samples"]])
    timing_table = table(["Metric", "Value"], [[key, value] for key, value in (timings or {}).items()])
    change_table = table(["Sample", "Label", "PyTorch", "tt-iree", "TTNN", "PyTorch Top-2 margin"], [
        [r["sample"], r["category"], categories[r["pytorch_top1"]], categories[r["ttiree_top1"]],
         categories[r["ttnn_top1"]], f"{r['pytorch_top2_margin']:.6f}"] for r in changes])
    top5_table = table(["Sample", "Label", "PyTorch rank", "tt-iree rank", "TTNN rank", "tt-iree 5/6 margin"], [
        [r["sample"], r["category"], r["pytorch_bf16"]["stable_label_rank"],
         r["ttiree_bf16"]["stable_label_rank"], r["ttnn_bf16"]["stable_label_rank"],
         f"{r['ttiree_bf16']['fifth_sixth_margin']:.6f}"] for r in top5_changes])
    dataset = manifest["source"].get("dataset", {})
    sampling = dataset.get("sampling", {})
    if template.exists():
        (out / "report-ko.md").write_text(template.read_text().format(
            sample_count=len(indices), class_count=len(set(labels)),
            one_image_percentage_points=100/len(indices),
            dataset=dataset.get("dataset", "manifest dataset"), revision=dataset.get("revision", "recorded in manifest"),
            sampling=json.dumps(sampling, ensure_ascii=False),
            accuracy_table=accuracy_table, metric_table=metric_table, pcc_table=pcc_table,
            worst_table=worst_table, timing_table=timing_table,
            change_table=change_table,
            top5_table=top5_table,
            disagreements=len(changes), matrix_engine=native_meta.get("matrix_engine", "recorded in run metadata"),
            dispatch_count=profile["dispatch_count"] if profile else len(indices)*58,
            golden=str(golden), iree_run=str(run), ttnn_run=str(nn),
            checkpoint_sha=manifest["source"]["weight_checkpoint"]["sha256"],
            weights_sha=manifest["state_dict"]["sha256"], input_sha=manifest["input"]["sha256"],
            torch_version=manifest["torch_version"], torchvision_version=manifest["torchvision_version"],
            ttnn_batch=nn_meta["batch"], ttnn_avgpool=json.dumps(nn_meta.get("avgpool_precision"), ensure_ascii=False),
        ), encoding="utf-8")
    print(json.dumps({"classification": accuracy,
                      "pcc": {k: v["aggregate"]["pcc"] for k, v in results.items()},
                      "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
