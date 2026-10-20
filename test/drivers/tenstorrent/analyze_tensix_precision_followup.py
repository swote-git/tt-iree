#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Post-hoc rounding/order hypotheses; never replace preregistered results."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

from tensix_precision_followup import MODEL_SHA256, error_metrics


def round_fp32(value, mode, frozen_round):
    if mode != "up":
        return frozen_round(value, mode == "away")
    nearest = value.astype(np.float32)
    higher = np.nextafter(nearest, np.float32(np.inf))
    midpoint = (nearest.astype(np.float64) + higher.astype(np.float64)) / 2
    change = (nearest.astype(np.float64) < value) & (value == midpoint)
    return np.where(change, higher, nearest).astype(np.float64)


def predict(a, b, data_format, slice_width, mode, merge, frozen_round):
    if data_format == "f32":
        a = (a.view(np.uint32) & 0xffffe000).view(np.float32)
        b = (b.view(np.uint32) & 0xffffc000).view(np.float32)
    ah = (a.view(np.uint32) & 0xfffe0000).view(np.float32)
    bh = (b.view(np.uint32) & 0xfff80000).view(np.float32)
    phases = ((ah, bh, 0), (ah, b - bh, 5), (a - ah, bh, 7), (a - ah, b - bh, 12))
    result = np.zeros((32, 32), dtype=np.float64)
    for begin in range(0, 32, slice_width):
        active = np.arange(32) // slice_width == begin // slice_width
        for pa, pb, offset in phases:
            parts = []
            for k in range(0, 32, 8):
                x = a[:, k:k + 8, None].astype(np.float64)
                y = b[None, k:k + 8, :].astype(np.float64)
                valid = (x != 0) & (y != 0) & active[None, k:k + 8, None]
                exponents = np.frexp(np.abs(x))[1] + np.frexp(np.abs(y))[1] - 2
                maximum = np.where(valid, exponents, -1000).max(axis=1)
                step = np.exp2(np.where(maximum > -1000, maximum - 10 - offset, 0))
                product = pa[:, k:k + 8, None].astype(np.float64) * pb[None, k:k + 8, :].astype(np.float64)
                units = np.where(valid, product, 0) / step[:, None, :]
                rounded = np.copysign(np.floor(np.abs(units) + .5), units)
                parts.append((rounded * step[:, None, :]).sum(axis=1))
            if merge == "sequential":
                for part in parts:
                    result = round_fp32(result + part, mode, frozen_round)
            else:
                for k in (0, 2):
                    combined = parts[k] + parts[k + 1]
                    if merge == "separated":
                        combined = round_fp32(combined, mode, frozen_round)
                    result = round_fp32(result + combined, mode, frozen_round)
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Frozen model changed")
    frozen = runpy.run_path(str(args.model))
    previous = json.loads((args.prior_report / "model-analysis-final.json").read_text())["measurements"]
    followup = json.loads((args.experiments / "measurements.json").read_text())["measurements"]
    records = []
    for record in previous:
        path = Path(record["directory"])
        records.append((path, "previous", path.parent.name, None))
    for record in followup:
        if record["family"] in ("term-prefix", "significand-sweep", "new-heldout"):
            records.append((args.experiments / record["name"], "followup", record["family"], record))
    candidates = [(mode, merge) for mode in ("away", "rne", "up") for merge in ("fused", "separated", "sequential")]
    results = []
    for path, population, family, original in records:
        meta = json.loads((path / "summary.json").read_text())
        a = np.fromfile(path / "lhs.f32", np.float32).reshape(32, 32)
        b = np.fromfile(path / "rhs.f32", np.float32).reshape(32, 32)
        actual = np.fromfile(path / "actual.f32", np.float32)[:1024].reshape(32, 32)
        models = {}
        for mode, merge in candidates:
            predicted = predict(a, b, meta["format"], meta["k_slice_width"], mode, merge, frozen["round_fp32"])
            if mode == "away" and merge == "fused":
                baseline = frozen["predict"](a, b, data_format=meta["format"], slice_width=meta["k_slice_width"])
                assert np.array_equal(baseline.view(np.uint32), predicted.view(np.uint32)), path
            models[f"{mode}-{merge}"] = error_metrics(predicted, actual)
        results.append(dict(name=path.name, path=str(path), population=population, family=family, details=original["details"] if original else None, models=models))

    rounding = []
    for record in followup:
        if record["family"] != "dst-rounding":
            continue
        path = args.experiments / record["name"]
        a = np.fromfile(path / "lhs.f32", np.float32).reshape(32, 32)[:, 0]
        actual = np.fromfile(path / "actual.f32", np.float32)[:1024].reshape(32, 32)[:, 0]
        exact = record["details"]["initial"] + a.astype(np.float64)
        predicted = round_fp32(exact, "up", frozen["round_fp32"]).astype(np.float32)
        rounding.append(dict(name=record["name"], **record["rounding"], ties_up_mismatches=int(np.count_nonzero(predicted.view(np.uint32) != actual.view(np.uint32)))))
    summaries = {}
    for population in ("previous", "followup"):
        rows = [r for r in results if r["population"] == population]
        summaries[population] = {}
        for mode, merge in candidates:
            name = f"{mode}-{merge}"
            values = [r["models"][name] for r in rows]
            summaries[population][name] = dict(tiles=len(values), exact_tiles=sum(v["bitwise_mismatches"] == 0 for v in values), bitwise_mismatches=sum(v["bitwise_mismatches"] for v in values), max_abs_difference=max(v["max_abs_difference"] for v in values))
        print(population, json.dumps(summaries[population], indent=2), flush=True)
    with args.output.open("x") as out:
        out.write(json.dumps(dict(note="Post-hoc hypotheses based on measured tie behavior; not preregistered successes or verified RTL. Original model and preregistered predictions are unchanged.", frozen_model_sha256=MODEL_SHA256, destination_rounding=rounding, summaries=summaries, measurements=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
