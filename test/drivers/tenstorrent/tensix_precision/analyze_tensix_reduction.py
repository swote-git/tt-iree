#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test an empirical alignment model; this is not an RTL specification."""

import argparse
import json
from pathlib import Path

import numpy as np


def round_fp32(value, ties_away):
    result = value.astype(np.float32)
    if ties_away:
        magnitude = np.abs(result)
        next_magnitude = np.nextafter(magnitude, np.float32(np.inf))
        # IEEE RNE already chose away-from-zero for half of the midpoint ties.
        # Only change exact midpoint cases for which it chose toward zero.
        midpoint = (magnitude.astype(np.float64) + next_magnitude.astype(np.float64)) / 2
        change = (magnitude.astype(np.float64) < np.abs(value)) & (np.abs(value) == midpoint)
        result = np.where(change, np.copysign(next_magnitude, result), result)
    return result.astype(np.float64)


def predict(a, b, *, data_format, slice_width=32, group_width=8, product_rounding="ties-away", dst_ties_away=True):
    if data_format == "f32":
        # Measured default Src unpack truncates to TF32. The additional SrcA
        # bit mask is an ISA-informed hypothesis, not an F32 semantics promise.
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
            for k in range(0, 32, group_width):
                x = a[:, k:k + group_width, None].astype(np.float64)
                y = b[None, k:k + group_width, :].astype(np.float64)
                valid = (x != 0) & (y != 0) & active[None, k:k + group_width, None]
                # Use exponent sums, not the exponent of normalized products.
                # A zero fidelity fragment still carries its original exponent
                # when the original operand was nonzero.
                exponents = np.frexp(np.abs(x))[1] + np.frexp(np.abs(y))[1] - 2
                maximum = np.where(valid, exponents, -1000).max(axis=1)
                step = np.exp2(np.where(maximum > -1000, maximum - 10 - offset, 0))
                product = pa[:, k:k + group_width, None].astype(np.float64) * pb[None, k:k + group_width, :].astype(np.float64)
                product = np.where(valid, product, 0)
                units = product / step[:, None, :]
                if product_rounding == "ties-away":
                    rounded = np.copysign(np.floor(np.abs(units) + 0.5), units)
                elif product_rounding == "rne":
                    rounded = np.rint(units)
                else:
                    rounded = np.trunc(units)
                parts.append((rounded * step[:, None, :]).sum(axis=1))
            # ISA/LLK MVMUL exposes K16. Treat its group partials and prior Dst
            # as one FP32 update; this final-rounding hypothesis remains open.
            for first in range(0, len(parts), max(1, 16 // group_width)):
                combined = sum(parts[first:first + max(1, 16 // group_width)])
                result = round_fp32(result + combined, dst_ties_away)
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    candidates = (("group8-ties-away", 8, "ties-away", True), ("group16-ties-away", 16, "ties-away", True), ("group8-product-rne", 8, "rne", True), ("group8-product-trunc", 8, "trunc", True), ("group8-dst-rne", 8, "ties-away", False))
    for directory in args.input_dir:
        for d in sorted(directory.iterdir()):
            if not (d / "summary.json").is_file():
                continue
            meta = json.loads((d / "summary.json").read_text())
            if meta["format"] == "int8" or meta["operation"] != "matmul" or meta.get("dst_check_enabled"):
                continue
            if meta["math_fidelity"] != 4:
                raise ValueError(f"This empirical model only analyzes HiFi4: {d}")
            a = np.fromfile(d / "lhs.f32", np.float32).reshape(32, 32)
            b = np.fromfile(d / "rhs.f32", np.float32).reshape(32, 32)
            actual = np.fromfile(d / "actual.f32", np.float32)[:1024].reshape(32, 32)
            if not np.isfinite(actual).all():
                raise ValueError(f"Nonfinite output needs separate analysis: {d}")
            metric = dict(directory=str(d), data_format=meta["format"], models=[])
            for name, width, product_rounding, dst_rounding in candidates:
                predicted = predict(a, b, data_format=meta["format"], slice_width=meta.get("k_slice_width", 32), group_width=width, product_rounding=product_rounding, dst_ties_away=dst_rounding)
                error = np.abs(predicted.astype(np.float64) - actual.astype(np.float64))
                metric["models"].append(dict(name=name, bitwise_mismatches=int(np.count_nonzero(predicted.view(np.uint32) != actual.view(np.uint32))), max_abs_difference=float(error.max()), rmse=float(np.sqrt(np.mean(error * error)))))
            records.append(metric)
    result = dict(note="Empirical model fitted to diagnostic data, not verified RTL or public Blackhole precision contract. Residual differences are preserved.", assumptions=dict(group_terms=8, source_significand_bits=[5, 7], phase_exponent_offsets=[0, 5, 7, 12], alignment_step="2**(max_nonzero_original_product_exponent_sum - 10 - phase_offset)", product_rounding="nearest ties away from zero", destination_rounding="FP32 nearest ties away hypothesis"), measurements=records)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    for name, *_ in candidates:
        rows = [m for r in records for m in r["models"] if m["name"] == name]
        print(f"{name}: exact tiles={sum(r['bitwise_mismatches'] == 0 for r in rows)}/{len(rows)}, differences={sum(r['bitwise_mismatches'] for r in rows)}, maximum={max((r['max_abs_difference'] for r in rows), default=0):.9g}")


if __name__ == "__main__":
    main()
