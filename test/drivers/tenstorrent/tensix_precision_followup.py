#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Preregister bit, destination-rounding, and frozen-model residual probes."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

import numpy as np


MODEL_SHA256 = "00f79e375ed23f98a4a3da99da74c93eca13b10245d25524eea2f4f1ea9dc825"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantize(a, data_format):
    a = np.ascontiguousarray(a, dtype=np.float32)
    if data_format == "bf16":
        bits = a.view(np.uint32)
        bits = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16
        a = bits.view(np.float32)
    return a


def error_metrics(predicted, actual):
    error = predicted.astype(np.float64) - actual.astype(np.float64)
    magnitude = np.abs(actual)
    ulp = np.nextafter(magnitude, np.float32(np.inf)) - magnitude
    normalized = np.abs(error) / ulp.astype(np.float64)
    norm = np.linalg.norm(actual.astype(np.float64))
    return {
        "bitwise_mismatches": int(np.count_nonzero(predicted.view(np.uint32) != actual.view(np.uint32))),
        "max_abs_difference": float(np.abs(error).max()),
        "mean_abs_difference": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_l2": float(np.linalg.norm(error) / norm) if norm else None,
        "max_output_ulp_difference": float(normalized.max()),
        "p99_output_ulp_difference": float(np.percentile(normalized, 99)),
    }


def exponent_quantum(a, b, data_format):
    if data_format == "f32":
        a = (a.view(np.uint32) & 0xffffe000).view(np.float32)
        b = (b.view(np.uint32) & 0xffffc000).view(np.float32)
    x = a[:, :, None].astype(np.float64)
    y = b[None, :, :].astype(np.float64)
    valid = (x != 0) & (y != 0)
    exp = np.frexp(np.abs(x))[1] + np.frexp(np.abs(y))[1] - 2
    maximum = np.where(valid, exp, -1000).max(axis=1)
    return np.exp2(np.where(maximum > -1000, maximum - 10, 0))


def residual_distribution(predicted, actual, a, b, data_format):
    error = np.abs(predicted.astype(np.float64) - actual.astype(np.float64))
    quantum = exponent_quantum(a, b, data_format)
    nonzero = error != 0
    magnitude = np.abs(actual.astype(np.float64))
    eligible = nonzero & (magnitude > 0)
    bins = []
    for lower, upper in ((-1000, 0), (0, 4), (4, 8), (8, 12), (12, 16), (16, 24), (24, 1000)):
        exp = np.frexp(magnitude)[1] - 1
        selected = (exp >= lower) & (exp < upper)
        if not selected.any():
            continue
        bins.append({
            "output_exponent_range": [lower, upper],
            "elements": int(selected.sum()),
            "nonzero_errors": int((selected & nonzero).sum()),
            "mean_abs_difference": float(error[selected].mean()),
            "max_abs_difference": float(error[selected].max()),
            "mean_error_over_phase0_quantum": float((error / quantum)[selected].mean()),
        })
    return {
        "nonzero_value_errors": int(nonzero.sum()),
        "max_error_over_phase0_quantum": float((error / quantum).max()),
        "mean_relative_error_nonzero_outputs": float((error[eligible] / magnitude[eligible]).mean()) if eligible.any() else 0,
        "bins": bins,
        "note": "Quantum is the maximum phase-0 operand-exponent quantum across K32, not a final-Dst rounding contract.",
    }


def experiments(prior):
    eye = np.eye(32, dtype=np.float32)
    fractions = [1 << bit for bit in range(23)]
    fractions += [0, 0x1fff, 0x2000, 0x3fff, 0x4000, 0x5fff, 0x7fff, 0x123456, 0x7fffff]
    rows = []
    for row in range(32):
        exponent = (0, -12, 12, -20, 20)[row % 5]
        sign = 0x80000000 if row % 2 else 0
        rows.append([sign | ((127 + exponent) << 23) | f for f in fractions])
    bitwalk = np.array(rows, dtype=np.uint32).view(np.float32)
    rng = np.random.default_rng(864209)
    bits = rng.integers(0, 1 << 23, (32, 32), dtype=np.uint32)
    bits |= rng.integers(107, 148, (32, 32), dtype=np.uint32) << 23
    bits |= rng.integers(0, 2, (32, 32), dtype=np.uint32) << 31
    for data_format in ("f32", "bf16"):
        for name, values in (("bitwalk", bitwalk), ("random", bits.view(np.float32))):
            for side in ("lhs", "rhs"):
                a, b = (values, eye) if side == "lhs" else (eye, values)
                yield f"{data_format}-{name}-{side}", "single-product", data_format, a, b, [], {"side": side}
    for op, extra in (("copy", []), ("copy-dst", ["--f32-unpack-to-dst"]), ("bypass", [])):
        yield f"f32-bitwalk-{op}", "transfer", "f32", bitwalk, eye, ["--op", "copy" if op == "copy-dst" else op] + extra, {}

    zero = np.zeros((32, 32), dtype=np.float32)
    yield "prefill-zero", "prefill-control", "bf16", zero, zero, ["--dst-prefill-bits", "3fc12345", "--require-exact"], {}
    yield "prefill-zero-check", "prefill-control", "bf16", zero, zero, ["--dst-prefill-bits", "3fc12345", "--dst-check-bits", "3fc12345", "--require-exact"], {}
    units = np.array([-3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5, -.75, -.25, .25, .75, -1, 0, 1, 2] * 2, dtype=np.float32)
    for exponent in (-10, 0, 10, 24):
        for sign in (-1, 1):
            for parity in (0, 1):
                magnitude = np.float32(np.ldexp(1.5, exponent))
                if parity:
                    magnitude = np.nextafter(magnitude, np.float32(np.inf))
                initial = np.float32(sign * magnitude)
                a, b = zero.copy(), zero.copy()
                a[:, 0] = np.ldexp(units, exponent - 23)
                b[0, :] = 1
                initial_bits = int(initial.view(np.uint32))
                yield f"dst-e{exponent}-s{sign}-p{parity}", "dst-rounding", "bf16", a, b, ["--dst-prefill-bits", f"{initial_bits:08x}"], {"exponent": exponent, "sign": sign, "parity": parity, "initial": float(initial)}

    sources = [("bf16-wave", prior / "bf16/dense-wave-slice32", "bf16")]
    sources += [(f"bf16-seed{seed}", prior / f"bf16-heldout/heldout-bf16-{seed}", "bf16") for seed in (17, 43, 101, 257, 509, 1031, 2053, 4099)]
    sources += [("f32-wave", prior / "f32/dense-wave-slice32", "f32")]
    for name, directory, data_format in sources:
        a = np.fromfile(directory / "lhs.f32", np.float32).reshape(32, 32)
        b = np.fromfile(directory / "rhs.f32", np.float32).reshape(32, 32)
        for terms in (1, 2, 3, 4, 8, 9, 16, 17, 32):
            x, y = a.copy(), b.copy()
            x[:, terms:] = 0
            y[terms:, :] = 0
            yield f"{name}-k{terms:02d}", "term-prefix", data_format, x, y, [], {"source": name, "terms": terms, "source_directory": str(directory)}
        if data_format != "bf16":
            continue
        for significant_bits in range(1, 9):
            mask = np.uint32((0xffffffff << (24 - significant_bits)) & 0xffffffff)
            x = (a.view(np.uint32) & mask).view(np.float32)
            y = (b.view(np.uint32) & mask).view(np.float32)
            yield f"{name}-sig{significant_bits}", "significand-sweep", data_format, x, y, [], {"source": name, "significant_bits": significant_bits}

    for seed in (314159, 271828, 161803):
        for distribution in ("uniform", "wide", "zeros", "paired-cancellation"):
            rng = np.random.default_rng(seed)
            limit = 24 if distribution == "wide" else 16 if distribution == "zeros" else 8
            def values():
                sig = rng.integers(128, 256, (32, 32)).astype(np.float32) / 128
                v = np.ldexp(sig, rng.integers(-limit, limit + 1, (32, 32)))
                v *= rng.choice(np.array([-1, 1], np.float32), (32, 32))
                if distribution == "zeros":
                    v[rng.random((32, 32)) < .35] = 0
                return v.astype(np.float32)
            a, b = values(), values()
            if distribution == "paired-cancellation":
                a[:, 16:] = a[:, :16]
                b[16:, :] = -b[:16, :]
            yield f"heldout-{seed}-{distribution}", "new-heldout", "bf16", a, b, [], {"seed": seed, "distribution": distribution}


def classify_previous(prior, output):
    records = json.loads((prior / "model-analysis-final.json").read_text())["measurements"]
    rows = []
    for record in records:
        model = record["models"][0]
        if not model["bitwise_mismatches"]:
            continue
        path = Path(record["directory"])
        family = "bf16-heldout" if path.parent.name == "bf16-heldout" else f"{record['data_format']}-wave"
        rows.append({**model, "model_name": model["name"], "name": path.name, "family": family})
    assert len(rows) == 20
    with (output / "previous-20-tiles.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model_hash = digest(args.model)
    if model_hash != MODEL_SHA256:
        raise ValueError("Frozen model SHA256 changed; do not silently refit the experiment")
    frozen = runpy.run_path(str(args.model))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = list(experiments(args.prior_report.resolve()))
    if len({item[0] for item in plan}) != len(plan):
        raise ValueError("Duplicate experiment name")
    (output / "predictions").mkdir()
    (output / "inputs").mkdir()
    previous_rows = classify_previous(args.prior_report, output)
    declarations = []
    for name, family, data_format, a, b, extra, details in plan:
        inputs = output / "inputs" / name
        inputs.mkdir()
        a.tofile(inputs / "lhs.f32")
        b.tofile(inputs / "rhs.f32")
        declarations.append(dict(name=name, family=family, data_format=data_format, extra=extra, details=details, lhs_sha256=digest(inputs / "lhs.f32"), rhs_sha256=digest(inputs / "rhs.f32")))
        if family in ("term-prefix", "significand-sweep", "new-heldout"):
            x, y = quantize(a, data_format), quantize(b, data_format)
            for suffix, dst_rounding in (("away", True), ("rne", False)):
                predicted = frozen["predict"](x, y, data_format=data_format, dst_ties_away=dst_rounding)
                predicted.tofile(output / "predictions" / f"{name}-{suffix}.f32")
    preregistration = dict(model_sha256=model_hash, generator_sha256=digest(Path(__file__)), probe_sha256=digest(args.probe), experiments=declarations, previous_mismatching_tiles=previous_rows, note="Saved all input hashes and frozen-model predictions before starting hardware. Numerical differences are not passes.")
    (output / "preregistered.json").write_text(json.dumps(preregistration, indent=2) + "\n")
    results = []
    rounding_rows = []
    env = os.environ.copy()
    env.setdefault("TT_METAL_RUNTIME_ROOT", str(Path(__file__).resolve().parents[3] / "third_party/tt-metal"))
    for name, family, data_format, _, _, extra, details in plan:
        inputs = output / "inputs" / name
        directory = output / name
        cmd = [str(args.probe.resolve()), "--case", "custom", "--format", data_format, "--lhs-file", str(inputs / "lhs.f32"), "--rhs-file", str(inputs / "rhs.f32"), "--output-dir", str(directory)] + extra
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError(f"Probe failed: {name}, exit={result.returncode}; see log")
        meta = json.loads((directory / "summary.json").read_text())
        actual = np.fromfile(directory / "actual.f32", np.float32)[:1024].reshape(32, 32)
        a = np.fromfile(directory / "lhs.f32", np.float32).reshape(32, 32)
        b = np.fromfile(directory / "rhs.f32", np.float32).reshape(32, 32)
        if meta["nonfinite"] or meta["different_from_first_repeat"]:
            raise ValueError(f"Nonfinite or unstable output: {name}")
        row = dict(name=name, family=family, details=details, command=cmd, probe_summary=meta)
        if family in ("single-product", "transfer"):
            source = a if details.get("side", "lhs") == "lhs" else b
            bits = source.view(np.uint32)
            candidates = {
                "unmasked": source,
                "truncate13-ffffe000": (bits & 0xffffe000).view(np.float32),
                "truncate14-ffffc000": (bits & 0xffffc000).view(np.float32),
                "tf32-rne": ((bits + 0xfff + ((bits >> 13) & 1)) & 0xffffe000).view(np.float32),
            }
            row["mask_candidates"] = {label: error_metrics(value, actual) for label, value in candidates.items()}
            row["matching_candidates"] = [label for label, value in row["mask_candidates"].items() if not value["bitwise_mismatches"]]
        elif family == "dst-rounding":
            exact = details["initial"] + a[:, 0].astype(np.float64)
            rne = exact.astype(np.float32)
            away = frozen["round_fp32"](exact, True).astype(np.float32)
            observed = actual[:, 0]
            discriminating = rne.view(np.uint32) != away.view(np.uint32)
            row["rounding"] = {
                "discriminating_rows": int(discriminating.sum()),
                "rne_mismatches": int(np.count_nonzero(observed.view(np.uint32) != rne.view(np.uint32))),
                "ties_away_mismatches": int(np.count_nonzero(observed.view(np.uint32) != away.view(np.uint32))),
                "all_columns_identical": bool(np.all(actual == observed[:, None])),
            }
            for i in range(32):
                rounding_rows.append(dict(name=name, row=i, initial=details["initial"], increment=float(a[i, 0]), exact_sum=float(exact[i]), rne=float(rne[i]), ties_away=float(away[i]), observed=float(observed[i]), observed_bits=f"{observed[i].view(np.uint32):08x}", discriminates=bool(discriminating[i])))
        elif family in ("term-prefix", "significand-sweep", "new-heldout"):
            row["models"] = {}
            for suffix in ("away", "rne"):
                predicted = np.fromfile(output / "predictions" / f"{name}-{suffix}.f32", np.float32).reshape(32, 32)
                row["models"][suffix] = error_metrics(predicted, actual)
                if suffix == "away":
                    row["residual_distribution"] = residual_distribution(predicted, actual, a, b, data_format)
            if family == "term-prefix" and details["terms"] == 32:
                previous_actual = np.fromfile(Path(details["source_directory"]) / "actual.f32", np.float32)[:1024].reshape(32, 32)
                row["unchanged_vs_previous_probe"] = bool(np.array_equal(actual.view(np.uint32), previous_actual.view(np.uint32)))
        results.append(row)
        (output / "measurements.json").write_text(json.dumps(dict(model_sha256=model_hash, measurements=results), indent=2) + "\n")
        status = row.get("matching_candidates", row.get("rounding", row.get("models", {}).get("away", {})))
        print(f"{name}: {status}", flush=True)
    with (output / "dst-rounding.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rounding_rows[0].keys())
        writer.writeheader()
        writer.writerows(rounding_rows)
    print(f"Recorded {len(results)} conditions; frozen model remains {model_hash}")


if __name__ == "__main__":
    main()
