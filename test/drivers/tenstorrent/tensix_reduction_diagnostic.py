#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record reduction-isolation experiments; numerical failures stay failures."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def pair(lhs=(-127, -90), rhs=(-9, 94), positions=(0, 1)):
    a = np.zeros((32, 32), dtype=np.float32)
    b = np.zeros((32, 32), dtype=np.float32)
    for k, x, y in zip(positions, lhs, rhs):
        a[:, k] = x
        b[k, :] = y
    return a, b


def cases():
    a, b = pair()
    yield "baseline", a, b, [], "baseline"
    yield "swapped-operands", b.T.copy(), a.T.copy(), [], "orientation"
    for width in (1, 2, 4, 8, 16):
        yield f"baseline-slice{width}", a, b, ["--k-slice-width", str(width)], "slicing"
    for p in range(32):
        # Each diagonal output combines the common term p and a different q.
        # Off-diagonal outputs are single-product controls, not two-term dots.
        x, y = pair(lhs=(-127,), rhs=(-9,), positions=(p,))
        for r, q in enumerate(k for k in range(32) if k != p):
            x[r, q] = -90
            y[q, r] = 94
        yield f"position-grid-p{p:02d}", x, y, [], "positions"
    for positions in ((0, 2), (0, 4), (0, 8), (0, 15), (0, 16), (0, 31), (15, 16), (16, 17)):
        x, y = pair(positions=positions)
        yield f"constant-positions-{positions[0]}-{positions[1]}", x, y, [], "position-controls"
    x, y = pair(lhs=(-127,), rhs=(-9,), positions=(0,))
    yield "single-product-first", x, y, [], "single-product"
    x, y = pair(lhs=(-90,), rhs=(94,), positions=(1,))
    yield "single-product-second", x, y, [], "single-product"
    for s0 in (-1, 1):
        for s1 in (-1, 1):
            x, y = pair(lhs=(-127 * s0, -90 * s1))
            yield f"signs-{s0}-{s1}", x, y, [], "signs"
    for exponent in (-12, -8, -4, 0, 4, 8, 12):
        yield f"uniform-scale-{exponent}", np.ldexp(a, exponent), b, [], "uniform-scale"
    for exponent in (-24, -16, -8, 0, 8, 16, 24):
        x, y = a.copy(), b.copy()
        x[:, 0] = np.ldexp(x[:, 0], exponent)
        y[0, :] = np.ldexp(y[0, :], -exponent)
        yield f"balanced-factor-scale-{exponent}", x, y, [], "factor-scale"
    for exponent in (8, 16, 32, 64):
        for operand in ("lhs", "rhs"):
            x, y = a.copy(), b.copy()
            if operand == "lhs":
                x[:, 2] = np.float32(2.0**exponent)
            else:
                y[2, :] = np.float32(2.0**exponent)
            yield f"inactive-{operand}-e{exponent}", x, y, [], "inactive-outlier"
    for index in (1, 7, 8, 15, 16, 31):
        x, y = a.copy(), b.copy()
        x[index, 2] = np.float32(2.0**32)
        yield f"neighbor-row-{index}", x, y, [], "neighbor-outlier"
        x, y = a.copy(), b.copy()
        y[2, index] = np.float32(2.0**32)
        yield f"neighbor-col-{index}", x, y, [], "neighbor-outlier"
    for dominant in (90, 91):
        for sign in (-1, 1):
            x = np.zeros((32, 32), dtype=np.float32)
            y = np.zeros((32, 32), dtype=np.float32)
            x[:, 0] = -(96 + np.arange(32))
            x[:, 1] = -dominant
            y[0, :] = -(1 + np.arange(32))
            y[1, :] = 92 * sign
            yield f"mantissa-grid-d{dominant}-s{sign}", x, y, [], "mantissa-grid"
            x_negative = x.copy()
            x_negative[:, 0] *= -1
            yield f"negative-mantissa-grid-d{dominant}-s{sign}", x_negative, y, [], "mantissa-grid-negative"
    for pattern in ("integer", "snapshot", "wave"):
        for width in (1, 2, 4, 8, 16, 32):
            # Select the built-in generator instead of custom input files.
            yield f"dense-{pattern}-slice{width}", None, None, ["--case", pattern, "--k-slice-width", str(width)], "dense-slicing"
    for seed in (17, 43, 101, 257, 509, 1031, 2053, 4099):
        rng = np.random.default_rng(seed)
        x = rng.integers(-127, 128, (32, 32)).astype(np.float32)
        y = rng.integers(-127, 128, (32, 32)).astype(np.float32)
        yield f"heldout-integer-{seed}", x, y, [], "heldout-integer"
        def bf16_values():
            significand = rng.integers(128, 256, (32, 32)).astype(np.float32) / 128
            exponents = rng.integers(-8, 9, (32, 32))
            signs = rng.choice(np.array([-1, 1], dtype=np.float32), (32, 32))
            return np.ldexp(significand, exponents).astype(np.float32) * signs
        yield f"heldout-bf16-{seed}", bf16_values(), bf16_values(), [], "heldout-bf16"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("bf16", "f32", "int8"), default="bf16")
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("TT_METAL_RUNTIME_ROOT", str(Path(__file__).resolve().parents[3] / "third_party/tt-metal"))
    results = []
    for name, a, b, extra, family in cases():
        if args.select and name not in args.select:
            continue
        inputs = root / "inputs" / name
        cmd = [str(args.probe.resolve()), "--format", args.format, "--repeats", str(args.repeats), "--output-dir", str(root / name)]
        if a is not None:
            inputs.mkdir(parents=True)
            a.tofile(inputs / "lhs.f32")
            b.tofile(inputs / "rhs.f32")
            cmd += ["--case", "custom", "--lhs-file", str(inputs / "lhs.f32"), "--rhs-file", str(inputs / "rhs.f32")]
        cmd += extra
        with (root / (name + ".log")).open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"Probe execution failed: {name}, exit={result.returncode}; see log")
        metric = json.loads((root / name / "summary.json").read_text())
        observed = np.fromfile(root / name / "actual.f32", np.float32)[:1024].reshape(32, 32)
        expected = np.fromfile(root / name / "expected.f32", np.float32).reshape(32, 32)
        delta = observed.astype(np.float64) - expected.astype(np.float64)
        metric.update(name=name, family=family, command=cmd, expected_0_0=float(expected[0, 0]), actual_0_0=float(observed[0, 0]), first_tile_mismatches=int(np.count_nonzero(observed.view(np.uint32) != expected.view(np.uint32))))
        if family == "positions":
            p = int(name[-2:])
            metric["position_pairs"] = [dict(p=p, q=q, expected=float(expected[r, r]), actual=float(observed[r, r]), delta=float(delta[r, r])) for r, q in enumerate(k for k in range(32) if k != p)]
        results.append(metric)
        (root / "measurements.json").write_text(json.dumps(dict(format=args.format, measurements=results, note="Completed observations are not unconditional correctness passes."), indent=2) + "\n")
        print(f"{name}: max={metric['max_abs_error_finite']:.9g}, mismatches/tile={metric['first_tile_mismatches']}, [0,0]={metric['actual_0_0']:.9g}/{metric['expected_0_0']:.9g}", flush=True)
    if not results:
        raise ValueError("No selected experiments matched")
    print(f"Recorded {len(results)} conditions in {root}")


if __name__ == "__main__":
    main()
