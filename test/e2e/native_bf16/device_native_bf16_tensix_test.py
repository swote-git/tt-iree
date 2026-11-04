#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exact BF16 Tensix transport/identity and bounded-integer contraction gates.

These are separate from the unchanged RISC-V FP32 bitwise fixtures. General
nonuniform/model accuracy is reported against PyTorch, never tolerance-fudged.
"""
import argparse
import csv
import json
import os
import pathlib
import struct
import subprocess

from device_native_bf16_test import bf16, bf16_bits, write_bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-compile", required=True)
    parser.add_argument("--iree-run-module", required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("TT_METAL_SIMULATOR", "TT_IREE_CHECKPOINT_OUTPUT"):
        environment.pop(name, None)
    environment.update(TT_IREE_REQUIRE_DEVICE_NATIVE="1",
                       TT_METAL_RUNTIME_ROOT=str(root / "third_party/tt-metal"))
    command = [args.iree_compile, str(root / "test/e2e/native_bf16/device_native_bf16_projection.mlir"),
               "--iree-hal-target-backends=tenstorrent",
               "--iree-tenstorrent-native-bf16-use-tensix=true", "-o", str(out / "module.vmfb")]
    with (out / "compile.log").open("w") as stream:
        subprocess.run(command, env=environment, check=True, stdout=stream,
                       stderr=subprocess.STDOUT, timeout=300)
    results = []
    for name in ("identity", "bounded_integer"):
        target = out / name
        target.mkdir(exist_ok=True)
        if name == "identity":
            bits = [((i % 2) << 15) | ((121 + (i * 3) % 13) << 7) | ((i * 19) % 128)
                    for i in range(35 * 49)]
            a = [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in bits]
            b = [float(n == k) for n in range(33) for k in range(35)]
            expected = bits[:33 * 49]
        else:
            a = [float((i * 13) % 5 - 2) for i in range(35 * 49)]
            b = [float((i * 17) % 5 - 2) for i in range(33 * 35)]
            expected = [bf16_bits(sum(a[k * 49 + m] * b[n * 35 + k] for k in range(35)))
                        for n in range(33) for m in range(49)]
        write_bf16(target / "a.bin", a)
        write_bf16(target / "b.bin", b)
        environment["TT_IREE_PROFILE_OUTPUT"] = str(target / "profile")
        command = [args.iree_run_module, "--device=tenstorrent", f"--module={out / 'module.vmfb'}",
                   "--function=bf16_projection", f"--input=35x49xbf16=@{target / 'a.bin'}",
                   f"--input=33x35xbf16=@{target / 'b.bin'}", f"--output=@{target / 'output.bin'}"]
        with (target / "run.log").open("w") as stream:
            subprocess.run(command, env=environment, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=120)
        data = (target / "output.bin").read_bytes()
        assert len(data) == len(expected) * 2, "BF16 output byte count"
        actual = struct.unpack("<" + "H" * len(expected), data)
        mismatches = sum(a != e for a, e in zip(actual, expected))
        with (target / "profile/ops_perf_results.csv").open() as stream:
            records = list(csv.DictReader(stream))
        assert len(records) == 1
        record = records[0]
        assert record["IREE EXECUTION DOMAIN"] == "device_native"
        assert record["IREE BUILTIN PROGRAM"] == "12"
        assert record["IREE COMPUTE DATATYPE"] == "BF16_HIFI4_FP32_DST_F32_EPILOGUE"
        assert int(record["CORE COUNT"]) == 4
        for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
            assert int(record[f"IREE {stage} DURATION [ns]"]) == 0
        results.append({"fixture": name, "elements": len(expected),
                        "bitwise_mismatches": mismatches, "cores": 4})
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        assert mismatches == 0, f"{name}: {mismatches} BF16 bitwise mismatches"
        print(f"PASS: {name}, {len(expected)} BF16 values, 4 cores, M/N/K tails", flush=True)

    # SFPU epilogues versus the device-RISC-V reference. Bounded integer
    # contractions are exact in both engines, so the epilogue alone decides
    # the bits.
    a = [float((i * 13) % 5 - 2) for i in range(35 * 49)]
    b = [float((i * 17) % 5 - 2) for i in range(33 * 35)]

    def channel(seed, scale, offset=0.0):
        return [bf16((offset + scale * ((n * seed) % 11 - 5)) * (1.0 + n / 97.0)) for n in range(33)]

    def compare_engines(name, mlir, function, operands):
        target = out / name
        target.mkdir(exist_ok=True)
        inputs = [("35x49xbf16", "a", a, "bf16"), ("33x35xbf16", "b", b, "bf16")] + operands
        for _, stem, values, kind in inputs:
            if kind == "f32":
                (target / f"{stem}.bin").write_bytes(struct.pack(f"<{len(values)}f", *values))
            else:
                write_bf16(target / f"{stem}.bin", values)
        outputs = {}
        for engine in ("tensix", "riscv"):
            vmfb = target / f"{engine}.vmfb"
            command = [args.iree_compile, str(root / "test/e2e/native_bf16" / mlir),
                       "--iree-hal-target-backends=tenstorrent",
                       f"--iree-tenstorrent-native-bf16-use-tensix={'true' if engine == 'tensix' else 'false'}",
                       "-o", str(vmfb)]
            with (target / f"{engine}.compile.log").open("w") as stream:
                subprocess.run(command, env=environment, check=True, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=300)
            environment["TT_IREE_PROFILE_OUTPUT"] = str(target / f"{engine}.profile")
            command = [args.iree_run_module, "--device=tenstorrent", f"--module={vmfb}",
                       f"--function={function}"]
            command += [f"--input={shape}=@{target / (stem + '.bin')}" for shape, stem, _, _ in inputs]
            command.append(f"--output=@{target / (engine + '.bin')}")
            with (target / f"{engine}.run.log").open("w") as stream:
                subprocess.run(command, env=environment, check=True, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=120)
            with (target / f"{engine}.profile/ops_perf_results.csv").open() as stream:
                records = list(csv.DictReader(stream))
            assert [r["IREE BUILTIN PROGRAM"] for r in records] == \
                (["12"] if engine == "tensix" else ["11"]), records
            outputs[engine] = (target / f"{engine}.bin").read_bytes()
        count = 33 * 49
        tensix = struct.unpack("<" + "H" * count, outputs["tensix"])
        riscv = struct.unpack("<" + "H" * count, outputs["riscv"])
        mismatches = sum(x != y for x, y in zip(tensix, riscv))
        results.append({"fixture": name, "elements": count,
                        "bitwise_mismatches_vs_riscv": mismatches,
                        "special_bits": [tensix[0], tensix[49]]})
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        assert mismatches == 0, f"{name}: {mismatches} BF16 mismatches versus RISC-V"
        print(f"PASS: {name}, {count} BF16 values equal to the RISC-V reference", flush=True)
        return tensix

    # FP32 chain with division; channel 0 divides by zero, channel 1 by NaN.
    scale = [bf16(0.37 + 0.731 * ((n * 7) % 11)) * (-1 if n % 3 == 0 else 1) for n in range(33)]
    divisor = [bf16(0.13 + 1.917 * ((n * 5) % 9)) for n in range(33)]
    divisor[0] = 0.0
    divisor[1] = float("nan")
    residual = [bf16(((i * 29) % 97 - 48) / 7.0) for i in range(33 * 49)]
    tensix = compare_engines("divide", "device_native_bf16_divide.mlir", "bf16_divide", [
        ("33xbf16", "scale", scale, "bf16"), ("33xbf16", "divisor", divisor, "bf16"),
        ("33x49xbf16", "residual", residual, "bf16")])
    assert tensix[49] == 0x7fc0, "NaN divisor must give the canonical NaN"

    # Nine operands (two batch norms, residual, ReLU) as ResNet's projection
    # shortcut: BF16 operands share Dst tiles in pairs. Zero high halves (mean,
    # beta) and a NaN low half (gamma) check the raw pair transport.
    mean0, gamma0, beta0 = channel(3, 2.5), channel(5, 0.21, 1.0), channel(7, 1.3)
    mean1, gamma1, beta1 = channel(4, 1.7), channel(2, 0.17, 0.9), channel(6, 0.9)
    inv0 = [1.0 / (0.7 + 0.113 * n) for n in range(33)]
    inv1 = [1.0 / (1.3 + 0.071 * n) for n in range(33)]
    mean0[2], mean0[3], beta0[6], beta0[7] = 0.0, -0.0, 0.0, -0.0
    gamma0[4] = float("nan")
    other = [bf16(((i * 37) % 89 - 44) / 5.0) for i in range(33 * 49)]
    compare_engines("shortcut", "device_native_bf16_shortcut.mlir", "bf16_shortcut", [
        ("33xbf16", "mean0", mean0, "bf16"), ("33xf32", "inv0", inv0, "f32"),
        ("33xbf16", "gamma0", gamma0, "bf16"), ("33xbf16", "beta0", beta0, "bf16"),
        ("33x49xbf16", "other", other, "bf16"), ("33xbf16", "mean1", mean1, "bf16"),
        ("33xf32", "inv1", inv1, "f32"), ("33xbf16", "gamma1", gamma1, "bf16"),
        ("33xbf16", "beta1", beta1, "bf16")])

    # v22 SFPU vector programs (GLOBAL_SUM, SCALAR_DIV, MAX_POOL_2D) versus the
    # device-RISC-V reference over multi-core tile splits with tails. Inputs mix
    # NaN payloads, a signalling NaN, infinities, signed zeros and the largest
    # finite value. The integer max is exact for every input, so the pool also
    # gets subnormals; the SFPU float adder flushes them, so the sums do not.
    vector_mlir = "device_native_bf16_vector.mlir"
    state = 12345

    def raw_bf16(count, subnormals):
        nonlocal state
        specials = [0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x7fc5, 0xffc1, 0x7f81,
                    0x7f7f, 0xff7f, 0x3f80, 0xbf80]
        if subnormals:
            specials += [0x0001, 0x8001, 0x007f, 0x807f]
        values = []
        for _ in range(count):
            state = (state * 1103515245 + 12345) & 0x7fffffff
            pick = state >> 8
            if pick % 97 == 0:
                values.append(specials[(pick // 97) % len(specials)])
            else:
                values.append(((pick & 1) << 15) | ((0x70 + (pick >> 1) % 32) << 7) | (pick >> 6) % 128)
        return values

    def compare_vector(name, function, shape, count, values, kinds, cores):
        target = out / name
        target.mkdir(exist_ok=True)
        (target / "input.bin").write_bytes(struct.pack(f"<{len(values)}H", *values))
        outputs = {}
        for engine in ("tensix", "riscv"):
            vmfb = out / f"vector-{engine}.vmfb"
            if not vmfb.exists():
                command = [args.iree_compile, str(root / "test/e2e/native_bf16" / vector_mlir),
                           "--iree-hal-target-backends=tenstorrent",
                           f"--iree-tenstorrent-native-bf16-use-tensix={'true' if engine == 'tensix' else 'false'}",
                           "-o", str(vmfb)]
                with (out / f"vector-{engine}.compile.log").open("w") as stream:
                    subprocess.run(command, env=environment, check=True, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=600)
            environment["TT_IREE_PROFILE_OUTPUT"] = str(target / f"{engine}.profile")
            command = [args.iree_run_module, "--device=tenstorrent", f"--module={vmfb}",
                       f"--function={function}", f"--input={shape}=@{target / 'input.bin'}",
                       f"--output=@{target / (engine + '.bin')}"]
            with (target / f"{engine}.run.log").open("w") as stream:
                subprocess.run(command, env=environment, check=True, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=120)
            with (target / f"{engine}.profile/ops_perf_results.csv").open() as stream:
                records = list(csv.DictReader(stream))
            assert [r["OP TYPE"] for r in records] == [kind for kind, _ in kinds], records
            for record, (kind, sfpu) in zip(records, kinds):
                program = "12" if engine == "tensix" and sfpu else "11"
                assert record["IREE BUILTIN PROGRAM"] == program, (engine, kind, record)
                if program == "12":
                    assert record["IREE COMPUTE DATATYPE"] == "BF16_SFPU_FP32_IR_ROUNDING", record
            if engine == "tensix":
                observed = {r["OP TYPE"]: int(r["CORE COUNT"]) for r in records}
                assert all(observed[kind] == n for kind, n in cores.items()), observed
            outputs[engine] = (target / f"{engine}.bin").read_bytes()
        tensix = struct.unpack(f"<{count}H", outputs["tensix"])
        riscv = struct.unpack(f"<{count}H", outputs["riscv"])
        mismatches = sum(x != y for x, y in zip(tensix, riscv))
        specials = sum((v & 0x7f80) == 0x7f80 for v in tensix)
        results.append({"fixture": name, "elements": count, "nan_or_inf_outputs": specials,
                        "bitwise_mismatches_vs_riscv": mismatches, "tensix_cores": cores})
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        assert mismatches == 0, f"{name}: {mismatches} BF16 mismatches versus RISC-V"
        assert specials, f"{name}: no NaN/infinity reached the output"
        print(f"PASS: {name}, {count} BF16 values equal to the RISC-V reference", flush=True)

    # 2500 channels: 40 GLOBAL_SUM tiles on 11x3 cores, 3 SCALAR_DIV tiles.
    compare_vector("vector_average", "bf16_average_wide", "2500x7x7xbf16", 2500,
                   raw_bf16(2500 * 49, False),
                   [("PADDED_COPY", False), ("GLOBAL_SUM", True), ("SCALAR_DIV", True)],
                   {"GLOBAL_SUM": 33, "SCALAR_DIV": 3})
    # 84 output rows of 57: 17 rows per tile, five tiles on five cores.
    compare_vector("vector_pool", "bf16_pool_wide", "6x29x115xbf16", 6 * 14 * 57,
                   raw_bf16(6 * 29 * 115, True), [("MAX_POOL_2D", True)], {"MAX_POOL_2D": 5})


if __name__ == "__main__":
    main()
