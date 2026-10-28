#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""CPU-only reference, profiler-audit, and CLI tests for the B2 BF16 runner."""

import contextlib
import copy
import csv
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import device_native_bf16_batch_test as batch


def operation(kind, m=0, n=0, k=0, call=1, grid=None):
    matrix = kind in ("CONV2D", "MATMUL", "LINEAR")
    if grid is None:
        grid = (min(11, (n + 31) // 32), min(10, (m + 31) // 32)) if matrix else (1, 1)
    program = 12 if matrix else 11
    record = {"OP TYPE": kind, "OP CODE": f"fixture_{call}", "GLOBAL CALL COUNT": str(call),
              "IREE M": str(m), "IREE N": str(n), "IREE K": str(k),
              "IREE GRID X": str(grid[0]), "IREE GRID Y": str(grid[1]), "IREE GRID Z": "1",
              "CORE COUNT": str(math.prod(grid)), "IREE EXECUTION DOMAIN": "device_native",
              "IREE BUILTIN PROGRAM": str(program), "IREE COMPUTE DATATYPE": batch.COMPUTE[program],
              "INPUT_0_DATATYPE": "BF16", "OUTPUT_0_DATATYPE": "BF16", "IREE DISPATCH KIND": "8"}
    record.update({f"IREE {stage} DURATION [ns]": "0" for stage in batch.STAGING})
    return record


def mappings(record, scalar_range=()):
    gx, gy = (int(record[f"IREE GRID {axis}"]) for axis in "XY")
    rows = []
    matrix = record["OP TYPE"] in ("CONV2D", "MATMUL", "LINEAR")
    if not matrix and not scalar_range:
        return rows
    for y in range(gy):
        for x in range(gx):
            row = {"OP CODE": record["OP CODE"], "GLOBAL CALL COUNT": record["GLOBAL CALL COUNT"],
                   "CORE X": str(x), "CORE Y": str(y), "M TILE BEGIN": "", "M TILE END": "",
                   "N TILE BEGIN": "", "N TILE END": "", "K TILE COUNT": "",
                   "ELEMENT BEGIN": "", "ELEMENT END": ""}
            if scalar_range:
                count, width = scalar_range
                index, size = y * gx + x, gx * gy
                row.update({"ELEMENT BEGIN": str((count // width * index // size) * width),
                            "ELEMENT END": str((count // width * (index + 1) // size) * width),
                            "PARTITION UNIT": "rows"})
            else:
                mt, nt, kt = ((int(record[f"IREE {dim}"]) + 31) // 32 for dim in "MNK")
                row.update({"M TILE BEGIN": str(mt * y // gy), "M TILE END": str(mt * (y + 1) // gy),
                            "N TILE BEGIN": str(nt * x // gx), "N TILE END": str(nt * (x + 1) // gx),
                            "K TILE COUNT": str(kt), "PARTITION UNIT": "tiles"})
            rows.append(row)
    return rows


def write_csv(path, records, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def profile_fixture(directory, case, edit=None):
    directory.mkdir(parents=True)
    records, rows = [], []
    for call, spec in enumerate(case.dispatches, 1):
        grid = (10, 3) if case.scalar_range else None
        record = operation(*spec, call=call, grid=grid)
        records.append(record)
        rows.extend(mappings(record, case.scalar_range))
    summary = {"dispatch_count": len(records), "all_dispatches_device_native": True}
    if edit:
        edit(records, rows, summary)
    (directory / "run_results.json").write_text(json.dumps(summary))
    write_csv(directory / "ops_perf_results.csv", records, list(operation("LINEAR")))
    fields = list(mappings(operation("LINEAR", 2, 33, 35))[0])
    write_csv(directory / "core_mapping.csv", rows, fields)


class ReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.name: case for case in batch.build_cases()}

    def test_exact_shapes_counts_and_batch_distinction(self):
        expected = {"conv_s1": ((2, 33, 7, 11), 5082),
                    "conv_s2": ((2, 35, 4, 7), 1960),
                    "projection": ((2, 33, 49), 3234),
                    "input_pad": ((2, 5, 10, 42), 4200),
                    "maxpool": ((2, 5, 3, 19), 570), "average_fc": ((2, 33), 66)}
        self.assertEqual(set(self.cases), set(expected))
        for name, (shape, count) in expected.items():
            with self.subTest(name=name):
                case = self.cases[name]
                self.assertEqual(case.shape, shape)
                self.assertEqual(len(case.expected), count)
                self.assertNotEqual(case.expected[:count // 2], case.expected[count // 2:])
                for tensor in case.inputs:
                    self.assertEqual(len(tensor.values), math.prod(tensor.shape))

    def test_rne_even_and_odd_ties_both_batches(self):
        self.assertEqual([batch.bf16_bits(x) for x in
                          (1 + 1 / 256, 1 + 3 / 256, -2 - 2 / 256, -2 - 6 / 256)],
                         [0x3f80, 0x3f82, 0xc000, 0xc002])
        self.assertEqual(batch.bf16_bits(-0.0), 0x8000)

    def test_convolution_spatial_tails_and_halo(self):
        for name in ("conv_s1", "conv_s2"):
            case = self.cases[name]
            _, m, n, k = case.dispatches[0]
            self.assertTrue(all(size % 32 for size in (m, n, k)))
            self.assertNotEqual(m // 2 % 32, 0)
        case = self.cases["conv_s1"]
        # Every channel of both batches has positive-zero borders, while the
        # interior is a true convolution, not an all-zero padding-only test.
        nonzero = 0
        for b in range(2):
            for c in range(33):
                for h in range(7):
                    for w in range(11):
                        value = case.expected[((b * 33 + c) * 7 + h) * 11 + w]
                        if h in (0, 6) or w in (0, 10):
                            self.assertEqual(value, 0)
                        else:
                            nonzero += value != 0
        self.assertGreater(nonzero, 1000)
        self.assertEqual(self.cases["conv_s2"].expected[:4], [0x4100, 0xc0e0, 0xbf80, 0x4100])
        self.assertEqual(self.cases["conv_s2"].expected[980:984], [0xc180, 0x4160, 0x4000, 0xc180])

    def test_projection_requires_ordered_bf16_boundaries_and_relu(self):
        case = self.cases["projection"]
        a, weight, mean, scale, bias, residual = (tensor.values for tensor in case.inputs)
        different = 0
        for b in range(2):
            for n in range(33):
                for m in range(49):
                    index = (b * 33 + n) * 49 + m
                    dot = sum(a[(b * 35 + k) * 49 + m] * weight[n * 35 + k] for k in range(35))
                    fused = max(0.0, (dot - mean[n]) * 0.5 * scale[n] + bias[n] + residual[index])
                    different += batch.bf16_bits(fused) != case.expected[index]
        self.assertGreater(different, 100)
        for values in (case.expected[:1617], case.expected[1617:]):
            self.assertIn(0, values)
            self.assertTrue(any(batch.from_bits(v) > 0 for v in values))

    def test_asymmetric_input_padding_and_distinct_maxpool_batches(self):
        case = self.cases["input_pad"]
        source = case.inputs[0].values
        for b in range(2):
            for c in range(5):
                for h in range(10):
                    for w in range(42):
                        expected = 0
                        if 1 <= h < 8 and 2 <= w < 41:
                            expected = batch.bf16_bits(source[((b * 5 + c) * 7 + h - 1) * 39 + w - 2])
                        self.assertEqual(case.expected[((b * 5 + c) * 10 + h) * 42 + w], expected)
        pooled = self.cases["maxpool"].expected
        self.assertTrue(all(batch.from_bits(v) > 0 for v in pooled[:285]))
        self.assertTrue(all(batch.from_bits(v) < 0 for v in pooled[285:]))

    def test_ordered_gap_49_division_and_nonzero_excluded_halo(self):
        case = self.cases["average_fc"]
        self.assertEqual((case.expected[0], case.expected[33]), (0x3ca7, 0xbd27))
        self.assertEqual(batch.ordered_average([1.0] + [1 / 512] * 48), batch.bf16(1 / 49))
        self.assertNotEqual(batch.ordered_average([1.0] + [1 / 512] * 48),
                            batch.bf16((1 + 48 / 512) / 49))
        values = case.inputs[0].values
        for b in range(2):
            for c in range(35):
                for h in range(8):
                    for w in range(8):
                        if h == 7 or w == 7:
                            self.assertEqual(values[((b * 35 + c) * 8 + h) * 8 + w], (1, -2)[b] * 64)

    def test_fc_is_single_product_permutation_with_active_k_tails(self):
        case = self.cases["average_fc"]
        source, weight, bias = (tensor.values for tensor in case.inputs)
        selected = []
        for n in range(33):
            row = weight[n * 35:(n + 1) * 35]
            self.assertEqual(row.count(1.0), 1)
            self.assertEqual(row.count(0.0), 34)
            selected.append(row.index(1.0))
        self.assertEqual(len(set(selected)), 33)
        self.assertTrue({0, 33, 34}.issubset(selected))
        for b in range(2):
            for n, c in enumerate(selected):
                valid = [source[((b * 35 + c) * 8 + h) * 8 + w]
                         for h in range(7) for w in range(7)]
                self.assertEqual(case.expected[b * 33 + n],
                                 batch.bf16_bits(batch.ordered_average(valid) + bias[n]))

    def test_raw_bf16_output_count_and_bit_mismatches_are_hard_failures(self):
        case = self.cases["average_fc"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.bin"
            batch.write_bits(output, case.expected)
            self.assertEqual(output.stat().st_size, 132)
            batch.compare_output(output, case)
            output.write_bytes(output.read_bytes()[:-2])
            with self.assertRaisesRegex(AssertionError, "BF16 bytes"):
                batch.compare_output(output, case)
            corrupted = list(case.expected)
            corrupted[33] ^= 1
            batch.write_bits(output, corrupted)
            with self.assertRaisesRegex(AssertionError, "1 bit mismatches.*33"):
                batch.compare_output(output, case)


class ProfileTest(unittest.TestCase):
    def test_uneven_matrix_rectangles_use_full_11_by_10_grid(self):
        record = operation("MATMUL", 415, 383, 35, grid=(11, 10))
        rows = mappings(record)
        self.assertEqual(len(rows), 110)
        batch.validate_mapping(record, rows)

    def test_scalar_row_coverage_can_exceed_64_cores(self):
        record = operation("PADDED_COPY", grid=(11, 10))
        scalar_range = (154 * 39, 39)
        batch.validate_mapping(record, mappings(record, scalar_range), scalar_range)

    def test_invalid_grids_and_tile_ranges_fail(self):
        original = operation("MATMUL", 98, 33, 35)
        original_rows = mappings(original)
        for field, value in (("IREE GRID X", "12"), ("IREE GRID Y", "11"),
                             ("IREE GRID Z", "2"), ("CORE COUNT", "64")):
            with self.subTest(field=field):
                record = dict(original, **{field: value})
                with self.assertRaises(AssertionError):
                    batch.validate_mapping(record, original_rows)
        for field, value in (("K TILE COUNT", "1"), ("M TILE END", "0"),
                             ("N TILE END", "3"), ("CORE X", "99")):
            with self.subTest(field=field):
                rows = copy.deepcopy(original_rows)
                rows[0][field] = value
                with self.assertRaises(AssertionError):
                    batch.validate_mapping(original, rows)
        rows = copy.deepcopy(original_rows)
        rows[-1].update({key: rows[0][key] for key in
                         ("M TILE BEGIN", "M TILE END", "N TILE BEGIN", "N TILE END")})
        with self.assertRaisesRegex(AssertionError, "overlapping"):
            batch.validate_mapping(original, rows)

    def test_scalar_gaps_overlap_split_rows_and_missing_core_fail(self):
        record = operation("PADDED_COPY", grid=(10, 7))
        scalar_range = (2730, 39)
        original = mappings(record, scalar_range)
        batch.validate_mapping(record, original, scalar_range)
        for begin in ("1", "0", "78"):
            rows = copy.deepcopy(original)
            rows[1]["ELEMENT BEGIN"] = begin
            with self.assertRaises(AssertionError):
                batch.validate_mapping(record, rows, scalar_range)
        with self.assertRaises(AssertionError):
            batch.validate_mapping(record, original[:-1], scalar_range)

    def test_all_case_profile_plans_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            for case in batch.build_cases():
                with self.subTest(case=case.name):
                    directory = Path(temporary) / case.name
                    profile_fixture(directory, case)
                    result = batch.validate_profile(directory, case)
                    self.assertTrue(result["all_dispatches_device_native"])
                    self.assertEqual(len(result["dispatches"]), len(case.dispatches))

    def test_host_fallback_storage_staging_and_routing_fail(self):
        case = batch.convolution(1)
        changes = [("IREE EXECUTION DOMAIN", "host_assisted_device"),
                   ("INPUT_0_DATATYPE", "F32"), ("OUTPUT_0_DATATYPE", "F32"),
                   ("IREE BUILTIN PROGRAM", "11"), ("IREE DISPATCH KIND", "2"),
                   ("IREE COMPUTE DATATYPE", "F32"), ("IREE M", "45")]
        changes += [(f"IREE {stage} DURATION [ns]", "1") for stage in batch.STAGING]
        for index, (field, value) in enumerate(changes):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / str(index)
                profile_fixture(directory, case,
                                lambda records, rows, summary: records[0].update({field: value}))
                with self.assertRaises(AssertionError):
                    batch.validate_profile(directory, case)

    def test_b1_unrolling_duplicate_calls_or_orphan_mappings_fail(self):
        case = batch.convolution(1)

        def unroll(records, rows, summary):
            records[0]["IREE M"] = "45"
            records.append(dict(records[0], **{"GLOBAL CALL COUNT": "2"}))
            summary["dispatch_count"] = 2

        def duplicate(records, rows, summary):
            records.append(dict(records[0]))

        def orphan(records, rows, summary):
            rows[0]["GLOBAL CALL COUNT"] = "99"

        def fallback(records, rows, summary):
            summary["all_dispatches_device_native"] = False

        for edit in (unroll, duplicate, orphan, fallback):
            with self.subTest(edit=edit.__name__), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "profile"
                profile_fixture(directory, case, edit)
                with self.assertRaises(AssertionError):
                    batch.validate_profile(directory, case)


class CommandTest(unittest.TestCase):
    def test_global_optimization_and_tensix_enabled(self):
        command = batch.compile_command("compiler", Path("batch.vmfb"))
        self.assertNotIn("--iree-tenstorrent-native-bf16-reuse-output-tiles=true", command)
        paired = batch.compile_command("compiler", Path("batch.vmfb"), True)
        self.assertIn("--iree-tenstorrent-native-bf16-reuse-output-tiles=true", paired)
        self.assertEqual(paired[-2:], command[-2:])
        self.assertIn("--iree-opt-level=O2", command)
        self.assertIn("--iree-opt-generalize-matmul=false", command)
        self.assertIn("--iree-tenstorrent-native-bf16-use-tensix=true", command)
        self.assertFalse(any("compile-from" in item or "global-opt=false" in item for item in command))

    def test_environment_requires_native_without_discarding_metal_cache(self):
        original = {"TT_METAL_SIMULATOR": "1", "TT_METAL_DEVICE_PROFILER": "1",
                    "TT_IREE_CHECKPOINT_OUTPUT": "bad", "TT_IREE_PROFILE_OUTPUT": "stale",
                    "TT_METAL_RUNTIME_ROOT": "/custom/metal", "TT_METAL_CACHE": "/custom/cache"}
        with mock.patch.dict(batch.os.environ, original, clear=True):
            env = batch.environment()
        for key in ("TT_METAL_SIMULATOR", "TT_METAL_DEVICE_PROFILER",
                    "TT_IREE_CHECKPOINT_OUTPUT", "TT_IREE_PROFILE_OUTPUT"):
            self.assertNotIn(key, env)
        self.assertEqual(env["TT_IREE_REQUIRE_DEVICE_NATIVE"], "1")
        self.assertEqual(env["TT_METAL_RUNTIME_ROOT"], "/custom/metal")
        self.assertEqual(env["TT_METAL_CACHE"], "/custom/cache")

    def test_prepare_only_is_stdlib_and_never_invokes_tools(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(batch, "run_logged") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                batch.main(["--prepare-only", "--output-dir", temporary])
            run.assert_not_called()
            manifest = json.loads((Path(temporary) / "expectations.json").read_text())
            self.assertEqual(len(manifest), 6)
            for case in batch.build_cases():
                target = Path(temporary) / case.name
                batch.compare_output(target / "expected.bin", case)
                for tensor in case.inputs:
                    raw = (target / f"{tensor.name}.bin").read_bytes()
                    self.assertEqual(list(struct.unpack("<" + "H" * len(tensor.values), raw)),
                                     [batch.bf16_bits(v) for v in tensor.values])

    def test_compile_only_then_reuse_runs_six_functions_not_a_batch_loop(self):
        cases = batch.build_cases()
        by_name = {f"batch_{case.name}": case for case in cases}

        def fake_run(command, log, env, timeout):
            if command[0] == "compiler":
                Path(command[command.index("-o") + 1]).write_bytes(b"mock vmfb, never executed")
                return
            function = next(value.split("=", 1)[1] for value in command if value.startswith("--function="))
            output = Path(next(value.split("=@", 1)[1] for value in command if value.startswith("--output=")))
            case = by_name[function]
            batch.write_bits(output, case.expected)
            profile_fixture(Path(env["TT_IREE_PROFILE_OUTPUT"]), case)
            self.assertEqual(sum(item.startswith("--input=") for item in command), len(case.inputs))
            for item in command:
                if item.startswith("--input="):
                    self.assertIn("bf16=@", item)

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(batch, "run_logged", side_effect=fake_run) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            batch.main(["--compile-only", "--iree-compile", "compiler", "--output-dir", temporary])
            self.assertEqual(run.call_count, 1)
            self.assertFalse((Path(temporary) / "results.json").exists())
            batch.main(["--skip-compile", "--iree-run-module", "runner", "--output-dir", temporary])
            self.assertEqual(run.call_count, 7)
            results = json.loads((Path(temporary) / "results.json").read_text())
            self.assertEqual(set(results), {case.name for case in cases})
            self.assertTrue(all(row["bitwise_mismatches"] == 0 for row in results.values()))

    def test_case_filter_and_missing_module_fail_without_hardware(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(batch, "run_logged") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                batch.main(["--prepare-only", "--case", "average_fc", "--output-dir", temporary])
            self.assertEqual(set(json.loads((Path(temporary) / "expectations.json").read_text())), {"average_fc"})
            with self.assertRaisesRegex(AssertionError, "missing compiled module"):
                batch.main(["--skip-compile", "--iree-run-module", "runner", "--output-dir", temporary])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
