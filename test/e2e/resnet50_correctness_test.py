#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import unittest


def _load_module(path):
    spec = importlib.util.spec_from_file_location("resnet50_correctness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_module(pathlib.Path(sys.argv.pop(1)))


class ResNet50CorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, shape, values):
        path = self.root / name
        TOOL.write_f32_npy(path, shape, values)
        return path

    def test_npy_round_trip(self):
        path = self.write("values.npy", (1, 4), [1.0, -2.5, 0.0, 4.25])
        array = TOOL.read_npy(path)
        self.assertEqual(array.shape, (1, 4))
        self.assertEqual(array.descr, "<f4")
        self.assertEqual(array.values, (1.0, -2.5, 0.0, 4.25))

    def test_tolerance_passes_and_reports_topk(self):
        expected = self.write("expected.npy", (1, 4), [0.0, 1.0, 2.0, 3.0])
        actual = self.write("actual.npy", (1, 4), [0.0, 1.001, 2.0, 3.0])
        result = TOOL.compare_files(expected, actual, atol=0.002, rtol=0.0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_top1"], 3)
        self.assertEqual(result["actual_top1"], 3)

    def test_value_mismatch_fails(self):
        expected = self.write("expected.npy", (2,), [1.0, 2.0])
        actual = self.write("actual.npy", (2,), [1.0, 2.1])
        result = TOOL.compare_files(expected, actual, atol=0.01, rtol=0.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_mismatch_index"], 1)
        self.assertEqual(result["failure"], "values outside tolerance")

    def test_shape_mismatch_fails(self):
        expected = self.write("expected.npy", (2,), [1.0, 2.0])
        actual = self.write("actual.npy", (1, 2), [1.0, 2.0])
        result = TOOL.compare_files(expected, actual, atol=0.0, rtol=0.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "shape mismatch")

    def test_nonfinite_output_fails(self):
        expected = self.write("expected.npy", (1,), [1.0])
        actual = self.write("actual.npy", (1,), [math.nan])
        result = TOOL.compare_files(expected, actual, atol=0.0, rtol=0.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["actual_nonfinite_count"], 1)
        self.assertEqual(result["failure"], "non-finite values")

    def write_checkpoints(self, name, records):
        directory = self.root / name
        directory.mkdir()
        manifest_records = []
        for index, (op_code, values) in enumerate(records, start=1):
            file_name = f"dispatch_{index:04d}_{op_code}.npy"
            TOOL.write_f32_npy(directory / file_name, (len(values),), values)
            manifest_records.append(
                {
                    "dispatch_index": index,
                    "op_code": op_code,
                    "builtin_program": 4,
                    "dispatch_kind": 1,
                    "output_binding": 2,
                    "shape": [len(values)],
                    "file": file_name,
                }
            )
        manifest = {
            "schema_version": 1,
            "dtype": "float32",
            "checkpoint_count": len(manifest_records),
            "records": manifest_records,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory

    def test_checkpoint_comparison_passes(self):
        expected = self.write_checkpoints(
            "expected", [("conv0", [1.0, 2.0]), ("conv1", [3.0])]
        )
        actual = self.write_checkpoints(
            "actual", [("conv0", [1.0, 2.0]), ("conv1", [3.00001])]
        )
        result = TOOL.compare_checkpoint_directories(
            expected, actual, atol=0.0001, rtol=0.0
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["compared_checkpoint_count"], 2)

    def test_checkpoint_comparison_finds_first_divergence(self):
        expected = self.write_checkpoints(
            "expected", [("conv0", [1.0, 2.0]), ("conv1", [3.0])]
        )
        actual = self.write_checkpoints(
            "actual", [("conv0", [1.0, 2.0]), ("conv1", [4.0])]
        )
        result = TOOL.compare_checkpoint_directories(
            expected, actual, atol=0.001, rtol=0.0
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_divergence"]["dispatch_index"], 2)
        self.assertEqual(result["first_divergence"]["op_code"], "conv1")


if __name__ == "__main__":
    unittest.main()
