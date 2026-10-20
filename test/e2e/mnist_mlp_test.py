#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Hardware-free dataset selection and native-profile contract tests."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_mnist_mlp import balanced_test_indices
from run_mnist_mlp import require_mnist_profile


class SelectionTest(unittest.TestCase):
    def test_balanced_unique_and_deterministic(self):
        labels = [digit for digit in range(10) for _ in range(20)]
        selected = balanced_test_indices(labels, 100, 20261020)
        self.assertEqual(len(set(selected)), 100)
        self.assertEqual(selected, balanced_test_indices(labels, 100, 20261020))
        self.assertNotEqual(selected, balanced_test_indices(labels, 100, 2))
        self.assertEqual([sum(labels[i] == digit for i in selected) for digit in range(10)], [10] * 10)

    def test_invalid_selection_is_rejected(self):
        for samples in (0, 1, 11, -10):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                balanced_test_indices(list(range(10)), samples, 0)
        with self.assertRaises(ValueError):
            balanced_test_indices(list(range(10)), 20, 0)


class ProfileTest(unittest.TestCase):
    def fixture(self, root, engine="tensix"):
        summary = {"all_dispatches_device_native": True, "dispatch_count": 2}
        rows = []
        for m, n, k in ((1, 128, 784), (1, 10, 128)):
            row = {"IREE M": str(m), "IREE N": str(n), "IREE K": str(k),
                   "OP TYPE": "LINEAR", "IREE EXECUTION DOMAIN": "device_native",
                   "INPUT_0_DATATYPE": "BF16", "OUTPUT_0_DATATYPE": "BF16",
                   "IREE BUILTIN PROGRAM": "12" if engine == "tensix" else "11",
                   "IREE COMPUTE DATATYPE": "BF16_HIFI4_FP32_DST_F32_EPILOGUE" if engine == "tensix"
                   else "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING"}
            for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
                row[f"IREE {stage} DURATION [ns]"] = "0"
            rows.append(row)
        return summary, rows

    def save(self, root, summary, rows):
        (root / "run_results.json").write_text(json.dumps(summary))
        with (root / "ops_perf_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_native_precisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for engine in ("tensix", "riscv"):
                summary, rows = self.fixture(root, engine)
                self.save(root, summary, rows)
                self.assertEqual(require_mnist_profile(root, 1, engine)["dispatch_count"], 2)

    def test_geometry_domain_storage_and_staging_must_match(self):
        cases = (("IREE K", "783"), ("OP TYPE", "MATMUL"),
                 ("IREE EXECUTION DOMAIN", "host"), ("INPUT_0_DATATYPE", "F32"),
                 ("OUTPUT_0_DATATYPE", "F32"), ("IREE BUILTIN PROGRAM", "11"),
                 ("IREE COMPUTE DATATYPE", "wrong"), ("IREE UPLOAD DURATION [ns]", "1"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value in cases:
                summary, rows = self.fixture(root)
                rows[0][key] = value
                self.save(root, summary, rows)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    require_mnist_profile(root, 1, "tensix")

    def test_incomplete_and_host_profiles_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in (("dispatch_count", 1), ("all_dispatches_device_native", False)):
                summary, rows = self.fixture(root)
                summary[field] = value
                self.save(root, summary, rows)
                with self.assertRaises(ValueError):
                    require_mnist_profile(root, 1, "tensix")


if __name__ == "__main__":
    unittest.main()
