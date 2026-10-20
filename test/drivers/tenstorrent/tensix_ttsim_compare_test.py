#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""CPU-only tests for precision replay artifact and failure classification."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

import tensix_ttsim_compare as replay


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.output = self.root / "output"
        self.meta = {
            "format": "bf16", "operation": "matmul", "math_fidelity": 4,
            "repeats": 2, "logical_k": 32, "k_slice_width": 32,
            "fill_bits": 0x3f812345, "rhs_component": "full",
            "fp32_dest_acc_en": True, "math_approx_mode": False,
            "bitwise_mismatches": 2048,
        }
        for directory in (self.reference, self.output):
            directory.mkdir()
            self.save_meta(directory)
            for artifact in replay.ARTIFACTS:
                (directory / artifact).write_bytes(b"identical")
            np.ones(2048, dtype=np.float32).tofile(directory / "actual.f32")

    def save_meta(self, directory):
        (directory / "summary.json").write_text(json.dumps(self.meta))

    def test_archived_match_is_not_mathematical_correctness(self):
        result = replay.compare(self.reference, self.output)
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["mathematical_reference_mismatches"], 2048)

    def test_numerical_mismatch(self):
        np.full(2048, 2, dtype=np.float32).tofile(self.output / "actual.f32")
        result = replay.compare(self.reference, self.output)
        self.assertEqual(result["status"], "numerical-mismatch")
        self.assertEqual(result["bitwise_mismatches"], 2048)

    def test_input_or_kernel_mismatch_is_not_numerical_failure(self):
        (self.output / "compute.cpp").write_bytes(b"changed kernel")
        result = replay.compare(self.reference, self.output)
        self.assertEqual(result["status"], "contract-mismatch")
        self.assertEqual(result["changed_input_or_kernel_artifacts"], ["compute.cpp"])

    def test_contract_mismatch(self):
        self.meta["math_fidelity"] = 2
        self.save_meta(self.output)
        result = replay.compare(self.reference, self.output)
        self.assertEqual(result["status"], "contract-mismatch")
        self.assertEqual(result["changed_contract_fields"], ["math_fidelity"])

    def test_output_shape_validation(self):
        np.ones(1024, dtype=np.float32).tofile(self.output / "actual.f32")
        with self.assertRaisesRegex(ValueError, "Output shape"):
            replay.compare(self.reference, self.output)

    def test_nonfinite_output_is_not_a_numeric_match(self):
        np.full(2048, np.nan, dtype=np.float32).tofile(self.output / "actual.f32")
        result = replay.compare(self.reference, self.output)
        self.assertEqual(result["status"], "nonfinite-or-unstable")
        self.assertIsNone(result["max_abs_difference"])

    def test_component_replay_is_rejected(self):
        self.meta["rhs_component"] = "low5"
        self.save_meta(self.reference)
        with self.assertRaisesRegex(ValueError, "unmodified RHS"):
            replay.replay_command(Path("probe"), self.reference, self.output)

    def test_unstable_output_is_not_a_stable_numerical_mismatch(self):
        values = np.ones(2048, dtype=np.float32)
        values[1024] = 2
        values.tofile(self.output / "actual.f32")
        self.assertEqual(replay.compare(self.reference, self.output)["status"], "nonfinite-or-unstable")

    def test_replay_preserves_prefill_and_prepack_check(self):
        self.meta.update(dst_prefill_enabled=True, dst_prefill_bits=0x4bc00000,
                         dst_check_enabled=True, dst_check_bits=0x4bc00001)
        self.save_meta(self.reference)
        command = replay.replay_command(Path("probe"), self.reference, self.output)
        self.assertEqual(command[command.index("--dst-prefill-bits") + 1], "4bc00000")
        self.assertEqual(command[command.index("--dst-check-bits") + 1], "4bc00001")
        self.assertNotIn("--require-exact", command)

    def test_environment_does_not_change_sfpu_code_or_enable_fast_paths(self):
        env = replay.environment(self.root, self.root / "cache", self.root / "sim.so", "slow")
        self.assertEqual(env["TT_METAL_SLOW_DISPATCH_MODE"], "1")
        self.assertNotIn("TT_METAL_DISABLE_SFPLOADMACRO", env)
        self.assertNotIn("TT_METAL_SIMULATOR_DIRECT_TENSOR_WRITES", env)
        self.assertNotIn("TT_SIMULATOR_DRAM_TELEPORT", env)

    def test_timeout_is_not_numerical_failure(self):
        result = replay.execute(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            os.environ.copy(), self.root / "timeout.log", 0.1,
        )
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
