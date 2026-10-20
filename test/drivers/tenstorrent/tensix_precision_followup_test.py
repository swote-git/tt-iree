#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""CPU-only checks for diagnostic arithmetic, not hardware correctness gates."""

from pathlib import Path
import runpy
import unittest

import numpy as np

from analyze_tensix_precision_followup import predict, round_fp32
from tensix_precision_followup import MODEL_SHA256, digest, error_metrics, quantize


class FollowupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = Path(__file__).with_name("analyze_tensix_reduction.py")
        cls.frozen = runpy.run_path(str(cls.model_path))

    def test_frozen_model_hash(self):
        self.assertEqual(digest(self.model_path), MODEL_SHA256)

    def test_bf16_rne_quantization(self):
        inputs = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], np.uint32)
        expected = np.array([0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000], np.uint32)
        np.testing.assert_array_equal(quantize(inputs.view(np.float32), "bf16").view(np.uint32), expected)
        np.testing.assert_array_equal(quantize(inputs.view(np.float32), "f32").view(np.uint32), inputs)

    def test_ties_up_both_signs_and_parities(self):
        exact = np.array([25165825, 25165827, -25165825, -25165827], np.float64)
        expected = np.array([25165826, 25165828, -25165824, -25165826], np.float64)
        np.testing.assert_array_equal(round_fp32(exact, "up", self.frozen["round_fp32"]), expected)

    def test_ties_up_preserves_nonties(self):
        exact = np.array([25165824.5, 25165825.5, -25165824.5, -25165825.5], np.float64)
        np.testing.assert_array_equal(round_fp32(exact, "up", self.frozen["round_fp32"]), exact.astype(np.float32).astype(np.float64))

    def test_posthoc_baseline_is_original_predictor(self):
        rng = np.random.default_rng(765432)
        for data_format in ("bf16", "f32"):
            a = quantize(rng.uniform(-5, 5, (32, 32)), data_format)
            b = quantize(rng.uniform(-5, 5, (32, 32)), data_format)
            for width in (1, 8, 32):
                expected = self.frozen["predict"](a, b, data_format=data_format, slice_width=width)
                actual = predict(a, b, data_format, width, "away", "fused", self.frozen["round_fp32"])
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_relative_metrics_zero_reference(self):
        zeros = np.zeros((32, 32), np.float32)
        metric = error_metrics(zeros, zeros)
        self.assertEqual(metric["bitwise_mismatches"], 0)
        self.assertIsNone(metric["relative_l2"])
        self.assertEqual(metric["max_output_ulp_difference"], 0)


if __name__ == "__main__":
    unittest.main()
