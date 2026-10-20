#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Hardware-free tests for TTNN benchmark metric semantics."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("benchmark", SCRIPTS / "benchmark_ttnn_resnet50.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
spec = importlib.util.spec_from_file_location("analysis", SCRIPTS / "analyze_ttnn_resnet50_benchmark.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class TimingTest(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(benchmark.percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(benchmark.percentile([4, 1, 3, 2], 95), 3.85)

    def test_batch_latency_is_not_per_image(self):
        result = benchmark.timing_summary([0.00143], 16)
        self.assertAlmostEqual(result["mean_batch_ms"], 1.43)
        self.assertAlmostEqual(result["amortized_ms_per_image_not_latency"], 1.43 / 16)
        self.assertAlmostEqual(result["images_per_second"], 16 / 0.00143)

    def test_stream_normalizes_by_batches_once(self):
        result = benchmark.timing_summary([0.02, 0.04], 16, 20)
        self.assertAlmostEqual(result["mean_batch_ms"], 1.5)
        self.assertAlmostEqual(result["images_per_second"], 16 / 0.0015)
        self.assertEqual(result["windows"], 2)

    def test_invalid_times(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark.timing_summary(values, 16)

    def test_invalid_counts(self):
        for batch, window in ((0, 1), (-1, 1), (16, 0)):
            with self.subTest(batch=batch, window=window), self.assertRaises(ValueError):
                benchmark.timing_summary([1], batch, window)

    def test_invalid_percentile(self):
        for values, percent in (([], 50), ([1], -1), ([1], 101)):
            with self.assertRaises(ValueError):
                benchmark.percentile(values, percent)


class DeviceLogTest(unittest.TestCase):
    HEADER = ("ARCH: blackhole, CHIP_FREQ[MHz]: 1000\n"
              "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],"
              "run host ID,trace id,trace id counter,zone name,type\n")

    def parse(self, text):
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "log.csv"
            filename.write_text(text)
            return analysis.read_kernel_intervals(filename)

    def test_unsynchronized_core_clocks_are_not_subtracted(self):
        durations, riscs, frequency = self.parse(self.HEADER +
            "0,1,1,BRISC,1,100,7,,,BRISC-KERNEL,ZONE_START\n"
            "0,1,1,BRISC,1,110,7,,,BRISC-KERNEL,ZONE_END\n"
            "0,2,1,BRISC,1,9000000,7,,,BRISC-KERNEL,ZONE_START\n"
            "0,2,1,BRISC,1,9000020,7,,,BRISC-KERNEL,ZONE_END\n")
        self.assertEqual(durations[7], 20)
        self.assertEqual(riscs[7]["BRISC"], 20)
        self.assertEqual(frequency, 1000)

    def test_truncated_log_fails(self):
        with self.assertRaisesRegex(ValueError, "unmatched kernel starts"):
            self.parse(self.HEADER + "0,1,1,BRISC,1,100,7,,,BRISC-KERNEL,ZONE_START\n")

    def test_unmatched_end_fails(self):
        with self.assertRaisesRegex(ValueError, "unmatched kernel end"):
            self.parse(self.HEADER + "0,1,1,BRISC,1,100,7,,,BRISC-KERNEL,ZONE_END\n")

    def test_negative_interval_fails(self):
        with self.assertRaisesRegex(ValueError, "negative same-core"):
            self.parse(self.HEADER +
                "0,1,1,BRISC,1,110,7,,,BRISC-KERNEL,ZONE_START\n"
                "0,1,1,BRISC,1,100,7,,,BRISC-KERNEL,ZONE_END\n")

    def test_markers_exclude_warmup(self):
        rows = [{"OP CODE": op, "OP TYPE": kind} for op, kind in (
            ("warmup", "tt_dnn_device"), ("benchmark_batch_0_start", "signpost"),
            ("conv", "tt_dnn_device"), ("benchmark_batch_0_stop", "signpost"))]
        self.assertEqual(analysis.marked_batches(rows)[0][0]["OP CODE"], "conv")

    def test_missing_marker_fails(self):
        with self.assertRaises(ValueError):
            analysis.marked_batches([{"OP CODE": "conv", "OP TYPE": "tt_dnn_device"}])


if __name__ == "__main__":
    unittest.main()
