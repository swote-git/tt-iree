#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    "transfer_analysis", Path(__file__).resolve().parents[3] / "tools/analyze_runtime_transfers.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
COPY_TEST = None
EXPECT_DEVICE_FILL = False


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


class TransferAnalysisTest(unittest.TestCase):
    def test_interval_union_and_intersection(self):
        self.assertEqual(ANALYSIS.duration([(0, 10), (5, 15), (20, 25)]), 20)
        self.assertEqual(ANALYSIS.intersection_duration([(0, 10), (8, 20)], [(5, 15)]), 10)
        with self.assertRaises(ValueError):
            ANALYSIS.duration([(2, 1)])

    def test_invoke_excludes_io_and_nested_stages_are_not_added_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def row(operation, start, end, phase="invoke", inside="0"):
                return {"OPERATION": operation, "INVOCATION ID": "1",
                        "WITHIN DISPATCH": inside, "INVOCATION PHASE": phase,
                        "HOST START TS [ns]": start, "HOST END TS [ns]": end,
                        "HOST DURATION [ns]": end - start, "STATUS CODE": 0,
                        "READFROMBUFFER BYTES": 4096, "WRITETOBUFFER BYTES": 4096,
                        "READ EXCLUSIVE [ns]": (end - start) // 2,
                        "OTHER [ns]": end - start - (end - start) // 2}

            write_csv(directory / "transfer_perf_results.csv", [
                row("UNMAP", 0, 20, "input"), row("FILL", 20, 50),
                row("MAP", 55, 60, inside="1"), row("FILL", 70, 100),
                row("MAP", 125, 150, "output"),
            ])
            write_csv(directory / "ops_perf_results.csv", [
                {"IREE INVOCATION ID": 1, "HOST START TS [ns]": 50, "HOST END TS [ns]": 70},
                {"IREE INVOCATION ID": 1, "HOST START TS [ns]": 100, "HOST END TS [ns]": 120},
            ])
            write_csv(directory / "timings.csv", [{"sample": 0, "invoke_seconds": "0.000000105"}])
            result = ANALYSIS.analyze(directory, directory / "timings.csv")["invocations"][0]
            self.assertEqual(result["dispatch_wall_union_ns"], 40)
            self.assertEqual(result["invoke_transfer_outside_dispatch_wall_union_ns"], 60)
            self.assertEqual(result["accounted_invoke_wall_union_ns"], 100)
            self.assertEqual(result["unattributed_invoke_ns"], 5)
            self.assertEqual(result["internal_dispatch_gap_ns"], 30)
            self.assertEqual(result["internal_gap_transfer_coverage_ns"], 30)
            self.assertEqual(result["invoke_operations"]["FILL"]["calls"], 2)
            self.assertEqual(result["readfrombuffer_bytes_invoke"], 3 * 4096)
            with (directory / "transfer_perf_results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["OTHER [ns]"] = "999"
            write_csv(directory / "transfer_perf_results.csv", rows)
            with self.assertRaisesRegex(ValueError, "exclusive transfer stages"):
                ANALYSIS.analyze(directory)

    def test_mock_hal_records_root_commands_and_resolved_ranges(self):
        if not COPY_TEST:
            self.skipTest("pass --copy-test for mock HAL integration")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            environment = os.environ.copy()
            environment["TT_IREE_PROFILE_OUTPUT"] = temporary
            run = subprocess.run([
                COPY_TEST, "--gtest_filter=CopyTest.RecordedFillHandlesDirectAndIndirectSubranges:"
                "CopyTest.RecordedUpdateSnapshotsHostData:CopyTest.DirectSubrangesPreserveSurroundingBytes",
            ], env=environment, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("[  PASSED  ] 3 tests.", run.stdout)
            result = ANALYSIS.analyze(directory)
            self.assertEqual(result["operation_counts_all_phases"]["FILL"], 2)
            self.assertEqual(result["operation_counts_all_phases"]["UPDATE"], 1)
            self.assertEqual(result["operation_counts_all_phases"]["COPY"], 1)
            with (directory / "transfer_perf_results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            fill_rows = [row for row in rows if row["OPERATION"] == "FILL"]
            self.assertEqual([int(row["TARGET OFFSET"]) for row in fill_rows], [256, 1088])
            self.assertEqual([int(row["REQUESTED BYTES"]) for row in fill_rows], [512, 384])
            self.assertEqual([int(row["COMMAND ORDINAL"]) for row in fill_rows], [0, 1])
            copy = next(row for row in rows if row["OPERATION"] == "COPY")
            self.assertEqual(int(copy["SOURCE OFFSET"]), 137)
            self.assertEqual(int(copy["TARGET OFFSET"]), 913)
            self.assertEqual(int(copy["REQUESTED BYTES"]), 777)
            self.assertNotEqual(copy["SOURCE BUFFER ID"], copy["TARGET BUFFER ID"])
            for row in rows:
                self.assertEqual(int(row["STATUS CODE"]), 0)
                if not EXPECT_DEVICE_FILL or row["OPERATION"] == "FILL":
                    self.assertEqual(int(row["READFROMBUFFER BYTES"]), 0)
                    self.assertEqual(int(row["WRITETOBUFFER BYTES"]), 0)
                if EXPECT_DEVICE_FILL and row["OPERATION"] == "FILL":
                    self.assertEqual(int(row["DEVICE FILL BYTES"]), int(row["REQUESTED BYTES"]))
            summary = json.loads((directory / "run_results.json").read_text())
            self.assertEqual(summary["transfer_count"], len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-test")
    parser.add_argument("--expect-device-fill", action="store_true")
    args, remaining = parser.parse_known_args()
    COPY_TEST = args.copy_test
    EXPECT_DEVICE_FILL = args.expect_device_fill
    unittest.main(argv=[sys.argv[0]] + remaining)
