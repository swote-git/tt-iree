#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""tools/tt_iree_profile.py on a synthetic TT-Metal device log (1 GHz, so one
cycle is one nanosecond)."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    "tt_iree_profile", Path(__file__).resolve().parents[3] / "tools/tt_iree_profile.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)

DEVICE_HEADER = ("PCIe slot, core_x, core_y, RISC processor type, timer_id, "
                 "time[cycles since reset], data, run host ID, trace id, trace id counter, "
                 "zone name, type, source line, source file, meta data")
HOST_FIELDS = ["OP CODE", "OP TYPE", "GLOBAL CALL COUNT", "DEVICE ID", "CORE COUNT",
               "HOST START TS [ns]", "HOST END TS [ns]", "HOST DURATION [ns]",
               "DEVICE KERNEL DURATION [ns]", "DEVICE KERNEL DURATION PER CORE MIN [ns]",
               "DEVICE KERNEL DURATION PER CORE MAX [ns]",
               "DEVICE KERNEL DURATION PER CORE AVG [ns]", "IREE M", "IREE N", "IREE K",
               "IREE PROGRAM SETUP DURATION [ns]", "IREE INVOCATION ID",
               "IREE DEVICE RUN ID"]


# TT-Metal logs (program id << 10) | device id as "run host ID".
def zone(rows, run, core, risc, name, start, end):
    for kind, t in (("ZONE_START", start), ("ZONE_END", end)):
        rows.append(f"0,{core[0]},{core[1]},{risc},1,{t},0,{run << 10},,,{name},{kind},1,x.cc,")


def phase(rows, run, core, risc, name, cycles, entries):
    rows.append(f"0,{core[0]},{core[1]},{risc},2,0,{(entries << 40) | cycles},{run << 10},,,"
                f"TTIREE {name},TS_DATA,1,x.cc,")


def device_log(path, with_phases=True):
    rows = []
    # Run 7: core (1,1) is critical (reader 900 cycles); core (2,1) is short.
    # Core clocks are deliberately offset: only same-core differences count.
    for core, base, rd, cmp, wr in (((1, 1), 10_000, 900, 850, 880),
                                    ((2, 1), 50_000, 400, 380, 390)):
        zone(rows, 7, core, "BRISC", "BRISC-FW", base, base + 1000)
        zone(rows, 7, core, "NCRISC", "NCRISC-KERNEL", base + 10, base + 10 + rd)
        zone(rows, 7, core, "TRISC_0", "TRISC-KERNEL", base + 100, base + 100 + cmp)
        zone(rows, 7, core, "TRISC_1", "TRISC-KERNEL", base + 100, base + 100 + cmp)
        zone(rows, 7, core, "BRISC", "BRISC-KERNEL", base + 15, base + 15 + wr)
        if with_phases:
            phase(rows, 7, core, "NCRISC", "rd.noc_wait", rd // 3, 4)
            phase(rows, 7, core, "NCRISC", "rd.tile", rd - rd // 3, 4)
            phase(rows, 7, core, "TRISC_0", "cmp.cb_wait", cmp - 50, 4)
            phase(rows, 7, core, "TRISC_0", "cmp.matmul", 50, 4)
            phase(rows, 7, core, "TRISC_1", "cmp.matmul", 50, 4)
            phase(rows, 7, core, "BRISC", "wr.out_wait", wr - 100, 1)
            phase(rows, 7, core, "BRISC", "wr.out_issue", 100, 1)
    # Run 8: one core, writer longest.
    zone(rows, 8, (1, 1), "NCRISC", "NCRISC-KERNEL", 20_000, 20_100)
    zone(rows, 8, (1, 1), "BRISC", "BRISC-KERNEL", 20_000, 20_300)
    # Run 0: a program without a host row (the device fill).
    zone(rows, 0, (1, 1), "BRISC", "BRISC-KERNEL", 30_000, 30_050)
    path.write_text("ARCH: blackhole, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 120\n"
                    + DEVICE_HEADER + "\n" + "\n".join(rows) + "\n")


def host_ops(path, run_ids=True):
    rows = [dict(zip(HOST_FIELDS, ["conv", "CONV2D", 5, 0, 2, 1000, 3000, 2000, "", "", "", "",
                                   64, 32, 576, 100, 1, 7 if run_ids else ""])),
            dict(zip(HOST_FIELDS, ["mm", "MATMUL", 6, 0, 1, 4000, 5000, 1000, "", "", "", "",
                                   32, 32, 32, 50, 1, 8 if run_ids else ""]))]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HOST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class DeviceProfileAnalysisTest(unittest.TestCase):
    def analyze(self, temporary, **kwargs):
        root = Path(temporary)
        device_log(root / "device.csv", kwargs.pop("with_phases", True))
        host_ops(root / "ops.csv", kwargs.pop("run_ids", True))
        report = PROFILE.analyze(root / "device.csv", root / "ops.csv", root / "out")
        with open(root / "out/ops_perf_results.csv", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return root / "out", report, rows

    def test_device_columns_follow_same_core_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, report, rows = self.analyze(temporary)
            conv, mm = rows
            self.assertEqual(conv["DEVICE KERNEL DURATION [ns]"], "900")
            # Core spans exceed every single zone: (1,1) 10..950, (2,1) 10..480.
            self.assertEqual(conv["DEVICE KERNEL DURATION PER CORE MAX [ns]"], "940")
            self.assertEqual(conv["DEVICE KERNEL DURATION PER CORE MIN [ns]"], "470")
            self.assertEqual(conv["DEVICE KERNEL DURATION PER CORE AVG [ns]"], "705")
            self.assertEqual(conv["DEVICE FW DURATION [ns]"], "1000")
            self.assertEqual(conv["DEVICE NCRISC KERNEL DURATION [ns]"], "900")
            self.assertEqual(conv["DEVICE TRISC1 KERNEL DURATION [ns]"], "850")
            self.assertEqual(conv["DEVICE TRISC0 KERNEL DURATION [ns]"], "850")
            self.assertEqual(conv["DEVICE TRISC2 KERNEL DURATION [ns]"], "")
            self.assertEqual(conv["DEVICE CRITICAL CORE"], "1-1")
            self.assertEqual(conv["DEVICE CRITICAL RISC"], "NCRISC")
            # Reader never waits for CB space; unpack waits 800 of 850 for
            # inputs; the writer waits 780 of 880 for output tiles.
            self.assertEqual(conv["DEVICE READER UTILIZATION"], "1.000")
            self.assertEqual(conv["DEVICE COMPUTE UTILIZATION"], "0.059")
            self.assertEqual(conv["DEVICE WRITER UTILIZATION"], "0.114")
            self.assertEqual(conv["DEVICE BOUND STAGE"], "reader")
            self.assertEqual(mm["DEVICE BOUND STAGE"], "")
            self.assertEqual(mm["DEVICE KERNEL DURATION [ns]"], "300")
            self.assertEqual(mm["DEVICE CRITICAL RISC"], "BRISC")
            self.assertIn("1 device runs without a host row (ids 0)", report)
            self.assertIn("phase markers   yes", report)
            self.assertIn("1.00/0.06/0.11", report)

    def test_phase_table_reports_critical_and_median_cores(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, report, _ = self.analyze(temporary)
            with open(out / "kernel_phases.csv", newline="") as stream:
                phases = {(r["RISC"], r["PHASE"]): r for r in csv.DictReader(stream)}
            tile = phases[("NCRISC", "rd.tile")]
            self.assertEqual(tile["CRITICAL CORE [ns]"], "600")
            self.assertEqual(tile["CRITICAL CORE ENTRIES"], "4")
            self.assertEqual(tile["MEDIAN CORE [ns]"], "434")  # median(600, 267)
            self.assertEqual(tile["MAX CORE [ns]"], "600")
            self.assertEqual(tile["CORES"], "2")
            self.assertIn("rd.tile 0.00 (67%)", report)
            self.assertIn("rd.noc_wait* 0.00 (33%)", report)

    def test_trace_keeps_core_offsets_and_names_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, _, _ = self.analyze(temporary)
            events = json.loads((out / "trace.json").read_text())["traceEvents"]
            names = {e["args"]["name"] for e in events if e.get("name") == "process_name"}
            self.assertEqual(names, {"host", "core 1-1", "core 2-1"})
            reader = [e for e in events if e.get("ph") == "X" and e["pid"] != 0
                      and e["tid"] == 1 and e["name"] == "CONV2D #5"]
            # Launch at host start + setup (100 ns), reader 10 cycles after FW start.
            self.assertEqual(sorted((e["ts"], e["dur"]) for e in reader),
                             [(0.11, 0.4), (0.11, 0.9)])
            self.assertEqual(reader[0]["args"]["phases [ns]"].keys(), {"rd.noc_wait", "rd.tile"})

    def test_logs_without_run_ids_match_in_order_by_core_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, report, rows = self.analyze(temporary, run_ids=False, with_phases=False)
            self.assertEqual([r["DEVICE KERNEL DURATION [ns]"] for r in rows], ["900", "300"])
            self.assertIn("phase markers   no", report)


if __name__ == "__main__":
    unittest.main()
