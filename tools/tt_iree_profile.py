#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""tt-iree device kernel profiler.

  run      Run a command (iree-run-module, tt-iree-resnet-eval, ...) built
           against a Tracy-enabled TT-Metal with the device profiler, the
           tt-iree host profiler and kernel phase markers enabled, then analyze.
  analyze  Join TT-Metal's profile_log_device.csv with tt-iree's
           ops_perf_results.csv.

Outputs in --output:
  ops_perf_results.csv  host rows with the DEVICE ... columns filled
  kernel_phases.csv     per dispatch, RISC and phase: critical-core and
                        median-core time
  report.txt            op, RISC and phase summary and the slowest dispatches
  trace.json            Chrome trace for ui.perfetto.dev

Timing definitions:
  - Cycle counters are per core, so only same-core differences are exact.
  - DEVICE KERNEL DURATION is the longest single *-KERNEL zone of a dispatch
    over all cores and RISCs (the TTNN comparison definition in reports/).
  - A core's kernel duration is its last kernel end minus its first kernel
    start over its RISCs. The critical core has the longest one.
  - Phases are cycle totals that tt-iree's generated kernels record per RISC
    when TT_IREE_KERNEL_PHASES=1 ("TTIREE <phase>" markers). Phase names that
    contain "wait" are time spent blocked on a CB, the NoC or another RISC.
    TRISC phases time instruction issue: CB waits poll on the RISC and are
    exact, while Tensix semaphore waits (Dst and source register handoffs)
    surface in the next issuing phase, so compare TRISC phases only within
    one thread.
  - Stage utilization (critical core) is the share of time a pipeline stage is
    not blocked by its neighbours: reader = 1 - rd.cb_wait (CB full);
    compute = 1 - unpack cmp.cb_wait (inputs empty) - pack cmp.out_wait
    (output full); writer = 1 - wr.out_wait - wr.operand_cb_wait. The stage
    with the highest utilization bounds the dispatch.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

RISCS = ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2")
RISC_COLUMN = {"BRISC": "BRISC", "NCRISC": "NCRISC", "TRISC_0": "TRISC0",
               "TRISC_1": "TRISC1", "TRISC_2": "TRISC2"}
# Kernel roles of tt-iree native programs.
RISC_ROLE = {"BRISC": "writer", "NCRISC": "reader", "TRISC_0": "unpack",
             "TRISC_1": "math", "TRISC_2": "pack"}
PHASE_PREFIX = "TTIREE "
CYCLE_MASK = (1 << 40) - 1
STAGES = ("reader", "compute", "writer")
# RISC whose phases describe each stage in the report.
STAGE_RISC = {"reader": "NCRISC", "compute": "TRISC_0", "writer": "BRISC"}
DEVICE_COLUMNS = ["DEVICE FW DURATION [ns]"] + [
    f"DEVICE {RISC_COLUMN[r]} KERNEL DURATION [ns]" for r in RISCS] + [
    "DEVICE CORE COUNT", "DEVICE CRITICAL CORE", "DEVICE CRITICAL RISC",
    "DEVICE BOUND STAGE"] + [f"DEVICE {s.upper()} UTILIZATION" for s in STAGES]


def base_program_id(run_host_id):
    """TT-Metal logs (program id << 10) | device id (EncodePerDeviceProgramID);
    tt-iree records the program id."""
    return (run_host_id & ((1 << 31) - 1)) >> 10


def is_wait(phase):
    return "wait" in phase


class DeviceRun:
    """Zones of one device program (one "run host ID")."""

    def __init__(self):
        self.kernel = defaultdict(list)  # (core, risc) -> [(start, end)]
        self.fw = defaultdict(list)
        self.phases = defaultdict(lambda: [0, 0])  # (core, risc, phase) -> [cycles, entries]

    def cores(self):
        return sorted({core for core, _ in self.kernel})


def read_device_log(path):
    """Returns (clock MHz, {run id: DeviceRun})."""
    starts, ends = defaultdict(list), defaultdict(list)
    runs = defaultdict(DeviceRun)
    with open(path, newline="") as stream:
        header = stream.readline()
        if "CHIP_FREQ[MHz]:" not in header:
            raise ValueError(f"{path}: not a TT-Metal device profile log")
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        for row in csv.DictReader(stream, skipinitialspace=True):
            zone, kind = row["zone name"], row["type"]
            run = base_program_id(int(row["run host ID"] or 0))
            core = (int(row["core_x"]), int(row["core_y"]))
            risc = row["RISC processor type"]
            if kind == "TS_DATA" and zone.startswith(PHASE_PREFIX):
                data = int(row["data"])
                total = runs[run].phases[(core, risc, zone[len(PHASE_PREFIX):])]
                total[0] += data & CYCLE_MASK
                total[1] += data >> 40
            elif kind in ("ZONE_START", "ZONE_END") and zone.endswith(("-KERNEL", "-FW")):
                key = (run, core, risc, "fw" if zone.endswith("-FW") else "kernel")
                (starts if kind == "ZONE_START" else ends)[key].append(
                    int(row["time[cycles since reset]"]))
    for key, begin in starts.items():
        run, core, risc, which = key
        intervals = list(zip(sorted(begin), sorted(ends.get(key, []))))
        getattr(runs[run], which)[(core, risc)].extend(intervals)
    return mhz, dict(runs)


def summarize_run(run, mhz):
    """Per-dispatch device metrics in ns (single launch per run id)."""
    ns = lambda cycles: cycles * 1000.0 / mhz
    span = {}
    for core in run.cores():
        intervals = [i for risc in RISCS for i in run.kernel.get((core, risc), [])]
        span[core] = max(e for _, e in intervals) - min(s for s, _ in intervals)
    longest = {key: max(e - s for s, e in v) for key, v in run.kernel.items() if v}
    critical_core = max(span, key=span.get)
    critical_risc = max((r for r in RISCS if (critical_core, r) in longest),
                        key=lambda r: longest[(critical_core, r)])
    fw = [max(e for _, e in v) - min(s for s, _ in v) for v in run.fw.values() if v]
    per_core = [ns(v) for v in span.values()]
    risc_max = {r: max((v for (c, rr), v in longest.items() if rr == r), default=None)
                for r in RISCS}
    # Pipeline stage utilization on the critical core.
    ph = {(rr, p): v[0] for (c, rr, p), v in run.phases.items() if c == critical_core}
    k = lambda r: longest.get((critical_core, r))
    util = {}
    if ph:
        if k("NCRISC"):
            util["reader"] = 1 - ph.get(("NCRISC", "rd.cb_wait"), 0) / k("NCRISC")
        if k("TRISC_0"):
            blocked = ph.get(("TRISC_0", "cmp.cb_wait"), 0) / k("TRISC_0")
            if k("TRISC_2"):
                blocked += ph.get(("TRISC_2", "cmp.out_wait"), 0) / k("TRISC_2")
            util["compute"] = max(0.0, 1 - blocked)
        if k("BRISC"):
            util["writer"] = 1 - (ph.get(("BRISC", "wr.out_wait"), 0) + ph.get(
                ("BRISC", "wr.operand_cb_wait"), 0)) / k("BRISC")
    return {
        "kernel_ns": ns(max(longest.values())),
        "per_core_min_ns": min(per_core), "per_core_max_ns": max(per_core),
        "per_core_avg_ns": statistics.mean(per_core),
        "fw_ns": ns(max(fw)) if fw else None,
        "risc_ns": {r: (ns(v) if v is not None else None) for r, v in risc_max.items()},
        "cores": len(span), "critical_core": critical_core,
        "critical_risc": critical_risc,
        "critical_risc_ns": {r: ns(longest[(critical_core, r)]) for r in RISCS
                             if (critical_core, r) in longest},
        "utilization": util,
        "bound": max(util, key=util.get) if util else "",
    }


def phase_rows(run, mhz, critical_core):
    """(risc, phase) -> critical-core ns/entries, median and max over cores."""
    ns = lambda cycles: cycles * 1000.0 / mhz
    grouped = defaultdict(dict)
    for (core, risc, phase), (cycles, entries) in run.phases.items():
        grouped[(risc, phase)][core] = (cycles, entries)
    out = {}
    for (risc, phase), by_core in grouped.items():
        values = [ns(c) for c, _ in by_core.values()]
        critical = by_core.get(critical_core, (0, 0))
        out[(risc, phase)] = {
            "critical_ns": ns(critical[0]), "critical_entries": critical[1],
            "median_ns": statistics.median(values), "max_ns": max(values),
            "cores": len(values)}
    return out


def match_runs(host_rows, runs):
    """Host row index -> device run id."""
    ids = [int(r.get("IREE DEVICE RUN ID") or 0) for r in host_rows]
    if any(ids):
        return {i: rid for i, rid in enumerate(ids) if rid in runs}
    # Logs without run ids: device runs in launch order, matched by core count.
    order = sorted(rid for rid in runs if rid)
    matched, cursor = {}, 0
    for i, row in enumerate(host_rows):
        cores = int(row.get("CORE COUNT") or 0)
        while cursor < len(order) and len(runs[order[cursor]].cores()) != cores:
            cursor += 1
        if cursor == len(order):
            break
        matched[i] = order[cursor]
        cursor += 1
    return matched


def fmt(value, digits=0):
    return "" if value is None else f"{value:.{digits}f}"


def analyze(device_log, ops, output, top=15, include_first=False, trace_invocations=None):
    output.mkdir(parents=True, exist_ok=True)
    mhz, runs = read_device_log(device_log)
    with open(ops, newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        host_rows = list(reader)
    matched = match_runs(host_rows, runs)
    summaries, phases = {}, {}
    for i, rid in matched.items():
        summaries[i] = summarize_run(runs[rid], mhz)
        phases[i] = phase_rows(runs[rid], mhz, summaries[i]["critical_core"])
    has_phases = any(phases.values())

    # ops_perf_results.csv with device columns.
    out_fields = fields + [c for c in DEVICE_COLUMNS if c not in fields]
    with open(output / "ops_perf_results.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(host_rows):
            row = dict(row)
            s = summaries.get(i)
            if s:
                row["DEVICE KERNEL DURATION [ns]"] = fmt(s["kernel_ns"])
                row["DEVICE KERNEL DURATION PER CORE MIN [ns]"] = fmt(s["per_core_min_ns"])
                row["DEVICE KERNEL DURATION PER CORE MAX [ns]"] = fmt(s["per_core_max_ns"])
                row["DEVICE KERNEL DURATION PER CORE AVG [ns]"] = fmt(s["per_core_avg_ns"])
                row["DEVICE FW DURATION [ns]"] = fmt(s["fw_ns"])
                for r in RISCS:
                    row[f"DEVICE {RISC_COLUMN[r]} KERNEL DURATION [ns]"] = fmt(s["risc_ns"][r])
                row["DEVICE CORE COUNT"] = s["cores"]
                row["DEVICE CRITICAL CORE"] = "%d-%d" % s["critical_core"]
                row["DEVICE CRITICAL RISC"] = s["critical_risc"]
                row["DEVICE BOUND STAGE"] = s["bound"]
                for stage in STAGES:
                    u = s["utilization"].get(stage)
                    row[f"DEVICE {stage.upper()} UTILIZATION"] = fmt(u, 3)
            writer.writerow(row)

    with open(output / "kernel_phases.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["GLOBAL CALL COUNT", "OP CODE", "OP TYPE", "IREE INVOCATION ID",
                         "RISC", "PHASE", "CRITICAL CORE [ns]", "CRITICAL CORE ENTRIES",
                         "MEDIAN CORE [ns]", "MAX CORE [ns]", "CORES"])
        for i in sorted(phases):
            row = host_rows[i]
            for (risc, phase), v in sorted(phases[i].items(),
                                           key=lambda kv: (RISCS.index(kv[0][0]), kv[0][1])):
                writer.writerow([row["GLOBAL CALL COUNT"], row["OP CODE"], row["OP TYPE"],
                                 row.get("IREE INVOCATION ID", ""), risc, phase,
                                 fmt(v["critical_ns"]), v["critical_entries"],
                                 fmt(v["median_ns"]), fmt(v["max_ns"]), v["cores"]])

    report = build_report(device_log, mhz, host_rows, runs, matched, summaries, phases,
                          has_phases, top, include_first)
    (output / "report.txt").write_text(report)
    write_trace(output / "trace.json", host_rows, runs, matched, summaries, phases, mhz,
                trace_invocations)
    return report


def invocations_of(host_rows, include_first):
    ids = sorted({int(r.get("IREE INVOCATION ID") or 0) for r in host_rows})
    if len(ids) > 1 and not include_first:
        ids = ids[1:]
    return ids


def build_report(device_log, mhz, host_rows, runs, matched, summaries, phases, has_phases,
                 top, include_first):
    lines = []
    out = lines.append
    inv_ids = invocations_of(host_rows, include_first)
    selected = [i for i, r in enumerate(host_rows)
                if int(r.get("IREE INVOCATION ID") or 0) in inv_ids]
    unmatched_runs = sorted(set(runs) - set(matched.values()))
    out("tt-iree device kernel profile")
    out(f"  device log      {device_log}")
    out(f"  clock           {mhz:.0f} MHz")
    out(f"  dispatches      {len(host_rows)} host rows, {len(matched)} with a device program")
    out(f"  other programs  {len(unmatched_runs)} device runs without a host row "
        f"(ids {', '.join(map(str, unmatched_runs[:8]))}{' ...' if len(unmatched_runs) > 8 else ''})")
    out(f"  phase markers   {'yes' if has_phases else 'no (run with TT_IREE_KERNEL_PHASES=1)'}")
    out(f"  summarized      invocations {', '.join(map(str, inv_ids))} "
        f"({'all' if include_first else 'first excluded'}), values per invocation")
    out("")
    n_inv = max(len(inv_ids), 1)

    out("Per invocation")
    out(f"  {'invocation':>10} {'dispatches':>10} {'host ms':>10} {'kernel ms':>10} {'kernel/host':>11}")
    for inv in sorted({int(r.get('IREE INVOCATION ID') or 0) for r in host_rows}):
        rows = [i for i, r in enumerate(host_rows) if int(r.get("IREE INVOCATION ID") or 0) == inv]
        host = sum(int(host_rows[i]["HOST DURATION [ns]"] or 0) for i in rows) / 1e6
        kernel = sum(summaries[i]["kernel_ns"] for i in rows if i in summaries) / 1e6
        out(f"  {inv:>10} {len(rows):>10} {host:>10.2f} {kernel:>10.2f} "
            f"{(kernel / host if host else 0):>10.1%}")
    out("")

    by_type = defaultdict(list)
    for i in selected:
        by_type[host_rows[i]["OP TYPE"]].append(i)
    total_kernel = sum(summaries[i]["kernel_ns"] for i in selected if i in summaries)
    out("Op types (kernel = sum of DEVICE KERNEL DURATION)")
    out(f"  {'op type':<14} {'calls':>6} {'kernel ms':>10} {'share':>7} {'host ms':>10} "
        f"{'mean cores':>10}  bound stage (calls)")
    for op, idx in sorted(by_type.items(), key=lambda kv: -sum(
            summaries[i]["kernel_ns"] for i in kv[1] if i in summaries)):
        kernel = sum(summaries[i]["kernel_ns"] for i in idx if i in summaries)
        host = sum(int(host_rows[i]["HOST DURATION [ns]"] or 0) for i in idx)
        cores = [summaries[i]["cores"] for i in idx if i in summaries]
        bound = defaultdict(int)
        for i in idx:
            if i in summaries and summaries[i]["bound"]:
                bound[summaries[i]["bound"]] += 1
        out(f"  {op:<14} {len(idx) / n_inv:>6.0f} {kernel / 1e6 / n_inv:>10.2f} "
            f"{(kernel / total_kernel if total_kernel else 0):>7.1%} {host / 1e6 / n_inv:>10.2f} "
            f"{(statistics.mean(cores) if cores else 0):>10.1f}  "
            + ", ".join(f"{b} {c / n_inv:.0f}" for b, c in sorted(bound.items(), key=lambda kv: -kv[1])))
    out(f"  {'total':<14} {len(selected) / n_inv:>6.0f} {total_kernel / 1e6 / n_inv:>10.2f}")
    out("")

    if has_phases:
        out("Phases on the critical core, summed over the dispatches of each op type")
        out("  (ms per invocation and share of that RISC's kernel time; * = waiting)")
        for op, idx in sorted(by_type.items(), key=lambda kv: -sum(
                summaries[i]["kernel_ns"] for i in kv[1] if i in summaries)):
            idx = [i for i in idx if i in summaries and phases.get(i)]
            if not idx:
                continue
            out(f"  {op}")
            for risc in RISCS:
                kernel = sum(summaries[i]["critical_risc_ns"].get(risc, 0) for i in idx)
                totals = defaultdict(float)
                for i in idx:
                    for (r, p), v in phases[i].items():
                        if r == risc:
                            totals[p] += v["critical_ns"]
                if not kernel or not totals:
                    continue
                parts = [f"{p}{'*' if is_wait(p) else ''} {v / 1e6 / n_inv:.2f} ({v / kernel:.0%})"
                         for p, v in sorted(totals.items(), key=lambda kv: -kv[1]) if v / kernel >= 0.01]
                out(f"    {risc:<8} {RISC_ROLE[risc]:<6} {kernel / 1e6 / n_inv:>8.2f} ms  "
                    + ", ".join(parts))
        out("")

    out(f"Slowest {top} dispatches")
    out("  (utilization r/c/w = reader/compute/writer stages; phases of the bound stage's RISC)")
    out(f"  {'call':>6} {'op':<12} {'M':>6} {'N':>5} {'K':>5} {'cores':>5} {'kernel us':>10} "
        f"{'host us':>9} {'crit core':>9} {'bound':>8} {'util r/c/w':>14}  top phases")
    ranked = sorted((i for i in selected if i in summaries),
                    key=lambda i: -summaries[i]["kernel_ns"])[:top]
    for i in ranked:
        r, s = host_rows[i], summaries[i]
        risc = STAGE_RISC.get(s["bound"], s["critical_risc"])
        ph = sorted(((p, v["critical_ns"]) for (rr, p), v in phases.get(i, {}).items()
                     if rr == risc), key=lambda kv: -kv[1])[:3]
        kernel = s["critical_risc_ns"].get(risc, 0) or 1
        util = "/".join(f"{s['utilization'][st]:.2f}" if st in s["utilization"] else "-"
                        for st in STAGES)
        out(f"  {r['GLOBAL CALL COUNT']:>6} {r['OP TYPE'][:12]:<12} {r.get('IREE M', ''):>6} "
            f"{r.get('IREE N', ''):>5} {r.get('IREE K', ''):>5} {s['cores']:>5} "
            f"{s['kernel_ns'] / 1e3:>10.1f} {int(r['HOST DURATION [ns]'] or 0) / 1e3:>9.1f} "
            f"{'%d-%d' % s['critical_core']:>9} {s['bound'] or '-':>8} {util:>14}  "
            + ", ".join(f"{p} {v / kernel:.0%}" for p, v in ph))
    out("")
    return "\n".join(lines) + "\n"


def write_trace(path, host_rows, runs, matched, summaries, phases, mhz, invocations):
    """Chrome trace: a host track plus one process per core, one thread per
    RISC. Each core's zones keep exact same-core offsets; cores are aligned at
    their firmware start, placed at the dispatch's host launch time."""
    if invocations is None:
        ids = sorted({int(r.get("IREE INVOCATION ID") or 0) for r in host_rows})
        invocations = ids[-1:]
    rows = [i for i, r in enumerate(host_rows) if int(r.get("IREE INVOCATION ID") or 0) in invocations]
    if not rows:
        path.write_text('{"traceEvents": []}\n')
        return
    base = min(int(host_rows[i]["HOST START TS [ns]"] or 0) for i in rows)
    events = [{"ph": "M", "pid": 0, "name": "process_name", "args": {"name": "host"}},
              {"ph": "M", "pid": 0, "name": "process_sort_index", "args": {"sort_index": 0}}]
    pids = {}
    for i in rows:
        r = host_rows[i]
        start = int(r["HOST START TS [ns]"] or 0) - base
        name = f"{r['OP TYPE']} #{r['GLOBAL CALL COUNT']}"
        args = {k: r.get(k, "") for k in ("OP CODE", "IREE M", "IREE N", "IREE K", "CORE COUNT",
                                          "IREE PROGRAM SETUP DURATION [ns]",
                                          "IREE ENQUEUE WAIT DURATION [ns]")}
        if i in summaries:
            args["DEVICE KERNEL DURATION [ns]"] = round(summaries[i]["kernel_ns"])
        events.append({"ph": "X", "pid": 0, "tid": 0, "name": name, "ts": start / 1e3,
                       "dur": int(r["HOST DURATION [ns]"] or 0) / 1e3, "args": args})
        if i not in matched:
            continue
        run = runs[matched[i]]
        launch = start + int(r.get("IREE PROGRAM SETUP DURATION [ns]") or 0)
        for core in run.cores():
            if core not in pids:
                pids[core] = len(pids) + 1
                events.append({"ph": "M", "pid": pids[core], "name": "process_name",
                               "args": {"name": "core %d-%d" % core}})
                events.append({"ph": "M", "pid": pids[core], "name": "process_sort_index",
                               "args": {"sort_index": pids[core]}})
                for t, risc in enumerate(RISCS):
                    events.append({"ph": "M", "pid": pids[core], "tid": t, "name": "thread_name",
                                   "args": {"name": f"{risc} ({RISC_ROLE[risc]})"}})
            origin = min(s for (c, _), v in list(run.fw.items()) + list(run.kernel.items())
                         if c == core for s, _ in v)
            for t, risc in enumerate(RISCS):
                for s, e in run.kernel.get((core, risc), []):
                    ph = {p: round(v[0] * 1000.0 / mhz) for (c, rr, p), v in run.phases.items()
                          if c == core and rr == risc}
                    events.append({"ph": "X", "pid": pids[core], "tid": t, "name": name,
                                   "ts": (launch + (s - origin) * 1000.0 / mhz) / 1e3,
                                   "dur": (e - s) / mhz, "args": {"phases [ns]": ph}})
    path.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}) + "\n")


def find_device_log(directory):
    for candidate in (directory / ".logs" / "profile_log_device.csv",
                      directory / "profile_log_device.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no profile_log_device.csv under {directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "analyze"):
        p = sub.add_parser(mode)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--top", type=int, default=15, help="slowest dispatches to list")
        p.add_argument("--include-first", action="store_true",
                       help="summarize the first invocation too (it includes JIT builds)")
        p.add_argument("--trace-invocation", type=int, action="append",
                       help="invocation(s) in trace.json (default: the last)")
        if mode == "run":
            p.add_argument("--no-phases", action="store_true",
                           help="kernel zones only, without phase markers")
            p.add_argument("command", nargs=argparse.REMAINDER,
                           help="-- command to profile")
        else:
            p.add_argument("--device-log", type=Path,
                           help="profile_log_device.csv (default: OUTPUT/device/.logs)")
            p.add_argument("--ops", type=Path,
                           help="host ops_perf_results.csv (default: OUTPUT/host)")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.mode == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("run needs a command after --")
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1",
                   TT_METAL_PROFILER_DIR=str(output / "device"),
                   TT_IREE_PROFILE_OUTPUT=str(output / "host"))
        if args.no_phases:
            env.pop("TT_IREE_KERNEL_PHASES", None)
        else:
            env["TT_IREE_KERNEL_PHASES"] = "1"
        status = subprocess.call(command, env=env)
        if status:
            print(f"tt_iree_profile: command exited with {status}", file=sys.stderr)
            return status
        device_log, ops = find_device_log(output / "device"), output / "host" / "ops_perf_results.csv"
    else:
        device_log = args.device_log or find_device_log(output / "device")
        ops = args.ops or output / "host" / "ops_perf_results.csv"
    report = analyze(device_log, ops, output, args.top, args.include_first, args.trace_invocation)
    sys.stdout.write(report)
    print(f"wrote {output}/ops_perf_results.csv, kernel_phases.csv, report.txt, trace.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
