#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Benchmark pinned upstream TTNN without changing its source or CPU goldens.

Run each precision/trace/CQ case in its own process. Profiler runs are diagnostic
only; host latency results must come from runs with the device profiler disabled.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from build_resnet50_precision_golden import (
    classifier_comparison, file_record, require_new_directory, write_json,
)
from run_ttnn_resnet50_golden import configure_stem_block, make_avgpool_precision_wrapper


def percentile(values, percent):
    if not values or not 0 <= percent <= 100:
        raise ValueError("requires nonempty values and percentile in [0, 100]")
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def timing_summary(seconds, batch, batches_per_window=1):
    if (not seconds or batch <= 0 or batches_per_window <= 0 or
            any(not math.isfinite(x) or x <= 0 for x in seconds)):
        raise ValueError("requires finite positive times, batch and window size")
    per_batch = [value / batches_per_window for value in seconds]
    mean = statistics.mean(per_batch)
    return {
        "windows": len(seconds), "batches_per_window": batches_per_window,
        "batch_size": batch, "mean_batch_ms": mean * 1000,
        "p50_batch_ms": percentile(per_batch, 50) * 1000,
        "p95_batch_ms": percentile(per_batch, 95) * 1000,
        "min_batch_ms": min(per_batch) * 1000,
        "max_batch_ms": max(per_batch) * 1000,
        "images_per_second": batch / mean,
        "amortized_ms_per_image_not_latency": mean * 1000 / batch,
        "raw_window_seconds": seconds,
    }


def write_timings(filename, rows):
    with filename.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--tt-metal-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precision", choices=("bf8_lofi", "bf16_hifi4"), required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--cqs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch", type=int, choices=(16,), default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--stream-windows", type=int, default=5)
    parser.add_argument("--stream-batches", type=int, default=20)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.stream_windows, args.stream_batches) < 1:
        parser.error("warmup, iterations and stream sizes must be positive")
    if args.cqs == 2 and not args.trace:
        parser.error("this benchmark matrix uses 2 CQs only with trace")
    if args.profile_only and (args.trace or args.cqs != 1):
        parser.error("operation profiling uses non-trace 1-CQ only")
    if args.profile_only != (os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"):
        parser.error("--profile-only requires profiler=1; timing runs require it disabled")
    return args


def main():
    args = parse_args()
    root, metal = args.golden_dir.resolve(), args.tt_metal_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for artifact in (manifest["state_dict"], manifest["input"], *manifest["logits"].values()):
        if file_record(artifact["path"])["sha256"] != artifact["sha256"]:
            raise ValueError(f"golden artifact hash changed: {artifact['path']}")
    require_new_directory(args.output_dir)
    output = args.output_dir.resolve()
    os.chdir(output)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(metal))
    os.environ["TT_METAL_HOME"] = str(metal)
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(metal)
    os.environ.setdefault("TT_METAL_PROFILER_DIR", str(output / "device-profile"))

    import numpy as np
    import torch
    import torchvision
    import ttnn
    from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common import resnet50_test_infra as infra_module
    from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

    if not Path(ttnn.__file__).resolve().is_relative_to(metal):
        raise ValueError(f"TTNN import does not match source root: {ttnn.__file__}")
    torch.set_num_threads(8)
    inputs = torch.from_numpy(np.load(root / "input.npy"))
    if len(inputs) % args.batch:
        raise ValueError("sample count must divide batch size; no sample duplication/padding")
    base = torchvision.models.resnet50(weights=None).eval()
    base.load_state_dict(torch.load(root / "state_dict.pt", weights_only=True))
    bf16 = args.precision == "bf16_hifi4"
    result = {
        "status": "started", "arguments": vars(args) | {
            key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
        "golden_manifest": file_record(root / "manifest.json"),
        "input_sha256": manifest["input"]["sha256"],
        "state_dict_sha256": manifest["state_dict"]["sha256"],
        "script": file_record(__file__), "ttnn_module": ttnn.__file__,
        "tt_metal_commit": subprocess.check_output(
            ["git", "-C", str(metal), "rev-parse", "HEAD"], text=True).strip(),
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "precision": args.precision, "batch": args.batch, "trace": args.trace, "cqs": args.cqs,
        "requested_activation_and_weight_dtype": "BF16" if bf16 else "BF8_B",
        "requested_matrix_fidelity": "HiFi4" if bf16 else "LoFi",
        "l1_small_size": 32768, "trace_region_size": 5554176,
        "device_profiler_enabled": args.profile_only,
        "timing_note": "host perf_counter wall with explicit device completion; not kernel-only",
        "input_policy": "preprocessed real images in host memory; disk/decode/preprocess excluded",
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith(("TT_", "TRACY", "ARCH_", "OMP_", "OPENBLAS_"))},
    }
    write_json(output / "results.json", result)
    saved_loader, saved_avgpool = infra_module.load_resnet50_model, ttnn.avg_pool2d
    infra_module.load_resnet50_model = lambda _location: copy.deepcopy(base)
    device, pipeline = None, None
    started = time.perf_counter()
    try:
        ttnn.avg_pool2d, result["avgpool_precision"] = make_avgpool_precision_wrapper(ttnn, bf16)
        init_start = time.perf_counter()
        device = ttnn.open_device(device_id=args.device, l1_small_size=32768,
                                  trace_region_size=5554176, num_command_queues=args.cqs)
        device.enable_program_cache()
        grid = device.compute_with_storage_grid_size()
        result["device_grid"] = [grid.x, grid.y]
        dtype = ttnn.bfloat16 if bf16 else ttnn.bfloat8_b
        with torch.inference_mode():
            infra = infra_module.create_test_infra(
                device, args.batch, dtype, dtype,
                ttnn.MathFidelity.HiFi4 if bf16 else ttnn.MathFidelity.LoFi,
                use_pretrained_weight=True, dealloc_input=True,
                final_output_mem_config=ttnn.L1_MEMORY_CONFIG)
        result["stem_block"] = configure_stem_block(infra.ttnn_resnet50_model, 512 if bf16 else None)
        result["device_and_model_init_seconds"] = time.perf_counter() - init_start
        host_inputs, dram_config, l1_config = [], None, None
        for batch in inputs.split(args.batch):
            host, dram_config, l1_config = infra.setup_dram_sharded_input(device, batch)
            host_inputs.append(host)

        def model(l1_input):
            infra.input_tensor = l1_input
            return infra.run()

        pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=args.trace, num_command_queues=args.cqs,
                                  all_transfers_on_separate_command_queue=False),
            model=model, device=device, dram_input_memory_config=dram_config,
            l1_input_memory_config=l1_config)
        compile_start = time.perf_counter()
        pipeline.compile(host_inputs[0])
        ttnn.synchronize_device(device)
        result["pipeline_compile_and_capture_seconds"] = time.perf_counter() - compile_start
        pipeline.preallocate_output_tensors_on_host(1)

        def to_numpy(tensor):
            return ttnn.to_torch(tensor, device=device).float().reshape(args.batch, 1000).clone().numpy()

        # Collect real-image logits outside all performance scopes.
        actual = []
        for host in host_inputs:
            actual.append(to_numpy(pipeline.enqueue([host]).pop_all()[0]))
            if args.profile_only:
                ttnn.ReadDeviceProfiler(device)
        actual = np.concatenate(actual)
        np.save(output / "logits.npy", actual)
        comparisons = {
            policy: classifier_comparison(np.load(root / policy / "logits.npy"), actual,
                                          manifest["categories"], manifest["labels"])
            for policy in ("bf16_module", "fp32")}
        write_json(output / "accuracy.json", comparisons)
        result["accuracy"] = {policy: values["aggregate"] for policy, values in comparisons.items()}
        expected = [actual[i:i + args.batch] for i in range(0, len(actual), args.batch)]

        def assert_output(tensor, index):
            if not np.array_equal(to_numpy(tensor), expected[index]):
                raise ValueError(f"replayed output changed for input batch {index}")

        if args.profile_only:
            # Dump at each boundary to avoid overflowing the finite device log.
            for _ in range(args.warmup):
                pipeline.enqueue([host_inputs[0]]).pop_all()
                ttnn.ReadDeviceProfiler(device)
            try:
                from tracy import signpost
            except ImportError:
                signpost = None
            for iteration in range(args.iterations):
                if signpost:
                    signpost(header=f"benchmark_batch_{iteration}_start")
                queued = pipeline.enqueue([host_inputs[0]]).pop_all()
                if signpost:
                    signpost(header=f"benchmark_batch_{iteration}_stop")
                ttnn.ReadDeviceProfiler(device)
                assert_output(queued[0], 0)
            result["profile_measurement_batches"] = args.iterations
            result["timings"] = None
        else:
            timings, rows = {}, []
            for _ in range(args.warmup):
                pipeline.enqueue([host_inputs[0]]).pop_all()
            elapsed = []
            for iteration in range(args.iterations):
                before = time.perf_counter_ns()
                queued = pipeline.enqueue([host_inputs[0]]).pop_all()
                seconds = (time.perf_counter_ns() - before) / 1e9
                elapsed.append(seconds)
                rows.append(dict(scope="host_io_single_batch", iteration=iteration,
                                 batches=1, seconds=seconds))
            assert_output(queued[0], 0)
            timings["host_io_single_batch"] = timing_summary(elapsed, args.batch)

            pipeline.preallocate_output_tensors_on_host(args.stream_batches)
            stream_inputs = [host_inputs[i % len(host_inputs)] for i in range(args.stream_batches)]
            pipeline.enqueue(stream_inputs).pop_all()
            elapsed = []
            for iteration in range(args.stream_windows):
                before = time.perf_counter_ns()
                queued = pipeline.enqueue(stream_inputs).pop_all()
                seconds = (time.perf_counter_ns() - before) / 1e9
                elapsed.append(seconds)
                rows.append(dict(scope="host_io_stream", iteration=iteration,
                                 batches=args.stream_batches, seconds=seconds))
                # All slots are checked; pointer aliasing must not masquerade as throughput.
                for index, tensor in enumerate(queued):
                    assert_output(tensor, index % len(host_inputs))
            timings["host_io_stream"] = timing_summary(elapsed, args.batch, args.stream_batches)
            timings["host_io_stream"]["percentile_scope"] = "window-averaged batch times, not individual latency"

            if args.trace:
                # Restore inputs from device DRAM before EACH replay. The model may
                # overwrite its deallocated L1 input; naked replay is not a valid test.
                executor = pipeline.executor
                pipeline.preallocate_output_tensors_on_host(1)
                pipeline.enqueue([host_inputs[0]]).pop_all()
                elapsed = []
                for iteration in range(args.warmup + args.iterations):
                    before = time.perf_counter_ns()
                    executor.l1_input_tensor = ttnn.reshard(
                        executor.dram_input_tensor, l1_config, executor.l1_input_tensor)
                    ttnn.execute_trace(device, executor.trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(device)
                    seconds = (time.perf_counter_ns() - before) / 1e9
                    if iteration >= args.warmup:
                        elapsed.append(seconds)
                        rows.append(dict(scope="resident_reshard_trace_sync", iteration=iteration - args.warmup,
                                         batches=1, seconds=seconds))
                assert_output(executor.output_tensor, 0)
                timings["resident_reshard_trace_sync"] = timing_summary(elapsed, args.batch)
                timings["resident_reshard_trace_sync"]["scope"] = (
                    "host wall: device DRAM-to-L1 reshard + model trace + synchronize; no host tensor I/O; not kernel-only")
            result["timings"] = timings
            write_timings(output / "timings.csv", rows)
        result["logits"] = file_record(output / "logits.npy")
        result["output_replay_check"] = "bitwise equality to this case's untimed 112-image pass"
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        infra_module.load_resnet50_model, ttnn.avg_pool2d = saved_loader, saved_avgpool
        try:
            if pipeline is not None:
                pipeline.cleanup()
        except BaseException as error:
            result.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if device is not None:
                    ttnn.close_device(device)
            except BaseException as error:
                result.update(status="failed", device_close_error=f"{type(error).__name__}: {error}")
                raise
            finally:
                result["total_process_work_seconds"] = time.perf_counter() - started
                write_json(output / "results.json", result)
    print(json.dumps({key: result[key] for key in ("status", "precision", "trace", "cqs", "timings")}))


if __name__ == "__main__":
    main()
