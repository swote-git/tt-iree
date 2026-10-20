#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run upstream TTNN ResNet50 against an immutable precision-golden package.

Imports TT-Metal read-only; run with PYTHONDONTWRITEBYTECODE=1. The hardware
result is a comparison artifact, never the source of a CPU golden.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
import time

from build_resnet50_precision_golden import (
    classifier_comparison, file_record, predictions, require_new_directory, write_json,
)


def tile_block_height(value):
    height = int(value)
    if height <= 0 or height % 32:
        raise argparse.ArgumentTypeError("stem block height must be a positive multiple of 32")
    return height


def configure_stem_block(model, block_h):
    if block_h is not None:
        tile_block_height(block_h)
    config = model.conv1_config
    previous = config.act_block_h_override
    if block_h is not None:
        config.act_block_h_override = block_h
    return {"requested_block_h": block_h, "upstream_block_h": previous,
            "effective_block_h": config.act_block_h_override}


def make_avgpool_precision_wrapper(ttnn, force_bf16=False):
    """Audit requested and returned AvgPool dtypes; optionally request BF16.

    Installation/restoration is process-local in main, never an upstream edit.
    Keep the upstream fidelity/accumulation configuration unchanged.
    """
    original = ttnn.avg_pool2d
    audit = {"override": "bf16" if force_bf16 else "upstream", "calls": []}

    def wrapped(*args, **kwargs):
        requested = kwargs.get("dtype")
        if force_bf16:
            kwargs["dtype"] = ttnn.bfloat16
        output = original(*args, **kwargs)
        config = kwargs.get("compute_kernel_config")
        audit["calls"].append({"upstream_dtype": str(requested),
                               "effective_dtype": str(kwargs.get("dtype")),
                               "output_dtype": str(output.dtype),
                               "math_fidelity": str(getattr(config, "math_fidelity", "unknown")),
                               "fp32_dest_acc_en": getattr(config, "fp32_dest_acc_en", None)})
        if force_bf16 and output.dtype != ttnn.bfloat16:
            raise ValueError("TTNN AvgPool did not produce requested BF16 storage")
        return output

    return wrapped, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", required=True, type=pathlib.Path)
    parser.add_argument("--tt-metal-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, choices=(16, 20, 32), default=16)
    parser.add_argument("--dtype", choices=("bf8b", "bf16"), default="bf8b")
    parser.add_argument("--fidelity", choices=("LoFi", "HiFi4"), default="LoFi")
    parser.add_argument("--avgpool-dtype", choices=("upstream", "bf16"), default="upstream",
                        help="upstream ResNet requests BF8 AvgPool; actual return dtype is audited; bf16 changes only that request in-process")
    parser.add_argument("--stem-act-block-h", type=tile_block_height,
                        help="explicit stem activation-block height; default preserves upstream config")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.avgpool_dtype == "bf16" and args.dtype != "bf16":
        parser.error("BF16 AvgPool comparison requires --dtype=bf16")
    root, source_root = args.golden_dir.resolve(), args.tt_metal_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for record in (manifest["state_dict"], manifest["input"], *manifest["logits"].values()):
        if file_record(record["path"])["sha256"] != record["sha256"]:
            raise ValueError(f"golden artifact hash changed: {record['path']}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))
    os.environ["TT_METAL_HOME"] = str(source_root)
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(source_root)
    import numpy as np
    import torch
    import torchvision
    import ttnn
    from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common import resnet50_test_infra as infra_module
    from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

    if not pathlib.Path(ttnn.__file__).resolve().is_relative_to(source_root):
        raise ValueError(f"TTNN import does not match requested source root: {ttnn.__file__}")
    torch.set_num_threads(args.threads)
    inputs = torch.from_numpy(np.load(root / "input.npy"))
    if len(inputs) % args.batch:
        raise ValueError("sample count must be divisible by TTNN batch; no hidden replication/padding")
    base = torchvision.models.resnet50(weights=None).eval()
    base.load_state_dict(torch.load(root / "state_dict.pt", weights_only=True))
    require_new_directory(args.output_dir)
    target = args.output_dir.resolve()
    os.chdir(target)
    saved_loader = infra_module.load_resnet50_model
    saved_avgpool = ttnn.avg_pool2d
    infra_module.load_resnet50_model = lambda _location: copy.deepcopy(base)
    device = None
    pipeline = None
    start = time.monotonic()
    results = {
        "golden_manifest": file_record(root / "manifest.json"),
        "weight_source": manifest["source"], "input_sha256": manifest["input"]["sha256"],
        "tt_metal_commit": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "ttnn_module": ttnn.__file__, "batch": args.batch, "samples": len(inputs),
        "activation_dtype": args.dtype, "weight_dtype": args.dtype,
        "input_dtype": "BF16", "math_fidelity": args.fidelity,
        "requested_stem_act_block_h": args.stem_act_block_h,
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "use_trace": False, "num_command_queues": 1,
        "timing_scope": "blocking host wall including transfers; not device kernel-only latency",
        "script": file_record(__file__), "status": "started",
    }
    write_json(target / "results.json", results)
    try:
        ttnn.avg_pool2d, results["avgpool_precision"] = make_avgpool_precision_wrapper(
            ttnn, args.avgpool_dtype == "bf16")
        device = ttnn.open_device(device_id=args.device, l1_small_size=24576, num_command_queues=1)
        device.enable_program_cache()
        grid = device.compute_with_storage_grid_size()
        results["device_grid"] = [grid.x, grid.y]
        dtype = ttnn.bfloat8_b if args.dtype == "bf8b" else ttnn.bfloat16
        with torch.inference_mode():
            infra = infra_module.create_test_infra(
                device, args.batch, dtype, dtype, getattr(ttnn.MathFidelity, args.fidelity),
                use_pretrained_weight=True, dealloc_input=True,
                final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
            )
        results["stem_block"] = configure_stem_block(
            infra.ttnn_resnet50_model, args.stem_act_block_h
        )
        def model_wrapper(l1_input):
            infra.input_tensor = l1_input
            return infra.run()
        batches = inputs.split(args.batch)
        host_input, dram_config, l1_config = infra.setup_dram_sharded_input(device, batches[0])
        pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=False, num_command_queues=1,
                                  all_transfers_on_separate_command_queue=False),
            model=model_wrapper, device=device,
            dram_input_memory_config=dram_config, l1_input_memory_config=l1_config,
        )
        compile_start = time.monotonic()
        pipeline.compile(host_input)
        results["pipeline_compile_wall_seconds"] = time.monotonic() - compile_start
        outputs, batch_timings = [], []
        for index, batch in enumerate(batches):
            if index:
                host_input, _, _ = infra.setup_dram_sharded_input(device, batch)
            batch_start = time.monotonic()
            queued = pipeline.enqueue([host_input]).pop_all()
            outputs.append(ttnn.to_torch(queued[0], device=device).float().reshape(args.batch, 1000).clone())
            batch_timings.append(time.monotonic() - batch_start)
            print(f"TTNN batch {index}: {batch_timings[-1]:.3f} s", flush=True)
            del queued
        actual = torch.cat(outputs).numpy()
        np.save(target / "logits.npy", actual)
        categories, labels = manifest["categories"], manifest["labels"]
        write_json(target / "predictions.json", predictions(actual, categories, labels))
        results["comparisons"] = {
            policy: classifier_comparison(np.load(root / policy / "logits.npy"), actual, categories, labels)
            for policy in manifest["policies"]
        }
        # Upstream's BF16 PyTorch reference uses the hardware batch size.
        cpu_bf16 = copy.deepcopy(base).to(torch.bfloat16).eval()
        with torch.inference_mode():
            official_reference = torch.cat([cpu_bf16(batch.to(torch.bfloat16)).float() for batch in batches]).numpy()
        np.save(target / "pytorch_bf16_batch_reference.npy", official_reference)
        results["comparisons"]["bf16_module_hardware_batch"] = classifier_comparison(
            official_reference, actual, categories, labels
        )
        results["cpu_batch_sensitivity"] = classifier_comparison(
            np.load(root / "bf16_module/logits.npy"), official_reference, categories, labels
        )
        results.update(status="completed", batch_host_wall_seconds=batch_timings,
                       output=file_record(target / "logits.npy"),
                       total_host_wall_seconds=time.monotonic() - start)
    except BaseException as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        infra_module.load_resnet50_model = saved_loader
        ttnn.avg_pool2d = saved_avgpool
        try:
            if pipeline is not None:
                pipeline.cleanup()
        finally:
            try:
                if device is not None:
                    ttnn.close_device(device)
            finally:
                write_json(target / "results.json", results)


if __name__ == "__main__":
    main()
