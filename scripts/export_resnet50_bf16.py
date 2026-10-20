#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export the immutable ResNet golden as an actual static batch-one BF16 graph.

Requires torch, torchvision and IREE Turbine. The upstream BF16 NumPy bridge
workaround is process-local and retains raw BF16 bits, not a dtype fallback.
Use MLIR bytecode at phase boundaries to retain dense resource payloads.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import pathlib
import subprocess

from build_resnet50_precision_golden import (
    file_record, require_new_directory, write_json,
)


def verify_record(filename, record):
    actual = file_record(filename)
    if actual["sha256"] != record["sha256"]:
        raise ValueError(f"golden hash mismatch: {filename}")
    return actual


@contextlib.contextmanager
def bf16_resource_bridge(ir_utils, torch):
    """Scope the Turbine 3.9 BF16 buffer adaptation to its global exporter."""
    original = ir_utils.np

    class Bridge:
        def array(self, value, *args, **kwargs):
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                if args or kwargs:
                    raise ValueError("unexpected BF16 resource array conversion options")
                return value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()
            return original.array(value, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(original, name)


    ir_utils.np = Bridge()
    try:
        yield
    finally:
        ir_utils.np = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iree-compile", required=True, type=pathlib.Path,
                        help="tt-iree compiler; the frontend wheel only lowers Torch to input IR")
    args = parser.parse_args()
    import json
    import torch
    import torchvision
    import iree.turbine.aot as aot
    from iree.turbine.aot.support import ir_utils
    from iree.compiler.tools.binaries import find_tool

    golden = args.golden_dir.resolve()
    manifest = json.loads((golden / "manifest.json").read_text())
    weights = verify_record(golden / "state_dict.pt", manifest["state_dict"])
    inputs = verify_record(golden / "input.npy", manifest["input"])
    reference = verify_record(golden / "bf16_module/logits.npy", manifest["logits"]["bf16_module"])
    require_new_directory(args.output_dir)
    out = args.output_dir.resolve()
    torch.set_num_threads(8)
    model = torchvision.models.resnet50(weights=None).eval()
    model.load_state_dict(torch.load(weights["path"], map_location="cpu", weights_only=True))
    model.to(torch.bfloat16)
    example = torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16)
    with bf16_resource_bridge(ir_utils, torch):
        exported = aot.export(model, args=(example,))
        exported.verify()
        exported.save_mlir(out / "resnet50_torch.mlirbc")

    commands = [
        [find_tool("iree-compile"), str(out / "resnet50_torch.mlirbc"),
         "--iree-input-type=torch", "--compile-to=input", "--emit-mlir-bytecode",
         "-o", str(out / "resnet50_input.mlirbc")],
        [str(args.iree_compile.resolve()), str(out / "resnet50_input.mlirbc"),
         "--compile-from=input", "--iree-hal-target-backends=tenstorrent",
         "--compile-to=global-optimization", "--emit-mlir-bytecode",
         "-o", str(out / "resnet50_global_opt.mlirbc")],
    ]
    for index, command in enumerate(commands):
        with (out / f"lowering-{index}.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=900)
    write_json(out / "manifest.json", {
        "golden_dir": str(golden), "source": manifest["source"],
        "state_dict": weights, "input": inputs, "bf16_reference": reference,
        "precision_policy": "bf16_module", "batch": 1,
        "input_shape": [1, 3, 224, 224], "input_dtype": "bf16",
        "output_shape": [1, 1000], "output_dtype": "bf16",
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "iree_turbine": importlib.metadata.version("iree-turbine"),
        "iree_base_compiler": importlib.metadata.version("iree-base-compiler"),
        "bf16_resource_bridge": "raw uint16 bits, scoped to Turbine global exporter",
        "commands": commands,
        "artifacts": {name: file_record(out / name) for name in (
            "resnet50_torch.mlirbc", "resnet50_input.mlirbc", "resnet50_global_opt.mlirbc")},
    })
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
