#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare paired BF16 logits and audit every static ResNet50 dispatch.

Uses saved hardware outputs, not a host execution fallback. Local replay feeds
measured device tensors into PyTorch CPU operators to remove upstream drift;
it is a diagnostic comparison, not a bit-exact oracle for device contractions.
The boundary map deliberately supports only the current batch-one v17 export.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import pathlib
from collections import Counter

from build_resnet50_precision_golden import (
    classifier_comparison, error_metrics, file_record, require_new_directory,
    write_json,
)
from export_resnet50_bf16 import verify_record


MAP_VERSION = "resnet50-static-b1-bf16-v17-v18-1"


@dataclasses.dataclass(frozen=True)
class Boundary:
    dispatch: int
    name: str
    op_type: str
    kind: str
    shape: tuple[int, ...]
    checkpoint_shape: tuple[int, ...]
    sources: tuple[int, ...]
    module: str = ""
    relu_call: int = 0
    export_index: int = 0
    export_suffix: str = ""

    @property
    def op_code(self):
        return f"main$async_dispatch_{self.export_index}_{self.export_suffix}"


def dispatch_boundaries():
    """Logical store slices, excluding padding halos, in dispatch call order.

    Export ordinals can repeat: executable deduplication is NOT call numbering.
    Exact names/shapes fail closed if a new compiler changes fusion or layout.
    """
    boundaries = []

    def add(name, op_type, kind, shape, checkpoint_shape, sources, module="",
            relu_call=0, suffix=""):
        call = len(boundaries) + 1
        boundaries.append(Boundary(call, name, op_type, kind, shape,
                                   checkpoint_shape, sources, module, relu_call,
                                   call - 1, suffix))
        return call

    add("input.bf16_copy", "PADDED_COPY", "copy", (1, 3, 224, 224),
        (3, 224, 224), (), suffix="slow_memcpy")
    add("stem.conv_bn_relu", "CONV2D", "stem", (1, 64, 112, 112),
        (64, 112, 112), (1,), "relu", 1,
        "conv_64x112x112x3x7x7_bf16xbf16xf32")
    previous = add("maxpool", "MAX_POOL_2D", "pool", (1, 64, 56, 56),
                   (64, 56, 56), (2,), "maxpool", suffix="conv_64x56x56x3x3_bf16")
    input_channels, spatial = 64, 56
    for layer, blocks in enumerate((3, 4, 6, 3), 1):
        channels = 64 * 2 ** (layer - 1)
        output_spatial = 56 // 2 ** (layer - 1)
        output_channels = channels * 4
        for block in range(blocks):
            name = f"layer{layer}.{block}"
            first = add(name + ".conv1_bn_relu", "MATMUL", "conv1",
                        (1, channels, spatial, spatial), (channels, spatial, spatial),
                        (previous,), name, 1,
                        f"matmul_like_{channels}x{spatial}x{spatial}x{input_channels}_bf16xbf16xf32")
            second = add(name + ".conv2_bn_relu", "CONV2D", "conv2",
                         (1, channels, output_spatial, output_spatial),
                         (channels, output_spatial, output_spatial), (first,), name, 2,
                         f"conv_{channels}x{output_spatial}x{output_spatial}x{channels}x3x3_bf16xbf16xf32")
            if block == 0:
                skip_shape = ((output_channels, output_spatial * output_spatial)
                              if layer == 1 else (output_channels, output_spatial, output_spatial))
                skip_suffix = (f"matmul_like_{output_channels}x{output_spatial * output_spatial}x{input_channels}"
                               if layer == 1 else
                               f"matmul_like_{output_channels}x{output_spatial}x{output_spatial}x{input_channels}")
                skip = add(name + ".downsample_conv_before_bn", "MATMUL", "downsample",
                           (1, output_channels, output_spatial, output_spatial),
                           skip_shape, (previous,), name + ".downsample.0",
                           suffix=skip_suffix + "_bf16xbf16xf32")
            else:
                skip = previous
            final_shape = ((output_channels, output_spatial, output_spatial)
                           if layer == 4 and block == 2 else
                           (output_channels, output_spatial * output_spatial))
            suffix = (f"matmul_like_{output_channels}x{output_spatial}x{output_spatial}x{channels}"
                      if len(final_shape) == 3 else
                      f"matmul_like_{output_channels}x{output_spatial * output_spatial}x{channels}")
            previous = add(name + ".conv3_bn_residual_relu", "MATMUL", "residual",
                           (1, output_channels, output_spatial, output_spatial),
                           final_shape, (second, skip), name, 3,
                           suffix + "_bf16xbf16xf32")
            input_channels, spatial = output_channels, output_spatial
    add("global_sum_before_divide", "GLOBAL_SUM", "sum", (1, 2048, 1, 1),
        (2048,), (previous,), suffix="reduction_2048x49_i1xbf16xbf16")
    add("avgpool", "SCALAR_DIV", "divide", (1, 2048, 1, 1), (2048,),
        (56,), "avgpool", suffix="elementwise_2048_bf16")
    add("fc", "LINEAR", "linear", (1, 1000), (1, 1000), (57,), "fc",
        suffix="matmul_1x1000x2048_bf16xbf16xf32")
    for call, export in {13: 9, 23: 19, 26: 19, 34: 30, 35: 31, 36: 32,
                         37: 30, 38: 31, 39: 32, 40: 30, 41: 31, 42: 32,
                         43: 30, 44: 31, 45: 32, 53: 49, 54: 50}.items():
        boundaries[call - 1] = dataclasses.replace(boundaries[call - 1], export_index=export)
    return boundaries


def require_bf16_values(values, context):
    import numpy as np
    if (values.dtype != np.dtype("float32") or not np.isfinite(values).all()
            or np.any(values.view(np.uint32) & 0xffff)):
        raise ValueError(f"expected finite BF16 values widened exactly to F32: {context}")


def load_dispatches(directory, boundaries, use_tensix=False):
    import numpy as np
    directory = pathlib.Path(directory)
    manifest_path = directory / "checkpoints/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with (directory / "profile/ops_perf_results.csv").open() as stream:
        profile = list(csv.DictReader(stream))
    records = manifest["records"]
    if (manifest["schema_version"] != 1 or manifest["dtype"] != "float32"
            or manifest["checkpoint_count"] != len(boundaries)
            or len(records) != len(boundaries) or len(profile) != len(boundaries)):
        raise ValueError("expected complete 58-dispatch checkpoint/profile coverage")
    actual, provenance = {}, [file_record(manifest_path),
                             file_record(directory / "profile/ops_perf_results.csv")]
    for boundary, record, operation in zip(boundaries, records, profile):
        expected_builtin = 12 if use_tensix and boundary.op_type in ("MATMUL", "CONV2D", "LINEAR") else 11
        if (record["dispatch_index"] != boundary.dispatch
                or int(operation["GLOBAL CALL COUNT"]) != boundary.dispatch
                or record["op_code"] != boundary.op_code
                or operation["OP CODE"] != boundary.op_code
                or operation["OP TYPE"] != boundary.op_type
                or record["builtin_program"] != expected_builtin or record["dispatch_kind"] != 8
                or operation["IREE EXECUTION DOMAIN"] != "device_native"
                or operation["INPUT_0_DATATYPE"] != "BF16"
                or operation["OUTPUT_0_DATATYPE"] != "BF16"
                or tuple(record["shape"]) != boundary.checkpoint_shape):
            raise ValueError(f"dispatch boundary contract changed: {boundary.dispatch} {boundary.name}")
        filename = record["file"]
        if pathlib.Path(filename).name != filename:
            raise ValueError("checkpoint file must be a basename")
        filename = directory / "checkpoints" / filename
        values = np.load(filename, allow_pickle=False)
        if values.shape != boundary.checkpoint_shape:
            raise ValueError(f"checkpoint shape does not match manifest: {filename}")
        require_bf16_values(values, filename)
        # The guarded map permits only singleton batch and CxHW flattening.
        actual[boundary.dispatch] = values.reshape(boundary.shape)
        provenance.append(file_record(filename))
    return actual, provenance


class MetricAccumulator:
    """Pool elements, not unweighted per-dispatch PCC/RMSE percentages."""
    def __init__(self):
        self.count = self.mismatches = 0
        self.max_error = self.abs_error = self.squared_error = 0.0
        self.e_sum = self.a_sum = self.e_sq = self.a_sq = self.dot = 0.0

    def add(self, expected, actual):
        import numpy as np
        if expected.shape != actual.shape or not expected.size:
            raise ValueError("comparison requires matching nonempty shapes")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("non-finite comparison values")
        self.mismatches += int(np.count_nonzero(expected != actual))
        e, a = expected.astype(np.float64).reshape(-1), actual.astype(np.float64).reshape(-1)
        diff = a - e
        self.count += e.size
        self.max_error = max(self.max_error, float(np.abs(diff).max()))
        self.abs_error += float(np.abs(diff).sum())
        self.squared_error += float(diff @ diff)
        self.e_sum += float(e.sum())
        self.a_sum += float(a.sum())
        self.e_sq += float(e @ e)
        self.a_sq += float(a @ a)
        self.dot += float(e @ a)

    def result(self):
        if not self.count:
            raise ValueError("cannot summarize empty comparisons")
        norm = math.sqrt(self.e_sq * self.a_sq)
        ev = max(0.0, self.e_sq - self.e_sum * self.e_sum / self.count)
        av = max(0.0, self.a_sq - self.a_sum * self.a_sum / self.count)
        centered_norm = math.sqrt(ev * av)
        clip = lambda value: max(-1.0, min(1.0, value))
        return {
            "elements": int(self.count), "unequal_elements": self.mismatches,
            "max_absolute_error": self.max_error,
            "mean_absolute_error": self.abs_error / self.count,
            "rmse": math.sqrt(self.squared_error / self.count),
            "relative_l2": math.sqrt(self.squared_error / self.e_sq) if self.e_sq else None,
            "cosine": clip(self.dot / norm) if norm else None,
            "pcc": clip((self.dot - self.e_sum * self.a_sum / self.count) / centered_norm)
                   if centered_norm else None,
            "scale": self.dot / self.e_sq if self.e_sq else None,
            "nonfinite_count": 0,
        }


def capture_pytorch(model, value, boundaries):
    """Capture reused ReLU invocations immediately, before in-place mutation."""
    import torch
    modules = dict(model.named_modules())
    captures, handles, invocations = {}, [], Counter()
    requested = {}
    for boundary in boundaries:
        if boundary.module:
            requested[(boundary.module + ".relu" if boundary.relu_call and boundary.kind != "stem"
                       else boundary.module, boundary.relu_call or 1)] = boundary.dispatch

    def hook(name):
        def save(_module, _args, output):
            invocations[name] += 1
            call = requested.get((name, invocations[name]))
            if call is not None:
                captures[call] = output.detach().clone()
        return save

    try:
        for name in {name for name, _ in requested}:
            handles.append(modules[name].register_forward_hook(hook(name)))
        with torch.inference_mode():
            output = model(value)
            captures[1] = value.detach().clone()
            # A diagnostic boundary: PyTorch AvgPool does not expose this sum.
            captures[56] = captures[55].float().sum((2, 3), keepdim=True).to(torch.bfloat16)
        if set(captures) != {b.dispatch for b in boundaries}:
            raise ValueError("PyTorch hooks did not cover all dispatch boundaries")
        return {call: tensor.float().numpy().copy() for call, tensor in captures.items()}, output
    finally:
        for handle in handles:
            handle.remove()


def replay_dispatch(model, boundary, measured, original_input):
    """No upstream substitution: each source is a saved DEVICE checkpoint."""
    import torch
    modules = dict(model.named_modules()) if model is not None else {}
    sources = [torch.from_numpy(measured[call]).to(torch.bfloat16) for call in boundary.sources]
    kind = boundary.kind
    with torch.inference_mode():
        if kind == "copy":
            return original_input.clone()
        if kind == "stem":
            return model.relu(model.bn1(model.conv1(sources[0])))
        if kind == "pool":
            return model.maxpool(sources[0])
        if kind in ("conv1", "conv2"):
            block = modules[boundary.module]
            number = int(kind[-1])
            return block.relu(getattr(block, f"bn{number}")(getattr(block, kind)(sources[0])))
        if kind == "downsample":
            return modules[boundary.module](sources[0])
        if kind == "residual":
            block = modules[boundary.module]
            output = block.bn3(block.conv3(sources[0]))
            skip = block.downsample[1](sources[1]) if block.downsample is not None else sources[1]
            return block.relu(output + skip)
        if kind == "sum":
            return sources[0].float().sum((2, 3), keepdim=True).to(torch.bfloat16)
        if kind == "divide":
            return sources[0] / 49
        if kind == "linear":
            return model.fc(sources[0].flatten(1))
        raise ValueError(f"unsupported replay: {kind}")


def replay_ir_policy(model, boundary, measured):
    """Exact scalar policies, including the narrow FC ordered-F32 dot."""
    if boundary.kind not in ("sum", "divide", "linear"):
        return None, ""
    import torch
    value = torch.from_numpy(measured[boundary.sources[0]]).to(torch.bfloat16)
    with torch.inference_mode():
        if boundary.kind == "sum":
            values = value.flatten(2)
            output = torch.zeros_like(values[:, :, :1])
            for index in range(values.shape[-1]):
                output = output + values[:, :, index:index + 1]
            return output.unsqueeze(-1), "49 BF16 adds, row-major order, RNE after each add"
        if boundary.kind == "divide":
            return value / 49, "measured BF16 sum divided by exactly 49, BF16 RNE"
        if boundary.kind == "linear":
            return ordered_linear_bf16(value.flatten(1), model.fc.weight, model.fc.bias), (
                "increasing-K FP32 product/add from zero, BF16 dot truncation, separate BF16 bias add")
    return None, ""


def ordered_linear_bf16(value, weight, bias):
    """Independent CPU reference for v17 FC, without BLAS reassociation/FMA."""
    import numpy as np
    import torch
    if (value.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or bias.dtype != torch.bfloat16 or value.ndim != 2 or weight.ndim != 2
            or value.shape[1] != weight.shape[1] or bias.shape != (weight.shape[0],)):
        raise ValueError("ordered linear reference requires matching BF16 tensors")
    inputs, weights = value.float().numpy(), weight.detach().float().numpy()
    accumulator = np.zeros((len(inputs), len(weights)), dtype=np.float32)
    for k in range(inputs.shape[1]):
        product = inputs[:, k:k + 1] * weights[:, k][None, :]
        np.add(accumulator, product, out=accumulator)
    return torch.from_numpy(accumulator).to(torch.bfloat16) + bias


def validate_run_metadata(golden, manifest, iree_results, ttnn_results, iree_logits, ttnn_logits):
    """Do not quietly pair different weights, inputs, precision or batch rows."""
    if (iree_results["status"] != "completed" or iree_results["batch"] != 1
            or iree_results["precision_policy"] != "bf16_module"
            or pathlib.Path(iree_results["golden_dir"]).resolve() != golden.resolve()):
        raise ValueError("requires a completed matching batch-one tt-iree BF16 run")
    indices = iree_results["indices"]
    count = len(indices)
    if (not count or any(type(i) is not int or i < 0 or i >= len(manifest["labels"]) for i in indices)
            or len(set(indices)) != count or iree_logits.shape != (count, 1000)
            or [s["index"] for s in iree_results["samples"]] != indices):
        raise ValueError("tt-iree sample indices or logits shape do not match")
    if (ttnn_results["status"] != "completed" or ttnn_results["activation_dtype"] != "bf16"
            or ttnn_results["weight_dtype"] != "bf16" or ttnn_results["input_dtype"] != "BF16"
            or ttnn_results["input_sha256"] != manifest["input"]["sha256"]
            or ttnn_results["golden_manifest"]["sha256"] != file_record(golden / "manifest.json")["sha256"]
            or ttnn_results["samples"] != len(manifest["labels"])
            or ttnn_logits.shape != (len(manifest["labels"]), 1000)):
        raise ValueError("TTNN inputs, weights, BF16 policy or full-dataset row order do not match")
    require_bf16_values(iree_logits, "tt-iree logits")
    require_bf16_values(ttnn_logits, "TTNN logits")
    return indices


def write_csv(filename, rows):
    with pathlib.Path(filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    import numpy as np
    import torch
    import torchvision
    torch.set_num_threads(args.threads)
    golden, run, ttnn = args.golden_dir.resolve(), args.ttiree_run.resolve(), args.ttnn_run.resolve()
    manifest = json.loads((golden / "manifest.json").read_text())
    iree_results = json.loads((run / "results.json").read_text())
    matrix_engine = iree_results.get("matrix_engine", "riscv_fp32")
    if matrix_engine not in ("riscv_fp32", "tensix_hifi4"):
        raise ValueError("unknown tt-iree BF16 matrix engine")
    use_tensix = matrix_engine == "tensix_hifi4"
    ttnn_results = json.loads((ttnn / "results.json").read_text())
    source = json.loads(pathlib.Path(iree_results["export"]["path"]).read_text())
    provenance = [verify_record(iree_results["export"]["path"], iree_results["export"]),
                  file_record(golden / "manifest.json"), file_record(run / "results.json"),
                  file_record(ttnn / "results.json"), file_record(run / "logits.npy"),
                  file_record(ttnn / "logits.npy"), file_record(__file__)]
    for key in ("state_dict", "input"):
        provenance.append(verify_record(golden / ("state_dict.pt" if key == "state_dict" else "input.npy"), manifest[key]))
        verify_record(manifest[key]["path"], source[key])
    for policy in ("bf16_module", "fp32"):
        provenance.append(verify_record(golden / policy / "logits.npy", manifest["logits"][policy]))
    native = np.load(run / "logits.npy", allow_pickle=False)
    nn_full = np.load(ttnn / "logits.npy", allow_pickle=False)
    indices = validate_run_metadata(golden, manifest, iree_results, ttnn_results, native, nn_full)
    inputs = np.load(golden / "input.npy", allow_pickle=False)
    bf16_full = np.load(golden / "bf16_module/logits.npy", allow_pickle=False)
    fp32 = np.load(golden / "fp32/logits.npy", allow_pickle=False)[indices]
    expected, nn_values = bf16_full[indices], nn_full[indices]
    batch_reference = np.load(ttnn / "pytorch_bf16_batch_reference.npy", allow_pickle=False)
    if batch_reference.shape != bf16_full.shape:
        raise ValueError("PyTorch TTNN-batch reference shape mismatch")
    require_bf16_values(batch_reference, "PyTorch TTNN-batch logits")
    provenance.append(file_record(ttnn / "pytorch_bf16_batch_reference.npy"))
    labels = [manifest["labels"][i] for i in indices]
    categories = manifest["categories"]
    comparisons = {}
    for name, reference, output in (
        ("pytorch_bf16_vs_pytorch_fp32", fp32, expected),
        ("ttiree_vs_pytorch_bf16", expected, native),
        ("ttnn_vs_pytorch_bf16_batch1", expected, nn_values),
        ("ttnn_vs_pytorch_bf16_hardware_batch", batch_reference[indices], nn_values),
        ("ttiree_vs_pytorch_fp32", fp32, native),
        ("ttnn_vs_pytorch_fp32", fp32, nn_values),
        ("ttnn_vs_ttiree", native, nn_values),
        ("pytorch_batch_sensitivity", expected, batch_reference[indices]),
    ):
        comparison = classifier_comparison(reference, output, categories, labels)
        for row, index in zip(comparison["per_image"], indices):
            row["source_index"] = index
        comparisons[name] = comparison
    baseline = None
    if args.ttiree_reference_run is not None:
        baseline = args.ttiree_reference_run.resolve()
        baseline_results = json.loads((baseline / "results.json").read_text())
        if (baseline_results["status"] != "completed" or baseline_results["batch"] != 1
                or baseline_results["precision_policy"] != "bf16_module"
                or baseline_results["indices"] != indices
                or baseline_results["export"]["sha256"] != iree_results["export"]["sha256"]
                or baseline_results.get("matrix_engine", "riscv_fp32") != "riscv_fp32"):
            raise ValueError("requires a paired matching RISC-V reference run")
        baseline_logits = np.load(baseline / "logits.npy", allow_pickle=False)
        require_bf16_values(baseline_logits, "RISC-V baseline logits")
        comparisons["ttiree_vs_riscv_baseline"] = classifier_comparison(
            baseline_logits, native, categories, labels)
        provenance.extend((file_record(baseline / "results.json"), file_record(baseline / "logits.npy")))
    for comparison in comparisons.values():
        for row, index in zip(comparison["per_image"], indices):
            row["source_index"] = index
    # Context only; never present TTNN's 32 rows as paired tt-iree coverage.
    nn_full_comparison = classifier_comparison(bf16_full, nn_full, categories, manifest["labels"])
    require_new_directory(args.output_dir)
    out = args.output_dir.resolve()
    summary = {"schema_version": 1, "status": "running", "map_version": MAP_VERSION,
               "indices": indices, "sample_count": len(indices),
               "comparison_scope": "paired demo subset; diagnostics, not ImageNet validation or a new tolerance gate",
               "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
               "threads": args.threads, "ttiree_batch": 1, "ttnn_batch": ttnn_results["batch"],
               "ttnn_fidelity": ttnn_results["math_fidelity"],
               "ttnn_avgpool_precision": ttnn_results.get("avgpool_precision"),
               "matrix_engine": matrix_engine,
               "ttiree_compute": ("HiFi4 BF16 Tensix contraction, FP32 Dst, device RISC-V IR epilogue"
                                  if use_tensix else "device RISC-V FP32 contraction with lowered BF16 rounding; NOT Tensix"),
               "local_ir_policy_scope": "v17 scalar/ordered-F32 reference; for Tensix this is a comparator, NOT a bit-exact hardware model",
               "comparisons": comparisons, "ttnn_full_dataset_context": nn_full_comparison,
               "dispatch_references": 0, "provenance": provenance}
    write_json(out / "summary.json", summary)
    boundaries = dispatch_boundaries()
    write_json(out / "boundary_map.json", {
        "version": MAP_VERSION, "layout": "singleton NCHW; CxHW flattened stores reshape only; no permutation",
        "padded_copy_scope": "logical stored slice only, not padding halo",
        "sum_reference": "derived FP32 sum then BF16 cast; not an exposed PyTorch AvgPool intermediate",
        "boundaries": [dataclasses.asdict(b) for b in boundaries],
    })
    rows, grouped, grouped_counts = [], {}, Counter()
    try:
        model = torchvision.models.resnet50(weights=None).eval()
        model.load_state_dict(torch.load(golden / "state_dict.pt", map_location="cpu", weights_only=True))
        model.to(torch.bfloat16)
        for index in indices:
            actual, files = load_dispatches(run / f"sample-{index:02d}", boundaries, use_tensix)
            summary["provenance"].extend(files)
            baseline_values = None
            if baseline is not None:
                baseline_values, baseline_files = load_dispatches(baseline / f"sample-{index:02d}", boundaries)
                summary["provenance"].extend(baseline_files)
            if not np.array_equal(actual[58], native[indices.index(index):indices.index(index) + 1]):
                raise ValueError("final checkpoint and recorded logits disagree")
            value = torch.from_numpy(inputs[index:index + 1].copy()).to(torch.bfloat16)
            input_bits = value.view(torch.uint16).numpy().astype("<u2").tobytes()
            recorded_input = run / f"sample-{index:02d}/input.bf16.bin"
            if recorded_input.read_bytes() != input_bits:
                raise ValueError(f"recorded device input differs from golden BF16 input: {index}")
            summary["provenance"].append(file_record(recorded_input))
            reference, logits = capture_pytorch(model, value, boundaries)
            if not np.array_equal(logits.float().numpy(), bf16_full[index:index + 1]):
                raise ValueError(f"fresh PyTorch BF16 logits disagree with immutable golden: {index}")
            for boundary in boundaries:
                if boundary.kind in ("pool", "residual", "divide", "linear"):
                    name = boundary.module
                    filename = golden / "bf16_module/checkpoints" / f"{index:05d}" / f"{name}.npy"
                    saved = np.load(filename, allow_pickle=False)
                    if not np.array_equal(saved, reference[boundary.dispatch]):
                        raise ValueError(f"fresh PyTorch checkpoint disagrees with golden: {index} {name}")
                    summary["provenance"].append(file_record(filename))
            target = out / "references" / f"{index:05d}"
            target.mkdir(parents=True)
            for boundary in boundaries:
                call = boundary.dispatch
                if reference[call].shape != boundary.shape:
                    raise ValueError(f"PyTorch boundary shape changed: {boundary.name}")
                np.save(target / f"dispatch_{call:04d}_pytorch.npy", reference[call])
                replay = replay_dispatch(model, boundary, actual, value).float().numpy()
                np.save(target / f"dispatch_{call:04d}_local_pytorch.npy", replay)
                policy, policy_description = replay_ir_policy(model, boundary, actual)
                pairs = [("cumulative", reference[call], "fresh PyTorch BF16 model, golden input"),
                         ("local_pytorch", replay, "PyTorch operator composition, measured device input(s)")]
                if baseline_values is not None:
                    pairs.append(("riscv_checkpoint", baseline_values[call],
                                  "paired v17 hardware checkpoint; includes upstream drift, NOT local engine error"))
                if policy is not None:
                    policy = policy.float().numpy()
                    np.save(target / f"dispatch_{call:04d}_local_ir_policy.npy", policy)
                    pairs.append(("local_ir_policy", policy, policy_description))
                if boundary.kind == "linear":
                    from torch.nn import functional as functional
                    with torch.inference_mode():
                        fc_input = torch.from_numpy(actual[57]).to(torch.bfloat16).flatten(1)
                        split_bias = (functional.linear(fc_input, model.fc.weight, None)
                                      + model.fc.bias).float().numpy()
                    np.save(target / f"dispatch_{call:04d}_local_pytorch_split_bias.npy", split_bias)
                    pairs.append(("local_pytorch_split_bias", split_bias,
                                  "PyTorch BF16 dot without bias, then BF16 bias add; dot order determined by PyTorch"))
                for scope, expected_values, description in pairs:
                    metrics = error_metrics(expected_values, actual[call])
                    rows.append({"source_index": index, "dispatch": call, "op_type": boundary.op_type,
                                 "boundary": boundary.name, "op_code": boundary.op_code,
                                 "scope": scope, "reference_policy": description,
                                 "source_dispatches": ";".join(map(str, boundary.sources)),
                                 **metrics, "unequal_elements": int(np.count_nonzero(expected_values != actual[call]))})
                    key = (scope, boundary.op_type)
                    grouped.setdefault(key, MetricAccumulator()).add(expected_values, actual[call])
                    grouped_counts[key] += 1
                summary["dispatch_references"] += 1
            write_json(out / "summary.json", summary)
            print(f"sample {index}: all 58 boundaries, cumulative and measured-input replay", flush=True)
        write_csv(out / "dispatch_accuracy.csv", rows)
        operation_rows = [{"scope": scope, "op_type": op, "calls": grouped_counts[(scope, op)],
                           **accumulator.result()} for (scope, op), accumulator in sorted(grouped.items())]
        write_csv(out / "operation_accuracy.csv", operation_rows)
        logits_rows = [{"comparison": name, **value["aggregate"]} for name, value in comparisons.items()]
        write_csv(out / "logits_accuracy.csv", logits_rows)
        summary.update(status="completed", operation_metrics=operation_rows,
                       coverage={"dispatches_per_image": 58, "cumulative": len(indices) * 58,
                                 "local_pytorch": len(indices) * 58, "local_ir_policy": len(indices) * 3,
                                 "local_pytorch_split_bias": len(indices),
                                 "riscv_checkpoint": len(indices) * 58 if baseline is not None else 0,
                                 "ttnn_intermediate_checkpoints": 0},
                       worst_local_dispatches=sorted((r for r in rows if r["scope"] == "local_pytorch"),
                                                     key=lambda r: r["relative_l2"] or 0, reverse=True)[:12])
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", required=True, type=pathlib.Path)
    parser.add_argument("--ttiree-run", required=True, type=pathlib.Path)
    parser.add_argument("--ttnn-run", required=True, type=pathlib.Path)
    parser.add_argument("--ttiree-reference-run", type=pathlib.Path,
                        help="optional paired v17 RISC-V run, with identical indices and export hash")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    analyze(args)


if __name__ == "__main__":
    main()
