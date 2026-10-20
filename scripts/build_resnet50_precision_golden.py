#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build additive ResNet50 references without changing the original golden.

Requires torch, torchvision, numpy and Pillow. No Tenstorrent device is opened.
The BF16 module reference follows upstream TTNN's PyTorch reference policy;
the BF16-operand reference runs FP32 torch operators, not a Tensix simulator.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime
from zoneinfo import ZoneInfo


POLICIES = {
    "fp32": "FP32 model, input, operators and outputs; eval mode",
    "bf16_module": (
        "model.to(bfloat16), input.to(bfloat16), eval mode; including BN "
        "parameters/buffers and operator outputs; accumulation is determined "
        "by the recorded PyTorch CPU implementation"
    ),
    "bf16_operands_fp32_ops": (
        "round Conv2d/Linear inputs and weights to BF16 then widen exactly to "
        "FP32; run torch FP32 operators; bias, BN, residual, pool, reduction "
        "and all outputs stay FP32; not bit-exact Tensix arithmetic"
    ),
}
CHECKPOINT_NAMES = (
    "maxpool",
    *(f"layer{layer}.{block}" for layer, blocks in enumerate((3, 4, 6, 3), 1)
      for block in range(blocks)),
    "avgpool",
    "fc",
)


def file_record(filename):
    filename = pathlib.Path(filename).resolve()
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(filename), "bytes": filename.stat().st_size,
            "sha256": digest.hexdigest()}


def write_json(filename, value):
    pathlib.Path(filename).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def require_new_directory(directory):
    directory = pathlib.Path(directory)
    if directory.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {directory}")
    directory.mkdir(parents=True)


def read_class_map(filename):
    """Read the upstream constant without importing the TT-Metal package."""
    tree = ast.parse(pathlib.Path(filename).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "IMAGENET2012_CLASSES"
            for target in node.targets
        ):
            value = node.value
            if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "OrderedDict" and len(value.args) == 1):
                raise ValueError("unexpected ImageNet class-map expression")
            classes = ast.literal_eval(value.args[0])
            if (len(classes) != 1000 or list(classes) != sorted(classes)
                    or not all(re.fullmatch(r"n\d{8}", key) for key in classes)):
                raise ValueError("class map must contain 1000 ordered ImageNet synsets")
            return classes
    raise ValueError("ImageNet class map not found")


def image_label(filename, class_map):
    match = re.fullmatch(r"ILSVRC2012_val_\d{8}_(n\d{8})", pathlib.Path(filename).stem)
    if not match or match[1] not in class_map:
        raise ValueError(f"image has no recognized filename synset: {filename}")
    return list(class_map).index(match[1])


def manifest_images(filename, class_map, limit):
    """Retain explicit source order and reject label/path ambiguity."""
    filename = pathlib.Path(filename).resolve()
    manifest = json.loads(filename.read_text())
    if manifest.get("split") != "validation":
        raise ValueError("requires a labeled validation manifest")
    paths, labels, seen = [], [], set()
    for row in manifest["images"][:limit]:
        image = (filename.parent / row["filename"]).resolve()
        label = row["label"]
        if (not image.is_relative_to(filename.parent) or image in seen
                or type(label) is not int or not 0 <= label < 1000
                or row["synset"] != list(class_map)[label]
                or file_record(image)["sha256"] != row["sha256"]):
            raise ValueError("invalid/duplicate image, label, synset or hash in manifest")
        seen.add(image)
        paths.append(image)
        labels.append(label)
    if not paths or len(paths) != min(limit, len(manifest["images"])):
        raise ValueError("empty or incomplete image manifest")
    return paths, labels, manifest


def error_metrics(expected, actual):
    import numpy as np

    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape or not expected.size:
        raise ValueError("comparison requires matching nonempty shapes")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("non-finite comparison values")
    e, a = expected.astype(np.float64).reshape(-1), actual.astype(np.float64).reshape(-1)
    diff = a - e
    en, an = float(np.linalg.norm(e)), float(np.linalg.norm(a))
    dot = float(e @ a)
    ec, ac = e - e.mean(), a - a.mean()
    denominator = float(np.linalg.norm(ec) * np.linalg.norm(ac))
    return {
        "elements": int(e.size),
        "max_absolute_error": float(np.abs(diff).max()),
        "mean_absolute_error": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_l2": float(np.linalg.norm(diff) / en) if en else None,
        "cosine": float(np.clip(dot / (en * an), -1, 1)) if en and an else None,
        "pcc": float(np.clip(ec @ ac / denominator, -1, 1)) if denominator else None,
        "scale": dot / (en * en) if en else None,
        "nonfinite_count": 0,
    }


def predictions(logits, categories, labels=None):
    import numpy as np

    logits = np.asarray(logits)
    if (logits.ndim != 2 or logits.shape[1] != len(categories)
            or logits.shape[1] < 5 or not np.isfinite(logits).all()):
        raise ValueError("invalid classifier logits")
    if labels is not None and (
        len(labels) != len(logits)
        or any(label < 0 or label >= len(categories) for label in labels)
    ):
        raise ValueError("invalid classifier labels")
    rows = []
    for index, row in enumerate(logits):
        top = np.argsort(-row, kind="stable")[:5].tolist()
        shifted = row.astype(np.float64) - float(row.max())
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        result = {"sample": index, "top1": top[0], "top5": top,
                  "categories": [categories[i] for i in top],
                  "logits": [float(row[i]) for i in top],
                  "probabilities": [float(probabilities[i]) for i in top]}
        if labels is not None:
            result.update(label=int(labels[index]), top1_correct=top[0] == labels[index],
                          top5_correct=labels[index] in top)
        rows.append(result)
    return rows


def classifier_comparison(expected, actual, categories, labels=None):
    ep, ap = predictions(expected, categories, labels), predictions(actual, categories, labels)
    summary = error_metrics(expected, actual)
    summary.update(
        samples=len(ep), top1_agreement=sum(e["top1"] == a["top1"] for e, a in zip(ep, ap)),
        mean_top5_overlap=sum(len(set(e["top5"]) & set(a["top5"])) for e, a in zip(ep, ap)) / len(ep),
    )
    if labels is not None:
        summary.update(expected_top1_correct=sum(r["top1_correct"] for r in ep),
                       actual_top1_correct=sum(r["top1_correct"] for r in ap),
                       expected_top5_correct=sum(r["top5_correct"] for r in ep),
                       actual_top5_correct=sum(r["top5_correct"] for r in ap))
    per_image = []
    for i, (e, a) in enumerate(zip(ep, ap)):
        per_image.append({"sample": i, **error_metrics(expected[i], actual[i]),
                          "expected_top1": e["top1"], "actual_top1": a["top1"],
                          "top1_agreement": e["top1"] == a["top1"],
                          "top5_overlap": len(set(e["top5"]) & set(a["top5"]))})
    return {"aggregate": summary, "per_image": per_image}


def quantize_operand_fp32(value):
    import torch

    return value.to(torch.bfloat16).to(torch.float32)


def prepare_model(base, policy):
    import torch
    from torch.nn import functional as functional

    model = copy.deepcopy(base).eval()
    if policy == "bf16_module":
        return model.to(torch.bfloat16)
    if policy == "bf16_operands_fp32_ops":
        def conv_forward(module, value):
            return module._conv_forward(quantize_operand_fp32(value),
                                        quantize_operand_fp32(module.weight), module.bias)

        def linear_forward(module, value):
            return functional.linear(quantize_operand_fp32(value),
                                     quantize_operand_fp32(module.weight), module.bias)

        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.forward = types.MethodType(conv_forward, module)
            elif isinstance(module, torch.nn.Linear):
                module.forward = types.MethodType(linear_forward, module)
    elif policy != "fp32":
        raise ValueError(f"unknown precision policy: {policy}")
    return model


def run_reference(model, inputs, policy, directory, checkpoints):
    import numpy as np
    import torch

    directory.mkdir(parents=True)
    current = [0]
    handles = []
    if checkpoints:
        for name, module in model.named_modules():
            if name in CHECKPOINT_NAMES:
                def hook(_module, _args, output, name=name):
                    target = directory / "checkpoints" / f"{current[0]:05d}"
                    target.mkdir(parents=True, exist_ok=True)
                    np.save(target / f"{name}.npy", output.detach().float().numpy())
                handles.append(module.register_forward_hook(hook))
    outputs = []
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for index, value in enumerate(inputs.split(1)):
                current[0] = index
                if policy == "bf16_module":
                    value = value.to(torch.bfloat16)
                output = model(value).detach()
                if not torch.isfinite(output).all():
                    raise ValueError(f"non-finite {policy} output at sample {index}")
                outputs.append(output)
    finally:
        for handle in handles:
            handle.remove()
    logits = torch.cat(outputs)
    torch.save(logits, directory / "logits.pt")
    np.save(directory / "logits.npy", logits.float().numpy())
    if logits.dtype == torch.bfloat16:
        np.save(directory / "logits.bf16_bits.npy", logits.contiguous().view(torch.uint16).numpy())
    return logits.float().numpy(), time.monotonic() - start


def build(args):
    import numpy as np
    import torch
    import torchvision
    from PIL import Image

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    base = torchvision.models.resnet50(weights=None).eval()
    source = {}
    image_records = []
    labels = None
    if args.random_golden:
        source_dir = args.random_golden.resolve()
        for name in ("state_dict.pt", "input.pt", "output.pt", "manifest.json"):
            source[name] = file_record(source_dir / name)
        base.load_state_dict(torch.load(source_dir / "state_dict.pt", weights_only=True))
        inputs = torch.load(source_dir / "input.pt", weights_only=True).float()
        categories = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
        scope = "existing_random_weight_fixed_input; not a classification accuracy test"
    else:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        base = torchvision.models.resnet50(weights=weights).eval()
        categories = weights.meta["categories"]
        classes = read_class_map(args.class_map)
        dataset_manifest = getattr(args, "image_manifest", None)
        if dataset_manifest:
            paths, labels, dataset = manifest_images(dataset_manifest, classes, args.limit)
        else:
            paths = sorted(args.images.glob("*.JPEG"))[:args.limit]
            if not paths:
                raise ValueError("no labeled ImageNet demo JPEGs found")
            labels = [image_label(filename, classes) for filename in paths]
        transform = weights.transforms()
        tensors = []
        for filename, label in zip(paths, labels):
            with Image.open(filename) as image:
                tensors.append(transform(image.convert("RGB")))
            image_records.append({**file_record(filename), "label": label,
                                  "synset": list(classes)[label],
                                  "category": categories[label]})
        inputs = torch.stack(tensors)
        checkpoint = pathlib.Path(torch.hub.get_dir()) / "checkpoints" / pathlib.Path(weights.url).name
        source = {"weights": str(weights), "weight_url": weights.url,
                  "weight_checkpoint": file_record(checkpoint),
                  "class_map": file_record(args.class_map),
                  "transform": repr(transform), "label_source": "upstream demo filename synset"}
        scope = "fixed upstream ImageNet validation demo subset; not full ImageNet accuracy"
        if dataset_manifest:
            source["image_manifest"] = file_record(dataset_manifest)
            source["dataset"] = {key: dataset[key] for key in (
                "dataset", "revision", "split", "source_url", "scope", "sampling")}
            source["label_source"] = "verified source Parquet ClassLabel and ordered synset map"
            scope = dataset["scope"]
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 224, 224) or not torch.isfinite(inputs).all():
        raise ValueError("expected finite Bx3x224x224 inputs")
    require_new_directory(args.output_dir)
    root = args.output_dir.resolve()
    torch.save(base.state_dict(), root / "state_dict.pt")
    torch.save(inputs, root / "input.pt")
    np.save(root / "input.npy", inputs.numpy())
    for index, value in enumerate(inputs.split(1)):
        target = root / "inputs"
        target.mkdir(exist_ok=True)
        np.save(target / f"{index:05d}.npy", value.numpy())
    if image_records:
        target = root / "images"
        target.mkdir()
        for record in image_records:
            original = pathlib.Path(record["path"])
            shutil.copyfile(original, target / original.name)
            record["saved_image"] = file_record(target / original.name)
    manifest = {
        "created_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "model": "torchvision.models.resnet50", "scope": scope, "source": source,
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "python": sys.version, "numpy_version": np.__version__, "cpu_threads": args.threads,
        "cpu_reference_batch": 1, "input_shape": list(inputs.shape), "categories": categories,
        "labels": labels, "images": image_records, "policies": POLICIES,
        "checkpoint_names": list(CHECKPOINT_NAMES) if args.checkpoints else [],
        "checkpoint_storage": "F32 NPY; BF16 values widened exactly; logits.pt retains dtype",
        "script": file_record(__file__), "state_dict": file_record(root / "state_dict.pt"),
        "input": file_record(root / "input.npy"), "comparisons_are_diagnostics_not_new_acceptance_gates": True,
    }
    write_json(root / "manifest.json", manifest)
    outputs, timings = {}, {}
    for policy in POLICIES:
        outputs[policy], timings[policy] = run_reference(
            prepare_model(base, policy), inputs, policy, root / policy, args.checkpoints
        )
        write_json(root / policy / "predictions.json", predictions(outputs[policy], categories, labels))
        print(f"{policy}: {len(inputs)} samples, {timings[policy]:.3f} s", flush=True)
    comparisons = {
        policy: classifier_comparison(outputs["fp32"], outputs[policy], categories, labels)
        for policy in POLICIES if policy != "fp32"
    }
    if args.random_golden:
        original = torch.load(args.random_golden / "output.pt", weights_only=True).float().numpy()
        comparisons["original_framework_vs_fresh_fp32"] = classifier_comparison(
            original, outputs["fp32"], categories
        )
    if args.checkpoints:
        checkpoint_rows = []
        for policy in POLICIES:
            if policy == "fp32":
                continue
            for index in range(len(inputs)):
                for name in CHECKPOINT_NAMES:
                    suffix = pathlib.Path("checkpoints") / f"{index:05d}" / f"{name}.npy"
                    checkpoint_rows.append({"policy": policy, "sample": index, "checkpoint": name,
                                            **error_metrics(np.load(root / "fp32" / suffix),
                                                            np.load(root / policy / suffix))})
        with (root / "checkpoint_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checkpoint_rows[0]))
            writer.writeheader()
            writer.writerows(checkpoint_rows)
    manifest["logits"] = {policy: file_record(root / policy / "logits.npy") for policy in POLICIES}
    manifest["reference_wall_seconds_including_checkpoint_io"] = timings
    write_json(root / "manifest.json", manifest)
    write_json(root / "comparisons.json", comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--random-golden", type=pathlib.Path)
    mode.add_argument("--images", type=pathlib.Path)
    mode.add_argument("--image-manifest", type=pathlib.Path)
    parser.add_argument("--class-map", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.threads < 1 or ((args.images or args.image_manifest) and not args.class_map):
        parser.error("positive limit/threads and --class-map for images are required")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new destination")
    build(args)


if __name__ == "__main__":
    main()
