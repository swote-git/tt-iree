#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Train an MNIST MLP and export a real BF16, static-B1 IREE input module.

Training uses only the official training split. Hardware samples are a seeded,
class-balanced subset of the held-out test split, never a training subset.
Requires the existing Torch/Torchvision/Turbine frontend environment.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
from pathlib import Path
import random
import subprocess
import time

from build_resnet50_precision_golden import file_record, require_new_directory, write_json
from export_resnet50_bf16 import bf16_resource_bridge


def balanced_test_indices(labels, samples, seed):
    if samples < 10 or samples % 10:
        raise ValueError("sample count must be a positive multiple of ten")
    rng, selected = random.Random(seed), []
    for digit in range(10):
        candidates = [i for i, label in enumerate(labels) if label == digit]
        if len(candidates) < samples // 10:
            raise ValueError("insufficient held-out samples for digit")
        selected.extend(rng.sample(candidates, samples // 10))
    rng.shuffle(selected)
    return selected


def make_model(torch):
    class MnistMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 128)
            self.fc2 = torch.nn.Linear(128, 10)

        def forward(self, images):
            return self.fc2(torch.relu(self.fc1(torch.flatten(images, 1))))

    return MnistMLP()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--iree-compile", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    if args.samples < 10 or args.samples % 10:
        parser.error("samples must be a positive multiple of ten")
    require_new_directory(args.output_dir)
    out = args.output_dir.resolve()
    import numpy as np
    import torch
    import torchvision
    import iree.turbine.aot as aot
    from iree.turbine.aot.support import ir_utils
    from iree.compiler.tools.binaries import find_tool

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train = torchvision.datasets.MNIST(args.data_dir, train=True, download=True)
    test = torchvision.datasets.MNIST(args.data_dir, train=False, download=True)
    assert len(train) == 60000 and len(test) == 10000
    train_x = (train.data.to(torch.float32)[:, None] / 255.0 - 0.1307) / 0.3081
    test_x = (test.data.to(torch.float32)[:, None] / 255.0 - 0.1307) / 0.3081
    indices = balanced_test_indices(test.targets.tolist(), args.samples, args.seed)
    model = make_model(torch).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(args.epochs):
        start, loss_sum = time.perf_counter(), 0.0
        order = torch.randperm(len(train))
        for batch in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[batch]), train.targets[batch])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        history.append(dict(epoch=epoch + 1, training_loss=loss_sum / len(train),
                            seconds=time.perf_counter() - start))
        print(history[-1], flush=True)
    model.eval()
    bf16_model = copy.deepcopy(model).to(torch.bfloat16).eval()
    torch.save(model.state_dict(), out / "state_dict_fp32.pt")
    torch.save(bf16_model.state_dict(), out / "state_dict_bf16.pt")
    inputs = test_x[indices].contiguous()
    np.save(out / "input.npy", inputs.numpy())
    np.save(out / "labels.npy", test.targets[indices].numpy())
    full_accuracy = {}
    with torch.inference_mode():
        for name, instance, dtype in (("fp32", model, torch.float32),
                                      ("bf16", bf16_model, torch.bfloat16)):
            correct = 0
            for begin in range(0, len(test), 512):
                logits = instance(test_x[begin:begin + 512].to(dtype))
                correct += (logits.argmax(1) == test.targets[begin:begin + 512]).sum().item()
            full_accuracy[name] = dict(correct=correct, total=len(test),
                                       scope="CPU reference, full held-out test split, batch up to 512")
            logits = torch.cat([instance(value[None].to(dtype)).float() for value in inputs])
            np.save(out / f"pytorch_{name}_logits.npy", logits.numpy())
    raw = inputs.to(torch.bfloat16).view(torch.uint16).numpy().astype("<u2")
    (out / "inputs").mkdir()
    for index, value in enumerate(raw):
        value.tofile(out / "inputs" / f"{index:03d}.bf16.bin")

    manifest = dict(
        dataset="MNIST", training_split="train", train_samples=len(train),
        hardware_split="test", test_samples=len(test), seed=args.seed,
        selection="seeded sample without replacement, equal number per digit",
        indices=indices, labels=test.targets[indices].tolist(),
        preprocessing="float32: (uint8 / 255 - 0.1307) / 0.3081; then BF16 input rounding",
        model="Flatten -> Linear(784,128) -> ReLU -> Linear(128,10); output logits",
        batch=1, input_shape=[1, 1, 28, 28], output_shape=[1, 10],
        input_dtype="bf16", output_dtype="bf16", hidden_size=128,
        training=dict(epochs=args.epochs, batch=256, optimizer="Adam", lr=0.001,
                      deterministic_algorithms=True, history=history),
        full_test_cpu_accuracy=full_accuracy,
        torch=torch.__version__, torchvision=torchvision.__version__,
        iree_turbine=importlib.metadata.version("iree-turbine"),
        iree_base_compiler=importlib.metadata.version("iree-base-compiler"),
        state_dict=file_record(out / "state_dict_bf16.pt"),
        fp32_state_dict=file_record(out / "state_dict_fp32.pt"),
        input=file_record(out / "input.npy"),
        reference={name: file_record(out / f"pytorch_{name}_logits.npy") for name in ("fp32", "bf16")},
        dataset_resources=[file_record(path) for path in sorted((args.data_dir / 'MNIST/raw').iterdir()) if path.is_file()],
        script=file_record(__file__), export_status="started")
    write_json(out / "manifest.json", manifest)
    with bf16_resource_bridge(ir_utils, torch):
        exported = aot.export(bf16_model, args=(inputs[:1].to(torch.bfloat16),))
        exported.verify()
        exported.save_mlir(out / "mnist_torch.mlirbc")
    commands = [
        [find_tool("iree-compile"), str(out / "mnist_torch.mlirbc"),
         "--iree-input-type=torch", "--compile-to=input", "--emit-mlir-bytecode",
         "-o", str(out / "mnist_input.mlirbc")],
        [str(args.iree_compile.resolve()), str(out / "mnist_input.mlirbc"),
         "--compile-from=input", "--iree-hal-target-backends=tenstorrent",
         "--compile-to=global-optimization", "-o", str(out / "mnist_global_opt.mlir")],
    ]
    for index, command in enumerate(commands):
        with (out / f"lowering-{index}.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    manifest.update(export_status="completed", commands=commands,
                    artifacts={name: file_record(out / name) for name in (
                        "mnist_torch.mlirbc", "mnist_input.mlirbc", "mnist_global_opt.mlir")})
    write_json(out / "manifest.json", manifest)
    print(full_accuracy, flush=True)


if __name__ == "__main__":
    main()
