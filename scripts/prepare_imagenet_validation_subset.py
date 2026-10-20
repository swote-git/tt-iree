#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Select new original-resolution validation images from a verified public shard.

Inputs are an explicitly downloaded Parquet and pinned Hugging Face API records.
This is a shard-local sample, not a random sample of the entire validation set.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import random

from build_resnet50_precision_golden import (
    file_record, image_label, read_class_map, require_new_directory, write_json,
)


def select_rows(row_count, count, seed):
    if count < 1 or count > row_count:
        raise ValueError("sample count must be within the source shard")
    return random.Random(seed).sample(range(row_count), row_count)


def rgb_digest(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode("ascii") + image.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parquet", "dataset-api", "tree-api", "class-map", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=pathlib.Path)
    parser.add_argument("--exclude-images", type=pathlib.Path)
    parser.add_argument("--count", type=int, default=112)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    from PIL import Image
    import pyarrow.parquet as pq

    api = json.loads(args.dataset_api.read_text())
    tree = json.loads(args.tree_api.read_text())
    if api["id"] != "benjamin-paine/imagenet-1k" or api.get("gated"):
        raise ValueError("requires the explicitly approved public ImageNet repack")
    records = [r for r in tree if r["path"] == f"data/{args.parquet.name}"]
    if len(records) != 1 or "validation-" not in args.parquet.name:
        raise ValueError("expected a validation Parquet listed in the pinned tree")
    source = file_record(args.parquet)
    if source["sha256"] != records[0]["lfs"]["oid"] or source["bytes"] != records[0]["size"]:
        raise ValueError("Parquet SHA256/size differs from the source repository")
    table = pq.read_table(args.parquet)
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    names = metadata["info"]["features"]["label"]["names"]
    classes = read_class_map(args.class_map)
    # The repack appends an unused sentinel for the unlabeled test split.
    if (names[:1000] != list(classes.values()) or len(names) not in (1000, 1001)
            or (len(names) == 1001 and names[-1] != "none")):
        raise ValueError("source labels do not match the 1000-class synset ordering")
    excluded = set()
    if args.exclude_images:
        for filename in args.exclude_images.glob("*.JPEG"):
            with Image.open(filename) as image:
                excluded.add(rgb_digest(image))
    chosen, seen = [], set()
    for row in select_rows(len(table), args.count, args.seed):
        record = table.slice(row, 1).to_pylist()[0]
        label = record["label"]
        if type(label) is not int or not 0 <= label < 1000:
            raise ValueError("unlabeled or out-of-range source image")
        if image_label(record["image"]["path"], classes) != label:
            raise ValueError("source filename synset disagrees with Parquet label")
        payload = record["image"]["bytes"]
        with Image.open(io.BytesIO(payload)) as image:
            digest = rgb_digest(image)
            if digest in seen or digest in excluded:
                continue
            chosen.append((row, label, record["image"].get("path"), payload,
                           list(image.size), image.mode, digest))
            seen.add(digest)
        if len(chosen) == args.count:
            break
    if len(chosen) != args.count:
        raise ValueError("not enough distinct, non-demo source images")
    require_new_directory(args.output_dir)
    out = args.output_dir.resolve()
    (out / "images").mkdir()
    images = []
    for i, (row, label, original, payload, size, mode, digest) in enumerate(chosen):
        filename = out / "images" / f"sample-{i:05d}.JPEG"
        filename.write_bytes(payload)
        images.append({"filename": str(filename.relative_to(out)),
                       "label": label, "synset": list(classes)[label],
                       "source_shard_row": row, "original_image_path": original,
                       "original_size": size, "original_mode": mode,
                       "rgb_sha256": digest, **file_record(filename)})
    write_json(out / "manifest.json", {
        "dataset": api["id"], "revision": api["sha"], "split": "validation",
        "source_url": f"https://huggingface.co/datasets/{api['id']}",
        "license": api["cardData"].get("license"),
        "license_details": api["cardData"].get("license_details"),
        "scope": "seeded sample within validation shard 0; not full ImageNet accuracy",
        "sampling": {"seed": args.seed, "algorithm": "Python random.Random.sample permutation; first distinct non-demo images",
                     "population_rows": len(table), "count": len(images),
                     "distinct_classes": len({r['label'] for r in images}),
                     "excluded_demo_rgb_hashes": len(excluded)},
        "images_not_resized_or_reencoded": True, "class_map": file_record(args.class_map),
        "parquet": source, "dataset_api": file_record(args.dataset_api),
        "tree_api": file_record(args.tree_api), "images": images,
        "script": file_record(__file__),
    })
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
