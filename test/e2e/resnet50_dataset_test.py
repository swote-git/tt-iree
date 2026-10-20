#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import importlib.util
import json
import pathlib
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from build_resnet50_precision_golden import manifest_images, file_record
from prepare_imagenet_validation_subset import select_rows
from compare_resnet50_dataset import REPORT_TEMPLATE, classifier_accuracy, comparison, label_boundary


class DatasetContractTest(unittest.TestCase):
    def test_report_template_is_a_tool_resource(self):
        scripts = pathlib.Path(__file__).resolve().parents[2] / "scripts"
        self.assertEqual(REPORT_TEMPLATE.parent, scripts / "templates")
        self.assertTrue(REPORT_TEMPLATE.is_file())
        self.assertIn("{accuracy_table}", REPORT_TEMPLATE.read_text())

    def test_seeded_selection_is_unique_and_independent_of_global_rng(self):
        import random
        first = select_rows(3572, 112, 20261015)
        random.seed(1)
        self.assertEqual(first, select_rows(3572, 112, 20261015))
        self.assertEqual(len(set(first[:112])), 112)
        self.assertNotEqual(first, select_rows(3572, 112, 2))

    def test_invalid_sample_counts(self):
        for count in (0, -1, 11):
            with self.assertRaises(ValueError):
                select_rows(10, count, 1)

    def test_manifest_labels_hashes_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            classes = {f"n{i:08d}": str(i) for i in range(1000)}
            rows = []
            for i in (37, 4):
                filename = root / f"image-{i}.JPEG"
                filename.write_bytes(str(i).encode())
                rows.append({"filename": filename.name, "label": i, "synset": list(classes)[i],
                             **file_record(filename)})
            manifest = root / "manifest.json"
            def save(images=rows, split="validation"):
                manifest.write_text(json.dumps({"split": split, "images": images}))
            save()
            paths, labels, _ = manifest_images(manifest, classes, 2)
            self.assertEqual(labels, [37, 4])
            self.assertEqual(paths[0].name, "image-37.JPEG")
            for key, value in (("label", 5), ("label", True), ("sha256", "bad"),
                               ("synset", "n00000001"), ("filename", "../outside.JPEG")):
                broken = [dict(rows[0]), rows[1]]
                broken[0][key] = value
                save(broken)
                with self.assertRaises(ValueError):
                    manifest_images(manifest, classes, 2)
            save([rows[0], rows[0]])
            with self.assertRaises(ValueError):
                manifest_images(manifest, classes, 2)
            save(split="train")
            with self.assertRaisesRegex(ValueError, "validation"):
                manifest_images(manifest, classes, 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_pcc_is_not_classifier_accuracy(self):
        import numpy as np
        logits = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.float32)
        categories, labels = list("abcde"), [0, 0]
        result = comparison(logits, logits * 2, categories, labels)
        self.assertAlmostEqual(result["aggregate"]["pcc"], 1)
        self.assertGreater(result["aggregate"]["rmse"], 0)
        self.assertEqual(classifier_accuracy(logits, categories, labels)["top1_accuracy"], .5)
        self.assertEqual(result["aggregate"]["top1_agreement"], 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_top5_cutoff_tie_is_reported_not_relaxed(self):
        import numpy as np
        logits = np.array([9, 8.75, 8.625, 8.5625, 8.375, 8.375], dtype=np.float32)
        result = label_boundary(logits, 5)
        self.assertEqual(result["stable_label_rank"], 6)
        self.assertTrue(result["label_ties_fifth"])
        self.assertEqual(result["fifth_sixth_margin"], 0)
        self.assertEqual(result["count_strictly_above_label"], 4)
        self.assertEqual(classifier_accuracy(logits[None], list("abcdef"), [5])["top5_correct"], 0)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_logits_report_with_synthetic_contract_fixture(self):
        import csv
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            golden, native, nn = (root / name for name in ("golden", "native", "ttnn"))
            for folder in (golden, native, nn):
                folder.mkdir()
            def save(filename, value):
                filename.write_text(json.dumps(value))
            weights = golden / "weights.pt"
            weights.write_bytes(b"synthetic fixture, not a trained model")
            np.save(golden / "input.npy", np.zeros((16, 3, 224, 224), dtype=np.float32))
            values = np.tile(np.arange(1000, dtype=np.float32) % 8, (16, 1))
            refs = {}
            for policy in ("fp32", "bf16_module"):
                (golden / policy).mkdir()
                np.save(golden / policy / "logits.npy", values)
                refs[policy] = file_record(golden / policy / "logits.npy")
            manifest = {"labels": [0] * 16, "categories": [str(i) for i in range(1000)],
                        "state_dict": file_record(weights), "input": file_record(golden / "input.npy"),
                        "logits": refs, "torch_version": "fixture", "torchvision_version": "fixture",
                        "source": {"weights": "synthetic", "weight_checkpoint": file_record(weights),
                                   "dataset": {"dataset": "synthetic", "revision": "fixture", "sampling": {}}}}
            save(golden / "manifest.json", manifest)
            export = root / "export.json"
            save(export, {"state_dict": manifest["state_dict"], "input": manifest["input"]})
            np.save(native / "logits.npy", values)
            np.save(nn / "logits.npy", values)
            np.save(nn / "pytorch_bf16_batch_reference.npy", values)
            save(native / "results.json", {
                "status": "completed", "batch": 1, "precision_policy": "bf16_module",
                "golden_dir": str(golden), "matrix_engine": "tensix_hifi4", "indices": list(range(16)),
                "samples": [{"index": i, "device_profile": {"total_dispatch_duration_ns": 58000}} for i in range(16)],
                "export": file_record(export), "session_runner": "synthetic", "session_process_wall_seconds": 16.,
            })
            save(nn / "results.json", {
                "status": "completed", "activation_dtype": "bf16", "weight_dtype": "bf16", "input_dtype": "BF16",
                "input_sha256": manifest["input"]["sha256"], "golden_manifest": file_record(golden / "manifest.json"),
                "samples": 16, "batch": 16, "batch_host_wall_seconds": [.008], "pipeline_compile_wall_seconds": .1,
                "output": file_record(nn / "logits.npy"),
            })
            (native / "profile").mkdir()
            save(native / "profile/run_results.json", {"all_dispatches_device_native": True, "dispatch_count": 16 * 58})
            ops = ["MATMUL"] * 36 + ["CONV2D"] * 17 + ["LINEAR", "PADDED_COPY", "MAX_POOL_2D", "GLOBAL_SUM", "SCALAR_DIV"]
            rows = []
            for op in ops * 16:
                matrix = op in ("MATMUL", "CONV2D", "LINEAR")
                row = {"OP TYPE": op, "INPUT_0_DATATYPE": "BF16", "OUTPUT_0_DATATYPE": "BF16",
                       "IREE EXECUTION DOMAIN": "device_native", "IREE BUILTIN PROGRAM": "12" if matrix else "11",
                       "IREE COMPUTE DATATYPE": "BF16_HIFI4_FP32_DST_F32_EPILOGUE" if matrix else "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING"}
                for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
                    row[f"IREE {stage} DURATION [ns]"] = "0"
                rows.append(row)
            with (native / "profile/ops_perf_results.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (native / "call_timings.csv").write_text("sample,invoke_seconds\n" + "".join(f"{i},1\n" for i in range(16)))
            analysis = root / "analysis"
            script = pathlib.Path(__file__).resolve().parents[2] / "scripts/compare_resnet50_dataset.py"
            subprocess.run([sys.executable, str(script), "--golden-dir", str(golden),
                            "--ttiree-run", str(native), "--ttnn-run", str(nn),
                            "--output-dir", str(analysis)], check=True, capture_output=True)
            summary = json.loads((analysis / "summary.json").read_text())
            self.assertEqual(summary["sample_count"], 16)
            self.assertEqual(summary["classification"]["ttiree_bf16"]["top1_correct"], 0)
            self.assertAlmostEqual(summary["comparisons"]["ttiree_bf16_vs_pytorch_bf16"]["aggregate"]["pcc"], 1)
            self.assertTrue((analysis / "report-ko.md").exists())
            with (analysis / "per_image_metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 16 * 8)


if __name__ == "__main__":
    unittest.main()
