#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Lightweight contract tests; numeric/model tests require optional dependencies."""

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_resnet50_precision_golden as tool
import run_ttnn_resnet50_golden as ttnn_tool


class GoldenContractTest(unittest.TestCase):
    def test_output_cannot_replace_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                tool.require_new_directory(pathlib.Path(directory))

    def test_file_record_hash_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "input"
            target.write_bytes(b"abc")
            result = tool.file_record(target)
            self.assertEqual(result["bytes"], 3)
            self.assertEqual(result["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_unknown_synset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "recognized filename synset"):
            tool.image_label("ILSVRC2012_val_00040000_n01797886.JPEG", {})

    def test_labels_use_map_order(self):
        labels = {"n00000001": "one", "n01797886": "two"}
        self.assertEqual(tool.image_label("ILSVRC2012_val_00040000_n01797886.JPEG", labels), 1)

    def test_class_map_rejects_missing_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "labels.py"
            target.write_text("IMAGENET2012_CLASSES = OrderedDict({'n00000001': 'one'})")
            with self.assertRaisesRegex(ValueError, "1000 ordered"):
                tool.read_class_map(target)

    def test_class_map_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "labels.py"
            target.write_text("raise RuntimeError('must not execute')\nIMAGENET2012_CLASSES = dangerous()")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                tool.read_class_map(target)

    def test_three_precision_policies_are_distinct(self):
        self.assertEqual(set(tool.POLICIES), {"fp32", "bf16_module", "bf16_operands_fp32_ops"})
        self.assertIn("not bit-exact Tensix", tool.POLICIES["bf16_operands_fp32_ops"])
        self.assertEqual(len(tool.CHECKPOINT_NAMES), 19)

    def test_stem_override_is_opt_in(self):
        model = types.SimpleNamespace(conv1_config=types.SimpleNamespace(act_block_h_override=0))
        result = ttnn_tool.configure_stem_block(model, None)
        self.assertEqual(result["effective_block_h"], 0)
        self.assertIsNone(result["requested_block_h"])

    def test_stem_override_is_recorded_and_local(self):
        model = types.SimpleNamespace(conv1_config=types.SimpleNamespace(act_block_h_override=0),
                                      other_conv=types.SimpleNamespace(act_block_h_override=0))
        result = ttnn_tool.configure_stem_block(model, 512)
        self.assertEqual(result, {"requested_block_h": 512, "upstream_block_h": 0,
                                  "effective_block_h": 512})
        self.assertEqual(model.other_conv.act_block_h_override, 0)

    def test_invalid_stem_override_is_rejected_before_mutation(self):
        import argparse
        model = types.SimpleNamespace(conv1_config=types.SimpleNamespace(act_block_h_override=0))
        for value in (0, -32, 33):
            with self.assertRaises(argparse.ArgumentTypeError):
                ttnn_tool.configure_stem_block(model, value)
            self.assertEqual(model.conv1_config.act_block_h_override, 0)

    def test_avgpool_override_is_explicit_and_does_not_change_fidelity(self):
        calls = []
        def original(**kwargs):
            calls.append(kwargs.copy())
            return types.SimpleNamespace(dtype=kwargs["dtype"])
        module = types.SimpleNamespace(avg_pool2d=original, bfloat16="BF16")
        config = types.SimpleNamespace(math_fidelity="LoFi", fp32_dest_acc_en=False)
        wrapper, audit = ttnn_tool.make_avgpool_precision_wrapper(module)
        self.assertEqual(wrapper(dtype="BF8", compute_kernel_config=config).dtype, "BF8")
        self.assertEqual(audit["calls"][0]["output_dtype"], "BF8")
        self.assertIs(module.avg_pool2d, original)
        wrapper, audit = ttnn_tool.make_avgpool_precision_wrapper(module, True)
        self.assertEqual(wrapper(dtype="BF8", compute_kernel_config=config).dtype, "BF16")
        self.assertIs(calls[-1]["compute_kernel_config"], config)
        self.assertEqual(audit["calls"][0]["math_fidelity"], "LoFi")
        self.assertEqual(audit["calls"][0]["upstream_dtype"], "BF8")

    def test_avgpool_override_does_not_mislabel_actual_output(self):
        module = types.SimpleNamespace(avg_pool2d=lambda **kwargs: types.SimpleNamespace(dtype="BF8"),
                                       bfloat16="BF16")
        wrapper, audit = ttnn_tool.make_avgpool_precision_wrapper(module, True)
        with self.assertRaisesRegex(ValueError, "did not produce"):
            wrapper(dtype="BF8")
        self.assertEqual(audit["calls"][0]["output_dtype"], "BF8")

    def test_avgpool_request_and_actual_dtype_can_differ(self):
        module = types.SimpleNamespace(avg_pool2d=lambda **kwargs: types.SimpleNamespace(dtype="BF16"),
                                       bfloat16="BF16")
        wrapper, audit = ttnn_tool.make_avgpool_precision_wrapper(module)
        wrapper(dtype="BF8")
        self.assertEqual(audit["calls"][0]["effective_dtype"], "BF8")
        self.assertEqual(audit["calls"][0]["output_dtype"], "BF16")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numeric tests require numpy")
class NumericGoldenTest(unittest.TestCase):
    def test_scale_error_is_not_hidden_by_correlation(self):
        metrics = tool.error_metrics([1.0, 2.0, 3.0], [1.5, 3.0, 4.5])
        self.assertAlmostEqual(metrics["pcc"], 1.0)
        self.assertAlmostEqual(metrics["scale"], 1.5)
        self.assertAlmostEqual(metrics["relative_l2"], 0.5)

    def test_shape_and_nonfinite_are_rejected(self):
        for expected, actual in (([1], [[1]]), ([1], [float("nan")]), ([], [])):
            with self.assertRaises(ValueError):
                tool.error_metrics(expected, actual)

    def test_zero_reference_has_no_invented_relative_metrics(self):
        metrics = tool.error_metrics([0.0], [1.0])
        self.assertIsNone(metrics["relative_l2"])
        self.assertIsNone(metrics["scale"])
        self.assertEqual(metrics["max_absolute_error"], 1.0)

    def test_classification_is_per_image_not_flattened(self):
        import numpy as np
        values = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=np.float32)
        result = tool.classifier_comparison(values, values, list("abcdef"), [5, 0])
        self.assertEqual(result["aggregate"]["actual_top1_correct"], 2)
        self.assertEqual(result["aggregate"]["top1_agreement"], 2)
        self.assertEqual([r["actual_top1"] for r in result["per_image"]], [5, 0])

    def test_invalid_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            tool.predictions([[1, 2, 3, 4, 5]], list("abcde"), [5])


@unittest.skipUnless(importlib.util.find_spec("torch"), "operand test requires torch")
class OperandGoldenTest(unittest.TestCase):
    def test_operand_rounding_preserves_fp32_output_and_bias(self):
        import torch
        base = torch.nn.Linear(2, 1).eval()
        with torch.no_grad():
            base.weight.copy_(torch.tensor([[1.003, -0.501]]))
            base.bias.copy_(torch.tensor([0.1234567]))
        original = base.weight.detach().clone()
        value = torch.tensor([[0.731, 0.137]])
        prepared = tool.prepare_model(base, "bf16_operands_fp32_ops")
        expected = torch.nn.functional.linear(tool.quantize_operand_fp32(value),
                                               tool.quantize_operand_fp32(base.weight), base.bias)
        with torch.no_grad():
            self.assertTrue(torch.equal(prepared(value), expected))
        self.assertEqual(prepared(value).dtype, torch.float32)
        self.assertTrue(torch.equal(base.weight, original))


if __name__ == "__main__":
    unittest.main()
