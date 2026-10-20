#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""BF16 comparison contracts; optional numeric tests never open a device."""
import dataclasses
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_resnet50_bf16 as tool
from build_resnet50_precision_golden import error_metrics, file_record

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None


class BoundaryTest(unittest.TestCase):
    def test_all_calls_and_operations_are_covered(self):
        from collections import Counter
        boundaries = tool.dispatch_boundaries()
        self.assertEqual([b.dispatch for b in boundaries], list(range(1, 59)))
        self.assertEqual(Counter(b.op_type for b in boundaries), {
            "MATMUL": 36, "CONV2D": 17, "PADDED_COPY": 1, "MAX_POOL_2D": 1,
            "GLOBAL_SUM": 1, "SCALAR_DIV": 1, "LINEAR": 1,
        })
        self.assertEqual(len({b.op_code for b in boundaries}), 41)
        self.assertEqual(len({b.name for b in boundaries}), 58)

    def test_only_channel_major_flattening_is_allowed(self):
        for boundary in tool.dispatch_boundaries():
            from math import prod
            self.assertEqual(prod(boundary.shape), prod(boundary.checkpoint_shape))
            if len(boundary.checkpoint_shape) == 2 and len(boundary.shape) == 4:
                self.assertEqual(boundary.checkpoint_shape, (boundary.shape[1],
                                 boundary.shape[2] * boundary.shape[3]))

    def test_raw_skip_branch_and_repeated_exports(self):
        boundaries = tool.dispatch_boundaries()
        self.assertEqual(boundaries[5].name, "layer1.0.downsample_conv_before_bn")
        self.assertEqual(boundaries[6].sources, (5, 6))
        self.assertEqual(boundaries[9].sources, (9, 7))
        self.assertEqual(boundaries[12].export_index, 9)
        self.assertEqual(boundaries[54].checkpoint_shape, (2048, 7, 7))
        for boundary in boundaries:
            self.assertTrue(all(source < boundary.dispatch for source in boundary.sources))


@unittest.skipUnless(HAS_NUMPY, "numeric tests require numpy")
class NumericTest(unittest.TestCase):
    def test_pooled_metrics_are_element_weighted(self):
        import numpy as np
        e1, a1 = np.array([1., 2.]), np.array([3., 4.])
        e2, a2 = np.arange(3., 100.), np.arange(3., 100.) / 2
        accumulator = tool.MetricAccumulator()
        accumulator.add(e1, a1)
        accumulator.add(e2, a2)
        metrics = accumulator.result()
        expected = error_metrics(np.concatenate((e1, e2)), np.concatenate((a1, a2)))
        for name, value in expected.items():
            self.assertAlmostEqual(metrics[name], value, places=12, msg=name)
        self.assertNotAlmostEqual(metrics["pcc"], (error_metrics(e1, a1)["pcc"]
                                                  + error_metrics(e2, a2)["pcc"]) / 2)

    def test_pcc_does_not_hide_scale_or_offset(self):
        import numpy as np
        accumulator = tool.MetricAccumulator()
        e = np.array([-2., -1., 0., 1., 2.])
        accumulator.add(e, e * 0.9 + 3)
        metrics = accumulator.result()
        self.assertAlmostEqual(metrics["pcc"], 1)
        self.assertAlmostEqual(metrics["scale"], 0.9)
        self.assertGreater(metrics["relative_l2"], 1)

    def test_constant_pcc_and_empty_comparisons(self):
        import numpy as np
        accumulator = tool.MetricAccumulator()
        with self.assertRaises(ValueError):
            accumulator.result()
        accumulator.add(np.ones(8), np.ones(8) * 2)
        self.assertIsNone(accumulator.result()["pcc"])
        for e, a in (([1], [[1]]), ([float("nan")], [0]), ([], [])):
            with self.assertRaises(ValueError):
                tool.MetricAccumulator().add(np.array(e), np.array(a))

    def test_bf16_storage_is_not_a_label_only(self):
        import numpy as np
        tool.require_bf16_values(np.array([1., -2., -0.], dtype=np.float32), "valid")
        for values in (np.array([1.001], dtype=np.float32),
                       np.array([float("inf")], dtype=np.float32), np.array([1.], dtype=np.float64)):
            with self.assertRaisesRegex(ValueError, "widened exactly"):
                tool.require_bf16_values(values, "invalid")

    def test_checkpoint_profile_contract_rejects_boundary_changes(self):
        import csv
        import numpy as np
        boundary = dataclasses.replace(tool.dispatch_boundaries()[0], shape=(1, 1),
                                       checkpoint_shape=(1,))
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "checkpoints").mkdir()
            (root / "profile").mkdir()
            np.save(root / "checkpoints/output.npy", np.array([1.], dtype=np.float32))
            record = {"dispatch_index": 1, "op_code": boundary.op_code, "builtin_program": 11,
                      "dispatch_kind": 8, "shape": [1], "file": "output.npy"}
            manifest = {"schema_version": 1, "dtype": "float32", "checkpoint_count": 1,
                        "records": [record]}
            operation = {"GLOBAL CALL COUNT": "1", "OP CODE": boundary.op_code,
                         "OP TYPE": "PADDED_COPY", "IREE EXECUTION DOMAIN": "device_native",
                         "INPUT_0_DATATYPE": "BF16", "OUTPUT_0_DATATYPE": "BF16"}
            def save():
                (root / "checkpoints/manifest.json").write_text(json.dumps(manifest))
                with (root / "profile/ops_perf_results.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(operation))
                    writer.writeheader()
                    writer.writerow(operation)
            save()
            actual, _ = tool.load_dispatches(root, [boundary])
            self.assertEqual(actual[1].shape, (1, 1))
            for key, value in (("OP TYPE", "CONV2D"), ("GLOBAL CALL COUNT", "2"),
                               ("IREE EXECUTION DOMAIN", "host"), ("OUTPUT_0_DATATYPE", "F32")):
                previous = operation[key]
                operation[key] = value
                save()
                with self.assertRaisesRegex(ValueError, "boundary contract changed"):
                    tool.load_dispatches(root, [boundary])
                operation[key] = previous
            manifest["checkpoint_count"] = 2
            save()
            with self.assertRaisesRegex(ValueError, "coverage"):
                tool.load_dispatches(root, [boundary])
            manifest["checkpoint_count"] = 1
            record["builtin_program"] = 12
            operation["OP TYPE"] = "MATMUL"
            matrix_boundary = dataclasses.replace(boundary, op_type="MATMUL")
            save()
            tool.load_dispatches(root, [matrix_boundary], use_tensix=True)
            with self.assertRaisesRegex(ValueError, "boundary contract changed"):
                tool.load_dispatches(root, [matrix_boundary])

    def test_matching_dataset_and_row_order_required(self):
        import copy
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            golden = pathlib.Path(directory)
            manifest = {"labels": [0, 1, 2, 3], "input": {"sha256": "inputhash"}}
            (golden / "manifest.json").write_text(json.dumps(manifest))
            iree = {"status": "completed", "batch": 1, "precision_policy": "bf16_module",
                    "golden_dir": directory, "indices": [1, 3], "samples": [{"index": 1}, {"index": 3}]}
            nn = {"status": "completed", "activation_dtype": "bf16", "weight_dtype": "bf16",
                  "input_dtype": "BF16", "input_sha256": "inputhash", "samples": 4,
                  "golden_manifest": file_record(golden / "manifest.json")}
            native, full = np.zeros((2, 1000), dtype=np.float32), np.zeros((4, 1000), dtype=np.float32)
            self.assertEqual(tool.validate_run_metadata(golden, manifest, iree, nn, native, full), [1, 3])
            for key, value in (("weight_dtype", "bf8b"), ("input_sha256", "other"), ("samples", 2)):
                changed = copy.deepcopy(nn)
                changed[key] = value
                with self.assertRaises(ValueError):
                    tool.validate_run_metadata(golden, manifest, iree, changed, native, full)
            for indices in ([1, 1], [1, 4], [3, 1], [True, 3]):
                changed = copy.deepcopy(iree)
                changed["indices"] = indices
                with self.assertRaises(ValueError):
                    tool.validate_run_metadata(golden, manifest, changed, nn, native, full)


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, "replay tests require numpy/torch")
class ReplayTest(unittest.TestCase):
    def test_non_policy_dispatch_has_no_implicit_replay(self):
        self.assertEqual(tool.replay_ir_policy(None, tool.dispatch_boundaries()[0], {}), (None, ""))

    def test_replay_uses_measured_input_not_original_input(self):
        import numpy as np
        import torch
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
                self.conv1.weight.data.fill_(3)
                self.bn1, self.relu = torch.nn.Identity(), torch.nn.ReLU()
        model = torch.nn.Module()
        model.layer1 = torch.nn.Sequential(Block())
        boundary = tool.dispatch_boundaries()[3]
        output = tool.replay_dispatch(model, boundary, {3: np.array([[2.]], dtype=np.float32)},
                                      torch.tensor([[99.]], dtype=torch.bfloat16))
        self.assertEqual(output.item(), 6)

    def test_bf16_sum_policy_is_different_from_fp32_sum(self):
        import numpy as np
        import torch
        measured = {55: np.array([1.] + [1 / 256] * 48, dtype=np.float32).reshape(1, 1, 7, 7)}
        boundary = tool.dispatch_boundaries()[55]
        reference, _ = tool.replay_ir_policy(None, boundary, measured)
        pytorch = tool.replay_dispatch(None, boundary, measured, None)
        self.assertEqual(reference.item(), 1)
        self.assertEqual(pytorch.item(), 1.1875)

    def test_linear_separate_rounding_policy(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        model = SimpleNamespace(fc=torch.nn.Linear(1, 1).to(torch.bfloat16))
        model.fc.weight.data.fill_(1)
        model.fc.bias.data.fill_(1 / 256)
        output, _ = tool.replay_ir_policy(model, tool.dispatch_boundaries()[57],
                                         {57: np.array([[[[1.]]]], dtype=np.float32)})
        self.assertEqual(output.item(), 1)

    def test_ordered_linear_does_not_reassociate_the_dot(self):
        import torch
        value = torch.tensor([[2 ** 24, 1, -(2 ** 24)]], dtype=torch.bfloat16)
        weight = torch.ones((1, 3), dtype=torch.bfloat16)
        bias = torch.zeros(1, dtype=torch.bfloat16)
        self.assertEqual(tool.ordered_linear_bf16(value, weight, bias).item(), 0)
        self.assertEqual((value.double() @ weight.double().T).item(), 1)
        with self.assertRaisesRegex(ValueError, "BF16 tensors"):
            tool.ordered_linear_bf16(value.float(), weight, bias)

    @unittest.skipUnless(HAS_TORCHVISION, "model hook test requires torchvision")
    def test_all_pytorch_hooks_cover_the_same_model_and_are_removed(self):
        import torch
        import torchvision
        torch.set_num_threads(1)
        model = torchvision.models.resnet50(weights=None).eval().to(torch.bfloat16)
        value = torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16)
        reference, output = tool.capture_pytorch(model, value, tool.dispatch_boundaries())
        self.assertEqual(len(reference), 58)
        for boundary in tool.dispatch_boundaries():
            self.assertEqual(reference[boundary.dispatch].shape, boundary.shape)
        self.assertEqual(tuple(output.shape), (1, 1000))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
