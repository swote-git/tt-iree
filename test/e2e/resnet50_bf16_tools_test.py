#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from export_resnet50_bf16 import bf16_resource_bridge, verify_record
from build_resnet50_precision_golden import file_record
from run_ttiree_resnet50_bf16 import decode_bf16, require_native_profile
from device_native_bf16_test import bf16_bits, bf16

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class Bf16ToolsTest(unittest.TestCase):
    def test_round_nearest_even_signed_ties(self):
        self.assertEqual(bf16_bits(1 + 1 / 256), 0x3f80)
        self.assertEqual(bf16_bits(1 + 3 / 256), 0x3f82)
        self.assertEqual(bf16_bits(-1 - 1 / 256), 0xbf80)
        self.assertEqual(bf16_bits(-1 - 3 / 256), 0xbf82)

    def test_special_values(self):
        self.assertEqual(bf16_bits(float("inf")), 0x7f80)
        self.assertEqual(bf16_bits(float("-inf")), 0xff80)
        self.assertEqual(bf16_bits(-0.0), 0x8000)
        self.assertEqual(bf16_bits(float("nan")) & 0x7fc0, 0x7fc0)

    def test_hash_rejects_mutated_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = pathlib.Path(directory) / "weights.pt"
            filename.write_bytes(b"original")
            record = file_record(filename)
            verify_record(filename, record)
            filename.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_record(filename, record)

    def test_compute_profile_cannot_silently_fall_back(self):
        import csv
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "run_results.json").write_text(json.dumps({
                "all_dispatches_device_native": True, "dispatch_count": 58}))
            rows = []
            for kind in (["MATMUL"] * 36 + ["CONV2D"] * 17 + ["LINEAR"]
                         + ["PADDED_COPY", "MAX_POOL_2D", "GLOBAL_SUM", "SCALAR_DIV"]):
                record = {"OP TYPE": kind, "INPUT_0_DATATYPE": "BF16", "IREE BUILTIN PROGRAM": "11",
                          "OUTPUT_0_DATATYPE": "BF16", "IREE EXECUTION DOMAIN": "device_native",
                          "IREE COMPUTE DATATYPE": "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING"}
                for stage in ("INPUT STAGING", "PACK", "UPLOAD", "DOWNLOAD", "UNPACK", "OUTPUT STAGING"):
                    record[f"IREE {stage} DURATION [ns]"] = "0"
                rows.append(record)
            def save():
                with (root / "ops_perf_results.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            save()
            require_native_profile(root)
            with self.assertRaisesRegex(ValueError, "routing"):
                require_native_profile(root, True)
            for row in rows[:54]:
                row.update({"IREE BUILTIN PROGRAM": "12",
                            "IREE COMPUTE DATATYPE": "BF16_HIFI4_FP32_DST_F32_EPILOGUE"})
            save()
            require_native_profile(root, True)
            with self.assertRaisesRegex(ValueError, "routing"):
                require_native_profile(root)
            rows[-1]["IREE BUILTIN PROGRAM"] = "12"
            save()
            with self.assertRaisesRegex(ValueError, "routing"):
                require_native_profile(root, True)
            rows[-1]["IREE BUILTIN PROGRAM"] = "11"
            rows[-1]["IREE EXECUTION DOMAIN"] = "host"
            save()
            with self.assertRaisesRegex(ValueError, "non-native"):
                require_native_profile(root, True)
            rows[-1]["IREE EXECUTION DOMAIN"] = "device_native"
            rows[-1]["OUTPUT_0_DATATYPE"] = "F32"
            save()
            with self.assertRaisesRegex(ValueError, "non-BF16"):
                require_native_profile(root, True)
            rows[-1]["OUTPUT_0_DATATYPE"] = "BF16"
            rows.extend(dict(row) for row in rows[:58])
            (root / "run_results.json").write_text(json.dumps({
                "all_dispatches_device_native": True, "dispatch_count": 116}))
            save()
            require_native_profile(root, True, sample_count=2)
            rows[-1]["OP TYPE"] = "MATMUL"
            save()
            with self.assertRaisesRegex(ValueError, "coverage"):
                require_native_profile(root, True, sample_count=2)

    @unittest.skipUnless(HAS_NUMPY and HAS_TORCH, "requires numpy/torch")
    def test_resource_bridge_keeps_bf16_bits_and_restores_on_error(self):
        import numpy as np
        import torch
        original = types.SimpleNamespace(array=np.array)
        ir_utils = types.SimpleNamespace(np=original)
        bits = torch.tensor([0x3f81, 0x8000, 0x7f80, 0xff80], dtype=torch.uint16)
        tensor = bits.view(torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "test error"):
            with bf16_resource_bridge(ir_utils, torch):
                result = ir_utils.np.array(tensor)
                np.testing.assert_array_equal(result, bits.numpy())
                self.assertEqual(result.dtype, np.dtype("uint16"))
                raise RuntimeError("test error")
        self.assertIs(ir_utils.np, original)

    @unittest.skipUnless(HAS_NUMPY, "requires numpy")
    def test_decode_uses_bits_not_numeric_uint16_cast(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            filename = pathlib.Path(directory) / "output.bin"
            np.array([0x3f80, 0xc000, 0x8000], dtype="<u2").tofile(filename)
            result = decode_bf16(filename, (1, 3))
            np.testing.assert_array_equal(result, [[1.0, -2.0, -0.0]])
            self.assertTrue(np.signbit(result[0, 2]))
            with self.assertRaisesRegex(ValueError, "output length"):
                decode_bf16(filename, (1, 2))

    @unittest.skipUnless(HAS_NUMPY, "requires numpy")
    def test_nonfinite_output_rejected(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            filename = pathlib.Path(directory) / "output.bin"
            np.array([0x7f80], dtype="<u2").tofile(filename)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                decode_bf16(filename, (1, 1))


if __name__ == "__main__":
    unittest.main()
