#!/usr/bin/env python3
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Standalone CPU-only tooling tests; no compiler build or hardware execution."""
import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import export_resnet50_bf16 as exporter
import run_ttiree_resnet50_bf16_batch as batch
from build_resnet50_precision_golden import file_record, write_json

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class ExportBatchTest(unittest.TestCase):
    def test_bounded_static_sizes(self):
        for value in (1, 2, 16):
            self.assertEqual(exporter.batch_size(str(value)), value)
        for value in ("0", "-1", "17", "1.5", "dynamic", "999999999999999999999"):
            with self.assertRaises(argparse.ArgumentTypeError):
                exporter.batch_size(value)

    def test_export_shape_and_immutable_sources_with_mock_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            golden = root / "golden"
            (golden / "bf16_module").mkdir(parents=True)
            for name in ("state_dict.pt", "input.npy", "bf16_module/logits.npy"):
                (golden / name).write_bytes(b"immutable fixture: " + name.encode())
            manifest = {"source": {"weights": "immutable fixture"},
                        "state_dict": file_record(golden / "state_dict.pt"),
                        "input": file_record(golden / "input.npy"),
                        "logits": {"bf16_module": file_record(golden / "bf16_module/logits.npy")}}
            write_json(golden / "manifest.json", manifest)
            before = {path: file_record(path) for path in golden.rglob("*") if path.is_file()}
            model = mock.Mock()
            model.eval.return_value = model
            torch = types.ModuleType("torch")
            torch.__version__, torch.bfloat16 = "fixture", "bf16"
            torch.set_num_threads, torch.load, torch.zeros = mock.Mock(), mock.Mock(), mock.Mock()
            torchvision = types.ModuleType("torchvision")
            torchvision.__version__ = "fixture"
            torchvision.models = types.SimpleNamespace(resnet50=mock.Mock(return_value=model))
            aot = types.ModuleType("iree.turbine.aot")
            exported = mock.Mock()
            exported.save_mlir.side_effect = lambda path: path.write_bytes(b"fixture torch IR")
            aot.export = mock.Mock(return_value=exported)
            support = types.ModuleType("iree.turbine.aot.support")
            support.ir_utils = types.SimpleNamespace(np=object())
            binaries = types.ModuleType("iree.compiler.tools.binaries")
            binaries.find_tool = lambda name: "fixture-frontend-compiler"
            modules = {"torch": torch, "torchvision": torchvision,
                       "iree": types.ModuleType("iree"), "iree.turbine": types.ModuleType("iree.turbine"),
                       "iree.turbine.aot": aot, "iree.turbine.aot.support": support,
                       "iree.compiler": types.ModuleType("iree.compiler"),
                       "iree.compiler.tools": types.ModuleType("iree.compiler.tools"),
                       "iree.compiler.tools.binaries": binaries}
            def lower(command, **kwargs):
                pathlib.Path(command[command.index("-o") + 1]).write_bytes(b"fixture lowered IR")
            for size in (1, 2, 16):
                out = root / f"export-{size}"
                argv = ["export", "--golden-dir", str(golden), "--output-dir", str(out),
                        "--iree-compile", str(root / "compiler")]
                if size != 1:
                    argv += ["--batch-size", str(size)]
                with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(exporter.subprocess, "run", side_effect=lower) as compiler, \
                        mock.patch.object(exporter.importlib.metadata, "version", return_value="fixture"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    exporter.main()
                self.assertEqual(compiler.call_count, 2)
                torch.zeros.assert_called_with((size, 3, 224, 224), dtype=torch.bfloat16)
                result = batch.read_json(out / "manifest.json")
                self.assertEqual(result["batch"], size)
                self.assertEqual(result["input_shape"], [size, 3, 224, 224])
                self.assertEqual(result["output_shape"], [size, 1000])
                self.assertEqual(result["state_dict"], manifest["state_dict"])
                self.assertEqual(result["bf16_reference"], manifest["logits"]["bf16_module"])
            self.assertEqual(before, {path: file_record(path) for path in before})


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class BatchDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)

    def fixture(self, count=32, size=16):
        import numpy as np
        golden, source, export = (self.root / name for name in ("golden", "source", "export"))
        (golden / "bf16_module").mkdir(parents=True)
        source.mkdir()
        export.mkdir()
        (golden / "state_dict.pt").write_bytes(b"synthetic immutable weights, not a model")
        inputs = np.broadcast_to((1 + np.arange(count, dtype=np.float32) / 128)[:, None, None, None],
                                 (count, 3, 224, 224)).copy()
        np.save(golden / "input.npy", inputs)
        logits = ((np.arange(1000)[None, :] + 13 * np.arange(count)[:, None]) % 128).astype(np.float32) / 16
        np.save(golden / "bf16_module/logits.npy", logits)
        np.save(source / "logits.npy", logits)
        manifest = {"labels": logits.argmax(1).tolist(), "state_dict": file_record(golden / "state_dict.pt"),
                    "input": file_record(golden / "input.npy"),
                    "logits": {"bf16_module": file_record(golden / "bf16_module/logits.npy")}}
        write_json(golden / "manifest.json", manifest)
        exported = {"golden_dir": str(golden), "batch": size, "precision_policy": "bf16_module",
                    "input_shape": [size, 3, 224, 224], "output_shape": [size, 1000],
                    "input_dtype": "bf16", "output_dtype": "bf16", "state_dict": manifest["state_dict"],
                    "input": manifest["input"], "bf16_reference": manifest["logits"]["bf16_module"], "artifacts": {}}
        write_json(export / "manifest.json", exported)
        old_export = {**exported, "batch": 1, "input_shape": [1, 3, 224, 224], "output_shape": [1, 1000]}
        write_json(source / "export.json", old_export)
        samples = []
        for index in range(count):
            sample = source / f"sample-{index:02d}"
            sample.mkdir()
            (inputs[index].view(np.uint32) >> 16).astype("<u2").tofile(sample / "input.bf16.bin")
            (logits[index].view(np.uint32) >> 16).astype("<u2").tofile(sample / "tenstorrent.bf16.bin")
            samples.append({"index": index, "input": file_record(sample / "input.bf16.bin")})
        results = {"status": "completed", "batch": 1, "precision_policy": "bf16_module",
                   "indices": list(range(count)), "golden_dir": str(golden), "samples": samples,
                   "export": file_record(source / "export.json")}
        write_json(source / "results.json", results)
        self.golden, self.source, self.export = golden, source, export
        self.out = self.root / "new-batch-run"
        self.logits = logits
        return batch.prepare(source, export, self.out, size)

    def save_profile(self, counts=(3, 4)):
        profile = self.out / "profile"
        profile.mkdir(exist_ok=True)
        write_json(profile / "run_results.json", {
            "dispatch_count": sum(counts), "all_dispatches_device_native": True,
            "execution_domain_counts": {"device_native": sum(counts), "host": 0}})
        rows = []
        for invocation, count in enumerate(counts, 1):
            for dispatch in range(count):
                builtin = "12" if dispatch % 2 else "11"
                row = {"IREE INVOCATION ID": invocation, "OP TYPE": "FUSED_NEW_OP",
                       "INPUT_0_DATATYPE": "BF16", "OUTPUT_0_DATATYPE": "BF16",
                       "IREE EXECUTION DOMAIN": "device_native", "IREE BUILTIN PROGRAM": builtin,
                       "IREE COMPUTE DATATYPE": {"11": "BF16_STORAGE_F32_DATAFLOW_IR_ROUNDING",
                                                 "12": "BF16_HIFI4_FP32_DST_F32_EPILOGUE"}[builtin]}
                row.update({f"IREE {stage} DURATION [ns]": 0 for stage in batch.STAGING_STAGES})
                rows.append(row)
        self.write_profile_rows(rows)
        return rows

    def write_profile_rows(self, rows):
        with (self.out / "profile/ops_perf_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def save_outputs(self, prepared):
        for row in prepared["batches"]:
            values = self.logits[row["indices"]]
            (values.view("uint32") >> 16).astype("<u2").tofile(row["output_path"])
        with (self.out / "call_timings.csv").open("w") as stream:
            stream.write("sample,input_read_upload_seconds,invoke_seconds,output_download_write_seconds,total_seconds\n")
            for i in range(prepared["batch_count"]):
                stream.write(f"{i},0.25,{4 + i},0.25,{4.5 + i}\n")

    def test_112_layout_has_seven_contiguous_b16_files_and_all_logits(self):
        prepared = self.fixture(count=112)
        self.assertEqual(prepared["batch_count"], 7)
        self.assertEqual(prepared["input_shape"], [16, 3, 224, 224])
        for row in prepared["batches"]:
            expected = b"".join((self.source / f"sample-{i:02d}/input.bf16.bin").read_bytes() for i in row["indices"])
            self.assertEqual(pathlib.Path(row["input"]["path"]).read_bytes(), expected)
        self.assertEqual(batch.validate_prepared(self.out), prepared)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            batch.prepare(self.source, self.export, self.out)
        self.save_outputs(prepared)
        self.save_profile(counts=(3, 4, 2, 5, 1, 6, 4))
        result = batch.analyze(self.out)
        self.assertEqual(result["sample_count"], 112)
        for values in result["comparisons"].values():
            self.assertEqual(values["aggregate"]["elements"], 112000)
            self.assertEqual(len(values["per_image"]), 112)

    def test_archived_run_with_parent_alias_keeps_original_manifests_valid(self):
        prepared = self.fixture(count=4, size=2)
        archive = self.root / "dated-archive"
        self.out.rename(archive)
        self.out.symlink_to(archive, target_is_directory=True)
        self.assertEqual(batch.validate_prepared(self.out), prepared)
        self.assertEqual(batch.validate_prepared(archive), prepared)

    def test_sources_remain_read_only_and_changes_after_preparation_are_rejected(self):
        self.fixture(count=4, size=2)
        for folder in (self.golden, self.source, self.export):
            with self.assertRaisesRegex(ValueError, "must be separate"):
                batch.prepare(self.source, self.export, folder / "new-run", 2)
            self.assertFalse((folder / "new-run").exists())
        weights = self.golden / "state_dict.pt"
        weights.write_bytes(b"mutated fixture weights")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            batch.validate_prepared(self.out)

    def test_default_b1_and_static_b2_preparation(self):
        prepared = self.fixture(count=4, size=2)
        self.assertEqual(prepared["batch_count"], 2)
        self.assertEqual(prepared["batches"][1]["indices"], [2, 3])
        # Batch one is also a real static graph, not a special padded path.
        metadata = batch.read_json(self.export / "manifest.json")
        metadata.update(batch=1, input_shape=[1, 3, 224, 224], output_shape=[1, 1000])
        write_json(self.export / "manifest.json", metadata)
        prepared = batch.prepare(self.source, self.export, self.root / "b1", 1)
        self.assertEqual(prepared["batch_count"], 4)

    def test_rejects_weight_shape_order_padding_and_saved_input_mismatch(self):
        self.fixture(count=4, size=2)
        export = batch.read_json(self.export / "manifest.json")
        source = batch.read_json(self.source / "results.json")
        for key, value in (("state_dict", {**export["state_dict"], "sha256": "different"}),
                           ("input_shape", [1, 3, 224, 224])):
            broken = copy.deepcopy(export)
            broken[key] = value
            write_json(self.export / "manifest.json", broken)
            with self.assertRaises(ValueError):
                batch.prepare(self.source, self.export, self.root / "rejected", 2)
            self.assertFalse((self.root / "rejected").exists())
        write_json(self.export / "manifest.json", export)
        for indices in ([1, 0, 2, 3], [0, 1, 2]):
            write_json(self.source / "results.json", {**source, "indices": indices})
            with self.assertRaisesRegex(ValueError, "ordered B1 dataset"):
                batch.prepare(self.source, self.export, self.root / "rejected", 2)
        write_json(self.source / "results.json", source)
        filename = self.source / "sample-00/input.bf16.bin"
        filename.write_bytes(b"\0" * batch.INPUT_SAMPLE_BYTES)
        source["samples"][0]["input"] = file_record(filename)
        write_json(self.source / "results.json", source)
        with self.assertRaisesRegex(ValueError, "differs from golden"):
            batch.prepare(self.source, self.export, self.root / "rejected", 2)

    def test_no_partial_batch_padding(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.fixture(count=3, size=2)
        self.assertFalse(self.out.exists())

    def test_analysis_covers_every_row_and_batch_not_b1_timing(self):
        prepared = self.fixture()
        self.save_outputs(prepared)
        self.save_profile()
        before = {path: file_record(path) for folder in (self.golden, self.source, self.export)
                  for path in folder.rglob("*") if path.is_file()}
        result = batch.analyze(self.out)
        self.assertEqual(result["sample_count"], 32)
        self.assertEqual(result["native_profile"]["dispatches_per_batch"], [3, 4])
        for values in result["comparisons"].values():
            self.assertEqual(len(values["per_image"]), 32)
            self.assertAlmostEqual(values["aggregate"]["pcc"], 1)
            self.assertEqual(values["aggregate"]["relative_l2"], 0)
            self.assertEqual(values["aggregate"]["top1_agreement"], 32)
            self.assertEqual(values["aggregate"]["bitwise_mismatches"], 0)
        invoke = result["timing"]["all_batches"]["invoke_seconds"]
        self.assertEqual(invoke["mean_per_batch"], 4.5)
        self.assertAlmostEqual(invoke["images_per_second"], 32 / 9)
        self.assertEqual(result["timing"]["excluding_first_batch"]["image_count"], 16)
        self.assertEqual(before, {path: file_record(path) for path in before})
        with self.assertRaisesRegex(ValueError, "overwrite analysis"):
            batch.analyze(self.out)

    def test_native_profile_rejects_fallback_staging_dtype_and_missing_batch(self):
        self.fixture(count=4, size=2)
        rows = self.save_profile()
        batch.require_native_profile(self.out / "profile", 2)
        for key, value in (("IREE EXECUTION DOMAIN", "host"), ("OUTPUT_0_DATATYPE", "F32"),
                           ("IREE PACK DURATION [ns]", 1), ("IREE BUILTIN PROGRAM", "3"),
                           ("IREE INVOCATION ID", 0), ("IREE COMPUTE DATATYPE", "F32")):
            broken = copy.deepcopy(rows)
            broken[0][key] = value
            self.write_profile_rows(broken)
            with self.assertRaises(ValueError):
                batch.require_native_profile(self.out / "profile", 2)
        self.write_profile_rows(rows[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            batch.require_native_profile(self.out / "profile", 2)

    def test_timing_rejects_missing_reordered_and_nonfinite_rows(self):
        prepared = self.fixture(count=4, size=2)
        self.save_outputs(prepared)
        path = self.out / "call_timings.csv"
        header, first, second = path.read_text().splitlines()
        for rows in ((first,), (second, first), (first.replace(",4,", ",nan,"), second),
                     (first.replace(",4,", ",0,"), second), (first.replace(",4,", ",-1,"), second)):
            path.write_text("\n".join((header, *rows)) + "\n")
            with self.assertRaises(ValueError):
                batch.timing_summary(path, 2, 2)

    def test_short_or_nonfinite_batch_logits_cannot_produce_analysis(self):
        prepared = self.fixture(count=4, size=2)
        self.save_outputs(prepared)
        self.save_profile()
        path = pathlib.Path(prepared["batches"][0]["output_path"])
        original = path.read_bytes()
        for payload in (original[:-2], b"\x80\x7f" + original[2:]):
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                batch.analyze(self.out)
            self.assertFalse((self.out / "analysis.json").exists())
            self.assertFalse((self.out / "logits.npy").exists())

    def test_changed_batched_input_and_tsv_rejected_before_run(self):
        prepared = self.fixture(count=4, size=2)
        tsv = self.out / "inputs.tsv"
        original = tsv.read_bytes()
        tsv.write_bytes(original[::-1])
        with mock.patch.object(batch.subprocess, "run") as invoke:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                batch.run(self.out, self.root / "missing.vmfb", self.root / "missing-runner", self.root)
            invoke.assert_not_called()
        tsv.write_bytes(original)
        filename = pathlib.Path(prepared["batches"][0]["input"]["path"])
        filename.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            batch.validate_prepared(self.out)

    def test_run_invokes_exactly_one_persistent_process_with_static_batch(self):
        prepared = self.fixture(count=4, size=2)
        module, runner = self.root / "module.vmfb", self.root / "session-runner"
        module.write_bytes(b"synthetic module")
        runner.write_bytes(b"synthetic runner")
        def fake_run(command, **kwargs):
            self.assertEqual(command[-3:], ["tenstorrent", "--batch-size", "2"])
            self.assertEqual(kwargs["env"]["TT_IREE_REQUIRE_DEVICE_NATIVE"], "1")
            self.assertNotIn("TT_METAL_SIMULATOR", kwargs["env"])
            self.assertNotIn("TT_IREE_CHECKPOINT_OUTPUT", kwargs["env"])
            self.save_outputs(prepared)
            self.save_profile()
        with mock.patch.object(batch.subprocess, "run", side_effect=fake_run) as invoke, \
                mock.patch.dict(batch.os.environ, {"TT_METAL_SIMULATOR": "1", "TT_IREE_CHECKPOINT_OUTPUT": "bad"}):
            result = batch.run(self.out, module, runner, self.root)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(result["status"], "completed")
        with mock.patch.object(batch.subprocess, "run") as invoke:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                batch.run(self.out, module, runner, self.root)
            invoke.assert_not_called()

    def test_failed_run_records_error_without_silent_retry(self):
        self.fixture(count=2, size=2)
        module, runner = self.root / "module.vmfb", self.root / "session-runner"
        module.write_bytes(b"synthetic module")
        runner.write_bytes(b"synthetic runner")
        with mock.patch.object(batch.subprocess, "run", side_effect=RuntimeError("injected failure")) as invoke:
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                batch.run(self.out, module, runner, self.root)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(batch.read_json(self.out / "results.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
