import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from dexgrasp_lab.config import load_config
from dexgrasp_lab.data import (REPRESENTATION, batch, build_kp_manifest, digest,
                               load_manifest, read_record, split_objects, validate_records)
from dexgrasp_lab.test import benchmark_plan, generate, metrics, reconstruct
from dexgrasp_lab.methods import alpha_bar, loss, sample, training_pair
from dexgrasp_lab.models import GraspModel
from dexgrasp_lab.train import atomic_checkpoint, validate, run
from dexgrasp_lab.prepare import catalog_manifest, forward_keypoints, convert_qpos
from dexgrasp_lab.references import PROFILES, plan as reference_plan

torch.set_num_threads(1)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.records = []
        for i, split in enumerate(("train", "val", "test")):
            path = self.root / f"{i}.npz"
            np.savez(path, point_cloud=np.ones((12, 3), np.float32) * i,
                     keypoints=np.ones((2, 3, 28, 3), np.float32) * (i + 0.1))
            self.records.append({"id": str(i), "object_id": str(i), "split": split,
                                 "storage": "npz", "path": str(path)})
        self.manifest = self.root / "manifest.json"
        self.write_manifest(self.records)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"manifest": str(self.manifest), "run_dir": str(self.root / "run"),
            "checkpoint_dir": str(self.root / "ckpt"), "device": "cpu", "width": 16, "depth": 1,
            "batch_size": 1, "num_points": 8, "val_objects": 1, "sample_steps": 2, "candidates": 2}))
        self.cfg = load_config(self.config)

    def write_manifest(self, records):
        self.manifest.write_text(json.dumps({"schema_version": 1, "representation": REPRESENTATION,
                                             "dataset_id": "synthetic_test_only", "records": records}))

    def checkpoint(self):
        model = GraspModel(width=16, depth=1)
        path = self.root / "untrained_test_fixture.pt"
        atomic_checkpoint(path, {"config": self.cfg, "model": model.state_dict(),
                                 "manifest_sha256": digest(self.manifest)})
        return path


class DataTests(Fixture):
    def test_split_reproducible_order_independent(self):
        a = split_objects(["a", "b", "c"], ["z"], seed=7)
        self.assertEqual(a, split_objects(["c", "b", "a"], ["z"], seed=7))
        self.assertEqual(a["z"], "test")
        self.assertEqual(set(a.values()), {"train", "val", "test"})

    def test_bad_split(self):
        for train, test in [(["a", "b"], ["a"]), (["a", "a"], ["z"]), (["a"], ["z"])]:
            with self.assertRaises(ValueError): split_objects(train, test)
        for fraction in (0, 1, -0.1):
            with self.assertRaises(ValueError): split_objects(["a", "b"], ["z"], fraction)

    def test_leakage_and_duplicate(self):
        for key in ("object_id", "id"):
            records = copy.deepcopy(self.records)
            records[1][key] = records[0][key]
            self.write_manifest(records)
            with self.assertRaises(ValueError): load_manifest(self.manifest)

    def test_missing_files(self):
        self.records[0]["path"] = "absent.npz"
        self.write_manifest(self.records)
        with self.assertRaises(FileNotFoundError): load_manifest(self.manifest)

    def test_evaluation_alias_normalizes_to_test(self):
        self.records[2]["split"] = "evaluation"
        self.write_manifest(self.records)
        self.assertEqual(load_manifest(self.manifest)["records"][2]["split"], "test")
        summary = generate(self.checkpoint(), self.manifest, self.root / "legacy", split="evaluation")
        self.assertEqual(summary["split"], "test")

    def test_same_file_cannot_cross_splits(self):
        self.records[1]["path"] = self.records[0]["path"]
        self.write_manifest(self.records)
        with self.assertRaises(ValueError): load_manifest(self.manifest)

    def test_catalog_grouped_split_and_unlabelled_test(self):
        np.savez(self.records[2]["path"], point_cloud=np.zeros((8, 3), np.float32))
        rows = copy.deepcopy(self.records)
        for row in rows:
            row["official_split"] = "test" if row.pop("split") == "test" else "train"
        catalog = self.root / "catalog.json"
        catalog.write_text(json.dumps(rows))
        output = self.root / "prepared.json"
        catalog_manifest(catalog, output)
        validate_records(load_manifest(output)["records"])
        pc, targets, _ = batch(self.records, [2], np.random.default_rng(0), 4, 0.1, "cpu", targets=False)
        self.assertIsNone(targets)

    def test_shapes_finite_and_normalization(self):
        validate_records(load_manifest(self.manifest)["records"])
        pc, kp, centre = batch(self.records, [0, 1], np.random.default_rng(1), 32, 0.1, "cpu")
        self.assertEqual(tuple(pc.shape), (2, 32, 3))
        self.assertTrue(torch.allclose(pc, torch.zeros_like(pc)))
        self.assertTrue(torch.allclose(kp, torch.ones_like(kp)))
        np.savez(self.records[0]["path"], point_cloud=np.zeros((4, 3)), keypoints=np.full((1, 3, 28, 3), np.nan))
        with self.assertRaises(ValueError): validate_records(self.records)

    def test_wrong_keypoints(self):
        np.savez(self.records[0]["path"], point_cloud=np.zeros((4, 3)), keypoints=np.zeros((1, 21, 3)))
        with self.assertRaises(ValueError): read_record(self.records[0], np.random.default_rng(0))

    def test_h5_manifest_and_scale(self):
        data = self.root / "dexonomy"
        (data / "dex_kp_h5").mkdir(parents=True)
        index = {}
        for ci, collection in enumerate(("DGN_5k", "objaverse_5k")):
            folder = data / "object" / collection
            (folder / "valid_split").mkdir(parents=True)
            (folder / "vision_data").mkdir()
            ids = [f"c{ci}_{i}" for i in range(4)]
            (folder / "valid_split/train.json").write_text(json.dumps(ids[:3]))
            (folder / "valid_split/test.json").write_text(json.dumps(ids[3:]))
            with h5py.File(folder / "vision_data/compl_pc.h5", "w") as f:
                f["data"] = np.ones((4, 8, 3), np.float32)
                f["filenames"] = np.array(ids, dtype="S")
            for i, oid in enumerate(ids): index[oid] = [ci * 4 + i, 1, "shard.h5"]
        (data / "dex_kp_h5/index_obj_grasp.json").write_text(json.dumps(index))
        with h5py.File(data / "dex_kp_h5/shard.h5", "w") as f:
            for phase in ("pregrasp", "grasp", "squeeze"): f[f"{phase}_kp"] = np.zeros((8, 28, 3))
            f["scene_scale"] = np.full(8, 0.08)
        output = self.root / "h5.json"
        build_kp_manifest(data, output)
        rows = load_manifest(output)["records"]
        self.assertEqual(len(rows), 8)
        pc, kp = read_record(rows[0], np.random.default_rng(0))
        self.assertTrue(np.allclose(pc, 0.08))
        rows[0]["count"] = 999
        with self.assertRaises(ValueError): validate_records(rows)
        with self.assertRaises(FileExistsError): build_kp_manifest(data, output)


class MethodTests(unittest.TestCase):
    def test_fm_endpoints(self):
        x, noise = torch.ones(2, 3, 28, 3), torch.zeros(2, 3, 28, 3)
        xt, t, target = training_pair(x, "flow_matching", noise=noise, time=torch.tensor([0., 1.]))
        self.assertTrue(torch.equal(xt[0], noise[0]))
        self.assertTrue(torch.equal(xt[1], x[1]))
        self.assertTrue(torch.equal(target, x - noise))

    def test_all_model_objective_gradients(self):
        for arch in ("pointnet_mlp", "pointnet_transformer"):
            for method in ("flow_matching", "diffusion"):
                with self.subTest(arch=arch, method=method):
                    model = GraspModel(arch, 16, 1)
                    value = loss(model, torch.randn(2, 8, 3), torch.randn(2, 3, 28, 3), method)
                    value.backward()
                    self.assertTrue(torch.isfinite(value))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_analytic_sampler_and_nfe(self):
        class Oracle:
            def __init__(self, method): self.method, self.calls = method, 0
            def encode(self, pc): return torch.zeros(len(pc), 1)
            def __call__(self, x, t, context):
                self.calls += 1
                if self.method == "flow_matching": return torch.ones_like(x) * 2
                a = alpha_bar(x.device)[(t * 999).round().long()].reshape(-1, 1, 1, 1)
                return (x - a.sqrt() * 2) / (1 - a).sqrt()
        for method in ("flow_matching", "diffusion"):
            for steps in (1, 2, 5, 20):
                model = Oracle(method)
                result = sample(model, torch.zeros(1, 8, 3), method, steps, 2,
                                noise=torch.zeros(2, 3, 28, 3))
                self.assertTrue(torch.allclose(result, torch.full_like(result, 2), atol=1e-3))
                self.assertEqual(model.calls, steps)

    def test_invalid_sampling(self):
        model = GraspModel(width=16, depth=1)
        for method, steps, candidates in [("bad", 5, 1), ("diffusion", 0, 1), ("flow_matching", 2, 0)]:
            with self.assertRaises(ValueError): sample(model, torch.zeros(1, 8, 3), method, steps, candidates)


class IntegrationTests(Fixture):
    def test_invalid_configs(self):
        raw = json.loads(self.config.read_text())
        for key, val in [("objective", "unknown"), ("batch_size", 0), ("scale_m", 0),
                         ("width", 7), ("lr", float("nan")), ("seed", -1), ("typo", True)]:
            self.config.write_text(json.dumps({**raw, key: val}))
            with self.assertRaises(ValueError): load_config(self.config)

    def test_validation_restores_rng_and_mode(self):
        model = GraspModel(width=16, depth=1)
        before = torch.get_rng_state()
        value = validate(model, self.records[1:2], self.cfg)
        self.assertEqual(value, validate(model, self.records[1:2], self.cfg))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_runner_checkpoint_resume_without_parameter_updates(self):
        # Control-flow test only: optimizer.step is disabled. No training takes place.
        cfg = {**self.cfg, "max_steps": 4, "val_every": 2, "save_every": 2, "accumulation": 2}
        def interrupt_after_checkpoint(path, state):
            atomic_checkpoint(path, state)
            if state["step"] == 2 and Path(path).name == "last.pt":
                raise RuntimeError("simulated interruption")
        with patch.object(torch.optim.AdamW, "step", return_value=None):
            with patch("dexgrasp_lab.train.atomic_checkpoint", side_effect=interrupt_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run(cfg)
            last = Path(cfg["checkpoint_dir"]) / "last.pt"
            before = torch.load(last, weights_only=False)
            result = run(cfg, str(last))
            after = torch.load(last, weights_only=False)
            self.assertEqual(result["steps"], 4)
            self.assertTrue(all(torch.equal(before["model"][k], after["model"][k]) for k in before["model"]))
            with self.assertRaises(ValueError): run({**cfg, "lr": 0.01}, str(last))
            with self.assertRaises(FileExistsError): run(cfg)

    def test_checkpoint_sample_metrics(self):
        path = self.checkpoint()
        out = self.root / "samples"
        summary = generate(path, self.manifest, out, device="cpu")
        self.assertEqual(summary["nfe_per_candidate"], 2)
        self.assertEqual(metrics(out)["grasps"], 2)
        self.assertIsNone(metrics(out)["physical_success_rate"])
        with self.assertRaises(FileExistsError): generate(path, self.manifest, out)
        with self.assertRaises(ValueError): reconstruct(out, self.root / "ik", "missing-reference")

    def test_manifest_mismatch(self):
        path = self.checkpoint()
        self.records[0]["id"] = "changed"
        self.write_manifest(self.records)
        with self.assertRaises(ValueError): generate(path, self.manifest, self.root / "samples")

    def test_train_defaults_to_plan(self):
        result = subprocess.run([sys.executable, "-m", "dexgrasp_lab", "train", "--config", str(self.config)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["training_started"])
        self.assertFalse(Path(self.cfg["run_dir"]).exists())

    def test_model_test_cli_and_metrics_alias(self):
        output = self.root / "evaluation_samples"
        generate(self.checkpoint(), self.manifest, output)
        results = []
        for command in ("test", "metrics"):
            process = subprocess.run([sys.executable, "-m", "dexgrasp_lab", command, "--samples", str(output)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            results.append(json.loads(process.stdout))
        self.assertEqual(results[0], results[1])
        process = subprocess.run([sys.executable, "-m", "dexgrasp_lab", "sample", "--checkpoint", "absent.pt",
                                  "--manifest", str(self.manifest), "--output", "unused", "--split", "evaluation"],
                                 capture_output=True, text=True)
        self.assertEqual(json.loads(process.stdout)["arguments"]["split"], "test")

    def test_benchmark_plan_no_delete(self):
        plan = benchmark_plan("/reference/DexGraspBench", "/env/bin/python", "/tmp/sample data", "/tmp/eval", "test")
        self.assertEqual(len(plan["commands"]), 3)
        self.assertIn("task.data_path=/tmp/sample data", plan["commands"][0])
        self.assertTrue(all("rm" not in command for command in plan["commands"]))

    def test_all_reference_plans(self):
        for profile in PROFILES:
            result = reference_plan(profile, sys.executable, "/data", "/outputs")
            self.assertEqual(result["status"], "reference_plan_only")
            self.assertEqual(Path(result["cwd"]).name, result["source"]["name"])
            self.assertEqual(result["argv"][0], sys.executable)

    def test_fk_joint_mapping_and_conversion(self):
        import mujoco
        root = Path(__file__).resolve().parents[2] / "reference/KPGrasp"
        xml = root / "assets/hand/shadow/right_hand_w_kp.xml"
        source = root / "src/dexgrasp/kinematics/ik.py"
        model = mujoco.MjModel.from_xml_path(str(xml))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        q = np.zeros((1, 3, 29)); q[..., 3] = 1
        q[..., 7:] = np.linspace(0, 0.1, 22)
        first = forward_keypoints(q, xml, source, names)
        reordered = q.copy(); reordered[..., 7:] = q[..., 7:][..., ::-1]
        second = forward_keypoints(reordered, xml, source, names[::-1])
        np.testing.assert_allclose(first, second)
        translated = q.copy(); translated[..., :3] = [1, 2, 3]
        np.testing.assert_allclose(forward_keypoints(translated, xml, source, names), first + [1, 2, 3], atol=1e-6)
        order_file = self.root / "joints.json"; order_file.write_text(json.dumps(names))
        poses = self.root / "poses.npz"
        np.savez(poses, **{f"{p}_qpos": q[:, i] for i, p in enumerate(("pregrasp", "grasp", "squeeze"))})
        cloud = self.root / "pc.npy"; np.save(cloud, np.ones((8, 3)))
        result = convert_qpos(poses, cloud, self.root / "converted.npz", root, order_file, 0.08)
        self.assertEqual(result["grasps"], 1)
        with self.assertRaises(ValueError): forward_keypoints(q, xml, source, names[:-1])
        q[..., 3] = 0
        with self.assertRaises(ValueError): forward_keypoints(q, xml, source, names)


if __name__ == "__main__":
    unittest.main()
