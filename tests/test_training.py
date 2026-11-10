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
from dexgrasp.config import load_config
from dexgrasp.data.dataset import sha256
from dexgrasp.train.runner import run, atomic_checkpoint, ResumeBatchSampler, initialize_student, train_mode
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.config import model_config
from dexgrasp.test.pipeline import generate, decode_file, evaluate
from helpers import dataset_fixture, run_config, FixtureEncoder


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hand = dataset_fixture(self.root/'dataset')
        self.cfg = run_config(self.root)

    def fake_train(self, cfg):
        # Software control-flow test: optimizer updates are explicitly disabled.
        with patch.object(torch.optim.AdamW, 'step', return_value=None):
            return run(cfg, encoder_factory=FixtureEncoder)

    def test_stateless_sampler_resume_ignores_prefetch(self):
        all_batches = list(ResumeBatchSampler(7, 3, 17, 0, 10))
        rest = list(ResumeBatchSampler(7, 3, 17, 4, 6))
        self.assertEqual(rest, all_batches[4:])
        self.assertEqual(sorted(sum(all_batches[:3], [])), list(range(7)))

    def test_teacher_initialization_copies_without_sharing(self):
        mc = model_config(self.cfg, self.hand.spec)
        teacher = KeypointModel(mc, 'fm', encoder_factory=FixtureEncoder)
        student = KeypointModel(mc, 'flowmap', encoder_factory=FixtureEncoder)
        initialize_student(student, teacher)
        for name, value in teacher.state_dict().items():
            target = student.state_dict()[name]
            if name == 'time.0.weight':
                torch.testing.assert_close(target[:, :128], value)
                self.assertEqual(int(target[:, 128:].count_nonzero()), 0)
            else:
                torch.testing.assert_close(target, value)
            self.assertNotEqual(target.data_ptr(), value.data_ptr())
        train_mode(student, {'freeze_encoders': True})
        self.assertFalse(student.object_encoder.training)
        self.assertFalse(student.scene_encoder.training)
        self.assertTrue(student.blocks.training)

    def test_training_resume_identical_without_optimizer_updates(self):
        def interrupt(path, state):
            atomic_checkpoint(path, state)
            if state['step'] == 2 and Path(path).name == 'last.pt':
                raise RuntimeError('simulated interruption')
        with patch.object(torch.optim.AdamW, 'step', return_value=None):
            with patch('dexgrasp.train.runner.atomic_checkpoint', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    run(self.cfg, encoder_factory=FixtureEncoder)
            checkpoint = self.root/'fm_ckpt/last.pt'
            before = torch.load(checkpoint, weights_only=False)
            result = run(self.cfg, resume=checkpoint, encoder_factory=FixtureEncoder)
            after = torch.load(checkpoint, weights_only=False)
        self.assertEqual(result['steps'], 4)
        self.assertTrue(all(torch.equal(before['model'][k], after['model'][k]) for k in before['model']))
        comparison = {**self.cfg, 'run_dir': str(self.root/'control_run'), 'checkpoint_dir': str(self.root/'control_ckpt')}
        self.fake_train(comparison)
        control = torch.load(self.root/'control_ckpt/last.pt', weights_only=False)
        self.assertTrue(torch.equal(control['torch_rng'], after['torch_rng']))
        self.assertEqual(control['best_val'], after['best_val'])
        self.assertIn('dexgrasp/models/keypoint.py', after['source']['source_sha256'])
        with self.assertRaises(FileExistsError): self.fake_train(self.cfg)
        with self.assertRaises(ValueError):
            run({**self.cfg, 'lr': .1}, resume=checkpoint, encoder_factory=FixtureEncoder)

    def test_fm_to_flowmap_distillation_pipeline(self):
        self.fake_train(self.cfg)
        teacher = self.root/'fm_ckpt/last.pt'
        cfg = run_config(self.root, 'flowmap', teacher)
        result = self.fake_train(cfg)
        self.assertEqual(result['method'], 'flowmap')
        self.assertIsNone(result['physical_success_rate'])
        state = torch.load(self.root/'flowmap_ckpt/last.pt', weights_only=False)
        self.assertIsNotNone(state['teacher_sha256'])
        self.assertEqual(state['validation_metric'], 'paired_endpoint_mse_diagnostic')
        bad = {**cfg, 'run_dir': str(self.root/'bad_run'), 'checkpoint_dir': str(self.root/'bad_ckpt'),
               'model': {**cfg['model'], 'coord_scale_m': .5}}
        with self.assertRaisesRegex(ValueError, 'scales'): self.fake_train(bad)

    def test_checkpoint_warm_start_has_new_optimizer_and_identity_guard(self):
        self.fake_train(self.cfg)
        parent = self.root/'fm_ckpt/last.pt'
        cfg = {**self.cfg, 'initialization': 'checkpoint', 'initial_checkpoint': str(parent),
               'run_dir': str(self.root/'warm_run'), 'checkpoint_dir': str(self.root/'warm_ckpt')}
        self.fake_train(cfg)
        state = torch.load(self.root/'warm_ckpt/last.pt', weights_only=False)
        self.assertEqual(state['initial_checkpoint_sha256'], sha256(parent))
        from dexgrasp.train.runner import initialize_checkpoint
        model = KeypointModel(model_config(cfg, self.hand.spec), encoder_factory=FixtureEncoder)
        with self.assertRaisesRegex(ValueError, 'dataset'):
            initialize_checkpoint(model, parent, {}, state['hand_spec_sha256'])

    def test_sample_decode_and_explicit_physics_coverage(self):
        self.fake_train(self.cfg)
        output = self.root/'generated.h5'
        result = generate(self.root/'fm_ckpt/last.pt', self.root/'dataset', output,
                          candidates=2, batch_size=2, steps=2, encoder_factory=FixtureEncoder)
        self.assertEqual(result['samples'], 2)
        with self.assertRaises(FileExistsError):
            generate(self.root/'fm_ckpt/last.pt', self.root/'dataset', output, encoder_factory=FixtureEncoder)
        # Substitute analytic reachable poses to test decoder independently of untrained network quality.
        with h5py.File(output, 'r+') as f:
            f['K'][:] = np.tile(self.hand.fk(np.zeros(22)), (2, 2, 1, 1))
        decoded = self.root/'decoded.jsonl'
        decode_file(output, decoded)
        metrics = evaluate(decoded)
        self.assertEqual(metrics['geometric_condition_pass_rate'], 1.)
        self.assertIsNone(metrics['joint_success_rate'])
        self.assertEqual(metrics['physics_evaluation_coverage'], 0.)
        verdicts = self.root/'verdicts.jsonl'
        verdicts.write_text(''.join(json.dumps(dict(sample_index=i, candidate_index=j,
                               collision_free=True, physical_success=(i == 0),
                               evaluator='synthetic-test-only', protocol_version='test',
                               decoded_sha256=sha256(decoded)))+'\n'
                               for i in range(2) for j in range(2)))
        self.assertEqual(evaluate(decoded, verdicts=verdicts)['joint_success_rate'], .5)
        rows = [json.loads(line) for line in decoded.read_text().splitlines()]
        rows[0]['palm_gap_valid'] = False
        decoded.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'different decoded'):
            evaluate(decoded, verdicts=verdicts)
        verdict_rows = [json.loads(line) for line in verdicts.read_text().splitlines()]
        for row in verdict_rows:
            row['decoded_sha256'] = sha256(decoded)
        verdicts.write_text(''.join(json.dumps(r)+'\n' for r in verdict_rows))
        self.assertEqual(evaluate(decoded, verdicts=verdicts)['joint_success_rate'], .25)

    def test_condition_override_is_inference_only(self):
        self.fake_train(self.cfg)
        output = self.root/'overridden.h5'
        generate(self.root/'fm_ckpt/last.pt', self.root/'dataset', output, steps=1,
                 candidates=1, palm_gap_m=.04, direction=[1, 0, 0], encoder_factory=FixtureEncoder)
        with h5py.File(output) as f:
            np.testing.assert_allclose(f['h'][:], .04)
            np.testing.assert_allclose(f['d'][:], [[1, 0, 0], [1, 0, 0]])
        with h5py.File(self.root/'dataset/shards/part-00000.h5') as f:
            np.testing.assert_allclose(f['grasps/h'][:], .02)

    def test_plan_only_cli_does_not_start_training(self):
        path = self.root/'config.json'
        path.write_text(json.dumps(self.cfg))
        process = subprocess.run([sys.executable, '-m', 'dexgrasp', 'train', '--config', str(path)],
                                 capture_output=True, text=True)
        report = json.loads(process.stdout)
        self.assertFalse(report['training_started'])
        self.assertFalse(Path(self.cfg['run_dir']).exists())

    def test_config_validation(self):
        path = self.root/'config.json'
        for key, value in [('method', 'diffusion'), ('precision', 'float16'), ('batch_size', 0),
                           ('full_interval_probability', 0), ('lr', float('nan')), ('typo', 1)]:
            path.write_text(json.dumps({**self.cfg, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError): load_config(path)


if __name__ == '__main__':
    unittest.main()
