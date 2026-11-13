"""Ablation behavior and integration checks; no optimizer updates or real training."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import torch

from dexgrasp.config import MODEL_VARIANTS, ModelConfig, load_config, model_config
from dexgrasp.data.dataset import GraspDataset
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.test.inference import predict
from dexgrasp.test.pipeline import generate
from dexgrasp.train.objectives import fm_loss, flowmap_loss, sample_keypoints
from dexgrasp.train.distributed import cache_training_data, cached_batch, FMLossModule
from dexgrasp.train.runner import run, load_model, initialize_checkpoint, train_mode
from helpers import FixtureEncoder, condition, small_config, dataset_fixture, run_config
from test_distributed import CachedFixtureEncoder


class AblationTests(unittest.TestCase):
    def model(self, variant, method='fm'):
        return KeypointModel(replace(small_config(), variant=variant), method,
                             encoder_factory=FixtureEncoder)

    def test_all_variants_have_only_active_parameters_and_gradients(self):
        for variant in MODEL_VARIANTS:
            for method in ('fm', 'flowmap'):
                with self.subTest(variant=variant, method=method):
                    model = self.model(variant, method)
                    cfg = model.cfg
                    self.assertEqual(model.scene_encoder is not None, cfg.use_scene)
                    self.assertEqual(model.gap is not None, cfg.use_gap)
                    self.assertEqual(model.direction_embedding is not None, cfg.direction_adaln)
                    for block in model.blocks:
                        self.assertEqual(len(block.cross), 2 if cfg.use_scene else 1)
                        self.assertEqual(len(block.modulations), len(block.cross)+2)
                        self.assertEqual(len(block.biases), 0 if cfg.direction_adaln else len(block.cross))
                    c = {k: v for k, v in condition().items() if k in cfg.condition_keys}
                    batch = dict(condition=c, keypoints_gt=torch.randn(2, 21, 3))
                    if method == 'fm':
                        loss = fm_loss(model, batch)
                    else:
                        teacher = self.model(variant).requires_grad_(False)
                        loss = flowmap_loss(model, teacher, batch, steps=2)
                        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                    loss.backward()
                    for name, param in model.named_parameters():
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                    train_mode(model, {'freeze_encoders': True})
                    self.assertFalse(model.object_encoder.training)

    def test_omitted_and_poisoned_inactive_inputs_are_identical(self):
        for variant in MODEL_VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant).eval()
                raw = condition()
                minimal = {k: raw[k] for k in model.cfg.condition_keys}
                poisoned = {k: v if k in minimal else torch.full_like(v, float('nan'))
                            for k, v in raw.items()}
                poisoned['scene_fps_indices'] = None  # Ignored only when scene is disabled.
                if model.cfg.use_scene:
                    poisoned.pop('scene_fps_indices')
                noise = torch.randn(2, 21, 3)
                expected = predict(model, minimal, noise=noise, steps=2)
                torch.testing.assert_close(predict(model, poisoned, noise=noise, steps=2), expected,
                                           atol=0, rtol=0)
                with torch.no_grad():
                    x, t = noise, torch.ones(2)*.3
                    torch.testing.assert_close(model(x, t, model.encode(minimal)),
                                               model(x, t, model.encode(poisoned)), atol=0, rtol=0)
                for key in model.cfg.condition_keys:
                    with self.assertRaisesRegex(ValueError, 'missing model conditions'):
                        model.encode({k: v for k, v in minimal.items() if k != key})

    def test_direction_adaln_replaces_geometry_and_remains_normalized_and_differentiable(self):
        for variant in ('object_direction_adaln', 'object_direction_adaln_h_less'):
            with self.subTest(variant=variant):
                model, c = self.model(variant), condition()
                c['grasp_direction_cmd'].requires_grad_(True)
                x, t = torch.randn(2, 21, 3), torch.full((2,), .4)
                with patch('dexgrasp.models.keypoint.direction_geometry', side_effect=AssertionError('old direction path')):
                    output = model(x, t, model.encode(c))
                    other = model(x, t, model.encode({**c, 'grasp_direction_cmd': torch.tensor([[1., 0., 0.]]).repeat(2, 1)}))
                    self.assertFalse(torch.allclose(output, other))
                    scaled = model(x, t, model.encode({**c, 'grasp_direction_cmd': c['grasp_direction_cmd']*7}))
                    torch.testing.assert_close(output, scaled)
                    output.square().mean().backward()
                self.assertGreater(float(c['grasp_direction_cmd'].grad.abs().sum()), 0)
                for p in model.direction_embedding.parameters():
                    self.assertGreater(float(p.grad.abs().sum()), 0)
                with self.assertRaisesRegex(ValueError, 'nonzero'):
                    model.encode({**c, 'grasp_direction_cmd': torch.zeros(2, 3)})

    def test_kv_cache_and_candidate_order_for_all_variants_and_methods(self):
        for variant in MODEL_VARIANTS:
            for method in ('fm', 'flowmap'):
                with self.subTest(variant=variant, method=method):
                    model, raw = self.model(variant, method), condition()
                    c = {k: raw[k] for k in model.cfg.condition_keys}
                    noise = torch.randn(4, 21, 3)
                    cached = sample_keypoints(model, c, steps=2, candidates=2, noise=noise)
                    self.assertEqual(model.object_encoder.calls, 1)
                    if model.cfg.use_scene:
                        self.assertEqual(model.scene_encoder.calls, 1)
                    uncached = sample_keypoints(model, c, steps=2, candidates=2, noise=noise, cache_kv=False)
                    torch.testing.assert_close(cached, uncached, atol=2e-6, rtol=1e-5)
                    separate = torch.cat([sample_keypoints(model, {k: v[i:i+1] for k, v in c.items()},
                                           steps=2, candidates=2, noise=noise[2*i:2*i+2]) for i in range(2)])
                    torch.testing.assert_close(cached, separate, atol=2e-6, rtol=1e-5)

    def test_training_checkpoint_and_sampling_integration_all_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_fixture(root/'dataset')
            for variant in MODEL_VARIANTS:
                with self.subTest(variant=variant):
                    cfg = run_config(root)
                    cfg.update(max_steps=1, accumulation=1, val_every=1,
                               run_dir=str(root/variant/'run'), checkpoint_dir=str(root/variant/'checkpoints'))
                    cfg['model']['variant'] = variant
                    with patch.object(torch.optim.AdamW, 'step', return_value=None):
                        run(cfg, encoder_factory=FixtureEncoder)
                    checkpoint = Path(cfg['checkpoint_dir'])/'last.pt'
                    model, state = load_model(checkpoint, 'cpu', encoder_factory=FixtureEncoder)
                    self.assertEqual(model.cfg.variant, variant)
                    self.assertEqual(state['model_config']['variant'], variant)
                    dataset = GraspDataset(root/'dataset', 'train', model.cfg.coord_scale_m,
                                           condition_keys=model.cfg.condition_keys)
                    try:
                        self.assertEqual(set(dataset[0]['condition']), set(model.cfg.condition_keys))
                    finally:
                        dataset.close()
                    output = root/variant/'samples.h5'
                    generate(checkpoint, root/'dataset', output, candidates=2, steps=1,
                             encoder_factory=FixtureEncoder)
                    with h5py.File(output) as f:
                        self.assertEqual(f.attrs['model_variant'], variant)
                        self.assertEqual(bool(f.attrs['palm_gap_conditioned']), model.cfg.use_gap)
                        self.assertEqual(f['K'].shape, (2, 2, 21, 3))
                    if variant == 'full':
                        # Old checkpoints omit variant; both loading and warm start must default to full.
                        state['model_config'].pop('variant')
                        legacy = root/'legacy.pt'
                        torch.save(state, legacy)
                        restored, _ = load_model(legacy, 'cpu', encoder_factory=FixtureEncoder)
                        self.assertEqual(restored.cfg.variant, 'full')
                        initialize_checkpoint(restored, legacy, state['data_identity'], state['hand_spec_sha256'])
                        other = KeypointModel(replace(restored.cfg, variant='object_only'), encoder_factory=FixtureEncoder)
                        with self.assertRaisesRegex(ValueError, 'variant'):
                            initialize_checkpoint(other, legacy, state['data_identity'], state['hand_spec_sha256'])

    def test_config_presets_use_equal_budgets_and_separate_outputs(self):
        folder = Path(__file__).resolve().parents[1]/'configs/ablations'
        controls, runs, checkpoints = [], set(), set()
        for variant in MODEL_VARIANTS:
            cfg = load_config(folder/f'{variant}.json')
            self.assertEqual(cfg['model']['variant'], variant)
            self.assertEqual(cfg['initialization'], 'random')
            self.assertEqual(model_config(cfg, {'reference_length_m': .1}).variant, variant)
            runs.add(cfg.pop('run_dir')); checkpoints.add(cfg.pop('checkpoint_dir'))
            cfg['model'].pop('variant')
            controls.append(cfg)
        self.assertEqual(len(runs), 6)
        self.assertEqual(len(checkpoints), 6)
        self.assertTrue(all(c == controls[0] for c in controls))
        with self.assertRaisesRegex(ValueError, 'variant'):
            ModelConfig(**{**asdict(small_config()), 'variant': 'typo'})
        with tempfile.TemporaryDirectory() as temp:
            cfg = json.loads((folder/'full.json').read_text())
            cfg['model']['variant'] = 'typo'
            path = Path(temp)/'bad.json'
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'variant'):
                load_config(path)

    def test_ddp_input_cache_only_contains_active_conditions_and_fps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_fixture(root/'dataset')
            cfg = run_config(root)
            for variant in MODEL_VARIANTS:
                with self.subTest(variant=variant):
                    model = KeypointModel(replace(small_config(), variant=variant), encoder_factory=CachedFixtureEncoder)
                    cache = cache_training_data(cfg, model)
                    expected_keys = set(model.cfg.condition_keys) | {'object_fps_indices'}
                    if model.cfg.use_scene:
                        expected_keys.add('scene_fps_indices')
                    self.assertEqual(set(cache['condition']), expected_keys)
                    batch = cached_batch(cache, [1, 0, 1])
                    x, t = torch.randn(3, 21, 3), torch.full((3,), .3)
                    c = batch['condition']
                    uncached = {k: v for k, v in c.items() if not k.endswith('_fps_indices')}
                    torch.testing.assert_close(model(x, t, model.encode(c)), model(x, t, model.encode(uncached)))
                    FMLossModule(model)(batch).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters()))
