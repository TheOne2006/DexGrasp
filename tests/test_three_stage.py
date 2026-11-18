"""Joint stage generation and paired shard regressions. No optimizer updates."""
from dataclasses import asdict, dataclass, replace
import json
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from dexgrasp.config import load_config
from dexgrasp.models.config import BlockConditions, ModelConfig, STAGE_ORDER
from dexgrasp.data.dataset import ThreeStageDataset, audit_dataset, sha256
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.models.registry import MODEL_BRANCHES, ModelBranch
from dexgrasp.test.inference import predict
from dexgrasp.test.physics_input import export_request
from dexgrasp.test.pipeline import generate, decode_file, evaluate
from dexgrasp.train.distributed import FMLossModule, streaming_loader
from dexgrasp.train.objectives import fm_pair, flowmap_loss
from dexgrasp.train.runner import load_model, run
from helpers import FixtureEncoder, condition, dataset_fixture, small_config, run_config


def model_for(variant='object_only', method='fm'):
    cfg = replace(small_config(), stages=3, self_attention_backend='centered_fp32', variant=variant)
    return KeypointModel(cfg, method, encoder_factory=FixtureEncoder)


def paired_shard(root):
    dataset_fixture(root, raw=True)
    spec_path = root/'hand/hand_spec.json'
    spec = json.loads(spec_path.read_text())
    basis = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    spec['official_from_grasp_rotation'] = basis
    spec_path.write_text(json.dumps(spec))
    hand = URDFHand(spec_path)
    rng = np.random.default_rng(124)
    # Four candidates sharing two observations, with distinct stage poses.
    rotations = np.tile(np.eye(3, dtype='f4'), (4, 3, 1, 1))
    rotations[:, 1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t = rng.uniform(0, .05, (4, 3, 3)).astype('f4')
    q = rng.uniform(-.03, .03, (4, 3, 22)).astype('f4')
    poses = np.array([[hand.world_keypoints(*stage) for stage in zip(r, p, j)]
                      for r, p, j in zip(rotations, t, q)], dtype='f4')
    (root/'shards').mkdir()
    (root/'splits').mkdir()
    with h5py.File(root/'shards/part.h5', 'x') as f:
        f.attrs.update(schema_version=2, stage_order=json.dumps(STAGE_ORDER), hand_spec_sha256=sha256(spec_path))
        obs = f.create_group('observations')
        obs['object_points'] = rng.uniform(-.1, .1, (2, 1024, 3)).astype('f4')
        obs['scene_id'], obs['view_id'] = [b's0', b's1'], [b'v0', b'v0']
        g = f.create_group('grasps')
        g['observation_index'] = np.array([0, 0, 1, 1], dtype='i8')
        g['candidate_id'] = [b'a', b'b', b'c', b'd']
        g['R'], g['t'], g['q'], g['keypoints_gt'] = rotations, t, q, poses
        g['grasp_direction_cmd'] = rotations[:, 1, :, 1]
        g['h32'] = np.full((4, 1), .02, dtype='f4')
    for split, ids in [('train', (0, 1)), ('test', (2, 3))]:
        (root/'splits'/f'{split}.jsonl').write_text(''.join(
            json.dumps(dict(shard='shards/part.h5', grasp_index=i))+'\n' for i in ids))
    return hand, poses


class ThreeStageModelTests(unittest.TestCase):
    def test_november_pair_recipes_and_shared_grasp_bias(self):
        root = Path(__file__).resolve().parents[1] / 'configs/paired_1125'
        self.assertEqual({p.stem for p in root.glob('*.json')}, {g+n for g in 'ABCD' for n in '12'})
        absolute = torch.randn(2, 3, 21, 3)
        for path in sorted(root.glob('*.json')):
            cfg = ModelConfig.from_file(path, hand_reference_length_m=.204)
            cfg = replace(cfg, width=32, depth=1, heads=4, tokens=8, backbone_channels=16)
            model = KeypointModel(cfg, encoder_factory=FixtureEncoder)
            name = path.stem
            self.assertEqual(cfg.target, 'absolute' if name in ('A1', 'B1') else 'residual')
            self.assertEqual((cfg.direction_adaln, cfg.geometry_bias, cfg.use_gap),
                             (name in ('B1','B2','C1','D1'), name in ('A1','A2','C2','D2'), name[0] == 'D'))
            state = cfg.training_state(absolute)
            torch.testing.assert_close(cfg.absolute_state(state), absolute)
            if name[1] == '2':
                torch.testing.assert_close(state[:, 0], absolute[:, 0] - absolute[:, 1])
            raw = dict(condition(), h32=torch.full((2, 1), .02))
            batch = dict(condition=raw, keypoints_gt=absolute)
            FMLossModule(model)(batch).backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            if cfg.geometry_bias:
                context = model.encode(raw)
                attention = model.blocks[0].cross[0]
                with patch.object(attention, 'forward', wraps=attention.forward) as call:
                    model(state, torch.tensor([.2,.6]), context)
                    bias = call.call_args.args[2]
                if cfg.stages == 3:
                    pre, grasp, squeeze = bias.chunk(3, 2)
                    torch.testing.assert_close(pre, grasp)
                    torch.testing.assert_close(squeeze, grasp)

    def test_shared_time_joint_attention_and_three_heads(self):
        state = torch.randn(2, 3, 21, 3)
        noise = torch.randn_like(state)
        mixed, time, velocity = fm_pair(state, noise=noise, time=torch.tensor([.2, .8]))
        for b in range(2):
            torch.testing.assert_close(mixed[b], noise[b].lerp(state[b], time[b]))
        torch.testing.assert_close(velocity, state-noise)
        cond = dict(condition(), h32=torch.tensor([[.03], [.04]]))
        for variant in ('object_only', 'object_only_h_less', 'object_direction_adaln', 'object_direction_adaln_h_less'):
            with self.subTest(variant=variant):
                model = model_for(variant)
                inputs = {k: cond[k] for k in model.cfg.condition_keys}
                output = model(mixed, time, model.encode(inputs))
                self.assertEqual(output.shape, state.shape)
                changed = mixed.clone()
                changed[:, 0] += .25
                other = model(changed, time, model.encode(inputs))
                self.assertFalse(torch.allclose(output[:, 2], other[:, 2], atol=1e-7, rtol=1e-6))
                FMLossModule(model)(dict(condition=inputs, keypoints_gt=state)).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for head in model.head:
                    self.assertGreater(head[-1].weight.grad.norm().item(), 0)
                with self.assertRaisesRegex(ValueError, 'shape'):
                    model(state[:, 1], time, model.encode(inputs))

    def test_sampling_candidate_order_and_distillation(self):
        c = dict(condition(), h32=torch.tensor([[.03], [.04]]))
        noise = torch.randn(4, 3, 21, 3)
        teacher = model_for().requires_grad_(False)
        student = model_for(method='flowmap')
        for model in (student, teacher):
            generated = predict(model, c, candidates=2, steps=2, noise=noise)
            self.assertEqual(generated.shape, (2, 2, 3, 21, 3))
            self.assertEqual(model.object_encoder.calls, 1)
            for b in range(2):
                raw = {k: v[b:b+1] for k, v in c.items()}
                expected = predict(model, raw, candidates=2, steps=2, noise=noise[2*b:2*b+2])
                torch.testing.assert_close(expected[0], generated[b])
            shift = torch.tensor([1., -2., .3])
            moved = predict(model, dict(c, object_points=c['object_points'] + shift), candidates=2, steps=2, noise=noise)
            torch.testing.assert_close(moved, generated + shift)
        loss = flowmap_loss(student, teacher, dict(condition=c, keypoints_gt=noise[:2]), steps=2)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        t = torch.tensor([.2, .7])
        torch.testing.assert_close(student(noise[:2], t, student.encode(c), t), noise[:2], atol=0, rtol=0)

    def test_composable_block_conditions(self):
        base = small_config()
        raw = dict(condition(), h32=torch.full((2, 1), .03))
        noise = torch.randn(4, 3, 21, 3)
        for adaln, bias, depth in product((False, True), repeat=3):
            for direction_adaln in ((False, True) if adaln else (False,)):
                switches = BlockConditions(adaln=adaln, direction_adaln=direction_adaln, geometry_bias=bias, depth=depth)
                with self.subTest(switches=switches):
                    cfg = replace(base, variant=None, stages=3, conditioning=switches)
                    model = KeypointModel(cfg, encoder_factory=FixtureEncoder)
                    c = {key: raw[key] for key in cfg.condition_keys}
                    self.assertEqual('grasp_direction_cmd' in c, bias or direction_adaln)
                    self.assertEqual('h32' in c, depth)
                    for block in model.blocks:
                        self.assertEqual(block.modulations is not None, adaln)
                        self.assertEqual(bool(block.biases), bias)
                    self.assertEqual(model.gap is not None, depth)
                    self.assertEqual(model.direction_embedding is not None, direction_adaln)
                    generated = predict(model, c, candidates=2, steps=2, noise=noise)
                    self.assertEqual(generated.shape, (2, 2, 3, 21, 3))
                    loss = FMLossModule(model)(dict(condition=c, keypoints_gt=noise[:2]))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_per_layer_switches_route_only_selected_embeddings(self):
        switches = [dict(adaln=False, geometry_bias=True), dict(direction_adaln=True, depth=True)]
        cfg = replace(small_config(), variant=None, stages=3, conditioning=switches)
        model = KeypointModel(cfg, encoder_factory=FixtureEncoder)
        state, time = torch.randn(2, 3, 21, 3), torch.tensor([.2, .7])
        raw = dict(condition(), h32=torch.full((2, 1), .02))
        layers = []
        for block in model.blocks:
            block.register_forward_hook(lambda module, inputs, output: layers.append(output.detach()))
        def forward(c, t=time):
            with torch.no_grad():
                return model(state, t, model.encode(c))
        expected = forward(raw)
        changed = forward(dict(raw, h32=raw['h32'] + .1))
        torch.testing.assert_close(layers[0], layers[2], atol=0, rtol=0)
        self.assertFalse(torch.allclose(expected, changed))
        forward(raw, 1-time)
        self.assertFalse(torch.allclose(layers[0], layers[4]))
        self.assertEqual([bool(block.biases) for block in model.blocks], [True, False])
        self.assertEqual(ModelConfig(**asdict(cfg)), cfg)
        with self.assertRaisesRegex(ValueError, 'each Transformer block'):
            replace(cfg, conditioning=switches[:1])
        with self.assertRaisesRegex(ValueError, 'requires AdaLN'):
            replace(cfg, conditioning=dict(direction_adaln=True, adaln=False))

    def test_model_files_and_separate_training_config(self):
        folder = Path(__file__).resolve().parents[1]/'configs/three_stage_v2'
        files = sorted(folder.glob('[1-4]_*.json'))
        presets = [ModelConfig.from_file(path, hand_reference_length_m=.204) for path in files]
        self.assertEqual([(mc.geometry_bias, mc.direction_adaln, mc.use_gap) for mc in presets],
                         [(True, False, False), (True, False, True), (False, True, False), (False, True, True)])
        self.assertEqual(ModelConfig.from_file(files[0]).state_shape, (3, 21, 3))
        with self.assertRaisesRegex(ValueError, 'unset V2 training settings.*batch_size'):
            load_config(folder.parent/'train/three_stage_v2.template.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'run.json'
            recipe = run_config(root)
            recipe.pop('model')
            recipe['microbatch_size_per_device'] = recipe.pop('batch_size')
            recipe['gradient_accumulation_steps'] = recipe.pop('accumulation')
            (root/'architecture.json').write_bytes(files[2].read_bytes())
            recipe.update(dataset_format='data_v2', model_config='architecture.json')
            path.write_text(json.dumps(recipe))
            resolved = load_config(path)
            self.assertEqual(resolved['model'], json.loads(files[2].read_text()))
            for key, internal in [('microbatch_size_per_device', 'batch_size'), ('gradient_accumulation_steps', 'accumulation'),
                                  ('lr', 'lr'), ('weight_decay', 'weight_decay'), ('max_steps', 'max_steps')]:
                self.assertEqual(resolved[internal], recipe[key])
                incomplete = dict(recipe)
                del incomplete[key]
                path.write_text(json.dumps(incomplete))
                with self.assertRaisesRegex(ValueError, internal):
                    load_config(path)


class PairedDataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)/'dataset'
        self.hand, self.poses = paired_shard(self.root)

    def test_paired_targets_streaming_and_rejection(self):
        model = model_for('object_direction_adaln_h_less')
        dataset = ThreeStageDataset(self.root, 'train', .2, condition_keys=model.cfg.condition_keys)
        sample = dataset[1]
        self.assertNotIn('h32', sample['condition'])
        self.assertAlmostEqual(sample['h32'].item(), .02)
        np.testing.assert_allclose(sample['keypoints_gt'].numpy() * .2 + sample['metadata']['center_m'].numpy(),
                                   self.poses[1], atol=1e-7)
        cfg = dict(run_config(self.root.parent), dataset_format='data_v2')
        loader = streaming_loader(cfg, model, 1, 0, 0, 1)
        batch = next(iter(loader))
        self.assertEqual(batch['keypoints_gt'].shape, (2, 3, 21, 3))
        samples = loader.dataset.__getitems__([1, 0, 1])
        for sample_index, source_index in enumerate([1, 0, 1]):
            torch.testing.assert_close(samples[sample_index]['keypoints_gt'], dataset[source_index]['keypoints_gt'])
        dataset.close()
        loader.dataset.close()
        identity = audit_dataset(self.root, dataset_format='data_v2')
        self.assertEqual(identity['stage_order'], list(STAGE_ORDER))
        path = self.root/'shards/part.h5'
        with h5py.File(path, 'r+') as f:
            f.attrs['stage_order'] = json.dumps(list(reversed(STAGE_ORDER)))
        with self.assertRaisesRegex(ValueError, 'stage_order'):
            audit_dataset(self.root, dataset_format='data_v2')
        with h5py.File(path, 'r+') as f:
            f.attrs['stage_order'] = json.dumps(STAGE_ORDER)
            f['grasps/keypoints_gt'][0, 2, 1, 0] += .01
        with self.assertRaisesRegex(ValueError, 'FK'):
            audit_dataset(self.root, dataset_format='data_v2')

    def test_official_test_object_excluded(self):
        official = self.root.parent/'official'
        for release in ('DGN_5k', 'objaverse_5k'):
            folder = official/release/'valid_split'
            folder.mkdir(parents=True)
            (folder/'test.json').write_text(json.dumps(['object0']))
        with self.assertRaisesRegex(ValueError, 'official test objects'):
            audit_dataset(self.root, dataset_format='data_v2', official_test_root=official)

    def test_another_family_uses_its_own_model_and_sampler(self):
        # Test-only FM branch with its own config, parameters and deterministic sampler.
        @dataclass
        class FixtureConfig:
            family: str = 'fixture_flow'
            gain: float = .5
            coord_scale_m: float = .2
            condition_keys = ('object_points',)
            checkpoint_schema = 'dexgrasp-fixture'

        class FixtureModel(torch.nn.Module):
            def __init__(self, cfg, method):
                super().__init__()
                self.cfg, self.method = cfg, method
                self.scale = torch.nn.Parameter(torch.tensor(cfg.gain))

            def encode(self, condition):
                return condition['object_points'].mean(1)

            def forward(self, state, time, context):
                return self.scale * state + context[:, None, None]

        def sample(model, condition, *, steps, candidates, noise=None):
            size = len(condition['object_points']) * candidates
            return torch.full((size, 3, 21, 3), steps * model.scale.item())

        branch = ModelBranch(FixtureConfig, FixtureModel, sample, {'fm': 7})
        cfg = run_config(self.root.parent)
        cfg.update(dataset_format='data_v2', model={'family': 'fixture_flow'}, max_steps=1)
        path = self.root.parent/'fixture.json'
        path.write_text(json.dumps(cfg))
        with patch.dict(MODEL_BRANCHES, fixture_flow=branch), patch.object(torch.optim.AdamW, 'step', return_value=None):
            run(load_config(path))
            checkpoint = Path(cfg['checkpoint_dir'])/'last.pt'
            loaded, _ = load_model(checkpoint, 'cpu')
            self.assertIsInstance(loaded, FixtureModel)
            points = torch.zeros(2, 1024, 3)
            pred = predict(loaded, {'object_points': points}, candidates=2)
            torch.testing.assert_close(pred, torch.full((2, 2, 3, 21, 3), .7))
            output = self.root.parent/'branch.h5'
            generate(checkpoint, self.root, output, candidates=2)
            with h5py.File(output) as f:
                self.assertEqual(f.attrs['model_family'], 'fixture_flow')
                self.assertEqual(f.attrs['steps'], 7)
                self.assertFalse(f.attrs['direction_conditioned'])

    def test_single_device_checkpoint_round_trip_without_optimizer_updates(self):
        cfg = run_config(self.root.parent)
        cfg.update(dataset_format='data_v2', max_steps=1)
        architecture = cfg.pop('model')
        architecture.update(stages=3, self_attention_backend='centered_fp32', conditioning=dict(geometry_bias=True,
                            direction_adaln=True, depth=True))
        model_file = self.root.parent/'model.json'
        model_file.write_text(json.dumps(architecture))
        cfg['model_config'] = model_file.name
        run_file = self.root.parent/'run.json'
        run_file.write_text(json.dumps(cfg))
        cfg = load_config(run_file)
        with patch.object(torch.optim.AdamW, 'step', return_value=None):
            run(cfg, encoder_factory=FixtureEncoder)
        model, state = load_model(Path(cfg['checkpoint_dir'])/'last.pt', 'cpu', encoder_factory=FixtureEncoder)
        self.assertEqual(model.cfg.state_shape, (3, 21, 3))
        self.assertEqual(state['schema'], 'dexgrasp-v2')
        self.assertEqual(model.cfg.conditioning, BlockConditions(geometry_bias=True, direction_adaln=True, depth=True))

    def test_checkpoint_sampling_decode_and_controls(self):
        identity = audit_dataset(self.root, dataset_format='data_v2')
        identity['hand_urdf_sha256'] = sha256(self.hand.urdf_path)
        model = model_for()
        checkpoint = self.root.parent/'random.pt'
        torch.save(dict(schema='dexgrasp-v2', method='fm', model=model.state_dict(), model_config=asdict(model.cfg),
                        hand_spec_sha256=sha256(self.hand.spec_path), data_identity=identity), checkpoint)
        output = self.root.parent/'samples.h5'
        generate(checkpoint, self.root, output, candidates=1, steps=2, palm_gap_m=.05, encoder_factory=FixtureEncoder)
        with h5py.File(output, 'r+') as f:
            self.assertEqual(f['K'].shape, (2, 1, 3, 21, 3))
            self.assertEqual(json.loads(f.attrs['stage_order']), list(STAGE_ORDER))
            np.testing.assert_allclose(f['h32_cmd'][:], .05)
            np.testing.assert_allclose(f['h32_gt'][:], .02)
            # Feed known poses and one degenerate squeeze to assess decoding independently of random weights.
            f['K'][:, 0] = self.poses[2:]
            f['K'][1, 0, 2] = 0.
        decoded = self.root.parent/'poses.jsonl'
        decode_file(output, decoded)
        rows = [json.loads(line) for line in decoded.read_text().splitlines()]
        self.assertTrue(rows[0]['ik_success'])
        self.assertFalse(rows[1]['ik_success'])
        self.assertTrue(rows[1]['stages']['grasp']['ik_success'])
        self.assertFalse(rows[1]['stages']['squeeze']['ik_success'])
        self.assertLess(rows[0]['direction_error_deg'], .01)
        report = evaluate(decoded)
        self.assertEqual(report['candidates'], 2)
        self.assertEqual(report['stage_ik_success_rate']['squeeze'], .5)
        self.assertEqual(report['raw_direction_pass_rate'], 1.)
        self.assertEqual(report['decoded_direction_pass_rate'], 1.)
        transform = dict(world_from_input_R=np.eye(3).tolist(), world_from_input_t=[.5, 0, 0],
                         official_from_decoded_R=self.hand.spec['official_from_grasp_rotation'])
        req = export_request(decoded, self.root.parent/'controls.json', protocol='floating', hand_xml='fixture.xml',
                             joint_names=self.hand.spec['joint_order'], scenes={'s1': 's1.npy'}, frame_transforms={'s1': transform})
        first, failed = req['candidates']
        for name in STAGE_ORDER:
            np.testing.assert_allclose(first[name]['t'], np.array(rows[0]['stages'][name]['t']) + [.5, 0, 0])
        self.assertIn('invalid_reason', failed)
