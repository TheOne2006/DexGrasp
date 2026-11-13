import json
import shutil
from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from dexgrasp.data.dataset import GraspDataset, audit_dataset, read_split, checkpoint_identity_matches, sha256
from dexgrasp.data.prepare import prepare_dataset
from helpers import dataset_fixture


class DataTests(unittest.TestCase):
    def test_deep_audit_receipt_rechecks_contents_and_splits(self):
        from dexgrasp.data.audit_receipt import create_receipt, verified_identity
        receipt = create_receipt(self.root)
        path = self.root/'audit.json'
        path.write_text(json.dumps(receipt))
        self.assertEqual(verified_identity(self.root, .2, path), receipt['identity'])
        with self.assertRaisesRegex(ValueError, 'scale'):
            verified_identity(self.root, .3, path)
        split = self.root/'splits/train.jsonl'
        split.write_text(split.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'split changed'):
            verified_identity(self.root, .2, path)

    def test_batched_training_reads_match_public_dataset_with_repeats(self):
        for keys in (None, ('object_points', 'grasp_direction_cmd')):
            normal = GraspDataset(self.root, 'train', .2, condition_keys=keys)
            fast = GraspDataset(self.root, 'train', .2, condition_keys=keys, training_only=True)
            try:
                indices = [1, 0, 1]
                actual = fast.__getitems__(indices)
                for index, batch in zip(indices, actual):
                    expected = normal[index]
                    self.assertEqual(set(expected['condition']), set(batch['condition']))
                    for key in batch['condition']:
                        torch.testing.assert_close(batch['condition'][key], expected['condition'][key], rtol=0, atol=0)
                    torch.testing.assert_close(batch['keypoints_gt'], expected['keypoints_gt'], rtol=0, atol=0)
                restored = pickle.loads(pickle.dumps(fast))
                self.assertEqual(list(restored.rows), list(normal.rows))
            finally:
                normal.close(); fast.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'dataset'
        dataset_fixture(self.root)

    def test_legacy_split_migration_preserves_identity_and_rejects_changes(self):
        archive = self.root/'split_history/three_way_v1'
        archive.mkdir(parents=True)
        shutil.copy2(self.root/'splits/train.jsonl', archive/'train.jsonl')
        rows = (self.root/'splits/test.jsonl').read_text().splitlines(keepends=True)
        (archive/'val.jsonl').write_text(rows[0])
        (archive/'test.jsonl').write_text(rows[1])
        current = audit_dataset(self.root)
        old = deepcopy(current)
        old['samples'] = {'train': 2, 'val': 1, 'test': 1}
        old['split_sha256'] = {s: sha256(archive/f'{s}.jsonl') for s in old['samples']}
        self.assertTrue(checkpoint_identity_matches(old, current, self.root))
        bad = deepcopy(current)
        bad['hand_spec_sha256'] = 'changed'
        self.assertFalse(checkpoint_identity_matches(old, bad, self.root))
        (self.root/'splits/test.jsonl').write_text(rows[0])
        self.assertFalse(checkpoint_identity_matches(old, current, self.root))
        with self.assertRaisesRegex(ValueError, 'train or test'):
            read_split(self.root, 'val')

    def test_roundtrip_and_common_normalization_with_shared_observation(self):
        ds = GraspDataset(self.root, 'train', .2)
        self.addCleanup(ds.close)
        first, second = ds[0], ds[1]
        torch.testing.assert_close(first['condition']['object_points'], second['condition']['object_points'])
        self.assertEqual(len(ds._observations), 1)
        self.assertEqual(len(ds._handles), 1)
        self.assertEqual(first['condition']['scene_mask'].sum(), 512)
        torch.testing.assert_close(first['condition']['palm_gap'], torch.tensor([.02]))
        torch.testing.assert_close(first['condition']['grasp_direction_cmd'], torch.tensor([0., 0., 1.]))
        obs, grasp = ds.raw(0)
        center = first['metadata']['center_m']
        torch.testing.assert_close(first['condition']['scene_points']*.2+center, torch.from_numpy(obs['scene_points']))
        torch.testing.assert_close(first['keypoints_gt']*.2+center, torch.from_numpy(grasp['K']), atol=1e-8, rtol=1e-5)
        self.assertEqual(audit_dataset(self.root)['samples'], {'train': 2, 'test': 2})

    def test_lazy_handles_are_worker_local_and_pickle_safe(self):
        ds = GraspDataset(self.root, 'train', .2)
        self.addCleanup(ds.close)
        ds[0]
        unpickled = pickle.loads(pickle.dumps(ds))
        self.addCleanup(unpickled.close)
        self.assertFalse(unpickled._handles)
        loader = DataLoader(ds, batch_size=1, num_workers=2)
        self.assertEqual(sum(len(b['keypoints_gt']) for b in loader), 2)
        self.assertEqual(ds[0]['keypoints_gt'].shape, (21, 3))

    def test_cross_object_split_leakage_rejected(self):
        with h5py.File(self.root/'shards/part-00000.h5', 'r+') as f:
            f['observations/object_id'][1] = 'object0'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            audit_dataset(self.root)

    def test_negative_indices_duplicate_refs_and_path_escape_rejected(self):
        path = self.root/'splits/train.jsonl'
        original = path.read_text()
        for data in [dict(shard='shards/part-00000.h5', grasp_index=-1),
                     dict(shard='../outside.h5', grasp_index=0)]:
            path.write_text(json.dumps(data)+'\n')
            with self.assertRaises(ValueError): read_split(self.root, 'train')
        path.write_text(original+original)
        with self.assertRaises(ValueError): read_split(self.root, 'train')

    def test_bad_direction_gap_dtype_and_observation_indices_rejected(self):
        path = self.root/'shards/part-00000.h5'
        for key, value in [('d', [1, 0, 0]), ('h', [-.1]), ('hit_ratio', .5), ('observation_index', -1)]:
            with self.subTest(key=key):
                with h5py.File(path, 'r+') as f:
                    previous = f[f'grasps/{key}'][0]
                    f[f'grasps/{key}'][0] = value
                ds = GraspDataset(self.root, 'train', .2)
                try:
                    with self.assertRaises(ValueError): ds[0]
                finally:
                    ds.close()
                with h5py.File(path, 'r+') as f:
                    f[f'grasps/{key}'][0] = previous
        with h5py.File(path, 'r+') as f:
            data = f['grasps/K'][:].astype(np.float64)
            del f['grasps/K']
            f['grasps/K'] = data
        with self.assertRaisesRegex(ValueError, 'float32'): audit_dataset(self.root)

    def test_hand_spec_change_rejected(self):
        path = self.root/'hand/hand_spec.json'
        path.write_text(path.read_text()+'\n')
        ds = GraspDataset(self.root, 'train', .2)
        self.addCleanup(ds.close)
        with self.assertRaisesRegex(ValueError, 'identity'): ds[0]

    def test_keypoint_labels_must_match_fk(self):
        with h5py.File(self.root/'shards/part-00000.h5', 'r+') as f:
            f['grasps/K'][0] = f['grasps/K'][0]+.01
        with self.assertRaisesRegex(ValueError, 'FK'): audit_dataset(self.root)

    def test_raw_mesh_preparation_and_sharding(self):
        root = Path(self.temp.name)/'raw'
        dataset_fixture(root, raw=True)
        result = prepare_dataset(root, grasps_per_shard=1)
        self.assertEqual(result['rejected'], 0)
        self.assertEqual(len(list((root/'shards').glob('*.h5'))), 4)
        self.assertEqual(audit_dataset(root)['samples']['train'], 2)
        ds = GraspDataset(root, 'train', .2, max_open_shards=1, observation_cache_size=1)
        self.addCleanup(ds.close)
        ds[0]; ds[1]
        self.assertEqual(len(ds._handles), 1)
        self.assertEqual(len(ds._observations), 1)
        with self.assertRaises(FileExistsError): prepare_dataset(root)

    def test_raw_split_leakage_does_not_publish_partial_data(self):
        root = Path(self.temp.name)/'bad_raw'
        dataset_fixture(root, raw=True)
        path = root/'scenes/s1.json'
        scene = json.loads(path.read_text())
        scene['object_id'] = 'object0'
        path.write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError, 'leakage'): prepare_dataset(root)
        self.assertFalse((root/'shards').exists())


if __name__ == '__main__':
    unittest.main()
