import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dexgrasp.data.formal_corpus import batched_fk, Groups, verified_packed_identity
from dexgrasp.geometry.hand import URDFHand

class FormalCorpusTests(unittest.TestCase):
    def test_transitive_identity_groups(self):
        groups=Groups(['a','b','c','d']);groups.join('b','c');groups.join('a','b')
        self.assertEqual(groups.root('c'),'a');self.assertEqual(groups.root('d'),'d')
    def test_vectorized_fk_matches_scalar_for_real_hand(self):
        path=Path(__file__).resolve().parents[2]/'data/sharpa-object-20261111/sharpa-object-surface-meanh-20261110-v1/hand/hand_spec.json'
        if not path.exists():
            path=Path(__file__).resolve().parents[2]/'data/sharpa_formal_20261111/local_cache/hand/hand_spec.json'
        if not path.exists():self.skipTest('real hand fixture unavailable')
        hand=URDFHand(path)
        q=np.random.default_rng(123).uniform(hand.lower,hand.upper,(8,22))
        np.testing.assert_allclose(batched_fk(hand,q),np.stack([hand.fk(row) for row in q]),atol=1e-12)
    def test_paired_split_exactly_matches_official_ids(self):
        from helpers import hand_fixture
        from dexgrasp.data.dataset import STAGE_ORDER, official_test_objects, sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hand = URDFHand(hand_fixture(root))
            for family, name in [('DGN_5k','held'), ('objaverse_5k','unseen')]:
                folder = root/family/'valid_split'
                folder.mkdir(parents=True)
                (folder/'test.json').write_text(json.dumps([name]))
            _, hashes = official_test_objects(root)
            protocol = dict(schema='dexgrasp-official-split-v2', audit=dict(full_fk=True,rows=2),
                            samples=dict(train=1,test=1), official_test_sha256=hashes)
            (root/'split_protocol.json').write_text(json.dumps(protocol))
            values = dict(keypoints_gt=np.zeros((2,3,21,3),'f4'), object_points=np.zeros((2,1024,3),'f4'),
                          grasp_direction_cmd=np.ones((2,3),'f4'), h32=np.zeros((2,1),'f4'),
                          observation_object_id=np.array(['train','held']), observation_index=np.arange(2))
            splits = {s:dict(array=s+'_rows',count=1) for s in ('train','test')}
            manifest = dict(schema='dexgrasp-packed-v1', split_status='official_object_ids_v2', stage_order=STAGE_ORDER,
                            coord_scale_m=.2, source_identity='fixture', samples=2, splits=splits,
                            split_protocol_sha256=sha256(root/'split_protocol.json'), hand_spec_sha256=sha256(hand.spec_path),
                            hand_urdf_sha256=sha256(hand.urdf_path), target={'array':'keypoints_gt'},
                            condition=dict(object_points=dict(array='object_points',index='observation_index'),
                                           grasp_direction_cmd=dict(array='grasp_direction_cmd'),h32=dict(array='h32')))
            for train, test in [(0,1),(1,0)]:
                values.update(train_rows=np.array([train]), test_rows=np.array([test]))
                arrays = {}
                for key, val in values.items():
                    path = root/(key+'.npy')
                    np.save(path,val)
                    arrays[key] = dict(path=path.name, dtype=val.dtype.str, shape=val.shape,
                                       bytes=path.stat().st_size,sha256=sha256(path))
                (root/'packed.json').write_text(json.dumps(dict(manifest,arrays=arrays)))
                if train == 0:
                    self.assertEqual(verified_packed_identity(root,.2,paired=True,official_test_root=root)['samples'],protocol['samples'])
                else:
                    with self.assertRaisesRegex(ValueError,'official object intersection'):
                        verified_packed_identity(root,.2,paired=True,official_test_root=root)
