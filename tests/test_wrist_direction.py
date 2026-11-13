import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dexgrasp.geometry.direction import hand_direction, direction_definition, WRIST_Z
from dexgrasp.geometry.hand import URDFHand, rotation_axis
from dexgrasp.geometry.decode import decode_keypoints
from dexgrasp.data.dataset import GraspDataset, validate_grasp, audit_dataset, sha256
from helpers import dataset_fixture
import h5py


class WristDirectionTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(approach_axis=[0,0,1],
                         official_from_grasp_rotation=[[0,0,1],[1,0,0],[0,1,0]],
                         direction_condition=dict(kind=WRIST_Z, local_axis=[0,1,0]))

    def test_official_axis_sign_and_legacy(self):
        official_from_grasp=np.asarray(self.spec['official_from_grasp_rotation'])
        np.testing.assert_allclose(hand_direction(official_from_grasp, self.spec), [0,0,1])
        np.testing.assert_allclose(hand_direction(official_from_grasp, dict(approach_axis=[0,0,1])), [1,0,0])
        bad=dict(self.spec, direction_condition=dict(kind=WRIST_Z,local_axis=[0,0,1]))
        with self.assertRaises(ValueError): direction_definition(bad)

    def test_translation_independent_orientation_sensitive_and_covariant(self):
        r=rotation_axis([1,0,0], .7)
        # Rotating around the palm normal changes +Zh while preserving the palm normal.
        changed=rotation_axis([0,0,1], .8)
        np.testing.assert_allclose(changed @ [0,0,1], [0,0,1])
        self.assertGreater(np.linalg.norm(hand_direction(changed,self.spec)-hand_direction(np.eye(3),self.spec)), .1)
        world=rotation_axis([0,1,0], .6)
        np.testing.assert_allclose(hand_direction(world@r,self.spec),world@hand_direction(r,self.spec))

    def test_dataset_decoder_and_translation_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dataset_fixture(root)
            path=root/'hand/hand_spec.json'
            spec=json.loads(path.read_text());spec.update(self.spec);path.write_text(json.dumps(spec))
            with h5py.File(root/'shards/part-00000.h5','r+') as shard:
                shard.attrs['hand_spec_sha256']=sha256(path)
                shard['grasps/d'][:]=np.tile([0,1,0],(4,1))
            audit_dataset(root)
            hand=URDFHand(path);ds=GraspDataset(root,'train',.2)
            _,grasp=ds.raw(0)
            np.testing.assert_array_equal(ds[0]['condition']['grasp_direction_cmd'],[0,1,0])
            for t in ([0,0,0],[.4,-.2,.3]):
                rotation=rotation_axis([1,0,0],.6)
                command=hand_direction(rotation,spec)
                result=decode_keypoints(hand.world_keypoints(rotation,t,np.zeros(22)),hand,command)
                self.assertLess(result['direction_error_deg'],1e-5)
                np.testing.assert_allclose(hand_direction(result['R'],spec),command,atol=1e-7)
                np.testing.assert_allclose(result['t'],t,atol=1e-7)
            grasp['d']=np.array([0,0,1],dtype=np.float32)
            with self.assertRaises(ValueError):validate_grasp(grasp,64,[0,1,0])
            ds.close()
