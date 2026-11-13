import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dexonomy.prediction_eval import load_request


class Contract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'input.json'
        pose=dict(R=np.eye(3).tolist(),t=[0,0,0],q=[0]*22)
        self.request=dict(schema='dexgrasp-physics/v1',frame='world',units=dict(length='m',angle='rad'),protocol='floating',hand_pose_frame='official_hand_root',
            joint_names=[str(i) for i in range(22)],hand_xml='hand.xml',decoded_sha256='a'*64,control_source='fixed',
            candidates=[dict(sample_index=0,candidate_index=0,scene='scene.npy',pregrasp=copy.deepcopy(pose),grasp=copy.deepcopy(pose),squeeze=copy.deepcopy(pose))])

    def load(self):
        self.path.write_text(json.dumps(self.request));return load_request(self.path)

    def test_valid_and_relative_paths(self):
        self.assertEqual(self.load()['hand_xml'],str(self.path.parent/'hand.xml'))

    def test_reflection_rejected(self):
        self.request['candidates'][0]['grasp']['R'][0][0]=-1
        with self.assertRaisesRegex(ValueError,'proper rotation'):self.load()

    def test_nonfinite_rejected(self):
        self.request['candidates'][0]['squeeze']['q'][3]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):self.load()

    def test_duplicate_ids_rejected(self):
        self.request['candidates']*=2
        with self.assertRaisesRegex(ValueError,'duplicate'):self.load()

    def test_joint_names_rejected(self):
        self.request['joint_names']=['same']*22
        with self.assertRaisesRegex(ValueError,'unique'):self.load()

    def test_failed_decode_is_not_dropped(self):
        self.request['candidates']=[dict(sample_index=0,candidate_index=0,scene='scene.npy',invalid_reason='failed')]
        self.assertEqual(len(self.load()['candidates']),1)

    def test_all_six_failed_direction_reports(self):
        from dexonomy.op.eval_force_closure_warp import _metrics_from_checkpoints
        poses=np.tile([1,0,0,1,0,0,0],(6,1))
        result=_metrics_from_checkpoints(np.array([0,0,0,1,0,0,0]),[poses]*10,.05,15)[0]
        self.assertFalse(result['passes']);self.assertEqual(len(result['direction_results']),6)
