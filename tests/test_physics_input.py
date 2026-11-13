import json
import tempfile
import unittest
from pathlib import Path
from dexgrasp.test.physics_input import export_request


class PhysicsExport(unittest.TestCase):
    def test_failed_decode_retained_and_controls_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'decoded.jsonl'
            row = dict(sample_index=0, candidate_index=0, scene_id='scene', ik_success=False)
            source.write_text(json.dumps(row)+'\n')
            result = export_request(source, root/'request.json', protocol='floating', hand_xml='hand.xml',
                joint_names=[f'j{i}' for i in range(22)], scenes={'scene':'scene.npy'}, controls={}, control_source='explicit-v1', frame_transforms={})
            self.assertEqual(len(result['candidates']), 1)
            self.assertIn('invalid_reason', result['candidates'][0])
            row.update(ik_success=True, joint_limits_valid=True)
            source.write_text(json.dumps(row)+'\n')
            with self.assertRaises(KeyError):
                export_request(source, root/'other.json', protocol='floating', hand_xml='hand.xml',
                    joint_names=[f'j{i}' for i in range(22)], scenes={'scene':'scene.npy'}, controls={}, control_source='explicit-v1', frame_transforms={})

    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input.jsonl'
            row=dict(sample_index=0,candidate_index=0,scene_id='s',ik_success=False)
            source.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                export_request(source,root/'out.json',protocol='tabletop',hand_xml='hand.xml',joint_names=[str(i) for i in range(22)],scenes={'s':'s.npy'},controls={},control_source='explicit', frame_transforms={})


class Report(unittest.TestCase):
    def test_unknown_stays_unknown_and_hash_is_checked(self):
        import hashlib
        from dexgrasp.test.physics_report import summarize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);decoded=root/'decoded';report=root/'report'
            decoded.write_text(json.dumps(dict(sample_index=0,candidate_index=0))+'\n')
            value=dict(schema='dexgrasp-physics-result/v1',decoded_sha256=hashlib.sha256(decoded.read_bytes()).hexdigest(),
                protocol='floating-noslip0/v1',backend='cpu',results=[dict(sample_index=0,candidate_index=0,status='simulation_error',physical_success=None)])
            report.write_text(json.dumps(value));result=summarize(decoded,report)
            self.assertEqual(result['physics_coverage'],0);self.assertIsNone(result['physical_success_rate'])
            decoded.write_text(decoded.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'match'):summarize(decoded,report)


class Frames(unittest.TestCase):
    def test_nonidentity_object_and_hand_transform(self):
        import numpy as np
        from dexgrasp.test.physics_input import world_hand_pose
        A=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
        W=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        source=np.eye(3);t=np.array([.1,.2,.3])
        out=world_hand_pose(dict(R=source@A,t=t,q=[0]*22),world_from_input_R=W,
            world_from_input_t=[1,2,3],official_from_decoded_R=A)
        np.testing.assert_allclose(out['R'],W)
        np.testing.assert_allclose(out['t'],W@t+[1,2,3])
