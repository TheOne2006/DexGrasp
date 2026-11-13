import unittest
from collections import Counter
import numpy as np
from dexgrasp.test.directions import cube_directions,direction_summary


class DirectionTests(unittest.TestCase):
    def test_commands_cover_all_nonzero_cube_vectors_with_antipodes(self):
        rows=cube_directions();vectors=np.array([r['command_direction'] for r in rows])
        self.assertEqual(Counter(r['direction_group'] for r in rows),{'axis':6,'face_diagonal':12,'space_diagonal':8})
        self.assertEqual(len({r['direction_id'] for r in rows}),26)
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1)
        for vector in vectors:
            self.assertEqual(int((np.linalg.norm(vectors+vector,axis=1)<1e-8).sum()),1)
        for axis in np.eye(3):self.assertTrue((np.linalg.norm(vectors-axis,axis=1)<1e-8).any())
        np.testing.assert_allclose(vectors.sum(0),0,atol=1e-14)

    def test_failed_decodes_remain_in_rate_denominators(self):
        rows=[dict(object_id='a',split='test',R=np.eye(3).tolist(),direction_error_deg=5,ik_success=True,
                   ik_rmse_mm=1,palm_gap_valid=True,palm_gap_error_mm=2,geometric_pass=True),
              dict(object_id='a',split='test',error='degenerate palm',ik_success=False)]
        summary=direction_summary(rows,'test')
        self.assertEqual(summary['samples'],2);self.assertEqual(summary['decoded'],1)
        for key in ('direction_pass_rate','ik_success_rate','palm_gap_coverage','geometric_pass_rate'):
            self.assertEqual(summary[key],.5)
        self.assertEqual(summary['direction_error_deg'],5)
        self.assertIsNone(summary['physical_success_rate'])
        self.assertIsNone(direction_summary(rows,'train')['direction_pass_rate'])


if __name__=='__main__':unittest.main()
