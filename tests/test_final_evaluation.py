"""Coverage and common-random-number invariants for the full inference shards."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('final_eval',Path(__file__).parents[1]/'scripts/evaluate_final.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FullEvaluationTests(unittest.TestCase):
    def test_partition_and_cube_controls(self):
        from dexgrasp.test.directions import cube_directions
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=np.array([1,3,5,7,9]);np.save(root/'test_rows.npy',rows)
            np.save(root/'object_indices.npy',np.array([0,0,1,1,1]));np.save(root/'direction_rows.npy',np.array([1,5]))
            gt=np.tile([0.,0.,1.],(10,1));np.save(root/'grasp_direction_cmd.npy',gt)
            plan=dict(root=str(root),directions=cube_directions())
            a,b=[m.inputs(plan,root,'test',r) for r in (0,1)]
            self.assertEqual(sorted(np.concatenate([a[0],b[0]]).tolist()),list(range(5)))
            self.assertFalse(set(a[0])&set(b[0]))
            self.assertTrue(np.array_equal(a[1],rows[::2]))
            a,b=[m.inputs(plan,root,'directions',r) for r in (0,1)]
            index=np.concatenate([a[0],b[0]]);order=np.argsort(index)
            packed=np.concatenate([a[1],b[1]])[order];dirs=np.concatenate([a[4],b[4]])[order]
            self.assertEqual(index.size,54)
            self.assertTrue(np.array_equal(packed,np.repeat([1,5],27)))
            for block in dirs.reshape(2,27,3):
                self.assertEqual(len(np.unique(block[:26],axis=0)),26)
                np.testing.assert_allclose(np.linalg.norm(block,axis=1),1,atol=1e-6)
                np.testing.assert_array_equal(block[26],[0,0,1])

    def test_aggregate_retains_failures(self):
        from dexgrasp.test.directions import cube_directions
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'A').mkdir()
            m.save(root/'split_protocol.json',dict(object_groups={'x':'x','y':'y'}))
            m.save(root/'plan.json',dict(root=str(root),objects=['x','y'],directions=cube_directions()))
            for rank in range(2):
                p=root/'A'/f'rank{rank}'/'test';p.mkdir(parents=True)
                a=np.zeros(2,dtype=m.DTYPE)
                a['finite']=True;a['decoded']=[True,False];a['fit']=[True,False];a['direction_pass']=[True,False]
                for k in ('mean_mm','max_mm','angle_deg'):a[k]=[1,np.nan]
                np.save(p/'decoded.npy',a);np.save(p/'keypoints.npy',np.zeros((2,21,3)))
                np.savez(p/'index.npz',objects=np.array([0,1]),commands=np.array([-1,-1]))
                m.save(p/'decoded.done.json',dict(count=2));m.save(p/'generation.json',dict(sampling_seconds=1))
            m.aggregate(root,'A','test');s=m.json.loads((root/'A/test_summary.json').read_text())
            self.assertEqual(s['candidates'],4)
            self.assertEqual(s['macro']['ik_fit_success_rate']['mean'],.5)
            self.assertEqual(s['macro']['mean_keypoint_error_mm']['coverage'],.5)

if __name__=='__main__':unittest.main()
