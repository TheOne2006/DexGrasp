"""Contact reduction semantics: missing fingers, palm penetration, self contacts."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))

@unittest.skipUnless(os.getenv('DEXGRASP_GPU_TESTS')=='1','explicit GPU test opt-in')
class QualityGPU(unittest.TestCase):
    def test_world_isolation_and_signed_distances(self):
        import warp as wp
        from quality_kernels import reduce_contacts
        wp.set_device('cuda:0')
        # Geoms: finger0, finger1, palm, object. Hand-self penetration is ignored.
        cats=wp.array([0,1,-1,5],dtype=int)
        geom=wp.array([[0,3],[1,3],[0,1],[2,3]],dtype=wp.vec2i)
        worlds=wp.array([0,0,0,1],dtype=int);dist=wp.array([-.003,.002,-.1,-.004],dtype=float)
        minima=wp.full((3,5),.01,dtype=float);pd=wp.zeros(3,dtype=float)
        wp.launch(reduce_contacts,4,inputs=[wp.array([4],dtype=int),worlds,geom,dist,cats,minima,pd])
        a=minima.numpy();b=pd.numpy()
        np.testing.assert_allclose((a.max(1)-a.min(1))*1000,[13,0,0],atol=1e-5)
        np.testing.assert_allclose(b*1000,[3,4,0],atol=1e-5)

if __name__=='__main__':unittest.main()
