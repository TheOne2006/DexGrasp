import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from helpers import hand_fixture
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.data.retarget import retarget, SHADOW_FROM_SHARPA

class RetargetTests(unittest.TestCase):
    def test_bounded_fit_and_world_frame_equivariance(self):
        with tempfile.TemporaryDirectory() as folder:
            hand=URDFHand(hand_fixture(folder))
            class Source:
                def __init__(self,R,t):self.points=hand.world_keypoints(R,t,np.full(22,.15))
                def world(self,q):return self.points
            R=Rotation.from_euler('xyz',[.3,-.2,.8]).as_matrix();t=np.array([.3,-.1,.2])
            src=np.r_[t,Rotation.from_matrix(R@SHADOW_FROM_SHARPA.T).as_quat(scalar_first=True),np.zeros(22)]
            result=retarget(hand,Source(R,t),src)
            self.assertTrue(result['optimizer_success'])
            self.assertTrue(hand.in_limits(result['q']))
            self.assertLess(result['mpjpe_m'],1e-4)
            np.testing.assert_allclose(result['R'],R,atol=1e-12)
            other_R=Rotation.from_euler('xyz',[-.6,.1,.2]).as_matrix();other_t=np.array([-.2,.4,.5])
            src2=np.r_[other_t,Rotation.from_matrix(other_R@SHADOW_FROM_SHARPA.T).as_quat(scalar_first=True),np.zeros(22)]
            result2=retarget(hand,Source(other_R,other_t),src2)
            np.testing.assert_allclose(result['q'],result2['q'],atol=1e-7)
            np.testing.assert_allclose((result['K']-t)@R,(result2['K']-other_t)@other_R,atol=1e-8)
            self.assertNotIn('physical_success',result)
