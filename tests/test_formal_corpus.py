import unittest
from pathlib import Path
import numpy as np
from dexgrasp.data.formal_corpus import batched_fk, Groups
from dexgrasp.geometry.hand import URDFHand

class FormalCorpusTests(unittest.TestCase):
    def test_transitive_identity_groups(self):
        groups=Groups(['a','b','c','d']);groups.join('b','c');groups.join('a','b')
        self.assertEqual(groups.root('c'),'a');self.assertEqual(groups.root('d'),'d')
    def test_vectorized_fk_matches_scalar_for_real_hand(self):
        path=Path(__file__).resolve().parents[2]/'data/sharpa-object-20261111/sharpa-object-surface-meanh-20261110-v1/hand/hand_spec.json'
        if not path.exists():self.skipTest('real hand fixture unavailable')
        hand=URDFHand(path)
        q=np.random.default_rng(123).uniform(hand.lower,hand.upper,(8,22))
        np.testing.assert_allclose(batched_fk(hand,q),np.stack([hand.fk(row) for row in q]),atol=1e-12)
