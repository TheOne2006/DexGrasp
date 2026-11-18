from pathlib import Path
import tempfile
import unittest

import numpy as np

from dexgrasp.test.frozen import decoded, digest, save


class FrozenArtifactTests(unittest.TestCase):
    def test_odd_shards_restore_order_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rank in (0, 1):
                folder = root / "A" / f"rank{rank}" / "test"
                folder.mkdir(parents=True)
                indices = np.arange(rank, 5, 2)
                np.save(folder / "decoded.npy", indices)
                np.savez(folder / "index.npz", index=indices)
                save(
                    folder / "decoded.done.json",
                    dict(count=len(indices), sha256=digest(folder / "decoded.npy")),
                )
            np.testing.assert_array_equal(decoded(root, "A"), np.arange(5))
            np.save(root / "A/rank1/test/decoded.npy", [7, 8])
            with self.assertRaisesRegex(ValueError, "Changed decoded poses"):
                decoded(root, "A")
