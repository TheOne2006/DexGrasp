import unittest

from scripts.sweep_fake_batches import exact_boundary, standard_boundary


class MicrobatchSweepTests(unittest.TestCase):
    def test_nearest_tier_and_adjacent_oom(self):
        for start, capacity, expected, visits in (
            (1079, 1079, (1024, 1280), [1024, 1280]),
            (976, 976, (640, 1024), [1024, 640]),
            (5952, 5952, (5120, 8192), [5120, 8192]),
            (1152, 640, (640, 1024), [1024, 640]),
            (3, 5, (5, 8), [2, 4, 5, 8]),
            (1, 0, (0, 1), [1]),
        ):
            with self.subTest(start=start):
                tried = []
                def fits(batch):
                    tried.append(batch)
                    return batch <= capacity
                self.assertEqual(standard_boundary(start, fits, fallback_1024=False), expected)
                self.assertEqual(tried, visits)

    def test_legacy_exact_grid(self):
        self.assertEqual(exact_boundary(1024, 32, 1, lambda b: b <= 1079), (1079, 1080))
        self.assertEqual(exact_boundary(1024, 32, 8, lambda b: b <= 976), (976, 984))

    def test_near_1024_fallback_is_used_only_after_1024_oom(self):
        for capacity, expected in ((1024, (1024, 1280)), (1003, (1000, 1024)),
                                   (980, (960, 1000)), (959, (640, 960))):
            with self.subTest(capacity=capacity):
                self.assertEqual(standard_boundary(1024, lambda b: b <= capacity), expected)
