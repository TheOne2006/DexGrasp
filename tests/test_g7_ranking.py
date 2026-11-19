import unittest

import torch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rank_g7_candidates import integrate


class RankingTests(unittest.TestCase):
    def test_constant_field_preserves_prior_and_changes_no_density(self):
        initial = torch.arange(12).float().reshape(2, 2, 3)
        probe = torch.ones_like(initial)
        final, prior, change = integrate(lambda x, t: x * 0 + 1, initial, probe, 5)
        torch.testing.assert_close(final, initial + 1)
        torch.testing.assert_close(prior, -0.5 * initial.square().sum((1, 2)))
        torch.testing.assert_close(change, torch.zeros(2))

    def test_linear_field_trace_and_euler_endpoint(self):
        initial = torch.randn(4, 3, 2)
        weights = torch.tensor([[.5, -.25], [2., 0.], [-1., .125]])
        probe = torch.randint(2, initial.shape).float() * 2 - 1
        final, _, change = integrate(lambda x, t: x * weights, initial, probe, 5)
        torch.testing.assert_close(final, initial * (1 + weights / 5)**5)
        torch.testing.assert_close(change, -weights.sum().expand(4))
