import unittest
from types import SimpleNamespace
from dexgrasp.config import DEFAULT_RUN
from dexgrasp.train.schedule import learning_rate, apply_learning_rate

class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.cfg={**DEFAULT_RUN,'lr_schedule':'warmup_cosine','max_steps':200000,'lr':2e-4,'muon_lr':2e-4}
    def test_boundaries_and_monotonicity(self):
        f=lambda s:learning_rate(self.cfg,s,2e-4)
        self.assertAlmostEqual(f(0),4e-5)
        self.assertAlmostEqual(f(5000),2e-4)
        self.assertAlmostEqual(f(200000),2e-5)
        self.assertLess(f(4999),f(5000));self.assertGreater(f(6000),f(100000))
    def test_hybrid_updates_both_branches_and_resumes_from_step(self):
        self.cfg['optimizer']='muon'
        opt=SimpleNamespace(muon=SimpleNamespace(param_groups=[{'lr':.02}]),adamw=SimpleNamespace(param_groups=[{'lr':.9}]))
        rates=apply_learning_rate(opt,self.cfg,73123)
        self.assertEqual(rates['muon'],rates['adamw'])
        self.assertEqual(opt.muon.param_groups[0]['lr'],rates['muon'])
        apply_learning_rate(opt,self.cfg,1)
        self.assertEqual(apply_learning_rate(opt,self.cfg,73123),rates)
    def test_constant_preserves_different_branch_rates(self):
        self.cfg.update(optimizer='muon',lr_schedule='constant',muon_lr=.02)
        self.assertEqual(learning_rate(self.cfg,10000,.02),.02)
