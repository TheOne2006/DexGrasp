import unittest
from dexonomy.preflight import check_device_build
from dexonomy.generation import validate_execution_window


class Portability(unittest.TestCase):
    def test_blackwell_rejects_old_torch_and_wrong_wheel(self):
        for version,cuda,arch in [('2.5.1','12.1',['sm_120']),('2.7.0','12.8',['sm_89'])]:
            with self.assertRaises(ValueError):check_device_build((12,0),version,cuda,arch,'570.86.10')
        check_device_build((12,0),'2.7.0','12.8',['sm_120'],'570.86.10')

    def test_ada_does_not_require_blackwell_wheel(self):
        check_device_build((8,9),'2.5.1+cu121','12.1',['sm_89'],'570.86.10')

    def test_product_can_exceed_task_budget(self):
        validate_execution_window(dict(hard_deadline_epoch=10000,case_limit_s=7200),now=100)

    def test_continuous_is_explicit_and_cannot_inherit_budget(self):
        validate_execution_window(dict(continuous=True),now=100)
        with self.assertRaises(ValueError):validate_execution_window(dict(continuous=True,case_limit_s=600),now=100)
        with self.assertRaises(ValueError):validate_execution_window({},now=100)

    def test_public_imports_are_lightweight(self):
        import subprocess, sys
        code = 'import sys; import dexonomy.prediction_eval, dexonomy.generation, dexonomy.preflight; assert "warp" not in sys.modules and "torch" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], check=True)
