"""Queue transition and failure gates without training or subprocess launches."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('queue_runner',Path(__file__).resolve().parents[1]/'scripts/schedule_experiments.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class SchedulerTests(unittest.TestCase):
    def test_recorded_numerical_failure_blocks_fresh_supervisor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lane,plan=self.fixture(root)
            out=root/'outputs'/'A';out.mkdir(parents=True)
            (out/'numerical_failure.json').write_text('{}')
            with patch.object(s.subprocess,'Popen') as launch:
                self.assertFalse(s.run_lane(lane,plan,root));launch.assert_not_called()

    def test_numerical_failure_does_not_retry_even_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lane,plan=self.fixture(root)
            def fail():
                (root/'A.log').write_text('RuntimeError: total norm is non-finite\n')
                return 1
            with patch.object(s,'checkpoint_step',return_value=90000),patch.object(s.subprocess,'Popen') as launch:
                launch.return_value.pid=123;launch.return_value.wait.side_effect=fail
                self.assertFalse(s.run_lane(lane,plan,root));self.assertEqual(launch.call_count,1)
            self.assertIn('automatic retry disabled',json.loads((root/'lane_AD.json').read_text())['reason'])

    def fixture(self,root):
        runs=[]
        for name in ('A','D'):
            path=root/(name+'.json');path.write_text(json.dumps(dict(run_dir=str(root/'outputs'/name),checkpoint_dir=str(root/'checkpoints'/name),max_steps=200000)))
            runs.append(dict(id=name,config=str(path)))
        return dict(name='lane_AD',gpus=[0,1,2,3],runs=runs),dict(environment={},python='python',source_root=str(root))
    def test_completed_checkpoint_advances_lane(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lane,plan=self.fixture(root)
            with patch.object(s,'checkpoint_step',return_value=200000),patch.object(s.subprocess,'Popen') as launch:
                self.assertTrue(s.run_lane(lane,plan,root));launch.assert_not_called()
            self.assertEqual(json.loads((root/'lane_AD.json').read_text())['status'],'completed')
    def test_failure_without_checkpoint_holds_successor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lane,plan=self.fixture(root)
            with patch.object(s,'checkpoint_step',return_value=None),patch.object(s.subprocess,'Popen') as launch:
                launch.return_value.pid=123;launch.return_value.wait.return_value=1
                self.assertFalse(s.run_lane(lane,plan,root));self.assertEqual(launch.call_count,1)
            state=json.loads((root/'lane_AD.json').read_text())
            self.assertEqual(state['run'],'A');self.assertEqual(state['status'],'failed')
