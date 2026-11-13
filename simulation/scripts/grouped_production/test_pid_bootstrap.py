import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('pid_bootstrap',Path(__file__).with_name('pid_bootstrap.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Driver:
 def __getattr__(self,name):return lambda *args:0
class Identity(unittest.TestCase):
 def test_ownership_uses_live_group_and_start_not_optional_rss(self):
  fields=['0']*20;fields[2]='42';fields[19]='123'
  self.assertTrue(m.owns_leader(fields,42,'123'))
  self.assertFalse(m.owns_leader(fields,43,'123'))
  self.assertFalse(m.owns_leader(fields,42,'124'))
  self.assertFalse(m.owns_leader([],42,'123'))
 def test_terminal_sample_never_allows_active_or_unknown_pid(self):
  self.assertTrue(m.completed_owner_sample(75,[496387],496387))
  self.assertTrue(m.completed_owner_sample(0,[],496387))
  self.assertFalse(m.completed_owner_sample(None,[496387],496387))
  self.assertFalse(m.completed_owner_sample(0,[496387,777],496387))
  self.assertFalse(m.completed_owner_sample(0,[777],496387))
 def check(self,rows,success):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m.ctypes,'CDLL',return_value=Driver()),patch.object(m,'gpu_pids',side_effect=rows):
   p=Path(tmp)/'identity.json'
   if success:
    _,_,r=m.establish('GPU-test',p);self.assertEqual(r['host_pid'],496387);self.assertEqual(r['local_pid'],m.os.getpid())
   else:
    with self.assertRaises((AssertionError,RuntimeError)):m.establish('GPU-test',p)
   self.assertEqual(json.loads(p.read_text())['status'],'PASS' if success else 'FAIL')
 def test_namespace_transition(self):self.check([[],[],[496387],[],[],[496387],[],[496387]],True)
 def test_busy_before_cuda(self):self.check([[777]],False)
 def test_real_competitor_during_retain(self):self.check([[],[],[496387,777]],False)
 def test_real_competitor_during_release(self):self.check([[],[],[496387],[777]],False)
 def test_changed_identity(self):self.check([[],[],[496387],[],[],[777]],False)
 def test_final_competitor(self):self.check([[],[],[496387],[],[],[496387],[],[496387,777]],False)
if __name__=='__main__':unittest.main()
