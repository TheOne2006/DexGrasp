import copy,unittest
from night_capacity_policy import *

def config():
 return dict(rank=2,groups=[{},{}],refine_batch=512,eval_batch=1280,capacity_policy=dict(mode='adaptive/v1',revision='test-v1',refine_ladder=[512,384,256,128,64,32],eval_ladder=[1280,640,320,160,32],window_jobs=20,window_min_jobs=8,deferred_fraction=.5,consecutive_deferred=4,no_progress_failures=12,no_progress_jobs=3,bank_window=20,bank_min_families=2))

class AdaptiveCapacity(unittest.TestCase):
 def setUp(self):self.c=config();self.s={};initialize(self.c,self.s)
 def fail(self,f='a',tag=None):
  return adapt_failure(self.c,self.s,0,f,'refine',tag=tag or f,attempt='attempt'+str(len(self.s['capacity_events'])),reason='CUDA_OOM',input_sha256='sha',input_path='jobs/'+f+'/init.npy',candidate_ids=['id1','id2'])
 def defer(self,f):
  self.s.setdefault('capacity_choices',{})[key(self.c,0,f,'refine')]={'batch':32,'validated':False};return self.fail(f)
 def test_legacy_history_not_reset_or_hard_limit(self):
  old=[dict(event='capacity_failure',key='legacy',attempt=str(i)) for i in range(120)]
  self.s['capacity_history']=copy.deepcopy(old)
  self.fail();self.assertEqual(self.s['capacity_history'][:120],old)
  self.assertEqual(selected(self.c,self.s,0,'a','refine'),384)
 def test_ladder_exhausts_then_deferred_without_completed_feedback(self):
  path=[]
  while not self.s['capacity_deferred']:
   path.append(self.fail()['batch'])
  self.assertEqual(path,[512,384,256,128,64,32]);self.assertNotIn('completed',self.s);self.assertNotIn('feedback_seen_ids',self.s)
  record=self.s['capacity_deferred'][key(self.c,0,'a','refine')]
  self.assertEqual(record['candidate_ids'],['id1','id2']);self.assertTrue(blocked(self.c,self.s,0,'a','a'))
  self.c['capacity_policy']['revision']='test-v2';self.assertTrue(blocked(self.c,self.s,0,'a','a'))
  self.c['capacity_policy']['retry_deferred_keys']=[record['key']]
  self.assertFalse(blocked(self.c,self.s,0,'a','a'));self.assertTrue(blocked(self.c,self.s,0,'a','new-tour-a'))
 def test_learn_bank_only_unvalidated_other_bank_unaffected(self):
  successful(self.c,self.s,0,'healthy','refine',512)
  self.fail('a');self.assertEqual(selected(self.c,self.s,0,'c','refine'),512)
  self.fail('b');self.assertEqual(selected(self.c,self.s,0,'c','refine'),384)
  self.assertEqual(selected(self.c,self.s,0,'healthy','refine'),512)
  self.assertEqual(selected(self.c,self.s,1,'c','refine'),512)
  self.assertEqual(selected(self.c,self.s,0,'c','eval'),1280)
 def test_single_outlier_does_not_drag_bank_to_floor(self):
  self.defer('outlier');self.fail('ordinary')
  self.assertEqual(selected(self.c,self.s,0,'unseen','refine'),384)
  self.assertEqual(selected(self.c,self.s,0,'ordinary','refine'),384)
 def test_recent_consecutive_minimum_failures_stop(self):
  for i in range(3):self.defer(str(i));self.assertIsNone(systemic_reason(self.c,self.s))
  self.defer('3');self.assertEqual(systemic_reason(self.c,self.s),'consecutive_minimum_capacity_failures')
 def test_window_fraction_not_diluted_by_old_success(self):
  for i in range(100):completed(self.c,self.s,'old'+str(i),1)
  for i in range(10):
   self.defer('d'+str(i));completed(self.c,self.s,'ok'+str(i),1)
  self.assertEqual(systemic_reason(self.c,self.s),'recent_minimum_capacity_failure_fraction')
 def test_failure_without_progress_boundary(self):
  self.c['capacity_policy']['bank_min_families']=99
  for f in ['a','b','c']:
   for i in range(4):self.fail(f)
  self.assertEqual(systemic_reason(self.c,self.s),'capacity_failures_without_committed_progress')
  completed(self.c,self.s,'good',1);self.assertIsNone(systemic_reason(self.c,self.s))
 def test_restore_no_double_count_and_fail_closed(self):
  self.fail();restored=copy.deepcopy(self.s)
  with self.assertRaisesRegex(RuntimeError,'already handled'):
   adapt_failure(self.c,self.s,0,'a','refine',tag='a',attempt='attempt0',reason='oom',input_sha256='sha',input_path='input',candidate_ids=[])
  self.assertEqual(restored,self.s)
  completed(self.c,self.s,'good',1);n=len(self.s['capacity_events']);completed(self.c,self.s,'good',1);self.assertEqual(n,len(self.s['capacity_events']))
  for reason in ['cgroup_memory_headroom','ram_available_less_than_16GiB','unknown_pid']:
   self.assertFalse(capacity_failure(dict(status='FAIL',stop_reason=reason),[],'CUDA error 2: out of memory'))
  for text in ['resident refinement non-finite','corrupt checkpoint','allocation overflow','identity mismatch']:
   self.assertFalse(capacity_failure(dict(status='FAIL',stop_reason='runtime_safety_error'),[],text+' CUDA error 2'))
  self.assertFalse(capacity_failure(dict(status='FAIL',stop_reason='gpu_headroom_less_than_2GiB'),[dict(ownership_live=False)],''))
 def test_all_deferred_no_busy_loop(self):
  self.c['groups']=[{}];self.defer('a');self.assertTrue(all_combinations_deferred(self.c,self.s,['a']));self.assertFalse(all_combinations_deferred(self.c,self.s,['a','healthy']))
 def test_recent_bank_window_expires(self):
  self.fail('a');self.fail('b')
  self.s['capacity_events'] += [dict(event='failure',bank='rank002/g000/refine',family='a',next_batch=384,batch=512,tag='a') for i in range(20)]
  self.assertIsNone(learned_start(self.c,self.s,0,'refine'))
if __name__=='__main__':unittest.main()
