import unittest,copy,tempfile,json
from pathlib import Path
from night_capacity_policy import selected,successful,lower,capacity_failure
from night_capacity_rank import locate
class Capacity(unittest.TestCase):
 def setUp(self):
  self.c=dict(rank=6,refine_batch=512,eval_batch=1280,groups=[{}],capacity_policy=dict(refine_ladder=[512,384,256,128,64],eval_ladder=[1280,640,320,32],max_failures_per_rank=4,max_failures_per_key=3,initial_choices={'rank006/g000/30_Palmar/refine':256}))
 def test_choice_does_not_degrade_healthy_family_or_bank(self):
  s={};self.assertEqual(selected(self.c,s,0,'30_Palmar','refine'),256);self.assertEqual(selected(self.c,s,0,'other','refine'),512)
  successful(self.c,s,0,'other','refine',512);record=lower(self.c,s,0,'30_Palmar','refine',attempt='one',reason='headroom',input_sha256='same');self.assertEqual(record['next_batch'],128)
  self.assertEqual(selected(self.c,s,0,'other','refine'),512);successful(self.c,s,0,'30_Palmar','refine',1492);n=len(s['capacity_history']);successful(self.c,s,0,'30_Palmar','refine',1492);self.assertEqual(n,len(s['capacity_history']))
 def test_fail_closed_and_bounded(self):
  head={'status':'FAIL','stop_reason':'gpu_headroom_less_than_2GiB'}
  self.assertTrue(capacity_failure(head,[{'unexpected_gpu_pids':[]}],''));self.assertFalse(capacity_failure(head,[{'unexpected_gpu_pids':[99]}],''));self.assertFalse(capacity_failure(head,[],'non-finite state'))
  self.assertFalse(capacity_failure(dict(status='FAIL',stop_reason='runtime_safety_error'),[],'allocation overflow'));self.assertTrue(capacity_failure(dict(status='FAIL',stop_reason=None),[],'OutOfMemoryError: CUDA out of memory'))
  s={}
  for i in range(2):lower(self.c,s,0,'30_Palmar','refine',attempt=str(i),reason='headroom',input_sha256='same')
  with self.assertRaises(RuntimeError):lower(self.c,s,0,'30_Palmar','refine',attempt='last',reason='headroom',input_sha256='same')
 def test_localization_requires_original_uncommitted_stage(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'audit').mkdir();cp=p/'checkpoint';job=cp/'jobs/t00-g000-e00-30_Palmar';job.mkdir(parents=True);(job/'init.npy').write_bytes(b'input')
   events=[dict(kind='vnext_job_start',time=1,tag=job.name,group=0,family='30_Palmar'),dict(kind='refine_batch_start',time=2)]
   (p/'audit/events-1.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events));_,stage,f=locate(p,cp);self.assertEqual(stage,'refine');self.assertEqual(f,job/'init.npy');(job/'refine.npy').touch()
   with self.assertRaises(RuntimeError):locate(p,cp)
 def test_tabletop_eval_marker_overrides_refine_stage(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'audit').mkdir();cp=p/'checkpoint';job=cp/'jobs/t00-g000-e00-30_Palmar';job.mkdir(parents=True)
   (job/'init.npy').write_bytes(b'init');(job/'refine.npy').write_bytes(b'refined')
   events=[dict(kind='vnext_job_start',time=1,tag=job.name,group=0,family='30_Palmar'),dict(kind='refine_batch_start',time=2),dict(kind='night_stage_start',stage='eval',time=3)]
   (p/'audit/events-1.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
   _,stage,f=locate(p,cp);self.assertEqual(stage,'eval');self.assertEqual(f,job/'refine.npy')
   (job/'eval.npy').touch()
   with self.assertRaises(RuntimeError):locate(p,cp)
if __name__=='__main__':unittest.main()
