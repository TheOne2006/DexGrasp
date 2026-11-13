import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from night_inventory import validate_inventory,validate_mode
class TabletopInventory(unittest.TestCase):
 def test_shards_and_all_pose_scales(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);base=r/'base.json';index=r/'index.json';out=r/'ranks'
   base.write_text(json.dumps({'mode':'tabletop','dataset_id':'tabletop-test','capacity_policy':{'refine_ladder':[512,256,128,64],'max_failures_per_rank':4},'capacity_fault_injection':True,'refine_batch':512,'eval_batch':192,'complexity_policy':{'hull_threshold':78,'objects_per_group':1,'capacity':{'refine_batch':128}},'object_hulls':{f'o{i}':100 if i%2 else 2 for i in range(19)}}))
   cases=[dict(object_id=f'o{i}',scene_id=f'o{i}/scale{s}_pose{j}',scale_code=s,official_pose_index=j) for i in range(19) for j in range(i%3+1) for s in ('005','008','011','014','017','020')];index.write_text(json.dumps({'cases':cases}))
   cmd=[sys.executable,str(Path(__file__).with_name('night_configure.py')),str(base),str(index),str(out),'--template-only','--continuous','--scales-per-group','6','--objects-per-group','3'];subprocess.run(cmd,check=True,capture_output=True)
   m=json.loads((out/'sharding.json').read_text());self.assertEqual(m['configurations'],len(cases));owned=set();paths=set();scene_ids=[];cpu=set()
   for row in m['assignments']:
    self.assertFalse(owned.intersection(row['objects']));owned.update(row['objects']);c=json.loads(Path(row['config']).read_text());self.assertEqual(validate_mode(c),'tabletop');self.assertNotIn(c['checkpoint_root'],paths);paths.add(c['checkpoint_root']);self.assertFalse(cpu.intersection(c['cpu_ids']));cpu.update(c['cpu_ids']);self.assertEqual(c['checkpoint_schema'],'tabletop-night/v1');self.assertTrue(c['dataset_id'].startswith('tabletop-test-rank')); self.assertEqual(c['capacity_policy']['refine_ladder'],[512,256,128,64]);self.assertNotIn('capacity_fault_injection',c)
    for g in c['groups']:
     cc=json.loads(Path(g['index']).read_text())['cases'];validate_inventory(cc,'tabletop');scene_ids.extend(x['scene_id'] for x in cc)
   self.assertEqual(set(scene_ids),{x['scene_id'] for x in cases});self.assertEqual(len(scene_ids),len(cases));self.assertEqual(len(owned),19)
 def test_reject_missing_scale_duplicate_and_mismatched_mode(self):
  c=[dict(object_id='a',scene_id=str(i),scale_code=s,official_pose_index=0) for i,s in enumerate(('005','008','011','014','017','020'))]
  validate_inventory(c,'tabletop')
  for bad in (c[:-1],c+[c[0]]):
   with self.assertRaises(AssertionError):validate_inventory(bad,'tabletop')
  with self.assertRaises(AssertionError):validate_mode(dict(mode='tabletop',checkpoint_schema='floating-night/v2'))
if __name__=='__main__':unittest.main()
