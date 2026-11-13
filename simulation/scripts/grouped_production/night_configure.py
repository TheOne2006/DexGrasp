"""Generate deterministic object shards. Never launch processes or submit ACP."""
import argparse,hashlib,json,os,time,datetime
from pathlib import Path
from night_inventory import SCHEMAS, scale_of, validate_inventory
p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('index',type=Path);p.add_argument('destination',type=Path);p.add_argument('--world-size',type=int,default=8);p.add_argument('--gpu-uuids',nargs='+');p.add_argument('--template-only',action='store_true');p.add_argument('--deadline-utc',help='Explicit ISO UTC timestamp; required for executable configurations');p.add_argument('--drain-seconds',type=int,default=600);p.add_argument('--tours',type=int,default=2);p.add_argument('--cpu-budget',type=int);p.add_argument('--objects-per-group',type=int,default=100);p.add_argument('--continuous',action='store_true');p.add_argument('--scales-per-group',type=int,choices=[2,6],default=2);a=p.parse_args()
assert a.world_size>0 and a.tours>0 and a.drain_seconds>0 and a.objects_per_group>0
assert not a.destination.exists(),'Fresh destination required'
c=json.loads(a.base.read_text());cases=json.loads(a.index.read_text())['cases'];objects=sorted({x['object_id'] for x in cases});assert len(objects)>=a.world_size
assert len({x['scene_id'] for x in cases})==len(cases), 'Duplicate scene ID in source index'
if a.template_only:uuids=[f'REPLACE_GPU_UUID_RANK_{i}' for i in range(a.world_size)]
else:
 uuids=a.gpu_uuids;assert uuids and len(uuids)==a.world_size and len(set(uuids))==a.world_size and all(x.startswith('GPU-') for x in uuids)
 assert a.deadline_utc or a.continuous,'Choose an explicit deadline or continuous mode'
assert not (a.continuous and a.deadline_utc),'Continuous mode has no deadline'
allowed=sorted(os.sched_getaffinity(0));quota=Path('/sys/fs/cgroup/cpu.max').read_text().split();quota_cores=len(allowed) if quota[0]=='max' else max(1,int(quota[0])//int(quota[1]));budget=min(len(allowed),quota_cores,a.cpu_budget or quota_cores);per_rank=min(8,budget//a.world_size);assert per_rank>=1,'Insufficient CPU budget for independent ranks'
parsed_deadline=datetime.datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')) if a.deadline_utc else None
if parsed_deadline is not None:assert parsed_deadline.tzinfo is not None, 'Deadline must include UTC offset'
deadline=parsed_deadline.timestamp() if parsed_deadline else None
if deadline is not None:assert deadline>time.time()+a.drain_seconds
mode=c.get('mode','floating')
validate_inventory(cases,mode)
root=a.destination.resolve();root.mkdir(parents=True);assignments=[];sets=[]
for rank in range(a.world_size):
 owned=objects[rank::a.world_size];sets.append(set(owned));selected=[x for x in cases if x['object_id'] in sets[-1]];base=root/f'rank-{rank:03d}';base.mkdir();groups=[];group_scenes=[]
 # Complexity tiers change capacity only, never object ownership or coverage.
 tiers=[(owned,a.objects_per_group,{})]
 if c.get('complexity_policy'):
  policy=c['complexity_policy'];hulls=c['object_hulls']
  assert set(owned)<=set(hulls)
  normal=[obj for obj in owned if hulls[obj]<=policy['hull_threshold']]
  complex_objects=[obj for obj in owned if hulls[obj]>policy['hull_threshold']]
  tiers=[(normal,a.objects_per_group,{}),(complex_objects,policy['objects_per_group'],policy['capacity'])]
 for inventory,group_size,capacity in tiers:
  assert group_size>0
  for offset in range(0,len(inventory),group_size):
   block=set(inventory[offset:offset+group_size])
   for scales in ([('005','008','011','014','017','020')] if a.scales_per_group==6 else [('005','020'),('008','017'),('011','014')]):
    subset=[x for x in selected if x['object_id'] in block and scale_of(x,mode) in scales]
    if not subset:continue
    group_scenes.extend(x['scene_id'] for x in subset)
    path=base/f'bank-{len(groups):03d}.json';path.write_text(json.dumps({'cases':subset},indent=2));groups.append(dict(index=str(path),**capacity))
 assert len(group_scenes)==len(set(group_scenes))==len(selected) and set(group_scenes)=={x['scene_id'] for x in selected}, 'Unsupported or missing scale mapping'
 rank_lock=base/'rank.lock';rank_lock.touch();gpu_lock='/tmp/dexonomy-'+uuids[rank].lower()+'.lock'
 if not a.template_only:Path(gpu_lock).touch(exist_ok=True)
 cfg=dict(c,worker='night_worker.py',mode=mode,checkpoint_schema=SCHEMAS[mode],rank=rank,world_size=a.world_size,rank_gpu_uuids=uuids,gpu_uuid=uuids[rank],groups=groups,output_root=str(base),checkpoint_root=str(base/'checkpoint'),dataset_id=c.get('dataset_id',root.name)+f'-rank{rank:03d}',locks=[str(rank_lock),gpu_lock],cpu_ids=allowed[rank*per_rank:(rank+1)*per_rank],blas_threads=1,loader_workers=1,family_capacity=512,refine_min_execution_slots=32,refine_small_batch_limit=32,refine_small_batch_capacity=2048,hard_deadline_epoch=deadline,stop_new_init_epoch=deadline-a.drain_seconds if deadline else None,continuous=a.continuous,tours=None if a.continuous else a.tours,max_group_visits=None if a.continuous else a.tours*len(groups),deployment_template_only=a.template_only,engineering_validation=False,validate_world_rows=False)
 for key in list(cfg):
  if key.startswith(('tail_','capacity_')) and key not in ('capacity_policy','capacity_memory_audit'):cfg.pop(key)
 cfg.pop('case_limit_s',None);cfg.pop('stop_after_jobs',None);cfg.pop('stop_after_stage',None)
 cfg['group_index_sha256']={g['index']:hashlib.sha256(Path(g['index']).read_bytes()).hexdigest() for g in groups}
 config=base/'config.json';config.write_text(json.dumps(cfg,indent=2));assignments.append({'rank':rank,'objects':owned,'configurations':len(selected),'config':str(config),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'gpu_uuid':uuids[rank],'cpu_ids':cfg['cpu_ids']})
assert set.union(*sets)==set(objects) and sum(map(len,sets))==len(objects)
assert all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i))
assert sum(x['configurations'] for x in assignments)==len(cases)
(root/'sharding.json').write_text(json.dumps({'status':'PASS','source_index':str(a.index.resolve()),'source_sha256':hashlib.sha256(a.index.read_bytes()).hexdigest(),'mode':mode,'checkpoint_schema':SCHEMAS[mode],'world_size':a.world_size,'object_count':len(objects),'configurations':len(cases),'cpu_quota_cores':quota_cores,'cpu_budget':budget,'assigned_cpu_cores':per_rank*a.world_size,'assignments':assignments,'boundary':'CPU partition proof only; no multi-GPU hardware execution. Template placeholders cannot launch.'},indent=2));print('PASS',len(objects),len(cases),'ranks',a.world_size,'CPU/rank',per_rank)
