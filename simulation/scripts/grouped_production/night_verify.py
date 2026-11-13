"""Validate v2 family archives in addition to existing stage/lineage accounting."""
import argparse,json,hashlib,subprocess,sys,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--family-capacity',type=int,default=512);a=p.parse_args();temp=a.output.with_suffix('.stages.json');subprocess.run([sys.executable,str(Path(__file__).with_name('vnext_verify.py')),str(a.checkpoint),'--output',str(temp),'--allow-interrupted-init'],check=True);s=np.load(a.checkpoint/'state.npy',allow_pickle=True).item();assert s['schema'] in ('floating-night/v2','tabletop-night/v1');ids=[];families={};bundles={}
for family,refs in s['family_archive'].items():
 for ref in refs:
  p=a.checkpoint/ref['path']
  if p not in bundles:
   assert hashlib.sha256(p.read_bytes()).hexdigest()==s['artifacts'][ref['path']];bundles[p]=np.load(p,allow_pickle=True).item()
  b=bundles[p];item=b['prepared'][ref['index']];assert item['hand_path']==ref['id'] and item['tmpl_name']==family;assert float(np.asarray(item['n_evo']).reshape(-1)[0])<=2;ids.append(ref['id'])
 families[family]=len(refs)
buffer=s['template_buffer'];nseed=len(set(buffer['tmpl_name']));assert len(set(buffer['tmpl_name'][:nseed]))==nseed
assert all(Path(x).is_file() for x in buffer['hand_path'][:nseed])
assert all(float(np.asarray(x).reshape(-1)[0])==0 for x in buffer['n_evo'][:nseed])
assert set(buffer['hand_path'][nseed:])<=set(ids)
assert all(n<=a.family_capacity+1 for n in collections.Counter(buffer['tmpl_name']).values())
assert len(ids)==len(set(ids)) and set(ids)==set(s['feedback_seen_ids']);report=json.loads(temp.read_text());report.update(schema=s['schema'],persistent_family_history=families,persisted_feedback_ids=len(ids),active_distribution=dict(collections.Counter(s['template_buffer']['tmpl_name'])));a.output.write_text(json.dumps(report,indent=2));print('PASS archive',families)
