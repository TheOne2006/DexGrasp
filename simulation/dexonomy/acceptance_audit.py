"""Task-local observations; disabled unless DEXONOMY_AUDIT_ROOT is set."""
import json,os,time,hashlib,threading
from pathlib import Path
from dexonomy.preparation_policy import validation_enabled
_seen=set();_lock=threading.Lock();_counter=0

def serial(v):
    if hasattr(v,'tolist'):return v.tolist()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)
def event(kind,**data):
    root=os.environ.get('DEXONOMY_AUDIT_ROOT')
    if not root:return
    path=Path(root);path.mkdir(parents=True,exist_ok=True)
    line=json.dumps({'time':time.time(),'pid':os.getpid(),'kind':kind,**data},default=serial,allow_nan=False)+'\n'
    with _lock:
        with (path/f'events-{os.getpid()}.jsonl').open('a') as f:f.write(line)
def sample(stage,scene,ok,relative,source,result):
    root=os.environ.get('DEXONOMY_AUDIT_ROOT');key=(stage,scene,bool(ok))
    if not validation_enabled() or not root or key in _seen:return
    _seen.add(key)
    import numpy as np
    dest=Path(root)/'samples';dest.mkdir(exist_ok=True)
    name=stage+'-'+hashlib.sha256(scene.encode()).hexdigest()[:12]+'-'+str(int(ok))+'.npy'
    np.save(dest/name,{'stage':stage,'scene_id':scene,'success':bool(ok),'relative':str(relative),'input':source,'result':result})
    event('sample',path=str(dest/name),stage=stage,scene_id=scene,success=bool(ok))
def checkpoints(worlds,poses,report):
    global _counter
    root=os.environ.get('DEXONOMY_AUDIT_ROOT')
    if not validation_enabled() or not root:return
    import numpy as np
    dest=Path(root)/'checkpoints';dest.mkdir(exist_ok=True);name=f'{os.getpid()}-{_counter:06d}';_counter+=1
    extra={k:report[k] for k in ('checkpoint_seen','terminal_qpos','terminal_qvel','terminal_steps') if k in report}
    np.savez(dest/(name+'.npz'),poses=poses,targets=np.stack([x.target_pose for x,d in worlds]),**extra)
    report={k:v for k,v in report.items() if k not in ('terminal_qpos','terminal_qvel')}
    event('eval_checkpoints',path=str(dest/(name+'.npz')),route=[{'id':x.source_name,'scene_path':x.data['scene_path'],'direction':d} for x,d in worlds],report=report)
