"""One authorized production attempt; process-group ownership and idle watchdog."""
import argparse,os,json,time,subprocess,fcntl,signal,hashlib,shutil
from pathlib import Path
from pid_bootstrap import owns_leader
from night_schedule import validate_schedule
from night_inventory import validate_mode
ROOT=Path(__file__).resolve().parents[2]
launcher_started=time.monotonic()
p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('--name',required=True);p.add_argument('--idle-seconds',type=int,default=1800);a=p.parse_args()
c=json.loads(a.config.read_text());BASE=Path(c.get('output_root',ROOT/'output/grouped')).resolve();UUID=c['gpu_uuid']
out=BASE/'runs'/a.name;out.mkdir(parents=True,exist_ok=False);audit=out/'audit';audit.mkdir()
assert len(set(c["locks"])) >= 3, "Three compatibility locks required"
locks=[]
for name in c['locks']:
 f=open(name,'r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
def smi(q):return subprocess.check_output(['nvidia-smi','-i',UUID,q,'--format=csv,noheader,nounits'],text=True).strip()
assert not smi('--query-compute-apps=pid'),'GPU busy';info=smi('--query-gpu=uuid,name,driver_version,memory.total,memory.used');total=int(info.split(',')[-2]);assert int(info.split(',')[-1])<256
available_cpus=set(os.sched_getaffinity(0))
quota=Path('/sys/fs/cgroup/cpu.max')
quota_fields=quota.read_text().split() if quota.exists() else ['max']
quota_count=len(available_cpus) if quota_fields[0]=='max' else max(1,int(quota_fields[0])//int(quota_fields[1]))
cpus=c.get('cpu_ids',sorted(available_cpus)[:min(len(available_cpus),quota_count)])
assert len(cpus)<=quota_count, 'CPU allocation exceeds cgroup quota'
assert cpus and len(cpus)==len(set(cpus)) and set(cpus)<=available_cpus, 'Invalid CPU allocation'
os.sched_setaffinity(0,cpus)
blas_threads=int(c.get('blas_threads',1));assert 1<=blas_threads<=len(cpus)
if c.get('checkpoint_schema') in ('floating-night/v2','tabletop-night/v1','tabletop-single-gpu/v1'):
 validate_mode(c)
 assert not c.get('deployment_template_only',False), 'Deployment template cannot launch'
 assert 0<=c['rank']<c['world_size'] and UUID==c['rank_gpu_uuids'][c['rank']]
 assert len(set(c['rank_gpu_uuids']))==c['world_size'], 'Ranks must use distinct GPUs'
 validate_schedule(c)
 assert c.get('hard_deadline_epoch') is None or c['hard_deadline_epoch']>time.time(), 'Expired run deadline'
 for path,expected in c.get('group_index_sha256',{}).items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, 'Rank bank index changed'

env=os.environ.copy();env.update(PYTHONPATH=str(Path(c['source']).resolve()),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl',XDG_CACHE_HOME=str(BASE/'cache'),CUDA_CACHE_PATH=str(BASE/'cache/cuda'),WARP_CACHE_PATH=str(BASE/'cache/warp'),MPLCONFIGDIR=str(BASE/'cache/mpl'),DEXONOMY_AUDIT_ROOT=str(audit))
env.update(OMP_NUM_THREADS=str(blas_threads),OPENBLAS_NUM_THREADS=str(blas_threads),MKL_NUM_THREADS=str(blas_threads),NUMEXPR_NUM_THREADS=str(blas_threads),DEXONOMY_STOP_REQUEST=str(out/'stop-request.json'))
env['DEXONOMY_HAND_SPEC_CACHE']='1' if c.get('hand_spec_cache',False) else '0'
env.pop('DEXONOMY_QP_CAPTURE',None)
env.pop('DEXONOMY_MJB_CACHE',None)
env['DEXONOMY_VALIDATE_WORLD_ROWS']='1' if c.get('validate_world_rows',False) else '0'
env['DEXONOMY_ENGINEERING_VALIDATION']='1' if c.get('engineering_validation',False) else '0'
if c.get('mjb_cache'):env['DEXONOMY_MJB_CACHE']=str(Path(c['mjb_cache']).resolve())
env['DEXONOMY_DATA_REUSE']='1' if c.get('data_reuse',False) else '0'
if c.get('capture_qp',True):env['DEXONOMY_QP_CAPTURE']=str(out/'qp-inputs')
env['DEXONOMY_PID_RECEIPT']=str(out/'pid-identity.json')
cmd=[c['python'],str(Path(__file__).with_name('pid_bootstrap.py')),str(Path(__file__).with_name(c.get('worker','worker.py'))),str(a.config.resolve()),str(out/'data')]

assert c.get('execution_mode', 'sequential') == 'sequential', 'Use the local sequential generation entry'
package_root=Path(c['source'])/'dexonomy'
assert package_root.is_dir(), 'Source must contain dexonomy package'
source={str(x.relative_to(Path(c['source']))):hashlib.sha256(x.read_bytes()).hexdigest() for x in package_root.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc'};receipt={'argv':cmd,'config':c,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'gpu_before':info,'cpu_affinity':cpus,'environment':{k:env[k] for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','WARP_CACHE_PATH','DEXONOMY_AUDIT_ROOT','DEXONOMY_ENGINEERING_VALIDATION','DEXONOMY_VALIDATE_WORLD_ROWS']},'source_hashes':source,'script_hashes':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*') if x.is_file()},'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'start_epoch':time.time(),'idle_limit_s':a.idle_seconds,'no_total_time_limit':not bool(c.get('case_limit_s')),'case_limit_s':c.get('case_limit_s'),'profile':False}
if c.get('archive_runtime',False):
 archive_started=time.monotonic()
 (out/'source-snapshot').mkdir()
 shutil.copytree(package_root,out/'source-snapshot/dexonomy',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 snapshot={str(x.relative_to(out/'source-snapshot')):hashlib.sha256(x.read_bytes()).hexdigest() for x in (out/'source-snapshot').rglob('*') if x.is_file()}
 assert snapshot==source,'Runtime changed while archiving'
 script_archive=out/'scripts-snapshot';script_archive.mkdir()
 for x in Path(__file__).parent.glob('*'):
  if x.is_file():shutil.copy2(x,script_archive/x.name)
 assert all(hashlib.sha256((script_archive/Path(rel).name).read_bytes()).hexdigest()==h for rel,h in receipt['script_hashes'].items()), 'Scripts changed while archiving'
 receipt['runtime_archive_s']=time.monotonic()-archive_started
receipt['launcher_setup_s']=time.monotonic()-launcher_started
(out/'command.json').write_text(json.dumps(receipt,indent=2)+'\n');started=time.monotonic();last_progress=started;signature=None;reason=None;peak=0;child=None;console_offset=0;console_tail=''
try:
 with (out/'console.log').open('w') as log,(out/'resources.jsonl').open('w') as monitor:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(f.fileno() for f in locks))
  (out/'process.json').write_text(json.dumps({'pid':child.pid,'pgid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split(')')[-1].split()[19]})+'\n')
  identity_deadline=time.monotonic()+40
  while not (out/'pid-identity.json').exists() and child.poll() is None and time.monotonic()<identity_deadline:time.sleep(.1)
  identity=json.loads((out/'pid-identity.json').read_text()) if (out/'pid-identity.json').exists() else {}
  owner=json.loads((out/'process.json').read_text())
  assert identity.get('status')=='PASS' and identity['local_pid']==child.pid and identity['start_ticks']==owner['start_ticks'], 'GPU PID identity not established'
  while child.poll() is None:
   now=time.monotonic();mem=int(smi('--query-gpu=memory.used'));peak=max(peak,mem)
   events=list(audit.rglob('events-*.jsonl'));sig=sum(x.stat().st_size for x in events)+(out/'console.log').stat().st_size
   if sig!=signature:signature=sig;last_progress=now
   members=[]
   for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
     fields=(path/'stat').read_text().split(')')[-1].split()
     if int(fields[2])==child.pid:
      rollup=(path/'smaps_rollup').read_text();pss=next((int(x.split()[1]) for x in rollup.splitlines() if x.startswith('Pss:')),0)
      st=(path/'status').read_text();rss=next((int(x.split()[1]) for x in st.splitlines() if x.startswith('VmRSS:')),0);members.append({'pid':int(path.name),'rss_kib':rss,'pss_kib':pss,'ticks':int(fields[11])+int(fields[12])})
    except (OSError,ValueError,ProcessLookupError):pass
   # NVML uses host PIDs; permit only the challenged, still-owned leader.
   try:leader_fields=Path(f'/proc/{child.pid}/stat').read_text().split(')')[-1].split()
   except FileNotFoundError:leader_fields=[]
   owned=owns_leader(leader_fields,child.pid,identity['start_ticks'])
   apps={int(x) for x in smi('--query-compute-apps=pid').splitlines() if x.strip()}
   from pid_bootstrap import completed_owner_sample
   terminal_code=child.poll()
   if completed_owner_sample(terminal_code,apps,identity['host_pid']):
    (out/'terminal-gpu-sample.json').write_text(json.dumps({'exit_code':terminal_code,'gpu_pids':sorted(apps),'challenged_host_pid':identity['host_pid'],'note':'Child already exited during sample; no unknown PID allowed'}))
    break
   allowed={identity['host_pid']} if owned else set()
   other=apps-allowed
   avail=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
   cg_current_path=Path('/sys/fs/cgroup/memory.current');cg_max_path=Path('/sys/fs/cgroup/memory.max')
   cg_current=int(cg_current_path.read_text()) if cg_current_path.exists() else None
   cg_limit=cg_max_path.read_text().strip() if cg_max_path.exists() else 'max'
   cg_remaining=int(cg_limit)-cg_current if cg_current is not None and cg_limit!='max' else None
   row={'gpu_process_ids':sorted(apps),'owned_host_pid':identity['host_pid'],'ownership_live':owned,'unexpected_gpu_pids':sorted(other),'cgroup_current_bytes':cg_current,'cgroup_headroom_bytes':cg_remaining,'elapsed_s':now-started,'gpu_mib':mem,'members':members,'ram_available_bytes':avail,'shm_free_bytes':shutil.disk_usage('/dev/shm').free,'idle_s':now-last_progress,'audit_bytes':sig};monitor.write(json.dumps(row)+'\n');monitor.flush()
   with (out/'console.log').open(errors='replace') as console:
    console.seek(console_offset);text=console_tail+console.read();console_offset=console.tell()
   console_tail=text[-512:]
   stop_request=json.loads((out/'stop-request.json').read_text()) if (out/'stop-request.json').exists() else {}
   deadline=min(c.get('hard_deadline_epoch') or float('inf'),stop_request.get('hard_deadline_epoch',float('inf')))
   if time.time()>=deadline:reason='wall_clock_deadline'
   elif mem>total-2048:reason='gpu_headroom_less_than_2GiB'
   elif other:reason='resource_competition_'+str(sorted(other))
   elif c.get('case_limit_s') and now-started>c['case_limit_s']:reason='case_time_limit'
   elif now-last_progress>a.idle_seconds:reason='no_progress_timeout'
   elif cg_remaining is not None and cg_remaining<16*1024**3:reason='cgroup_memory_headroom'
   elif avail<16*1024**3 or row['shm_free_bytes']<4*1024**3:reason='host_resource_headroom'
   elif any(x in text for x in ['CUDA_ERROR_OUT_OF_MEMORY','CUDA error 2','OutOfMemoryError','allocation overflow','non-finite state','The simulation is unstable']):reason='runtime_safety_error'
   if reason:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
   time.sleep(2)
  code=child.wait()
finally:
 if child is not None:
  try:os.killpg(child.pid,0)
  except ProcessLookupError:pass
  else:
   os.killpg(child.pid,signal.SIGTERM);time.sleep(1)
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
result={'status':'PASS' if code==0 and reason is None else ('UNKNOWN' if reason=='no_progress_timeout' else 'FAIL'),'exit_code':code,'stop_reason':reason,'wall_s':time.monotonic()-started,'peak_sampled_gpu_mib':peak,'gpu_after':smi('--query-gpu=memory.used'),'end_epoch':time.time(),'note':'Process result only; final output/physics/provenance acceptance separate.'}
result['launcher_total_wall_s']=time.monotonic()-launcher_started
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise SystemExit(0 if result['status']=='PASS' else 1)
