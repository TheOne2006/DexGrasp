"""Bounded CUDA-primary-context challenge to bind local PID to NVML PID.
No arbitrary NVML PID is accepted: empty -> one -> empty, twice, same PID.
This is an operational ownership challenge on an exclusively allocated GPU,
not a security boundary against an adversary deliberately imitating transitions.
"""
import ctypes, json, os, runpy, subprocess, sys, time
from pathlib import Path


def gpu_pids(uuid):
    raw = subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True, timeout=5)
    return sorted({int(x) for x in raw.splitlines() if x.strip()})


def establish(uuid, receipt):
    started=time.monotonic(); observations=[]; candidate=None; retained=False
    lib=ctypes.CDLL('libcuda.so.1')
    def call(name,*args):
        code=getattr(lib,name)(*args)
        if code: raise RuntimeError((name,code))
    def observe(label, expected=None):
        # Allow only bounded NVML teardown visibility delay. Never accept extras.
        end=min(started+25,time.monotonic()+3)
        while True:
            pids=gpu_pids(uuid);observations.append({'phase':label,'pids':pids,'at':time.time()})
            if expected is None:
                if len(pids)==1:return pids[0]
                if pids:raise RuntimeError(('ambiguous GPU ownership',pids))
            elif pids==expected:return None
            elif pids and pids!=[candidate]:raise RuntimeError(('unexpected GPU process',pids))
            if time.monotonic()>=end:raise RuntimeError(('ownership transition timeout',label,pids))
            time.sleep(.1)
    result={'status':'FAIL','local_pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split(')')[-1].split()[19],
            'gpu_uuid':uuid,'proc_status':Path('/proc/self/status').read_text(),'observations':observations}
    ctx=ctypes.c_void_p();dev=ctypes.c_int()
    try:
        assert gpu_pids(uuid)==[], 'GPU not idle before identity probe'
        call('cuInit',0);call('cuDeviceGet',ctypes.byref(dev),0)
        for n in range(2):
            assert gpu_pids(uuid)==[], 'GPU became busy before challenge'
            call('cuDevicePrimaryCtxRetain',ctypes.byref(ctx),dev);retained=True
            seen=observe(f'retain-{n}')
            if candidate is None:candidate=seen
            assert candidate==seen, 'Host PID changed across own context transitions'
            call('cuDevicePrimaryCtxRelease_v2',dev);retained=False
            observe(f'release-{n}',[])
        call('cuDevicePrimaryCtxRetain',ctypes.byref(ctx),dev);retained=True
        assert observe('retain-for-worker')==candidate
        result.update(status='PASS',host_pid=candidate,method='two primary-context retain/release challenges, exclusive GPU')
        return lib,dev,result
    except BaseException:
        if retained:lib.cuDevicePrimaryCtxRelease_v2(dev)
        raise
    finally:
        result['wall_s']=time.monotonic()-started
        temp=receipt.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2));os.replace(temp,receipt)


def main():
    worker=Path(sys.argv[1]).resolve();args=sys.argv[2:]
    receipt=Path(os.environ['DEXONOMY_PID_RECEIPT']);uuid=os.environ['CUDA_VISIBLE_DEVICES']
    lib,dev,_=establish(uuid,receipt)
    try:
        from dexonomy.preflight import inspect_runtime
        preflight = inspect_runtime(gpu=True)
        receipt.with_name('environment-preflight.json').write_text(json.dumps(preflight, indent=2))
        sys.argv=[str(worker),*args];sys.path.insert(0,str(worker.parent));runpy.run_path(str(worker),run_name='__main__')
    finally:lib.cuDevicePrimaryCtxRelease_v2(dev)

def owns_leader(fields,pid,start_ticks):
    return len(fields)>19 and int(fields[2])==pid and fields[19]==start_ticks

def completed_owner_sample(exit_code, gpu_pids, challenged_host_pid):
    """An exited child may remain briefly in NVML; never allow a new PID."""
    return exit_code is not None and not (set(gpu_pids)-{challenged_host_pid})

if __name__=='__main__':main()
