"""One bounded allocation cache; reset using MuJoCo Warp's official API."""
import os,time
from dexonomy.acceptance_audit import event
_SLOT=None

def acquire(model,warp_model,*,nworld,**allocation):
    global _SLOT
    import mujoco_warp as mjw
    import warp as wp
    begin=time.perf_counter()
    if os.environ.get('DEXONOMY_DATA_REUSE')!='1':
        return mjw.make_data(model,nworld=nworld,**allocation)
    key=(id(model),nworld,tuple(sorted(allocation.items())))
    hit=_SLOT is not None and _SLOT[0]==key and _SLOT[1] is model
    if hit:
        data=_SLOT[2];mjw.reset_data(warp_model,data)
    else:
        _SLOT=None
        import gc
        gc.collect()
        data=mjw.make_data(model,nworld=nworld,**allocation)
        _SLOT=(key,model,data)
    wp.synchronize()
    event('data_workspace',hit=hit,nworld=nworld,allocation=allocation,wall_s=time.perf_counter()-begin)
    return data

def clear():
    global _SLOT
    _SLOT=None
