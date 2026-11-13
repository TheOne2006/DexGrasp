"""Pure contracts: padding is computation only; deadlines never alter physics."""
import copy

def execution_plan(count, minimum, maximum):
    if not 1 <= minimum <= maximum or not 0 <= count <= maximum:
        raise ValueError('invalid logical/execution batch bounds')
    return list(range(count)) + ([i % count for i in range(max(0, minimum-count))] if count else [])

def padded_inputs(data, names, minimum, maximum):
    if len(data)!=len(names) or len(set(names))!=len(names):
        raise ValueError('logical IDs must be unique')
    indices=execution_plan(len(data),minimum,maximum)
    if len(indices)>len(data) and not all(x.get('audit_init_hard_all') for x in data):
        raise ValueError('only strict-qualified candidates may be duplicated')
    physical=[copy.deepcopy(data[i]) for i in indices]
    physical_names=[names[i] if slot<len(data) else names[i]+'/__compute_pad_'+str(slot) for slot,i in enumerate(indices)]
    if len(set(physical_names))!=len(physical_names):raise ValueError('padding ID collision')
    return physical,physical_names,indices

def small_batch_allocation(allocation,nworld,limit,per_world):
    allocation=dict(allocation)
    if limit<0 or not 0<=per_world<=4096:raise ValueError('bounded small-batch capacity required')
    if 0<nworld<=limit and per_world:
        allocation['nconmax']=max(allocation['nconmax'],per_world)
        allocation['nccdmax']=max(allocation['nccdmax'],per_world)
    if allocation['nccdmax']>allocation['nconmax']:raise ValueError('CCD per-world capacity exceeds contact capacity')
    return allocation


def constraint_allocation(allocation, requested=0):
    """Explicit per-world constraint storage; never reduce the existing capacity."""
    result=dict(allocation)
    if isinstance(requested,bool) or not isinstance(requested,int):
        raise ValueError('njmax must be an integer')
    if requested:
        if not result['njmax'] <= requested <= 32768:
            raise ValueError('njmax must preserve default capacity and be <=32768')
        result['njmax']=requested
    return result


def contact_allocation(allocation, nconmax=0, nccdmax=0):
    """Explicit storage floors for contact and CCD candidates, not solver settings."""
    result=dict(allocation)
    for key,value in [('nconmax',nconmax),('nccdmax',nccdmax)]:
        if isinstance(value,bool) or not isinstance(value,int):raise ValueError(key+' must be an integer')
        if value:
            if not result[key]<=value<=8192:raise ValueError(key+' must preserve default and be <=8192')
            result[key]=value
    if result['nccdmax']>result['nconmax']:raise ValueError('CCD storage must fit contact storage')
    return result
