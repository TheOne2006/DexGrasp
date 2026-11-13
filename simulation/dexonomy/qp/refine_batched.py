"""Exact-contact-count batching of the official RELUQP; no padded contacts."""
from collections import defaultdict
import time
import numpy as np
import torch
from dexonomy.qp.qp_batched import build_constraint_batched, build_target_batched, _RELUQP

def build_refine_target(pos, normal, gravity, center, miu_coef):
    # CPU ContactQP uses Y as its first reference axis, Z near parallel.
    # Init's Torch default uses the opposite basis; preserve Refine's pyramid.
    from dexonomy.util.torch_util import torch_normal_to_rot
    rot = torch_normal_to_rot(normal, rot_base1=torch.tensor([0.,1.,0.]),
                               rot_base2=torch.tensor([0.,0.,1.]))
    return build_target_batched(pos, normal, gravity, center, miu_coef, contact_rot=rot)

@torch.no_grad()
def solve_contacts(records, miu_coef, device='cuda:0', chunk_size=32):
    results=[None]*len(records); groups=defaultdict(list)
    for i,(pos,normal,wrench,center) in enumerate(records):
        if len(pos): groups[len(pos)].append(i)
        else: results[i]={'wrench':None,'error':float('inf'),'finite':False,'converged':False}
    for n,indices in groups.items():
        for offset in range(0,len(indices),chunk_size):
            ids=indices[offset:offset+chunk_size]; started=time.perf_counter()
            pos,normal,gravity,center=[torch.as_tensor(np.stack([records[i][j] for i in ids]),dtype=torch.float32,device=device) for j in range(4)]
            A,l,u=build_constraint_batched(len(ids),n,device)
            H,g,mapping=build_refine_target(pos,normal,gravity,center,miu_coef)
            solver=_RELUQP(len(ids),H.shape[-1],A.shape[-2],device)
            # Init tolerances are too loose for the unchanged Refine 1e-3 test.
            solver.eps_abs=1e-6
            x=solver.solve(H,g,A,l,u)
            force=(mapping@x.reshape(len(ids),n,6,1)).squeeze(-1)
            error=(force.sum(1)+gravity).norm(dim=-1)
            ax=(A@x[...,None]).squeeze(-1)
            violation=torch.maximum((l-ax).clamp_min(0).amax(-1),(ax-u).clamp_min(0).amax(-1))
            z=solver.output[:,solver.nx:solver.nx+solver.nc,0]
            lam=solver.output[:,solver.nx+solver.nc:,0]
            primal=(ax-z).abs().amax(-1)
            dual=((H@x[...,None]).squeeze(-1)+(A.transpose(-1,-2)@lam[...,None]).squeeze(-1)+g).abs().amax(-1)
            f,e,v,p,d,params=[t.cpu().numpy() for t in (force,error,violation,primal,dual,x)]
            elapsed=time.perf_counter()-started
            for k,i in enumerate(ids):
                finite=bool(np.isfinite(f[k]).all() and np.isfinite(e[k]) and np.isfinite(v[k]))
                results[i]={'wrench':f[k], 'error':float(e[k]),'finite':finite,
                    'converged':bool(p[k]<solver.eps_abs*solver.nc**.5 and d[k]<solver.eps_abs*solver.nx**.5),
                    'constraint_violation':float(v[k]),'primal_residual':float(p[k]),'dual_residual':float(d[k]),
                    'parameters':params[k], 'group_size':len(ids),'contact_count':n,'amortized_wall_s':elapsed/len(ids)}
    return results
