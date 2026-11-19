"""Score the joint conditional grasp and two residual flows before physics or IK."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from evaluate_g7_anchors import BASE, FROZEN, OLD, REPO
from dexgrasp.models.checkpoint import load_model
from dexgrasp.models.sampling import repeat_context
from dexgrasp.test.frozen import digest, save
from dexgrasp.train.execution import configure_execution

SEED = 628153
ARMS = ('random', 'paper', 'released')


def integrate(field, initial, probe, steps):
    """Gaussian log prior (up to a constant) and signed CNF log-density change."""
    x = initial.float()
    prior = -0.5 * x.flatten(1).square().sum(1)
    change = torch.zeros_like(prior)
    for step in range(steps):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            t = x.new_full((len(x),), step / steps)
            velocity = field(x, t).float()
            vjp, = torch.autograd.grad((velocity * probe).sum(), x)
        with torch.no_grad():
            change -= (probe * vjp).flatten(1).sum(1) / steps
            x = x + velocity / steps
    return x.detach(), prior, change


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    plan = json.loads((OLD/'official/plan.json').read_text())
    rng = np.random.default_rng(SEED)
    selected = []
    for dataset in ('DGN_5k','objaverse_5k'):
        keys = sorted({s['object'] for s in plan['scenes'] if s['source']==dataset})
        objects = set(rng.choice(keys,32,replace=False))
        selected.extend(i for i,s in enumerate(plan['scenes']) if s['source']==dataset and s['object'] in objects)
    assert len(selected)==384
    protocol = dict(seed=SEED,scene_ids=selected,objects_per_catalog=32,
                    reference_plan=digest(OLD/'official/plan.json'),steps=5,pool=100,retained=10,
                    direction='one fixed uniform direction per scene, first archived direction',
                    score='sum of three conditional flow log densities; frozen G7 encoder conditioned on actual generated grasp',
                    arms=list(ARMS),paper='https://arxiv.org/html/2606.09314v1',
                    released_solver=digest(BASE/'reference/KPGrasp/src/dexgrasp/network/ode_solver.py'),
                    implementation=digest(__file__))
    path=out/'design.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Frozen ranking design changed')
    save(path,protocol)


def configure(path,expected):
    model,state=load_model(path,torch.device('cuda:0'))
    if state['step']!=expected:
        raise ValueError('Unexpected checkpoint step')
    configure_execution(model,dict(state['config'],compile_transformer=False))
    return model.eval().requires_grad_(False)


def generate(args):
    torch.set_num_threads(1)
    design=json.loads((args.output/'design.json').read_text())
    ids=design['scene_ids'][args.rank::2]
    if args.limit:
        ids=ids[:args.limit]
    points=np.load(OLD/'official/points.npy',mmap_mode='r')
    dirs=np.load(OLD/'official/directions.npy',mmap_mode='r')
    g7=configure(args.checkpoint,args.step)
    g7_hash=digest(args.checkpoint)
    jobs=json.loads((FROZEN/'plan.json').read_text())['jobs']
    for job in jobs:
        group=job['group']
        if group not in args.groups:
            continue
        dest=args.output/group
        dest.mkdir(exist_ok=True)
        if digest(job['checkpoint'])!=job['checkpoint_sha256']:
            raise ValueError('Anchor checksum mismatch')
        model=configure(job['checkpoint'],200000)
        scale=model.cfg.coord_scale_m
        assert g7.cfg.coord_scale_m==scale
        protocol=dict(design=digest(args.output/'design.json'),source=digest(__file__),grasp=job['checkpoint_sha256'],
                      g7=g7_hash,step=args.step,rank=args.rank,scenes=ids)
        marker=dest/f'protocol-{args.rank}.json'
        if marker.exists() and json.loads(marker.read_text())!=protocol:
            raise ValueError('Ranking resume mismatch')
        save(marker,protocol)
        for sid in ids:
            path=dest/f'{sid}.npz'
            if path.exists():
                continue
            tick=time.perf_counter()
            rng=torch.Generator().manual_seed(SEED+sid)
            init=torch.randn(100,3,21,3,generator=rng).cuda()
            probe=(torch.randint(0,2,init.shape,generator=rng)*2-1).float().cuda()
            random=torch.randperm(100,generator=rng)[:10].numpy()
            pts=torch.from_numpy(points[sid].copy()).cuda()[None]
            center=pts.double().mean(1,keepdim=True).float()
            condition=dict(object_points=(pts-center)/scale,
                           grasp_direction_cmd=torch.from_numpy(dirs[10*sid].copy()).cuda()[None])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.no_grad():
                    main=repeat_context(model.encode(condition,cache_kv=True),100)
                    context=repeat_context(g7.encode(condition,cache_kv=True),100)
                grasp,prior,change=integrate(lambda x,t:model(x,t,main),init[:,1],probe[:,1],5)
                with torch.no_grad():
                    hidden=g7.grasp.features(grasp,grasp.new_ones(100),context)
                stages=[None,grasp,None]
                for j,flow in [(0,g7.pregrasp),(2,g7.squeeze)]:
                    with torch.no_grad():
                        features=flow.prepare(hidden,context.tokens[0],True)
                    residual,p,d=integrate(lambda x,t:flow(x,t,grasp,features),init[:,j],probe[:,j],5)
                    stages[j]=grasp+residual
                    prior+=p
                    change+=d
            scores=(prior+change).cpu().numpy()
            released=change.cpu().numpy()
            if not np.isfinite(scores).all() or not np.isfinite(released).all():
                raise FloatingPointError('Invalid candidate ranking')
            selected=np.stack([random,np.argsort(-scores,kind='stable')[:10],
                               np.argsort(-released,kind='stable')[:10]])
            unique=np.unique(selected)
            keypoints=(torch.stack(stages,dim=1)[unique]*scale+center).cpu().numpy()
            with path.with_suffix('.partial').open('wb') as f:
                np.savez(f,selected=selected,unique=unique,keypoints=keypoints,paper=scores,
                         released=released,prior=prior.cpu().numpy(),direction=dirs[10*sid],
                         seconds=time.perf_counter()-tick)
            path.with_suffix('.partial').replace(path)
        del model


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','generate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--step',type=int)
    p.add_argument('--rank',type=int,choices=[0,1])
    p.add_argument('--groups',default='ABCD')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.action=='prepare':
        prepare(args.output)
    else:
        generate(args)
