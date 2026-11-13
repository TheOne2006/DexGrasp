"""CPU-only ADMM projection workers; no CUDA state or mutable mesh sharing.

Spawned processes receive model geometry snapshots, never a live GPU context.
Every iteration waits for all projections before advancing the hand dynamics.
"""
import os
from types import SimpleNamespace


def project_candidate(s, data, host, mesh_info, hand_info, cfg, position, rotation, iteration):
    import numpy as np
    import torch
    from dexonomy.op.gen_grasp_admm import get_closest_triangles, point_to_triangle_simple, project_to_hand
    mesh, verts, normals, faces = mesh_info
    cp = s['cp_o'].detach().clone().requires_grad_(True)
    _, _, tri, normal = get_closest_triangles(mesh, cp.detach().numpy(), verts, normals, faces)
    objective = None
    for _ in range(int(cfg.step_obj)):
        cp.grad = None
        _, cp_o, cn = point_to_triangle_simple(cp, tri, normal)
        cn = -cn
        try:
            if getattr(cfg, 'inject_qp_failure', False):
                raise RuntimeError('QP returned None')
            res = s['qp'].solve(cp_o, cn, data['ext_wrench'], data['ext_center'])
            if res is None:
                raise RuntimeError('QP returned None')
        except Exception as exc:
            s['fallback'] = True
            s['qp_error'] = repr(exc)
            return None
        if cfg.square_res or data['hand_name'] == 'shadow':
            res = res ** 2
        res.backward()
        objective = float(res.detach())
        with torch.no_grad():
            cp.grad += float(cfg.rho_admm) * (cp - s['cp_h'] - s['lam'])
            cp.grad -= torch.sum(cp.grad * cn, dim=-1, keepdim=True) * cn
            cp -= float(cfg.lr_obj) * cp.grad
            closest, _, tri, normal = get_closest_triangles(mesh, cp.detach().numpy(), verts, normals, faces)
            cp.copy_(torch.from_numpy(closest))
            s['cp_o'] = cp.detach().clone()
    target = (s['cp_o'] - s['lam']).numpy()
    host._data.xpos[:] = position
    host._data.xmat[:] = rotation.reshape(-1, 9)
    if not cfg.fix_hand_local_cp:
        meshes, queries = hand_info
        _, local = project_to_hand(target, data['hand_cbody'], host, meshes, queries, 'collision')
        s['local'][:, :3] = local
    s['trace'].append(dict(iteration=iteration, objective=objective))
    return target


class ModelSnapshot(SimpleNamespace):
    def body(self, name):
        return SimpleNamespace(id=self.body_ids[name])


def snapshot_host(host):
    """Only immutable fields used by the unchanged upstream projection helpers."""
    model = host._model
    fields = ['geom_bodyid', 'geom_contype', 'geom_conaffinity', 'geom_quat',
              'geom_pos', 'geom_type', 'geom_size', 'geom_dataid', 'mesh_vertadr',
              'mesh_vertnum', 'mesh_faceadr', 'mesh_facenum', 'mesh_vert', 'mesh_face']
    return dict(model={**{k: getattr(model, k).copy() for k in fields},
                       'ngeom': model.ngeom,
                       'body_ids': {model.body(i).name: i for i in range(model.nbody)}},
                prefix=host._cfg.hand_prefix, xpos=host._data.xpos.copy(), xmat=host._data.xmat.copy())


_CONTEXT = None


def initialize(snapshots, config):
    global _CONTEXT
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import trimesh
    from dexonomy.op.gen_grasp_admm import load_obj_trimesh, create_hand_trimesh
    torch.set_num_threads(1)
    hosts = []
    hands = []
    for spec in snapshots:
        host = SimpleNamespace(_model=ModelSnapshot(**spec['model']),
                               _cfg=SimpleNamespace(hand_prefix=spec['prefix']),
                               _data=SimpleNamespace(xpos=spec['xpos'], xmat=spec['xmat']))
        meshes = create_hand_trimesh(host)
        hosts.append(host)
        hands.append((meshes, [trimesh.proximity.ProximityQuery(m) for m in meshes]))
    _CONTEXT = (hosts, {}, hands, SimpleNamespace(**config), {})


def project_chunk(items):
    import torch
    from dexonomy.qp.qp_single import ContactQPTorch
    from dexonomy.op.gen_grasp_admm import load_obj_trimesh
    hosts, meshes, hands, cfg, qps = _CONTEXT
    results = []
    for w, data, host_id, payload, position, rotation, iteration in items:
        if data['scene_path'] not in meshes:
            meshes[data['scene_path']] = load_obj_trimesh(data)
        if w not in qps:
            qps[w] = ContactQPTorch(cfg.opt_miu_coef)
        state = {k: torch.from_numpy(payload[k]) for k in ['cp_o', 'cp_h', 'lam']}
        state.update(local=payload['local'], qp=qps[w], trace=[], fallback=False)
        target = project_candidate(state, data, hosts[host_id], meshes[data['scene_path']],
                                   hands[host_id], cfg, position, rotation, iteration)
        results.append((w, state['cp_o'].numpy(), state['local'], state['trace'],
                        state['fallback'], state.get('qp_error'), target))
    return results


class ProjectionPool:
    """Explicit model-bank lifetime; call close before releasing a bank or rank."""
    def __init__(self, hosts, cfg, workers):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from omegaconf import OmegaConf
        from dexonomy.util.cpu_slots import detect_available_cpu_slots
        if not 1 <= workers <= detect_available_cpu_slots():
            raise ValueError('ADMM CPU workers exceed the current affinity/cgroup budget')
        self.workers = workers
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.host_ids = {id(h): i for i, h in enumerate(hosts)}
        self.executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=([snapshot_host(h) for h in hosts], self.config))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
