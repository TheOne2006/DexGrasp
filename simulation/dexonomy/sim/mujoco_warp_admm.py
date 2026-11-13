"""Opt-in GraspADMM: CPU QP/surface projections, batched MJWarp hand dynamics.

Equations and dual annealing follow JYChen18/Dexonomy d561ff9. No gradient
through the physics or QP solution; the latter is detached by ContactQPTorch.
"""
import time
import numpy as np
import torch
import trimesh
import warp as wp
import mujoco_warp as mjw
from dexonomy.sim.mujoco_warp_resident_env import ResidentRefinementBatch, invert_mask
from dexonomy.op.gen_grasp_admm import load_obj_trimesh, create_hand_trimesh
from dexonomy.qp.qp_single import ContactQPTorch


@wp.kernel
def admm_springs(active: wp.array[int], bodies: wp.array2d[int], counts: wp.array[int],
                 local: wp.array2d[wp.vec3], targets: wp.array2d[wp.vec3],
                 xpos: wp.array2d[wp.vec3], xmat: wp.array2d[wp.mat33],
                 xipos: wp.array2d[wp.vec3], forces: wp.array2d[wp.spatial_vector]):
    w, body = wp.tid()
    if active[w] != 0:
        f = wp.vec3(0.0)
        t = wp.vec3(0.0)
        for j in range(counts[w]):
            if bodies[w,j] == body:
                p = xmat[w,body] @ local[w,j] + xpos[w,body]
                spring = (targets[w,j] - p) * 500.0
                f += spring
                t += wp.cross(p-xipos[w,body], spring)
        forces[w,body] = wp.spatial_vector(f,t)


class ADMMBatch(ResidentRefinementBatch):
    def pregrasp(self, passing, contact_sets):
        # The ordinary optimizer creates this graph during refine(); ADMM has
        # its own hand graph and must provide the shared pregrasp graph too.
        def physics():
            wp.launch(invert_mask,self.nworld,inputs=[self.active,self.inactive])
            self._copy(self.saved,to_saved=False,mask=self.inactive)
            for _ in range(int(self.cfg.op.grasp.substep)):
                mjw.step(self.m,self.d);self._errors()
        self._capture('physics',physics)
        return super().pregrasp(passing,contact_sets)

    def refine(self):
        self._cpu_pool = None
        self._owns_cpu_pool = False
        try:
            return self._refine_admm()
        finally:
            if self._cpu_pool is not None and self._owns_cpu_pool:
                self._cpu_pool.close()

    def _refine_admm(self):
        cfg=self.cfg.op.grasp
        if min(int(cfg.step),int(cfg.step_obj),int(cfg.step_hand),int(cfg.substep))<1:
            raise ValueError('ADMM iteration counts must be positive')
        begin=time.perf_counter();states=[];mesh_cache={};hand_cache={}
        for w,(data,host) in enumerate(zip(self.candidates,self.world_hosts)):
            key=data['scene_path']
            if key not in mesh_cache:mesh_cache[key]=load_obj_trimesh(data)
            if id(host) not in hand_cache:
                meshes=create_hand_trimesh(host)
                hand_cache[id(host)]=(meshes,[trimesh.proximity.ProximityQuery(m) for m in meshes])
            n=len(data['hand_cbody'])
            states.append(dict(cp_o=torch.tensor(data['obj_cpn_w'][:,:3],dtype=torch.float32),
                cp_h=torch.tensor(data['hand_cpn_w'][:,:3],dtype=torch.float32),
                lam=torch.zeros((n,3)),local=np.concatenate([self.local_host[w,:n],self.local_normals_host[w,:n]],axis=1),
                qp=ContactQPTorch(cfg.opt_miu_coef),trace=[],fallback=False))
        from dexonomy.sim.admm_cpu_projection import project_candidate, ProjectionPool, project_chunk
        workers = int(self.cfg.get('admm_cpu_workers', 0))
        if workers < 0:
            raise ValueError('admm_cpu_workers must be nonnegative')
        if workers:
            from omegaconf import OmegaConf
            cache = getattr(self, 'admm_model_cache', None)
            existing = getattr(cache, 'admm_cpu_pool', None)
            if existing is not None:
                if existing.workers != workers or existing.config != OmegaConf.to_container(cfg, resolve=True):
                    raise ValueError('Close the existing CPU pool before changing workers or ADMM config')
                self._cpu_pool = existing
            else:
                hosts = cache.hosts if cache is not None else list({id(h): h for h in self.world_hosts}.values())
                self._cpu_pool = ProjectionPool(hosts, cfg, workers)
                if cache is not None:
                    cache.admm_cpu_pool = self._cpu_pool
                else:
                    self._owns_cpu_pool = True
        self.timings['admm_cpu_setup_s']=time.perf_counter()-begin
        def force_step():
            self._controls()
            wp.launch(admm_springs,(self.nworld,self.m.nbody),inputs=[self.active,self.bodies,self.counts,self.local,self.targets,self.d.xpos,self.d.xmat,self.d.xipos,self.d.xfrc_applied])
            self.d.qfrc_applied.zero_();mjw.xfrc_accumulate(self.m,self.d,self.d.qfrc_applied);self.d.xfrc_applied.zero_()
        def hand_step():
            force_step()
            for _ in range(int(cfg.substep)):
                mjw.step(self.m,self.d);self._errors()
        self._capture('admm_hand',hand_step)
        begin=time.perf_counter()
        for iteration in range(int(cfg.step)):
            cpu_start=time.perf_counter()
            positions=self.d.xpos.numpy();rotations=self.d.xmat.numpy()
            targets=np.zeros((self.nworld,self.width,3),np.float32)
            if self._cpu_pool is not None:
                items = [(w, self.candidates[w], self._cpu_pool.host_ids[id(self.world_hosts[w])], {k: s[k].numpy().copy() for k in ['cp_o', 'cp_h', 'lam']} |
                          {'local': s['local'].copy()}, positions[w], rotations[w], iteration)
                         for w, s in enumerate(states) if not s['fallback']]
                chunks = [items[i::workers] for i in range(workers)]
                futures = [self._cpu_pool.executor.submit(project_chunk, chunk) for chunk in chunks if chunk]
                for future in futures:
                    for w, cp_o, local, trace, fallback, error, target in future.result():
                        s = states[w]
                        if self.cfg.get('admm_verify_projection', False):
                            import copy
                            reference = copy.deepcopy(s)
                            expected = project_candidate(reference, self.candidates[w], self.world_hosts[w],
                                mesh_cache[self.candidates[w]['scene_path']], hand_cache[id(self.world_hosts[w])],
                                cfg, positions[w], rotations[w], iteration)
                            if reference['fallback'] != fallback:
                                raise RuntimeError('CPU parallel fallback differs from serial')
                            if not fallback:
                                for actual, wanted in [(cp_o, reference['cp_o'].numpy()),
                                                       (local, reference['local']), (target, expected)]:
                                    if not np.array_equal(actual, wanted):
                                        raise RuntimeError('CPU parallel projection differs from serial')
                                if trace[-1] != reference['trace'][-1]:
                                    raise RuntimeError('CPU parallel objective differs from serial')
                            self.timings['admm_verified_projections'] += 1
                        s['cp_o'] = torch.from_numpy(cp_o)
                        s['local'] = local
                        s['trace'].extend(trace)
                        s['fallback'] = fallback
                        if fallback:
                            s['qp_error'] = error
                        else:
                            n = len(target)
                            self.local_host[w, :n] = local[:, :3]
                            targets[w, :n] = target
            else:
                for w,(s,data,host) in enumerate(zip(states,self.candidates,self.world_hosts)):
                    if s['fallback']:continue
                    target = project_candidate(s, data, host, mesh_cache[data['scene_path']],
                        hand_cache[id(host)], cfg, positions[w], rotations[w], iteration)
                    if s['fallback']:continue
                    n=len(target);self.local_host[w,:n]=s['local'][:,:3];targets[w,:n]=target
            self.local.assign(self.local_host);self.targets.assign(targets)
            self.active.assign(np.array([not s['fallback'] for s in states],np.int32))
            elapsed_projection = time.perf_counter()-cpu_start
            self.timings['admm_cpu_projection_qp_s']+=elapsed_projection
            if iteration == 0:self.timings['admm_first_projection_s']=elapsed_projection
            if iteration == 0 and self.cfg.get('admm_check_force', False):
                # Independently compare GPU Cartesian accumulation against native
                # mj_applyFT using the very same exported derived state.
                import mujoco
                force_step()
                actual = self.d.qfrc_applied.numpy()
                derived = {name: getattr(self.d,name).numpy() for name in
                           ['xpos','xmat','xipos','subtree_com','cdof']}
                for w,(s,data,host) in enumerate(zip(states,self.candidates,self.world_hosts)):
                    if s['fallback']: continue
                    for name,values in derived.items():
                        getattr(host._data,name)[:] = values[w].reshape(getattr(host._data,name).shape)
                    expected = np.zeros(host._model.nv)
                    for j,body in enumerate(data['hand_cbody']):
                        bid=host._model.body(host._cfg.hand_prefix+body).id
                        point=host._data.xmat[bid].reshape(3,3) @ s['local'][j,:3] + host._data.xpos[bid]
                        force=500*(targets[w,j]-point)
                        mujoco.mj_applyFT(host._model,host._data,force,np.zeros(3),point,bid,expected)
                    error=float(np.max(np.abs(actual[w]-expected)))
                    s['trace'][-1]['force_mapping_max_abs']=error
                    if not np.allclose(actual[w],expected,rtol=2e-4,atol=2e-5):
                        raise RuntimeError(f'ADMM force mapping mismatch world {w}: {error}')
            for _ in range(int(cfg.step_hand)):
                if any(s['fallback'] for s in states):
                    # Keep retired worlds near their original reset state until
                    # the wrapper runs the official fallback on those inputs.
                    wp.launch(invert_mask,self.nworld,inputs=[self.active,self.inactive])
                    self._copy(self.saved,to_saved=False,mask=self.inactive)
                self._launch('admm_hand')
            self._check()
            positions=self.d.xpos.numpy();rotations=self.d.xmat.numpy()
            for w,(s,data,host) in enumerate(zip(states,self.candidates,self.world_hosts)):
                if s['fallback']:continue
                host._data.xpos[:]=positions[w];host._data.xmat[:]=rotations[w].reshape(-1,9)
                cp_h=host.transform_cpn_b2w(data['hand_cbody'],s['local'])[:,:3]
                s['cp_h'].copy_(torch.from_numpy(cp_h))
                dist=s['cp_h']-s['cp_o'];condition=torch.linalg.vector_norm(s['lam'],dim=-1,keepdim=True)>2e-2
                s['lam']=torch.where(condition,0.,s['lam']+dist)
                s['trace'][-1].update(primal_max=float(torch.linalg.vector_norm(dist,dim=-1).max()),dual_max=float(torch.linalg.vector_norm(s['lam'],dim=-1).max()))
                if not all(torch.isfinite(s[k]).all() for k in ['cp_o','cp_h','lam']):raise RuntimeError('nonfinite ADMM state')
        self.timings['admm_loop_wall_s']=time.perf_counter()-begin
        # Failure worlds are explicitly returned for the reference 20-step fallback.
        self.admm_diagnostics=[{k:v for k,v in s.items() if k in ['trace','fallback','qp_error']} for s in states]
        self.grasp_iterations=np.array([len(s['trace'])*int(cfg.step_hand) for s in states])
        self.active.fill_(1)
        self._copy(self.saved,to_saved=True)
        self._margin(float(cfg.filter.contact.thre));self._forward();self._errors();self._check()
        self._margin(float(cfg.cdist))
        return self.export_grasps()
