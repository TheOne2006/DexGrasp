"""Bounded MJWarp 3.17 batched Slide/FC; bounded protocol-owned Data and Graph reuse.

This module is lazy imported. CPU contract checks do not import or initialize Warp.
"""
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from dexonomy.tabletop_protocol import STATE, check_snapshot
from dexonomy.sim.mujoco_warp_mixed import MixedModelBank
from dexonomy.sim.mujoco_warp_env import check_warp_allocation
from dexonomy.sim.fc_execution import checked_steps, capture_graph, launch_graph, check_execution
from dexonomy import tabletop_kernels as k
from dexonomy.sim.eval_workspace import EvalWorkspace, eval_allocation

class WarpFactory:
    def __init__(self,hosts,capacities=None):
        if mujoco.__version__!='3.17.0' or mjw.__version__!='3.17.0':raise ValueError('requires pinned MuJoCo/MJWarp3.17.0')
        self.capacities=capacities or {}
        self.bank=MixedModelBank(hosts,device='cuda:0')
        self.workspace=EvalWorkspace('tabletop-slide-fc/v1')
        self.counters={'graph_hits':0,'graph_builds':0}
    def release_execution(self):self.workspace.release()
    @property
    def stats(self):return {**self.bank.cache_stats, **self.workspace.stats, **self.counters}
    def __call__(self,hosts,indices,**kwargs):
        if len(hosts)!=len(self.bank.hosts) or any(a is not b for a,b in zip(hosts,self.bank.hosts)):
            raise ValueError('factory host inventory changed')
        # Exact scene rows retain parameter buffers captured by graphs. Slide and
        # continuation have separate keys even at identical world capacity.
        key=(tuple(indices), tuple(sorted(self.capacities.items())),
             'fc' if kwargs.get('snapshots') is not None else 'slide')
        batch=self.workspace.acquire(key,
            lambda: WarpBatch(self.bank,hosts,indices,capacities=self.capacities,counters=self.counters),
            lambda b: mjw.reset_data(b.model,b.data))
        try:batch.initialize(**kwargs)
        except BaseException:
            self.release_execution()
            raise
        return batch

class WarpBatch:
    backend='mujoco-warp-3.17'
    def __init__(self,bank,hosts,indices,capacities=None,counters=None):
        self.hosts=[hosts[i] for i in indices];self.n=len(indices)
        self.packed=bank.bind(indices);self.model=self.packed.warp_model
        native=self.packed.carrier_model;self.native=native
        self.capacities=eval_allocation(native,capacities)
        self.data=mjw.make_data(native,nworld=self.n,**self.capacities)
        self.packed.initialize_data(self.data)
        self.bodies=wp.array([h._obj_active_bodyid for h in self.hosts],dtype=int)
        cats=np.zeros((self.n,native.ngeom),np.int32)
        for w,i in enumerate(indices):cats[w,self.packed.geom_maps[i]]=hosts[i].tabletop_categories
        self.categories=wp.array(cats,dtype=int)
        self.flags=wp.zeros((self.n,2),dtype=int);self.near=wp.zeros_like(self.flags)
        self.minimum=wp.array(np.full((self.n,2),np.inf,np.float32),dtype=float)
        self.invalid=wp.zeros(1,dtype=int);self.error=wp.zeros(self.n,dtype=int);self.active=wp.ones(self.n,dtype=int)
        self.graphs={}
        self.counters=counters if counters is not None else {'graph_hits':0,'graph_builds':0}
    def initialize(self,starts=None,snapshots=None,candidate_ids=None):
        self.packed.initialize_data(self.data)
        self.flags.zero_();self.near.zero_();self.invalid.zero_();self.error.zero_()
        self.minimum.fill_(float('inf'))
        if snapshots is None:
            self.data.qpos.assign(np.stack([np.concatenate([q,h._obj_init_qpos]) for q,h in zip(starts,self.hosts)]).astype(np.float32))
            self.control(starts)
            self.data.eq_active.assign(np.stack([h._model.eq_active0 for h in self.hosts]))
            mjw.forward(self.model,self.data)
        else:
            for saved,h,name in zip(snapshots,self.hosts,candidate_ids):check_snapshot(saved,h,name,self.backend)
            for name in STATE:
                values=np.stack([s['fields'][name] for s in snapshots]);dest=getattr(self.data,name)
                if dest.numpy().shape!=values.shape:raise ValueError('GPU snapshot shape mismatch: '+name)
                dest.assign(values)
        self.data.qfrc_applied.zero_();self.data.xfrc_applied.zero_()
        check_warp_allocation(self.data,mjw.OverflowType,context='tabletop batch initialization')
    def control(self,targets):
        self.data.ctrl.assign(np.stack([h._hand_qpos2ctrl_mat[:,6:]@q[7:] for h,q in zip(self.hosts,targets)]).astype(np.float32))
        self.data.mocap_pos.assign(np.asarray(targets,np.float32)[:,None,:3])
        self.data.mocap_quat.assign(np.asarray(targets,np.float32)[:,None,3:7])
    def force(self,directions):
        forces=wp.array(np.stack([np.asarray(f)*h._model.body_subtreemass[h._obj_active_bodyid]*9.8 for h,f in zip(self.hosts,directions)]).astype(np.float32),dtype=wp.vec3)
        self.data.qfrc_applied.zero_();self.data.xfrc_applied.zero_()
        wp.launch(k.apply_force,self.n,inputs=[self.bodies,forces,self.data.subtree_com,self.data.xipos,self.data.xfrc_applied])
        mjw.xfrc_accumulate(self.model,self.data,self.data.qfrc_applied);self.data.xfrc_applied.zero_()
    def record_contacts(self,tolerance,data=None):
        d=self.data if data is None else data
        wp.launch(k.table_contacts,d.naconmax,inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.dist,self.categories,float(tolerance),self.flags,self.near,self.minimum,self.invalid])
    def step(self,n,contact=False,tolerance=0.):
        d=self.data
        key=(int(n),bool(contact),float(tolerance))
        graph=self.graphs.get(key)
        if graph is None:
            # Only the protocol's 5/50-step calls are retained. Bound to this
            # Data, model and contact buffers; discarded with the workspace.
            graph=capture_graph(lambda: checked_steps(self.model,d,n,self.active,self.error,
                (lambda: self.record_contacts(tolerance)) if contact else None))
            if len(self.graphs)>=2:self.graphs.clear()
            self.graphs[key]=graph;self.counters['graph_builds']+=1
        else:self.counters['graph_hits']+=1
        launch_graph(graph)
        wp.synchronize_device('cuda:0')
        check_execution(d,self.error,'tabletop steps')
        if np.any(self.invalid.numpy()):raise RuntimeError('tabletop invalid contact')
    def poses(self):
        pos=self.data.xpos.numpy();quat=self.data.xquat.numpy()
        return np.stack([np.concatenate([pos[w,h._obj_active_bodyid],quat[w,h._obj_active_bodyid]]) for w,h in enumerate(self.hosts)])
    def qpos(self):return self.data.qpos.numpy()
    def states(self):
        values={name:getattr(self.data,name).numpy() for name in STATE}
        return [{name:np.array(value[w],copy=True) for name,value in values.items()} for w in range(self.n)]
    def endpoint_contacts(self,tolerance):
        query=mjw.make_data(self.native,nworld=self.n,**self.capacities)
        self.packed.initialize_data(query)
        for name in STATE:wp.copy(getattr(query,name),getattr(self.data,name))
        mjw.forward(self.model,query)
        check_warp_allocation(query,mjw.OverflowType,context='Tabletop endpoint contact query')
        self.record_contacts(tolerance,query)
        wp.synchronize_device('cuda:0')
        if np.any(self.invalid.numpy()):raise ValueError('invalid terminal GPU contact')
    def end_contacts(self,tolerance):
        self.flags.zero_();self.near.zero_();self.endpoint_contacts(tolerance)
        return self.flags.numpy()
    def contacts(self,tolerance=0.):
        self.endpoint_contacts(tolerance)
        return self.flags.numpy(),self.near.numpy(),self.minimum.numpy()
