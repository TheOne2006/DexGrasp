"""Thin adapter for the fixed vNext Init/Refine scheduler; no budget changes."""
from pathlib import Path
import time
import uuid
import numpy as np
from dexonomy.tabletop_protocol import make_host,evaluate,Protocol,digest

class TabletopSlideFCBatcher:
    def __init__(self,cfg,world_batch_size=8,*,scene_paths=None):
        if getattr(cfg.op,'noslip_iterations',None)!=0:raise ValueError('Tabletop Eval requires explicit noslip0')
        if int(world_batch_size)<1:raise ValueError('world_batch_size must be positive')
        self.cfg=cfg;self.world_batch_size=int(world_batch_size);self.pending=[];self.closed=False
        paths=list(dict.fromkeys(str(Path(p).resolve()) for p in scene_paths or []))
        if not paths:raise ValueError('Tabletop requires explicit scene inventory')
        start=time.perf_counter();self.hosts=[make_host(p,cfg.hand.xml_path) for p in paths]
        self.path_indices={p:i for i,p in enumerate(paths)}
        self.factory=None
        self.cache_stats={'host_model_builds':len(paths),'duration_s':time.perf_counter()-start,'gpu_bank':'lazy persistent bank; bounded protocol workspace'}
        self.last_report=None;self.flush_index=0
    @property
    def model_bank(self):return None if self.factory is None else self.factory.bank
    def release_execution(self):
        if self.factory is not None:self.factory.release_execution()
    def push(self,values,source_names):
        if self.closed or len(values)!=len(source_names):raise ValueError('invalid push state')
        known={x['id'] for x in self.pending}
        for data,name in zip(values,source_names):
            if name in known:raise ValueError('duplicate candidate ID')
            path=str(Path(data['scene_path']).resolve())
            if path not in self.path_indices:raise ValueError('unbound Tabletop scene')
            self.pending.append({'id':name,'data':data,'host_index':self.path_indices[path]});known.add(name)
        return {'results':[]}
    def flush(self):
        if self.closed:raise ValueError('batcher already closed')
        batch=self.pending
        root=Path(self.cfg.get('tabletop_output','output/tabletop-slide-fc/results'))
        checkpoint=root/'slide-checkpoints'/(digest([x['id'] for x in batch])+'.npy')
        if batch and self.factory is None:
            from dexonomy.tabletop_warp import WarpFactory
            self.factory=WarpFactory(self.hosts,capacities={k:int(self.cfg.get('eval_'+k,0)) for k in ('njmax','nconmax','nccdmax')})
        try:
            report=evaluate(batch,self.hosts,[x['host_index'] for x in batch],batch_size=self.world_batch_size,backend='warp',
                policy=Protocol(),trace=bool(self.cfg.get('tabletop_trace',False)),slide_checkpoint_path=checkpoint,
                factory=self.factory,isolation_cfg=self.cfg,trace_candidates=([x['id'] for x in batch if x['id'] in self.cfg.get('tabletop_trace_candidates',[])] if self.cfg.get('tabletop_trace_candidates') else None),
                capacities={k:int(self.cfg.get('eval_'+k,0)) for k in ('njmax','nconmax','nccdmax')})
        except BaseException:
            self.release_execution()
            raise
        if self.factory is not None:report["runtime"]=self.factory.stats
        self.last_report=report
        # Persist independently of optional engineering audit: these are protocol
        # results and resumable Slide states, not diagnostic checkpoint copies.
        root=Path(self.cfg.get('tabletop_output','output/tabletop-slide-fc/results'))
        root.mkdir(parents=True,exist_ok=True);path=root/f'protocol-{uuid.uuid4().hex}.npy'
        if path.exists():raise FileExistsError(path)
        with path.with_suffix('.partial').open('wb') as f:np.save(f,report,allow_pickle=True)
        path.with_suffix('.partial').replace(path)
        self.flush_index+=1
        by_id={r['candidate_id']:r for r in report['combined']}
        slides={r['candidate_id']:r for r in report['slide']}
        errors={r['source_name']:r for r in report.get('simulation_errors',[])}
        results=[]
        for entry in batch:
            name=entry['id']
            if name in errors:results.append(errors[name]);continue
            ok=by_id[name]['passes']
            metrics={'slide':slides[name],'directions':[r for r in report['directions'] if r['candidate_id']==name],
                     'protocol':'tabletop-slide-fc/v1','artifact':str(path),'joint_success':ok}
            terminal=report['snapshots'].get(name)
            results.append({'source_name':name,'success':ok,'metrics':metrics,
                            'output_data':dict(entry['data'],tabletop_metrics=metrics,tabletop_initial_scene=self.hosts[entry['host_index']].tabletop_identity,
                                tabletop_terminal_state=terminal,tabletop_control_target=terminal['hold_target'],
                                tabletop_actual_object_pose=slides[name]['final_pose'],
                                tabletop_integrated_object_freejoint_pose=terminal['fields']['qpos'][-7:].copy(),
                                tabletop_pose_convention='actual_object_pose is official last-step derived world pose; integrated_object_freejoint_pose is terminal qpos; grasp_qpos remains pre-Slide Refine pose') if ok else None})
        self.pending=[];self.closed=True
        return {'results':results}
