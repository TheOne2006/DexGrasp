"""Execute the actual worker main with synthetic stages, never GPU imports.

AST replaces imports with supplied bindings; control flow and file operations are real.
Every name must be supplied explicitly. This tests orchestration, not physics.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import unittest
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from dexonomy.op.simulation_isolation import nonfinite_stop_reason
from dexonomy.acceptance_audit import serial
from night_checkpoint_identity import token
from night_inventory import validate_mode
from night_schedule import validate_schedule, tour_numbers, planned_visits, reached
from night_capacity_policy import selected, successful, initialize, blocked, completed as capacity_completed, systemic_reason, all_combinations_deferred

ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).with_name('night_worker.py')
TAG = 't00-g000-e00-family'

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class RemoveImports(ast.NodeTransformer):
    def visit_Import(self, node): return None
    def visit_ImportFrom(self, node):
        # Preserve Python local binding (notably capacity selected vs Init selected).
        return [ast.copy_location(ast.Assign(targets=[ast.Name(id=a.asname or a.name, ctx=ast.Store())],
            value=ast.Subscript(value=ast.Call(func=ast.Name(id='globals',ctx=ast.Load()),args=[],keywords=[]),
                slice=ast.Constant(a.asname or a.name),ctx=ast.Load())),node) for a in node.names]

class Fixture:
    def __init__(self, root):
        self.root = root
        self.out = root / 'checkpoint'
        self.counts = Counter()
        self.events = []
        self.routes = []
        self.reject_after = None
        self.append_calls = 0
        self.index = root / 'index.json'
        self.index.write_text(json.dumps({'cases':[{'scene_id':'scene','grasp_scene':'fake-grasp','stability_scene':'fake-stability'}]}))
        self.config = dict(source=str(ROOT), checkpoint_root=str(self.out), dataset_id='synthetic',
            checkpoint_schema='floating-night/v2', rank=0, world_size=1, mode='floating',
            rank_gpu_uuids=['GPU-fixture'], gpu_uuid='GPU-fixture', target=4, sample_points=16,
            object_batch=1, refine_batch=4, eval_batch=24, epochs_per_visit=1, tours=1,
            hard_deadline_epoch=4000000000, stop_new_init_epoch=3999999990,
            groups=[{'index':str(self.index)}], asset_root='fake-assets', seed=4,
            families=['family'], refine_min_execution_slots=32, refine_small_batch_limit=32,
            refine_small_batch_capacity=32, stop_after_stage=None)
        self.runtime = self.namespace()
        tree = ast.parse(DRIVER.read_text())
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        tree = RemoveImports().visit(tree)
        exec(compile(ast.fix_missing_locations(tree), str(DRIVER), 'exec'), self.runtime)

    def namespace(self):
        fixture = self
        class Loader:
            def __init__(self, **kwargs):
                self.family_capacity = kwargs['family_capacity']
                self.xml_path = 'fake'; self.data_buffer = []; self.feedback_stats = {}; self.feedback_seen_ids = set()
            def activate_family(self, *args): return []
            def distribution(self): return {}
            def memory_estimate(self): return {}
            def stop(self): pass
            def _prepare_data(self, vis, data, name): return {'id':name,'index':data['index']}
            def _append_hand_data(self, data):
                fixture.append_calls += 1
                if fixture.reject_after == fixture.append_calls: raise RuntimeError('synthetic append interruption')
                if data['index'] == 1: return False
                self.data_buffer.append(data); return True
            def _record_feedback_stat(self, name): self.feedback_stats[name] = self.feedback_stats.get(name,0)+1
            def _record_family_feedback(self, *args): pass
        class Cache:
            cache_stats = {}
            def __init__(self, *args, **kw): pass
            def close_traditional_qp_pool(self): pass
        class Batcher(Cache):
            pending = False
            def __init__(self, *args, **kwargs): fixture.routes.append('floating')
            def release_execution(self): pass
            def push(self, values, *, source_names):
                fixture.counts['eval'] += 1
                return {'results':[dict(source_name=n,success=d['index']!=2,output_data=d,
                    metrics={'type':'simulation_error' if d['index']==2 else 'fc'}) for n,d in zip(source_names,values)]}
            def flush(self): return {'results':[]}
        class TabletopBatcher(Batcher):
            def __init__(self, *args, **kwargs): fixture.routes.append('tabletop')
        def compose(overrides):
            fixture.events.append(('config', list(overrides)))
            return Config(init_tmpl_dir=str(fixture.root),hand=SimpleNamespace(xml_path='fake'),
                          hand_skt_path='fake',tmpl_max_evo=2)
        def synthetic_initialize(cfg,scene_epoch_data_callback,**kw):
            fixture.counts['init'] += 1
            entries = [{'data':dict(index=i,audit_init_hard_all=True,tmpl_name='family',
                grasp_qpos=np.array([i,0,0,1,0,0,0.]),n_evo=0)} for i in range(4)]
            scene_epoch_data_callback('scene',None,entries)
        def refine(values,cfg,**kw):
            fixture.counts['refine'] += 1
            # Actual small ledger files enter the worker checkpoint artifact map.
            for folder in ('simulation-errors','refine-simulation-errors'):
                p=fixture.out/'jobs'/TAG/folder/'synthetic.npy';p.parent.mkdir(parents=True,exist_ok=True)
                np.save(p,{'type':'simulation_error','synthetic':True})
            return [dict(success=d['index']!=3,output_data=d,
                metrics={'type':'simulation_error' if d['index']==3 else 'refine'}) for d in values]
        noop=lambda *a,**kw:None
        cuda=SimpleNamespace(empty_cache=noop,memory_allocated=lambda:0,memory_reserved=lambda:0)
        ns=dict(argparse=argparse,copy=copy,gc=gc,hashlib=hashlib,json=json,os=os,random=random,time=time,
            Path=Path,SimpleNamespace=SimpleNamespace,np=np,torch=SimpleNamespace(cuda=cuda,manual_seed=noop),
            importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda n: {'mujoco':'3.17.0','mujoco-warp':'3.17.0','warp-lang':'1.16.0'}.get(n,'fixture'))),
            validate_mode=validate_mode,validate_schedule=validate_schedule,tour_numbers=tour_numbers,
            planned_visits=planned_visits,reached=reached,install_snapshot=noop,checkpoint_token=token,
            selected=selected,successful=successful,check_assets=noop,open_dict=nullcontext,
            initialize=initialize,blocked=blocked,capacity_completed=capacity_completed,
            systemic_reason=systemic_reason,all_combinations_deferred=all_combinations_deferred,
            nonfinite_stop_reason=nonfinite_stop_reason,
            _compose_cfg=compose,_common_overrides=lambda args,op,exp:[op],_grasp_overrides=lambda args:[],
            FamilyPoolLoader=Loader,MixedRefinementCache=Cache,MixedForceClosureBatcher=Batcher,
            TabletopSlideFCBatcher=TabletopBatcher,operate_init=synthetic_initialize,refine_grasp_batch_mixed=refine,
            release_warp_scene_memory=noop,clear=noop,MuJoCo_VisEnv=lambda **kw:None,HandCfg=lambda **kw:None,
            event=lambda kind,**kw:fixture.events.append((kind,kw)),serial=serial,__file__=str(DRIVER))
        # Once extracted, actual pure helpers remain under test, not replacements.
        config_helper = DRIVER.with_name('night_config.py')
        if config_helper.exists():
            from night_config import validate_worker_mode, stage_configs
            ns.update(validate_worker_mode=validate_worker_mode,stage_configs=stage_configs)
        transaction_helper = DRIVER.with_name('night_transactions.py')
        if transaction_helper.exists():
            from night_transactions import checkpoint_artifacts, apply_feedback, completion_row, record_completion
            ns.update(checkpoint_artifacts=checkpoint_artifacts,apply_feedback=apply_feedback,
                      completion_row=completion_row,record_completion=record_completion)
        return ns

    def run(self, stop=None):
        self.config['stop_after_stage'] = None if stop is None else TAG+'/'+stop
        path=self.root/'config.json';path.write_text(json.dumps(self.config))
        with patch.object(sys,'argv',['night_worker.py',str(path),str(self.root/'run')]):
            self.runtime['main']()

    def state(self): return np.load(self.out/'state.npy',allow_pickle=True).item()

class TransactionTests(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        return Fixture(Path(tmp.name))

    def assert_complete(self, f):
        s=f.state();self.assertEqual(s['completed'],[TAG]);self.assertEqual(len(s['rows']),1)
        row=s['rows'][0]
        for key,value in dict(init=4,refined=3,stable=2,feedback=1,feedback_rejected=1,simulation_error=2,refine_simulation_error=1).items(): self.assertEqual(row[key],value,key)
        self.assertEqual(s['template_stats']['received'],2)
        self.assertEqual(len(s['feedback_seen_ids']),1)
        self.assertEqual(len(s['family_archive']['family']),1)
        self.assertEqual(len(s['template_buffer']),1)
        self.assertEqual(len(list((f.out/'success').rglob('*.npy'))),2)
        self.assertEqual(len(s['artifacts']),6)
        self.assertTrue(all((f.out/p).is_file() for p in s['artifacts']))

    def test_every_persisted_boundary_resume_and_duplicate_resume(self):
        for mode in ('floating','tabletop'):
            for stage in ('init','refine','eval','feedback_prepared','complete'):
                with self.subTest(mode=mode,stage=stage):
                    f=self.fixture();f.config.update(mode=mode,checkpoint_schema=mode+'-night/'+('v2' if mode=='floating' else 'v1'))
                    with self.assertRaises(SystemExit) as e:f.run(stage)
                    self.assertEqual(e.exception.code,75)
                    self.assertEqual(f.state()['completed'],[TAG] if stage=='complete' else [])
                    if stage=='feedback_prepared':
                        self.assertEqual(len(list((f.out/'success').rglob('*.npy'))),2)
                        self.assertEqual(f.append_calls,0)
                    f.run();self.assert_complete(f)
                    before=f.counts.copy();calls=f.append_calls;f.run();self.assert_complete(f)
                    self.assertEqual(before,f.counts);self.assertEqual(calls,f.append_calls)
                    self.assertEqual(f.counts,dict(init=1,refine=1,eval=1))
                    self.assertTrue(f.routes)
                    self.assertEqual(set(f.routes),{mode})

    def test_adaptive_completion_commits_once_in_both_modes(self):
        from test_night_adaptive_capacity import config
        for mode in ('floating', 'tabletop'):
            f=self.fixture()
            policy=config()['capacity_policy']
            policy.update(refine_ladder=[4,2,1],eval_ladder=[24,12,6])
            f.config.update(mode=mode,checkpoint_schema=mode+'-night/'+('v2' if mode=='floating' else 'v1'),capacity_policy=policy)
            with self.assertRaises(SystemExit):f.run('feedback_prepared')
            self.assertEqual(f.state()['capacity_events'],[])
            f.run();f.run();self.assert_complete(f)
            self.assertEqual(f.state()['capacity_events'],[{'event':'completed','tag':TAG,'init_count':4}])

    def test_feedback_interruption_rolls_back_uncommitted_memory(self):
        f=self.fixture();f.reject_after=2
        with self.assertRaisesRegex(RuntimeError,'synthetic append'):f.run()
        self.assertEqual(f.state()['completed'],[]);self.assertNotIn('template_buffer',f.state())
        f.reject_after=None;f.run();self.assert_complete(f)
        self.assertEqual(f.counts,dict(init=1,refine=1,eval=1))

    def test_atomic_completion_replace_failure_keeps_previous_checkpoint(self):
        f=self.fixture();replace=os.replace
        def fail(src,dst):
            if Path(dst).name=='state.npy' and np.load(src,allow_pickle=True).item()['completed']:
                raise OSError('synthetic replace failure')
            return replace(src,dst)
        with patch.object(os,'replace',side_effect=fail):
            with self.assertRaisesRegex(OSError,'synthetic replace'):f.run()
        self.assertEqual(f.state()['completed'],[])
        self.assertTrue((f.out/'state.npy.partial').exists())
        f.run();self.assert_complete(f)

    def test_corrupt_stage_or_error_ledger_rejected_before_consumption(self):
        for rel in ('init.npy','simulation-errors/synthetic.npy','refine-simulation-errors/synthetic.npy'):
            with self.subTest(rel=rel):
                f=self.fixture()
                with self.assertRaises(SystemExit):f.run('eval')
                p=f.out/'jobs'/TAG/rel;p.write_bytes(p.read_bytes()+b'corrupt')
                before=f.counts.copy()
                with self.assertRaisesRegex(AssertionError,'Checkpoint changed'):f.run()
                self.assertEqual(before,f.counts);self.assertEqual(f.state()['completed'],[])

    def test_schema_and_source_mismatch_rejected_before_stages(self):
        for field in ('schema','config_hash'):
            f=self.fixture()
            with self.assertRaises(SystemExit):f.run('init')
            s=f.state();s[field]='old-source-or-schema';np.save(f.out/'state.npy',s)
            before=f.counts.copy()
            with self.assertRaises(AssertionError):f.run()
            self.assertEqual(before,f.counts)

    def test_atomic_write_fsync_replace_event_order(self):
        f=self.fixture();calls=[];save=np.save;fsync=os.fsync;replace=os.replace
        def saved(*a,**kw):calls.append('save');return save(*a,**kw)
        def synced(*a,**kw):calls.append('fsync');return fsync(*a,**kw)
        def replaced(*a,**kw):calls.append('replace');return replace(*a,**kw)
        f.runtime['event']=lambda *a,**kw:calls.append('event')
        with patch.object(np,'save',side_effect=saved),patch.object(os,'fsync',side_effect=synced),patch.object(os,'replace',side_effect=replaced):
            f.runtime['atomic'](f.root/'artifact.npy',{'ok':True})
        self.assertEqual(calls,['save','fsync','replace','event'])

if __name__=='__main__':unittest.main()
