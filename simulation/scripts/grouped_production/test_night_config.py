"""Config golden records from the unmodified 114985a worker AST."""
import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from night_config import stage_configs, validate_worker_mode
from night_checkpoint_identity import script_dependencies

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class ConfigTests(unittest.TestCase):
    def test_original_order_defaults_and_mode_fields(self):
        golden=json.loads((Path(__file__).parent/'fixtures/night_config_baseline.json').read_text())
        for case in golden['cases']:
            with self.subTest(tabletop=case['tabletop'],config=case['config']):
                actual=stage_configs(case['config'],Path('fixture-checkpoint'),case['tabletop'],
                    lambda x:Config(overrides=x),
                    lambda args,op,exp:[op,exp,json.dumps(vars(args),sort_keys=True)],
                    lambda args:['grasp-overrides'],nullcontext)
                self.assertEqual(dict(zip(('refine_cfg','eval_cfg','base_init','seed_cfg'),actual)),case['expected'])
                self.assertIn('op.noslip_iterations=0',actual[1]['overrides'])
                self.assertIn('op.reuse_fc_preamble=false',actual[1]['overrides'])
                self.assertIn('+op.filter.general.strict=true',actual[2])
    def test_routing_and_rejections(self):
        base=dict(rank=0,world_size=1,rank_gpu_uuids=['GPU-test'],gpu_uuid='GPU-test',
                  tours=1,hard_deadline_epoch=100,stop_new_init_epoch=90)
        for mode,schema in [('floating','floating-night/v2'),('tabletop','tabletop-night/v1'),('tabletop','tabletop-single-gpu/v1')]:
            c=dict(base,mode=mode,checkpoint_schema=schema)
            self.assertEqual(validate_worker_mode(c),mode)
            with self.assertRaises(AssertionError):validate_worker_mode(dict(c,deployment_template_only=True))
        self.assertEqual(validate_worker_mode(dict(base,checkpoint_schema='floating-night/v2')),'floating')
        for changes in [dict(mode='tabletop'),dict(gpu_uuid='wrong'),dict(rank=2),dict(hard_deadline_epoch=None)]:
            with self.assertRaises(AssertionError):validate_worker_mode(dict(base,checkpoint_schema='floating-night/v2',**changes))
    def test_candidate_examples_are_bounded_templates_and_protocol_specific(self):
        from night_capacity_policy import initialize, selected
        for mode in ('floating','tabletop'):
            c=json.loads((Path(__file__).resolve().parents[2]/'examples/unified-night'/f'{mode}.base.json').read_text())
            self.assertTrue(c['deployment_template_only']);self.assertEqual(len(c['families']),35)
            c.update(rank=0,world_size=1,rank_gpu_uuids=['GPU-fixture'],gpu_uuid='GPU-fixture',
                     groups=[{}],hard_deadline_epoch=100,stop_new_init_epoch=90)
            with self.assertRaises(AssertionError):validate_worker_mode(c)
            c['deployment_template_only']=False
            self.assertEqual(validate_worker_mode(c),mode)
            state={};initialize(c,state)
            for stage in ('refine','eval'):self.assertEqual(selected(c,state,0,'test',stage),c[stage+'_batch'])
            if mode=='tabletop':
                for option in ('isolate_eval_nonfinite','eval_failed_world_retirement'):
                    with self.assertRaises(ValueError):validate_worker_mode(dict(c,**{option:True}))

    def test_new_helpers_and_transitive_routing_are_identity_bound(self):
        root=Path(__file__).parent
        dependencies=script_dependencies(root/'night_worker.py',root)
        for name in ('night_config.py','night_transactions.py','night_inventory.py','night_schedule.py','night_capacity_policy.py'):
            self.assertIn(name,dependencies)

if __name__=='__main__':unittest.main()
