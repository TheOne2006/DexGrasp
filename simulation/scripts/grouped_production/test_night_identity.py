"""Synthetic source trees only; never load a production checkpoint."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from night_checkpoint_identity import token, script_dependencies


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'dexonomy').mkdir()
        (self.root / 'dexonomy/runtime.py').write_text('value = 1\n')
        for name in ('night_schedule', 'night_templates', 'night_inventory', 'night_checkpoint_identity', 'night_capacity_policy'):
            (self.root / (name + '.py')).write_text('# fixture\n')
        self.driver = self.root / 'night_worker.py'
        self.driver.write_text('from night_config import configure\ndef run():\n from night_transactions import complete\n')
        (self.root / 'night_config.py').write_text('from nested_helper import value\n')
        (self.root / 'night_transactions.py').write_text('def complete(): pass\n')
        (self.root / 'nested_helper.py').write_text('value = 1\n')
        self.config = {'source': str(self.root), 'mode': 'floating'}

    def test_each_runtime_dependency_changes_token(self):
        before = token(self.config, self.driver)
        for name in ('night_worker.py', 'night_config.py', 'night_transactions.py', 'nested_helper.py', 'night_checkpoint_identity.py', 'dexonomy/runtime.py'):
            with self.subTest(name=name):
                p = self.root / name; original = p.read_text()
                p.write_text(original + '# changed\n')
                self.assertNotEqual(before, token(self.config, self.driver))
                p.write_text(original)
        self.assertIn('nested_helper.py', script_dependencies(self.driver, self.root))

    def test_removed_helper_changes_token_and_missing_mandatory_fails(self):
        before = token(self.config, self.driver)
        (self.root / 'night_transactions.py').unlink()
        self.assertNotEqual(before, token(self.config, self.driver))
        (self.root / 'night_inventory.py').unlink()
        with self.assertRaises(FileNotFoundError): token(self.config, self.driver)

    def test_source_archive_and_config_boundaries(self):
        before = token(self.config, self.driver)
        sources = {'runtime.py': 'old'}
        self.assertNotEqual(token(self.config, self.driver, sources=sources), before)
        for field in ('stop_after_stage', 'stop_after_jobs', 'case_limit_s', 'hard_deadline_epoch', 'stop_new_init_epoch'):
            self.assertEqual(before, token(dict(self.config, **{field: 123}), self.driver))
        for field in ('mode', 'checkpoint_schema', 'refine_batch', 'eval_batch'):
            self.assertNotEqual(before, token(dict(self.config, **{field: 'changed'}), self.driver))

    def test_legacy_algorithm_is_explicit_and_not_current_identity(self):
        old = dict(self.config)
        old['driver_sha256'] = hashlib.sha256(self.driver.read_bytes()).hexdigest()
        old['driver_dependencies'] = {n: hashlib.sha256((self.root / n).read_bytes()).hexdigest() for n in ('night_schedule.py','night_templates.py','night_inventory.py')}
        old['runtime_source'] = {'fixed.py': 'frozen'}
        expected = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
        self.assertEqual(expected, token(self.config, self.driver, sources=old['runtime_source'], legacy=True))
        self.assertNotEqual(expected, token(self.config, self.driver, sources=old['runtime_source']))

if __name__ == '__main__': unittest.main()
