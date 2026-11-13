import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from dexgrasp.train.checkpoints import atomic_checkpoint, retain_live_checkpoint


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.last = self.root / 'last.pt'

    def save(self, step, path=None):
        atomic_checkpoint(path or self.last, {
            'step': step, 'config': {'max_steps': 2500},
            'model': {'weight': torch.tensor([step])},
            'optimizer': {'momentum': torch.tensor([step + 1])},
        })

    def read(self, path):
        return torch.load(path, weights_only=False)

    def test_milestones_survive_replacement_and_final_is_retained(self):
        for step in (1000, 1250, 2000, 2500):
            self.save(step)
        self.assertEqual(sorted(p.name for p in self.root.glob('step_*.pt')),
                         ['step_001000.pt', 'step_002000.pt', 'step_002500.pt'])
        for step in (1000, 2000, 2500):
            state = self.read(self.root / f'step_{step:06d}.pt')
            self.assertEqual(state['step'], step)
            self.assertEqual(state['model']['weight'].item(), step)
            self.assertEqual(state['optimizer']['momentum'].item(), step + 1)
        self.assertEqual(self.read(self.last)['step'], 2500)
        self.assertEqual(self.last.stat().st_ino, (self.root/'step_002500.pt').stat().st_ino)

    def test_explicit_5k_policy_does_not_archive_1k(self):
        for step in (1000, 5000, 10000, 12000):
            atomic_checkpoint(self.last, {'step':step, 'config':{'save_every':5000, 'max_steps':12000}})
        self.assertEqual(sorted(p.name for p in self.root.glob('step_*.pt')),
                         ['step_005000.pt', 'step_010000.pt', 'step_012000.pt'])

    def test_failed_serialization_preserves_existing_files(self):
        self.save(1000)
        with patch('dexgrasp.train.checkpoints.torch.save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.save(2000)
        self.assertEqual(self.read(self.last)['step'], 1000)
        self.assertFalse((self.root/'step_002000.pt').exists())

    def test_existing_milestone_cannot_be_overwritten(self):
        self.save(1000)
        self.save(1250)
        with self.assertRaises(FileExistsError):
            self.save(1000)
        self.assertEqual(self.read(self.last)['step'], 1250)
        self.assertEqual(self.read(self.root/'step_001000.pt')['step'], 1000)
        self.assertFalse((self.root/'last.pt.tmp').exists())

    def test_best_checkpoint_does_not_create_milestone(self):
        self.save(1000, self.root/'best.pt')
        self.assertEqual(list(self.root.glob('step_*.pt')), [])

    def test_live_archive_pins_before_concurrent_last_replacement(self):
        torch.save({'step': 1000}, self.last)
        original_load = torch.load

        def replace_during_read(path, **kwargs):
            replacement = self.root/'replacement.pt'
            torch.save({'step': 2000}, replacement)
            replacement.replace(self.last)
            return original_load(path, **kwargs)

        with patch('dexgrasp.train.checkpoints.torch.load', side_effect=replace_during_read):
            destination, _ = retain_live_checkpoint(self.last)
        self.assertEqual(self.read(destination)['step'], 1000)
        self.assertEqual(self.read(self.last)['step'], 2000)
        next_destination, inode = retain_live_checkpoint(self.last)
        self.assertEqual(self.read(next_destination)['step'], 2000)
        self.assertEqual(retain_live_checkpoint(self.last), (next_destination, inode))
        self.assertEqual(list(self.root.glob('.retention-*')), [])
