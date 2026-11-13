"""Atomic checkpoint replacement with permanent optimizer-step snapshots."""
import os
from pathlib import Path
import torch


ARCHIVE_EVERY = 1000


def retain_checkpoint(path, step):
    """Keep the completed inode; replacing last.pt cannot change this snapshot."""
    path = Path(path)
    destination = path.parent / f'step_{step:06d}.pt'
    try:
        os.link(path, destination)
    except FileExistsError:
        if not os.path.samefile(path, destination):
            raise FileExistsError(f'Refusing to overwrite retained checkpoint: {destination}')
    return destination


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(state, temporary)
        step = state['step']
        if path.name == 'last.pt' and (step % state.get('config', {}).get('save_every', ARCHIVE_EVERY) == 0
                                     or step == state.get('config', {}).get('max_steps')):
            retain_checkpoint(temporary, step)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def retain_live_checkpoint(path):
    """Pin last.pt before reading metadata so concurrent replacement is safe."""
    path = Path(path)
    pinned = path.parent / f'.retention-{os.getpid()}.pt'
    os.link(path, pinned)
    try:
        state = torch.load(pinned, map_location='cpu', weights_only=False, mmap=True)
        destination = retain_checkpoint(pinned, state['step'])
        return destination, pinned.stat().st_ino
    finally:
        pinned.unlink(missing_ok=True)
