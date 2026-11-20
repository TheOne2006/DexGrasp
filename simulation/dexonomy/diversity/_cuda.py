"""Optional FP64 Torch backend; importing the public CPU API never imports Torch.

Promoted from the validated 2026-11-20 pair-budget experiment. Pair generation
stays in NumPy PCG64, with one bounded producer preserving its original stream.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np


def _prefetched(batches):
    iterator = iter(batches)
    # Only this producer advances the iterator, never concurrently. At most two
    # batches are queued; exceptions propagate and the executor always joins.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = deque(pool.submit(next, iterator, None) for _ in range(2))
        while pending:
            batch = pending.popleft().result()
            if batch is None:
                return
            yield batch
            pending.append(pool.submit(next, iterator, None))


def accumulate(features, batches, *, device, prefetch):
    """Return count, kernel means, centered sums of squares and execution timing."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('CUDA diversity requires PyTorch in the simulation environment') from error
    setup_started = time.perf_counter()
    target = torch.device(device)
    # The public dispatcher only supplies CUDA. CPU is useful for backend parity
    # unit tests without acquiring a GPU or importing Torch on the CPU API path.
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; choose device="cpu" explicitly')
    if target.type == 'cuda':
        torch.cuda.synchronize(target)
    with torch.inference_mode():
        # Accept read-only mmap and negative-stride views just like the CPU API.
        tensor = torch.tensor(np.ascontiguousarray(features), dtype=torch.float64, device=target)
        if target.type == 'cuda':
            torch.cuda.synchronize(target)
        setup_seconds = time.perf_counter() - setup_started
        compute_started = time.perf_counter()
        seen = 0
        mean = np.zeros(4, dtype=np.float64)
        m2 = np.zeros(4, dtype=np.float64)
        stream = _prefetched(batches) if prefetch else iter(batches)
        try:
            for first, second in stream:
                first = torch.as_tensor(first, device=target)
                second = torch.as_tensor(second, device=target)
                difference = tensor[first] - tensor[second]
                translation = difference[:, :3].square().sum(dim=1)
                rotation = difference[:, 3:12].square().sum(dim=1)
                hand = difference[:, 12:].square().sum(dim=1)
                values = torch.exp(-torch.stack((translation + rotation + hand, translation, rotation, hand), dim=1))
                batch_mean = values.mean(dim=0)
                batch_m2 = (values - batch_mean).square().sum(dim=0)
                # Copying the moments back synchronizes the completed GPU work;
                # reported pair_seconds is wall time, not kernel enqueue time.
                batch_mean = batch_mean.cpu().numpy()
                batch_m2 = batch_m2.cpu().numpy()
                count = len(first)
                total = seen + count
                delta = batch_mean - mean
                m2 += batch_m2 + delta * delta * (seen * count / total)
                mean += delta * (count / total)
                seen = total
        finally:
            close = getattr(stream, 'close', None)
            if close is not None:
                close()
        pair_seconds = time.perf_counter() - compute_started
    return seen, mean, m2, dict(backend='torch', device=str(tensor.device), dtype='float64',
        pair_prefetch=prefetch, feature_upload_seconds=setup_seconds, pair_seconds=pair_seconds)
