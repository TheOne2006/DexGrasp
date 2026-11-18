"""Shared I/O for immutable two-rank final evaluation artifacts."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
        return checksum.hexdigest()


def load_decoded_shard(folder):
    path = folder / "decoded.npy"
    receipt = json.loads((folder / "decoded.done.json").read_text())
    if receipt["sha256"] != digest(path):
        raise ValueError(f"Changed decoded poses in {folder}")
    part = np.load(path, mmap_mode="r")
    if receipt["count"] != len(part):
        raise ValueError(f"Decoded count mismatch in {folder}")
    return part


def decoded(root, group, mode="test"):
    parts = []
    for rank in (0, 1):
        folder = root / group / f"rank{rank}" / mode
        part = load_decoded_shard(folder)
        with np.load(folder / "index.npz") as index:
            np.testing.assert_array_equal(index["index"], rank + 2 * np.arange(len(part)))
        parts.append(part)
    result = np.empty(sum(map(len, parts)), dtype=parts[0].dtype)
    result[::2], result[1::2] = parts
    return result


DTYPE = np.dtype(
    [
        ("finite", "?"),
        ("decoded", "?"),
        ("fit", "?"),
        ("direction_pass", "?"),
        ("mean_mm", "f4"),
        ("max_mm", "f4"),
        ("angle_deg", "f4"),
        ("ik_seconds", "f4"),
        ("q", "f4", (22,)),
        ("R", "f4", (3, 3)),
        ("t", "f4", (3,)),
    ]
)


def save(path, value):
    p = Path(path)
    tmp = p.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def now():
    return datetime.now(timezone.utc).isoformat()


_DEC = None
_DECODE_CACHE = {}


def init_decode(spec):
    global _DEC
    from dexgrasp.test.formal import official_hand

    _DEC = official_hand(spec)


def decode_chunk(job):
    from dexgrasp.test.metrics import decode_metrics

    path, start, end = job
    p = Path(path)
    if path not in _DECODE_CACHE:
        _DECODE_CACHE.clear()
        with np.load(p / "index.npz") as index:
            dirs = index["directions"]
        _DECODE_CACHE[path] = (np.load(p / "keypoints.npy", mmap_mode="r"), dirs)
    k, d = _DECODE_CACHE[path]
    result = np.zeros(end - start, dtype=DTYPE)
    for i in range(start, end):
        r = decode_metrics(k[i], _DEC, d[i])
        j = i - start
        for a, b in [
            ("finite", "finite_output"),
            ("decoded", "decoded"),
            ("fit", "fit_success"),
            ("direction_pass", "direction_pass"),
        ]:
            result[a][j] = r[b]
        for a, b in [
            ("mean_mm", "mean_keypoint_error_mm"),
            ("max_mm", "max_keypoint_error_mm"),
            ("angle_deg", "direction_error_deg"),
            ("ik_seconds", "ik_seconds"),
        ]:
            result[a][j] = r.get(b, np.nan)
        for key in ("q", "R", "t"):
            result[key][j] = r.get(key, np.nan)
    return start, result


def decode_shard(folder, pool, *, chunk_size, progress_workers=None):
    """Decode one inference rank with an initialized pool, then publish its receipt."""
    count = len(np.load(folder / "keypoints.npy", mmap_mode="r"))
    poses = np.empty(count, dtype=DTYPE)
    jobs = ((str(folder), i, min(i + chunk_size, count)) for i in range(0, count, chunk_size))
    started = time.monotonic()
    last_update = None
    for start, chunk in pool.map(decode_chunk, jobs):
        stop = start + len(chunk)
        poses[start:stop] = chunk
        if progress_workers is not None:
            tick = time.monotonic()
            if last_update is None or tick - last_update >= 2 or stop == count:
                save(folder / "decode_progress.json", dict(
                    completed=stop, total=count, workers=progress_workers,
                    seconds=tick - started, updated_at=now(),
                ))
                last_update = tick
    path = folder / "decoded.npy"
    np.save(path, poses)
    receipt = dict(count=count, sha256=digest(path))
    if progress_workers is not None:
        receipt["seconds"] = time.monotonic() - started
    save(folder / "decoded.done.json", receipt)
