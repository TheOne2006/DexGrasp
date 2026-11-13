"""Task-local preparation policy. Does not gate physics or safety checks."""
from collections import OrderedDict
from pathlib import Path
import os
import threading
import time
import xml.etree.ElementTree as ET

_CACHE = OrderedDict()
_LOCK = threading.RLock()
_CACHE_LIMIT = 128


def validation_enabled():
    value = os.environ.get('DEXONOMY_ENGINEERING_VALIDATION', '0')
    if value not in ('0', '1'):
        raise ValueError('DEXONOMY_ENGINEERING_VALIDATION must be 0 or 1')
    return value == '1'


def _stamp(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _xml_dependencies(path, seen=None):
    # Includes affect parsed declarations. Mesh files are opened by each compile;
    # no compiled model or mesh payload is cached here.
    seen = {} if seen is None else seen
    path = path.resolve()
    if path in seen:
        return seen
    seen[path] = _stamp(path)
    for node in ET.parse(path).iter('include'):
        _xml_dependencies(path.parent / node.attrib['file'], seen)
    return seen


def load_spec(xml_path):
    """Share immutable parsed declarations across stages, copy for each caller.

    Revalidate all included XML identities on every hit. Bound cache lifetime and
    refuse concurrent dependency changes; no stale parsed object is returned.
    """
    import mujoco
    path = Path(xml_path).resolve()
    start = time.perf_counter()
    with _LOCK:
        saved = _CACHE.get(path)
        hit = saved is not None and all(_stamp(p) == stamp for p, stamp in saved[0].items())
        if not hit:
            before = _xml_dependencies(path)
            spec = mujoco.MjSpec.from_file(str(path))
            after = _xml_dependencies(path)
            if before != after:
                raise RuntimeError('XML dependencies changed while parsing')
            saved = (after, spec)
            _CACHE[path] = saved
        _CACHE.move_to_end(path)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
        result = saved[1].copy()
    from dexonomy.acceptance_audit import event
    event('shared_spec_cache', hit=hit, path=str(path), wall_s=time.perf_counter()-start)
    return result


def clear_preparation_cache():
    with _LOCK:
        _CACHE.clear()
