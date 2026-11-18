"""Resolve object and mesh identities from the manifests used to pack the formal corpus."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from .dataset import sha256


def split_objects(root, identity, split):
    file = Path(root) / "split_protocol.json"
    if sha256(file) != identity["split_protocol_sha256"]:
        raise ValueError("Packed object split differs from its frozen identity")
    mapping = json.loads(file.read_text())["object_splits"]
    return sorted(obj for obj, value in mapping.items() if value == split)


def observation_objects(packages, identity, ids, *, workers=24):
    """Read source identities in the order of the supplied observation indices."""
    paths = []
    for name, source in identity["source_identity"].items():
        package = Path(packages) / name
        manifest = package / "MANIFEST.json"
        if sha256(manifest) != source["manifest_sha256"]:
            raise ValueError(f"Changed source manifest {manifest}")
        paths.extend(
            (package / f["path"], f["sha256"])
            for f in json.loads(manifest.read_text())["files"]
            if f["path"].startswith("observations/") and f["path"].endswith(".npz")
        )

    def read(i):
        path, expected = paths[i]
        if sha256(path) != expected:
            raise ValueError(f"Changed observation {path}")
        with np.load(path, allow_pickle=False) as obs:
            return str(obs["object_id"]), str(obs["source_mesh_sha256"])

    ids = np.asarray(ids)
    if (ids < 0).any() or (ids >= len(paths)).any():
        raise ValueError("Packed observation index outside source manifests")
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(read, ids))
