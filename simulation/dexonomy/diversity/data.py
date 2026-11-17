"""Read trusted local DexGrasp generation NPY records without simulation imports.

Native Dexonomy NPY files contain pickled dictionaries. Loading these files has
Python pickle's trust requirements; this module is for the user's local outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GraspRecord:
    path: Path
    scene_path: Path
    mesh_path: Path
    hand_name: str
    protocol: str
    task_key: str
    object_key: str
    scale: tuple[float, float, float]
    root_translation: np.ndarray
    root_rotation: np.ndarray
    q: np.ndarray
    object_translation: np.ndarray
    object_rotation: np.ndarray
    object_center: np.ndarray
    object_radius: float
    group_key: str
    candidate_identity: str | None = None


def _candidate_identity(data: dict, path: Path) -> str | None:
    """Keep the full native dataset/job/candidate ID, never only its fingerprint."""
    identities = {}
    for field in ("provenance", "experiment_provenance"):
        provenance = data.get(field)
        if provenance is None:
            continue
        if not isinstance(provenance, dict):
            raise ValueError(f"{path}: {field} must be a dictionary")
        value = provenance.get("candidate_id")
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {field}.candidate_id must be a nonempty string")
        identities[field] = value
    if len(set(identities.values())) > 1:
        raise ValueError(f"{path}: conflicting candidate identities: "
                         f"provenance.candidate_id={identities['provenance']!r}, "
                         f"experiment_provenance.candidate_id={identities['experiment_provenance']!r}")
    return identities.get("provenance", identities.get("experiment_provenance"))


def register_candidate_identity(seen: dict[str, str], identity: str | None,
                                path: str | Path, *, scope: str) -> None:
    """Reject redundant files within one admitted pool; never deduplicate poses."""
    if identity is None:
        return
    source = str(path)
    previous = seen.get(identity)
    if previous is not None and previous != source:
        raise ValueError(f"Duplicate candidate identity {identity!r} in {scope}: "
                         f"{previous} and {source}. Each candidate must have one source file "
                         "within this pool; equal poses with different candidate IDs remain valid.")
    seen[identity] = source


def discover_grasps(inputs: Iterable[str | Path] | str | Path) -> list[Path]:
    """Find one-record NPY files, preferring a production run's success trees.

    Explicit files and a directory of grasp files are accepted. A run/rank root
    with success directories reads only those directories, never stage ledgers,
    trajectories, or template feedback copies. Duplicate paths are read once.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    found: set[Path] = set()
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_file():
            if path.suffix != ".npy":
                raise ValueError(f"Expected a native grasp .npy file: {path}")
            found.add(path)
            continue
        success_roots = ([path] if path.name == "success" else
                         sorted(p for p in path.rglob("success") if p.is_dir()))
        roots = success_roots or [path]
        for root in roots:
            found.update(p.resolve() for p in root.rglob("*.npy")
                         if p.is_file() and not p.name.startswith("._"))
    if not found:
        raise ValueError("No native grasp .npy files found in the supplied inputs")
    return sorted(found)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _json(value) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _canonical_numbers(value):
    """Discard float32 serialization noise in task metadata, preserving units."""
    value = _jsonable(value)
    if isinstance(value, float):
        return float(f"{value:.7g}") if value else 0.0
    if isinstance(value, dict):
        return {key: _canonical_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_numbers(item) for item in value]
    return value


def _dictionary(path: Path) -> dict:
    try:
        value = np.load(path, allow_pickle=True)
        if not isinstance(value, np.ndarray) or value.shape != ():
            raise ValueError("expected a scalar NPY dictionary")
        value = value.item()
        if not isinstance(value, dict):
            raise ValueError("expected a dictionary")
        return value
    except Exception as exc:
        raise ValueError(f"Cannot read native dictionary {path}: {exc}") from exc


def _asset_suffix(path: Path) -> Path | None:
    # Both old .../Dexonomy/assets/object/... and the canonical assets/object/...
    # resolve against a caller-supplied asset root after relocation.
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i] == "assets" and parts[i + 1] in {"object", "hand"}:
            return Path(*parts[i + 1:])
    for i, part in enumerate(parts):
        if part in {"object", "hand"}:
            return Path(*parts[i:])
    return None


@lru_cache(maxsize=32)
def _canonical_asset_root(path: Path) -> Path:
    return path.resolve()


@lru_cache(maxsize=None)
def _resolve_cached(candidate: Path, asset_root: Path) -> Path:
    """Cache referenced files for this evaluation, independent of grasp parents."""
    if candidate.is_file():
        return candidate.resolve()
    suffix = _asset_suffix(candidate)
    if suffix is not None:
        remapped = asset_root / suffix
        if remapped.is_file():
            return remapped.resolve()
    raise FileNotFoundError(
        f"Referenced file not found: {candidate}; with asset root {asset_root}. "
        "Preserve scene files when moving outputs."
    )


def _resolve(value, *, base: Path, asset_root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"Expected a nonempty file path, got {value!r}")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else base / raw
    # In particular, an absolute scene reference must not be keyed by the
    # different parent directory of every candidate in a million-grasp run.
    return _resolve_cached(candidate, asset_root)


@lru_cache(maxsize=None)
def _scene(path: Path) -> dict:
    value = _dictionary(path)
    if not isinstance(value.get("scene"), dict) or not isinstance(value.get("task"), dict):
        raise ValueError(f"Invalid scene dictionary: {path}")
    # Keep only metadata used for grouping/geometry. Do not retain collision
    # annotations, auxiliary records, or other potentially large scene payloads.
    fields = {"type", "file_path", "scale", "pose", "size"}
    scene = {}
    for name, obj in value["scene"].items():
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid object {name!r} in scene: {path}")
        scene[name] = {key: obj[key] for key in fields if key in obj}
    return {"scene_id": value.get("scene_id"), "task": value["task"], "scene": scene}


def _vector(value, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite vector of length {size}, got {result.shape}")
    return result


def _rotation(quaternion, label: str) -> np.ndarray:
    q = _vector(quaternion, 4, label)
    length = np.linalg.norm(q)
    if length < 1e-12 or not np.isclose(length, 1., atol=1e-3, rtol=0):
        raise ValueError(f"{label} must be a unit MuJoCo wxyz quaternion")
    w, x, y, z = q / length
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _scale(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.repeat(result, 3)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(f"{label} must contain three positive finite scale values")
    return result


@lru_cache(maxsize=None)
def _geometry(mesh_path: Path, scale: tuple[float, float, float]) -> tuple[np.ndarray, float]:
    """Exact area-weighted center/RMS of the scaled triangular mesh surface.

    Integrating over triangle interiors avoids vertex-density bias and random
    per-run sampling noise. This is the continuous surface counterpart of V0's
    uniform-surface-point RMS radius, with the center defined by surface area.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    triangles = np.asarray(mesh.triangles, dtype=np.float64) * np.asarray(scale)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError(f"Mesh has no triangular surface: {mesh_path}")
    if not np.isfinite(triangles).all():
        raise ValueError(f"Mesh contains nonfinite vertices: {mesh_path}")
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Mesh has no positive surface area: {mesh_path}")
    weights = areas / total
    center = np.einsum("n,nj->j", weights, triangles.mean(axis=1))
    centered = triangles - center
    moments = (np.sum(centered**2, axis=(1, 2)) +
               np.sum(centered.sum(axis=1)**2, axis=1)) / 12
    radius = float(np.sqrt(np.dot(weights, moments)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Mesh has no positive RMS radius: {mesh_path}")
    center.setflags(write=False)
    return center, radius


def clear_caches() -> None:
    """Start a fresh evaluation after on-disk assets may have changed.

    Scene, geometry, and path caches cover all distinct inputs in one evaluation
    (including 5,751 objects at six scales), and retain only small results. The
    public dataset evaluator calls this once at its start. Call it explicitly
    between direct ``load_grasp`` runs when changing assets in a long-lived
    process. Do not mutate assets while an evaluation is running.
    """
    _canonical_asset_root.cache_clear()
    _resolve_cached.cache_clear()
    _scene.cache_clear()
    _geometry.cache_clear()


def _tabletop_geometry(scene: dict, target: str, scene_pose: np.ndarray) -> list[dict]:
    """Canonical support geometry in the initial object frame, never terminal."""
    object_rotation = _rotation(scene_pose[3:], "scene quaternion")
    support = []

    def rounded(value):
        array = np.round(np.asarray(value, dtype=np.float64), decimals=6)
        array[array == 0] = 0.0  # JSON -0.0 and +0.0 must name the same group.
        return array.tolist()

    for name, obj in sorted(scene["scene"].items()):
        if name == target:
            continue
        if obj.get("type") != "plane":
            raise ValueError("Tabletop grouping supports only the target rigid object "
                             "and support planes; other environment objects are unsupported")
        pose = _vector(obj.get("pose"), 7, f"plane {name} pose")
        rotation = _rotation(pose[3:], f"plane {name} quaternion")
        size = _vector(obj.get("size"), 3, f"plane {name} size")
        if np.any(size < 0):
            raise ValueError(f"plane {name} size must be nonnegative")
        support.append({
            "name": name, "type": "plane",
            "translation_object_m": rounded(object_rotation.T @ (pose[:3] - scene_pose[:3])),
            "rotation_object": rounded(object_rotation.T @ rotation),
            "size": rounded(size),
        })
    if not support:
        raise ValueError("Tabletop grouping requires at least one support plane")
    return support


def _success(data: dict, path: Path) -> bool:
    verdicts = []
    for field, flag in (("stability_metrics", "passes"),
                        ("tabletop_metrics", "joint_success")):
        if field in data:
            metrics = data[field]
            if not isinstance(metrics, dict) or flag not in metrics:
                raise ValueError(f"{path}: {field} is missing its {flag} verdict")
            verdict = metrics[flag]
            if not isinstance(verdict, (bool, np.bool_)):
                raise ValueError(f"{path}: {field}.{flag} must be a boolean")
            verdicts.append(bool(verdict))
    if verdicts:
        return all(verdicts)
    if "success" in path.parts:
        return True
    raise ValueError(
        f"{path}: success is unknown; provide the generator's success directory "
        "or native stability/tabletop metrics, or explicitly select all grasps"
    )


def _protocol(data: dict, scene: dict, requested: str, path: Path) -> str:
    clues = set()
    if "tabletop_metrics" in data or "tabletop_initial_scene" in data:
        clues.add("tabletop")
    if "stability_metrics" in data or data.get("pipeline_stage") == "stable_grasp":
        clues.add("floating")
    task = scene["task"]
    planes = [v for v in scene["scene"].values()
              if isinstance(v, dict) and v.get("type") == "plane"]
    if task.get("type") == "slide" and planes:
        clues.add("tabletop")
    if task.get("type") == "force_closure" and not planes:
        clues.add("floating")
    if len(clues) > 1:
        raise ValueError(f"{path}: conflicting Floating/Tabletop protocol metadata")
    if requested != "auto":
        if clues and requested not in clues:
            raise ValueError(f"{path}: requested {requested} contradicts native protocol metadata")
        return requested
    if not clues:
        raise ValueError(f"{path}: cannot infer protocol; specify floating or tabletop explicitly")
    return clues.pop()


def load_grasp(path: str | Path, *, asset_root: str | Path,
               protocol: str = "auto", selection: str = "success") -> GraspRecord | None:
    """Load a single generated grasp; return None only for an explicit failure.

    Unknown formats and malformed/missing metadata fail visibly. ``selection``
    is either ``success`` (default) or ``all``; it never substitutes pregrasp,
    squeeze, or Tabletop terminal control targets for the actual grasp pose.
    ``object_center`` is in scaled object-local meters. To center a palm point,
    use R_object.T @ (t_palm - object_translation) - object_center.
    """
    if protocol not in {"auto", "floating", "tabletop"}:
        raise ValueError("protocol must be auto, floating, or tabletop")
    if selection not in {"success", "all"}:
        raise ValueError("selection must be success or all")
    path = Path(path).expanduser()
    # discover_grasps already canonicalizes each path. Avoid repeating resolve's
    # path-component stat calls for every grasp; relative direct calls still work.
    if not path.is_absolute():
        path = path.resolve()
    asset_root = Path(asset_root).expanduser()
    if not asset_root.is_absolute():
        asset_root = Path.cwd() / asset_root
    asset_root = _canonical_asset_root(asset_root)
    data = _dictionary(path)
    if "grasp_qpos" not in data:
        raise ValueError(f"{path}: not a native grasp record (missing grasp_qpos); "
                         "pass success/ or a directory of individual grasps, not a stage ledger")
    if selection == "success" and not _success(data, path):
        return None
    candidate_identity = _candidate_identity(data, path)
    pose = np.asarray(data["grasp_qpos"], dtype=np.float64)
    if pose.shape == (1, 29):
        pose = pose[0]
    pose = _vector(pose, 29, "Sharpa grasp_qpos [xyz,wxyz,q22]")
    hand = data.get("hand_name")
    if hand != "sharpa_v2":
        raise ValueError(f"{path}: unsupported hand_name {hand!r}; expected sharpa_v2")
    if "scene_path" not in data:
        raise ValueError(f"{path}: missing scene_path")
    scene_path = _resolve(data["scene_path"], base=path.parent, asset_root=asset_root)
    scene = _scene(scene_path)
    target = scene["task"].get("obj_name")
    if not isinstance(target, str) or target not in scene["scene"]:
        raise ValueError(f"{scene_path}: task.obj_name does not identify an object")
    obj = scene["scene"][target]
    if obj.get("type") != "rigid_object":
        raise ValueError(f"{scene_path}: only rigid_object targets are supported")
    scale_value = _scale(obj.get("scale"), "scene object scale")
    if "obj_scale" in data and not np.allclose(
            _scale(data["obj_scale"], "grasp obj_scale"), scale_value, atol=1e-8, rtol=1e-6):
        raise ValueError(f"{path}: obj_scale differs from the referenced scene")
    # Current generator serializes scale once as float32 and once as float64.
    # Seven significant digits preserve source precision without splitting them.
    scale = tuple(float(f"{x:.7g}") for x in scale_value)
    mesh_path = _resolve(obj.get("file_path"), base=scene_path.parent, asset_root=asset_root)
    center, radius = _geometry(mesh_path, scale)
    object_pose = _vector(data.get("obj_pose", obj.get("pose")), 7, "obj_pose [xyz,wxyz]")
    object_rotation = _rotation(object_pose[3:], "obj_pose quaternion")
    scene_pose = _vector(obj.get("pose"), 7, "scene object pose")
    if (not np.allclose(object_pose[:3], scene_pose[:3], atol=1e-6, rtol=1e-6)
            or not np.allclose(object_rotation, _rotation(scene_pose[3:], "scene quaternion"),
                               atol=1e-5, rtol=1e-5)):
        raise ValueError(f"{path}: obj_pose differs from the initial scene; cannot pair "
                         "grasp_qpos with a different object time/frame")
    protocol_name = _protocol(data, scene, protocol, path)
    task = _canonical_numbers({k: v for k, v in scene["task"].items() if k != "obj_name"})
    if protocol_name == "tabletop":
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"{scene_path}: Tabletop grouping requires scene_id for its initial pose")
        task["initial_scene_id"] = scene_id
        task["initial_scene_geometry"] = _tabletop_geometry(scene, target, scene_pose)
    task_key = _json(task)
    mesh_identity = _asset_suffix(mesh_path)
    object_key = f"{target}:{mesh_identity.as_posix() if mesh_identity else mesh_path.as_posix()}"
    group_key = _json(dict(object=object_key, scale=scale, hand=hand,
                           protocol=protocol_name, task=task))
    return GraspRecord(
        path=path, scene_path=scene_path, mesh_path=mesh_path, hand_name=hand,
        protocol=protocol_name, task_key=task_key, object_key=object_key, scale=scale,
        root_translation=pose[:3].copy(), root_rotation=_rotation(pose[3:7], "hand quaternion"),
        q=pose[7:].copy(), object_translation=object_pose[:3].copy(),
        object_rotation=object_rotation, object_center=center.copy(), object_radius=radius,
        group_key=group_key, candidate_identity=candidate_identity,
    )
