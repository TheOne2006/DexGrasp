"""Validate an upstream planning request and convert it to Dexonomy assets.

The public boundary is JSON plus OBJ meshes.  Dexonomy's pickle-backed ``.npy``
scene file is deliberately kept as an internal artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import trimesh


SCHEMA_VERSION = "dexonomyopt-scene/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RequestError(ValueError):
    """Raised when an upstream request violates the v1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RequestError(message)


def _no_extra(value: dict[str, Any], allowed: set[str], field: str) -> None:
    extra = sorted(set(value) - allowed)
    _require(not extra, f"{field} contains unsupported fields: {', '.join(extra)}")


def _vector(value: Any, length: int, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{field} must contain numbers") from exc
    _require(result.shape == (length,), f"{field} must have shape ({length},)")
    _require(np.isfinite(result).all(), f"{field} must contain only finite values")
    return result


def _pose(value: Any, field: str) -> np.ndarray:
    _require(isinstance(value, dict), f"{field} must be an object")
    _no_extra(value, {"position", "quaternion_wxyz"}, field)
    position = _vector(value.get("position"), 3, f"{field}.position")
    quaternion = _vector(
        value.get("quaternion_wxyz"), 4, f"{field}.quaternion_wxyz"
    )
    norm = float(np.linalg.norm(quaternion))
    _require(norm > 1e-8, f"{field}.quaternion_wxyz must be non-zero")
    _require(
        abs(norm - 1.0) <= 1e-3,
        f"{field}.quaternion_wxyz must be normalized (norm={norm:.6g})",
    )
    return np.concatenate([position, quaternion / norm]).astype(np.float32)


def _asset_path(request_dir: Path, raw_path: Any, field: str) -> Path:
    _require(isinstance(raw_path, str) and raw_path, f"{field} must be a path string")
    path = (request_dir / raw_path).resolve() if not os.path.isabs(raw_path) else Path(raw_path).resolve()
    _require(path.is_file(), f"{field} does not exist: {path}")
    _require(path.suffix.lower() == ".obj", f"{field} must be an OBJ file: {path}")
    return path


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def _mesh_properties(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception as exc:
        raise RequestError(f"failed to load mesh {mesh_path}: {exc}") from exc
    _require(isinstance(mesh, trimesh.Trimesh), f"asset is not a mesh: {mesh_path}")
    _require(len(mesh.vertices) > 0 and len(mesh.faces) > 0, f"mesh is empty: {mesh_path}")
    _require(np.isfinite(mesh.vertices).all(), f"mesh has non-finite vertices: {mesh_path}")

    center = np.asarray(mesh.center_mass, dtype=np.float64)
    if not np.isfinite(center).all():
        center = np.asarray(mesh.centroid, dtype=np.float64)
    try:
        obb = np.asarray(mesh.bounding_box_oriented.extents, dtype=np.float64)
    except Exception:
        obb = np.asarray(mesh.bounding_box.extents, dtype=np.float64)

    volume = abs(float(mesh.volume)) if mesh.is_volume else 0.0
    if not math.isfinite(volume) or volume <= 1e-12:
        try:
            volume = abs(float(mesh.convex_hull.volume))
        except Exception:
            volume = 0.0
    _require(volume > 1e-12, f"cannot estimate a positive volume for mesh: {mesh_path}")
    return center, obb, volume


def _write_mjcf(
    path: Path, entity_id: str, surface_mesh: Path, collision_meshes: list[Path]
) -> None:
    mesh_paths = [surface_mesh, *collision_meshes]
    names = ["visual", *[f"collision_{i:03d}" for i in range(len(collision_meshes))]]
    assets = "\n".join(
        f'    <mesh name="{name}" file="{escape(_relative(mesh, path.parent))}"/>'
        for name, mesh in zip(names, mesh_paths)
    )
    collisions = "\n".join(
        f'      <geom name="collision_{i:03d}" type="mesh" mesh="collision_{i:03d}"/>'
        for i in range(len(collision_meshes))
    )
    content = f"""<mujoco model="{escape(entity_id)}">
  <compiler angle="radian" meshdir="."/>
  <asset>
{assets}
  </asset>
  <worldbody>
    <body name="object">
      <geom name="visual" type="mesh" mesh="visual" contype="0" conaffinity="0" density="0"/>
{collisions}
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(content, encoding="utf-8")


def _write_urdf(path: Path, entity_id: str, collision_meshes: list[Path]) -> None:
    links: list[str] = []
    joints: list[str] = []
    for index, mesh in enumerate(collision_meshes):
        link = f"collision_{index:03d}"
        filename = escape(_relative(mesh, path.parent))
        links.append(
            f"""  <link name="{link}">
    <visual><geometry><mesh filename="{filename}"/></geometry></visual>
    <collision><geometry><mesh filename="{filename}"/></geometry></collision>
  </link>"""
        )
        if index:
            joints.append(
                f"""  <joint name="{link}_joint" type="fixed">
    <parent link="collision_000"/><child link="{link}"/>
  </joint>"""
            )
    content = "\n".join(
        [f'<robot name="{escape(entity_id)}">', *links, *joints, "</robot>", ""]
    )
    path.write_text(content, encoding="utf-8")


def _write_info(
    path: Path,
    center: np.ndarray,
    obb: np.ndarray,
    reference_volume: float,
) -> None:
    # Dexonomy expects mass/density to describe the unscaled source mesh.  The
    # requested physical mass is passed separately as obj_mass at runtime.
    data = {
        "gravity_center": center.tolist(),
        "obb": obb.tolist(),
        "scale": 1.0,
        "density": 1.0,
        "mass": reference_volume,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def convert_request(request_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Convert one v1 upstream request and return the generated artifact paths."""
    request_path = Path(request_path).resolve()
    output_dir = Path(output_dir).resolve()
    _require(request_path.is_file(), f"request file does not exist: {request_path}")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RequestError(f"cannot read request JSON: {exc}") from exc

    _require(isinstance(request, dict), "request root must be an object")
    _no_extra(
        request,
        {"schema_version", "request_id", "world_frame", "units", "scene"},
        "request",
    )
    _require(request.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION!r}")
    request_id = request.get("request_id")
    _require(isinstance(request_id, str) and _SAFE_ID.fullmatch(request_id) is not None,
             "request_id must use only letters, digits, '.', '_' or '-'")
    _require(request.get("world_frame") == "world", "world_frame must be 'world' in v1")
    units = request.get("units")
    _require(
        units == {"length": "m", "mass": "kg", "time": "s", "angle": "rad"},
        "units must be exactly m/kg/s/rad",
    )

    scene = request.get("scene")
    _require(isinstance(scene, dict), "scene must be an object")
    _no_extra(scene, {"entities", "visualization_camera"}, "scene")
    visualization_camera = scene.get("visualization_camera")
    dex_visualization = None
    if visualization_camera is not None:
        _require(
            isinstance(visualization_camera, dict),
            "scene.visualization_camera must be an object",
        )
        _no_extra(
            visualization_camera,
            {"position", "xyaxes"},
            "scene.visualization_camera",
        )
        camera_position = _vector(
            visualization_camera.get("position"),
            3,
            "scene.visualization_camera.position",
        )
        camera_xyaxes = _vector(
            visualization_camera.get("xyaxes"),
            6,
            "scene.visualization_camera.xyaxes",
        )
        camera_x = camera_xyaxes[:3]
        camera_y = camera_xyaxes[3:]
        _require(
            np.linalg.norm(camera_x) > 1e-6
            and np.linalg.norm(camera_y) > 1e-6
            and np.linalg.norm(np.cross(camera_x, camera_y)) > 1e-6,
            "scene.visualization_camera.xyaxes must contain two non-collinear axes",
        )
        dex_visualization = {
            "camera_pos": camera_position.astype(np.float32),
            "camera_xyaxes": camera_xyaxes.astype(np.float32),
        }
    entities = scene.get("entities")
    _require(isinstance(entities, list) and entities, "scene.entities must be a non-empty array")
    ids: set[str] = set()
    targets = []
    for i, entity in enumerate(entities):
        field = f"scene.entities[{i}]"
        _require(isinstance(entity, dict), f"{field} must be an object")
        _no_extra(
            entity,
            {"id", "role", "type", "pose", "scale", "half_size_m", "assets", "physical"},
            field,
        )
        entity_id = entity.get("id")
        _require(isinstance(entity_id, str) and _SAFE_ID.fullmatch(entity_id) is not None,
                 f"{field}.id contains unsupported characters")
        _require(entity_id not in ids, f"duplicate entity id: {entity_id}")
        ids.add(entity_id)
        _require(entity.get("role") in {"target", "static"}, f"{field}.role must be target or static")
        _require(entity.get("type") in {"rigid_mesh", "plane"}, f"{field}.type must be rigid_mesh or plane")
        _pose(entity.get("pose"), f"{field}.pose")
        if entity["role"] == "target":
            targets.append(entity)
            _require(entity["type"] == "rigid_mesh", "the target must be a rigid_mesh")
        if entity["type"] == "plane":
            _require(entity["role"] == "static", f"{field}: a plane must be static")
            half_size = _vector(entity.get("half_size_m"), 2, f"{field}.half_size_m")
            _require((half_size > 0).all(), f"{field}.half_size_m must be positive")
            plane_pose = _pose(entity["pose"], f"{field}.pose")
            _require(
                abs(float(plane_pose[3])) >= 1.0 - 1e-5
                and np.linalg.norm(plane_pose[4:]) <= 1e-5,
                f"{field}: v1 planes must be horizontal; use a static rigid_mesh for tilted surfaces",
            )
    _require(len(targets) == 1, "scene.entities must contain exactly one target")
    target = targets[0]

    output_scene = output_dir / "scene.npy"
    grasp_scene_path = output_dir / "grasp_scene.npy"
    stability_scene_path = output_dir / "stability_scene.npy"
    existing_outputs = [
        path
        for path in (
            output_scene,
            grasp_scene_path,
            stability_scene_path,
        )
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise RequestError(
            f"output already exists (pass --overwrite): {existing_outputs[0]}"
        )
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    dex_scene: dict[str, dict[str, Any]] = {}
    request_dir = request_path.parent
    target_mass = None
    target_friction = np.full(3, np.nan, dtype=np.float32)
    for i, entity in enumerate(entities):
        entity_id = entity["id"]
        pose = _pose(entity["pose"], f"scene.entities[{i}].pose")
        if entity["type"] == "plane":
            half_size = _vector(entity["half_size_m"], 2, f"scene.entities[{i}].half_size_m")
            dex_scene[entity_id] = {
                "type": "plane",
                "pose": pose,
                "size": np.asarray([half_size[0], half_size[1], 0.1], dtype=np.float32),
            }
            continue

        assets = entity.get("assets")
        _require(isinstance(assets, dict), f"entity {entity_id}.assets must be an object")
        _no_extra(assets, {"surface_mesh", "collision_meshes"}, f"entity {entity_id}.assets")
        surface = _asset_path(request_dir, assets.get("surface_mesh"), f"entity {entity_id}.assets.surface_mesh")
        raw_collision = assets.get("collision_meshes")
        _require(isinstance(raw_collision, list) and raw_collision,
                 f"entity {entity_id}.assets.collision_meshes must be a non-empty array")
        collision = [
            _asset_path(request_dir, value, f"entity {entity_id}.assets.collision_meshes[{j}]")
            for j, value in enumerate(raw_collision)
        ]
        for collision_mesh in collision:
            _mesh_properties(collision_mesh)
        scale = _vector(entity.get("scale"), 3, f"entity {entity_id}.scale")
        _require((scale > 0).all(), f"entity {entity_id}.scale must be positive")
        center, obb, volume = _mesh_properties(surface)

        physical = entity.get("physical", {})
        _require(isinstance(physical, dict), f"entity {entity_id}.physical must be an object")
        _no_extra(
            physical,
            {"mass_kg", "center_of_mass_local_m", "friction"},
            f"entity {entity_id}.physical",
        )
        friction = physical.get("friction", {})
        _require(isinstance(friction, dict), f"entity {entity_id}.physical.friction must be an object")
        _no_extra(
            friction,
            {"sliding", "torsional", "rolling"},
            f"entity {entity_id}.physical.friction",
        )
        friction_values = []
        for friction_name in ["sliding", "torsional", "rolling"]:
            value = friction.get(friction_name, np.nan)
            _require(
                isinstance(value, (int, float, np.floating))
                and (math.isnan(value) or (math.isfinite(value) and value >= 0)),
                f"entity {entity_id}.physical.friction.{friction_name} must be non-negative",
            )
            friction_values.append(float(value))
        if "center_of_mass_local_m" in physical:
            center_m = _vector(physical["center_of_mass_local_m"], 3,
                               f"entity {entity_id}.physical.center_of_mass_local_m")
            center = center_m / scale
        if entity["role"] == "target":
            mass = physical.get("mass_kg")
            _require(isinstance(mass, (int, float)) and math.isfinite(mass) and mass > 0,
                     f"target {entity_id}.physical.mass_kg must be positive")
            target_mass = float(mass)
            target_friction = np.asarray(friction_values, dtype=np.float32)

        xml_path = generated_dir / f"{entity_id}.xml"
        urdf_path = generated_dir / f"{entity_id}.urdf"
        info_path = generated_dir / f"{entity_id}.info.json"
        _write_mjcf(xml_path, entity_id, surface, collision)
        _write_urdf(urdf_path, entity_id, collision)
        _write_info(info_path, center, obb, volume)
        dex_scene[entity_id] = {
            "type": "rigid_object",
            "file_path": _relative(surface, output_dir),
            "xml_path": _relative(xml_path, output_dir),
            "urdf_path": _relative(urdf_path, output_dir),
            "info_path": _relative(info_path, output_dir),
            "scale": scale.astype(np.float32),
            "pose": pose,
            "fix_root": entity["role"] == "static",
        }

    common_cfg = {"scene_id": request_id}
    if dex_visualization is not None:
        common_cfg["visualization"] = dex_visualization

    grasp_cfg = {
        **common_cfg,
        "scene": dex_scene,
        "task": {
            "type": "force_closure",
            "obj_name": target["id"],
        },
        "pipeline_stage": "grasp_generation",
    }
    stability_cfg = {
        **common_cfg,
        # Force-closure robustness must not receive support or friction from a
        # table, shelf, drawer, or any other fixture.  The free target is the
        # only scene entity; MuJoCo supplies the synthetic ±g fields.
        "scene": {target["id"]: dex_scene[target["id"]]},
        "task": {
            "type": "force_closure",
            "obj_name": target["id"],
        },
        "pipeline_stage": "grasp_stability",
    }
    np.save(grasp_scene_path, grasp_cfg, allow_pickle=True)
    np.save(stability_scene_path, stability_cfg, allow_pickle=True)
    np.save(output_scene, grasp_cfg, allow_pickle=True)

    normalized_path = output_dir / "normalized_request.json"
    normalized_path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = output_dir / "conversion_report.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "target_id": target["id"],
        "target_mass_kg": target_mass,
        "dexonomy_scene": output_scene.name,
        "grasp_scene": grasp_scene_path.name,
        "stability_scene": stability_scene_path.name,
        "dexonomy_runtime_overrides": {"obj_mass": target_mass},
        "notes": [
            "Run Dexonomy init and grasp refinement with grasp_scene.npy, which retains scene geometry.",
            "Rebind refined grasps to stability_scene.npy before six-direction force-closure evaluation.",
            "stability_scene.npy contains only the free target object; no fixture can support the object.",
            "Static rigid meshes use fix_root=true and require this repository's static-object patch.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "scene": output_scene,
        "grasp_scene": grasp_scene_path,
        "stability_scene": stability_scene_path,
        "normalized_request": normalized_path,
        "report": report_path,
    }
