"""Explicit canonical-data conversion. Simulation is not used for training."""
import ast
import json
from pathlib import Path

import numpy as np

from .data import PHASES, REPRESENTATION, digest, load_manifest, normalize_split, split_objects, validate_records, write_json


def catalog_manifest(catalog, output, fraction=0.1, seed=0):
    """Catalog declares train/test; the earlier evaluation label is accepted as an alias."""
    catalog, output = Path(catalog).resolve(), Path(output).resolve()
    rows = json.loads(catalog.read_text())
    objects = {}
    for row in rows:
        split = normalize_split(row["official_split"])
        if split not in ("train", "test"):
            raise ValueError("catalog official_split must be train/test")
        if row["object_id"] in objects and objects[row["object_id"]] != split:
            raise ValueError("official split leakage")
        objects[row["object_id"]] = split
    splits = split_objects([k for k, v in objects.items() if v == "train"],
                           [k for k, v in objects.items() if v == "test"], fraction, seed)
    for row in rows:
        row["split"] = splits[row["object_id"]]
        del row["official_split"]
        for key in ("path", "point_cloud_h5", "grasp_h5", "scene_path"):
            if row.get(key): row[key] = str((catalog.parent / row[key]).resolve())
    data = {"schema_version": 1, "representation": REPRESENTATION,
            "dataset_id": catalog.stem, "split_seed": seed, "validation_fraction": fraction,
            "catalog_sha256": digest(catalog), "records": rows}
    # Validate a private temporary manifest before publishing the requested file.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory) / "manifest.json"
        write_json(temp, data)
        validate_records(load_manifest(temp)["records"])
    write_json(output, data)
    return data


def forward_keypoints(qpos, xml, ik_source, joint_names):
    """Map explicitly named input joints to the reference hand, then perform FK."""
    import mujoco
    tree = ast.parse(Path(ik_source).read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "kp_name_lst" for t in n.targets))
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nq != 22 or len(names) != 28:
        raise ValueError("expected the 22-joint, 28-site reference Shadow Hand")
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if len(joint_names) != 22 or len(set(joint_names)) != 22 or set(joint_names) != set(model_names):
        raise ValueError("input joint names must match the reference hand exactly")
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape[-1] != 29 or not np.isfinite(qpos).all():
        raise ValueError("expected finite [..., xyz+wxyz+22 joints]")
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in names]
    if min(sites) < 0:
        raise ValueError("keypoint sites missing in XML")
    data = mujoco.MjData(model)
    result = []
    for pose in qpos.reshape(-1, 29):
        quat = pose[3:7]
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
            raise ValueError("wxyz quaternion must have unit norm")
        for j, name in enumerate(model_names):
            data.qpos[model.jnt_qposadr[j]] = pose[7 + joint_names.index(name)]
        mujoco.mj_forward(model, data)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
        result.append(data.site_xpos[sites] @ rotation.reshape(3, 3).T + pose[:3])
    return np.asarray(result, dtype=np.float32).reshape(*qpos.shape[:-1], 28, 3)


def convert_qpos(source, cloud, output, kpgrasp_root, joint_order, scale=1.0):
    """Input is pickle-free NPZ with *_qpos arrays; cloud is unscaled NPY."""
    root, output = Path(kpgrasp_root), Path(output)
    if output.exists() or Path(str(output) + ".provenance.json").exists():
        raise FileExistsError(output)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    xml = root / "assets/hand/shadow/right_hand_w_kp.xml"
    ik = root / "src/dexgrasp/kinematics/ik.py"
    names = json.loads(Path(joint_order).read_text())
    with np.load(source, allow_pickle=False) as f:
        poses = np.stack([f[f"{phase}_qpos"] for phase in PHASES], axis=1)
    if poses.ndim != 3 or poses.shape[1:] != (3, 29) or not len(poses):
        raise ValueError("each phase must have shape (G,29), G>0")
    points = np.load(cloud, allow_pickle=False).astype(np.float32) * scale
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("invalid point cloud")
    kp = forward_keypoints(poses, xml, ik, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, point_cloud=points, keypoints=kp)
    write_json(str(output) + ".provenance.json", {
        "input_sha256": digest(source), "cloud_sha256": digest(cloud), "xml_sha256": digest(xml),
        "ik_source_sha256": digest(ik), "joint_order": names, "cloud_scale": scale,
        "representation": REPRESENTATION, "quaternion": "wxyz",
        "note": "FK conversion, not physical validation. Pose translations already in metres."})
    return {"status": "converted", "grasps": len(poses), "output": str(output)}
