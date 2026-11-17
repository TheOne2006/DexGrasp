"""Sharpa V2 forward kinematics, using its compiled MJCF as the source of truth.

Only CPU MuJoCo is used to compile static model metadata. Evaluation uses batched
NumPy transforms; it does not initialize Dexonomy, Torch, Warp, or a renderer.
"""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path

import mujoco
import numpy as np


JOINT_NAMES = (
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE",
    "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA",
    "right_pinky_PIP", "right_pinky_DIP",
)
PALM_BODY = "right_hand_C_MC"
DEFAULT_KEYPOINT_SCHEME = "project-urdf-landmarks-diagnostic-v1"
DEFAULT_KEYPOINT_COUNT = 21
LEGACY_KEYPOINT_SCHEME = "segment20_legacy"
LANDMARK_RESOURCE = "resources/sharpa21_v1.json"
LANDMARK_RESOURCE_SHA256 = "42fb61d5d7d48d2160be76e46b6f778a253626dc2464c22d9346771a56e53e05"


def _rpy_rotation(rpy):
    """URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class SharpaHand:
    """Fixed right Sharpa V2, with 22 independent revolute joints in MJCF order.

    ``keypoints`` accepts radians in shape (N, 22), or a single (22,) vector,
    and returns metres in the fixed palm frame. Joint values are not clipped:
    the metric describes the recorded pose, including small simulation limit
    excursions. The default is the supplied project 21-point definition,
    equally weighted 1/21, including the wrist and rigid palm anchors. Points
    remain in the official palm frame used by native generation; no diagnostic
    root rotation is applied to the native grasp pose.

    ``keypoint_scheme='segment20_legacy'`` explicitly restores the historical
    20 segment landmarks for reproducibility. These schemes define different
    metrics even when the outer hand kernel weight is unchanged.
    """

    def __init__(self, asset_root: str | Path, *, keypoint_scheme=DEFAULT_KEYPOINT_SCHEME):
        if keypoint_scheme not in (DEFAULT_KEYPOINT_SCHEME, LEGACY_KEYPOINT_SCHEME):
            raise ValueError(f"Unknown Sharpa keypoint scheme: {keypoint_scheme}")
        self.keypoint_scheme = keypoint_scheme
        self._landmark_resource = None
        self._source_points = None
        self._kinematic_validation = None
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.xml_path = self.asset_root / "hand" / "sharpa_v2" / "right_kp.xml"
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"Sharpa V2 model not found: {self.xml_path}")
        self.model_sha256 = hashlib.sha256(self.xml_path.read_bytes()).hexdigest()
        self._model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        model = self._model
        self.joint_names = [model.joint(i).name for i in range(model.njnt)]
        if tuple(self.joint_names) != JOINT_NAMES or model.nq != 22 or model.nv != 22:
            raise ValueError("Expected the 22-joint Sharpa V2 MJCF in generation qpos order")
        if np.any(model.jnt_type != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Sharpa evaluator requires revolute joints")
        if model.neq or model.ntendon or np.any(model.body_jntnum > 1):
            raise ValueError("Unexpected coupled joints or multi-joint bodies in Sharpa model")
        if not np.array_equal(model.jnt_qposadr, np.arange(22)):
            raise ValueError("Unexpected Sharpa qpos ordering")
        self.joint_ranges = np.array(model.jnt_range, dtype=np.float64, copy=True)
        self._palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
        if self._palm_id < 0:
            raise ValueError(f"Missing palm body {PALM_BODY}")
        palm = self._palm_id
        if (model.body_parentid[palm] != 0 or model.body_jntnum[palm] != 0
                or not np.allclose(model.body_pos[palm], 0, atol=1e-12)
                or not np.allclose(model.body_quat[palm], [1, 0, 0, 0], atol=1e-12)):
            raise ValueError("Sharpa palm must coincide with the generation hand root frame")
        self._rest_rotations = np.empty((model.nbody, 3, 3), dtype=np.float64)
        for body_id in range(model.nbody):
            mujoco.mju_quat2Mat(self._rest_rotations[body_id].reshape(9),
                               model.body_quat[body_id])
        self.keypoint_names: list[str] = []
        self.keypoint_bodies: list[str] = []
        self._point_body_ids: list[int] = []
        points: list[np.ndarray] = []
        if keypoint_scheme == DEFAULT_KEYPOINT_SCHEME:
            self._add_project21_points(points)
        else:
            self._add_legacy20_points(points)
        self.keypoint_local_positions = np.asarray(points, dtype=np.float64)
        self._point_body_ids = np.asarray(self._point_body_ids, dtype=np.int64)
        self.keypoint_weights = np.full(len(points), 1.0 / len(points), dtype=np.float64)
        self._points_by_body = {
            i: np.flatnonzero(self._point_body_ids == i) for i in range(model.nbody)
        }

    def _add_project21_points(self, points):
        raw = files("dexonomy.diversity").joinpath(LANDMARK_RESOURCE).read_bytes()
        if hashlib.sha256(raw).hexdigest() != LANDMARK_RESOURCE_SHA256:
            raise ValueError("Bundled Sharpa 21-point definition SHA256 mismatch")
        resource = json.loads(raw)
        if (resource["schema"] != "sharpa21-landmark-resource/v1"
                or resource["scheme"] != DEFAULT_KEYPOINT_SCHEME
                or tuple(resource["joint_order"]) != JOINT_NAMES
                or len(resource["keypoints"]) != DEFAULT_KEYPOINT_COUNT
                or resource["output_frame"] != PALM_BODY):
            raise ValueError("Invalid bundled Sharpa landmark/frame/joint contract")
        self._landmark_resource = resource
        self._source_points = resource["keypoints"]
        self._validate_project_kinematics(resource)
        for point in resource["keypoints"]:
            body = point["link"]
            local = np.asarray(point["local_xyz_m"], dtype=np.float64)
            if body in resource["fixed_tip_chains"]:
                chain = resource["fixed_tip_chains"][body]
                rotation, translation = np.eye(3), np.zeros(3)
                parent = chain[0]["parent"]
                for edge in chain:
                    if edge["type"] != "fixed" or edge["parent"] != parent:
                        raise ValueError("Invalid fixed Sharpa fingertip chain")
                    translation += rotation @ np.asarray(edge["origin_xyz_m"])
                    rotation = rotation @ _rpy_rotation(edge["origin_rpy_rad"])
                    parent = edge["child"]
                if parent != body:
                    raise ValueError("Fixed fingertip chain ends at the wrong link")
                body = chain[0]["parent"]
                local = rotation @ local + translation
            self._add_point(point["name"], body, local, points)

    def _validate_project_kinematics(self, resource):
        """Reject a different hand/frame; allow only known MJCF decimal rounding."""
        model = self._model
        tolerances = resource["native_model_validation"]
        A = np.asarray(resource["official_from_grasp_rotation"])
        root = resource["diagnostic_root_joint"]
        if (root["parent"] != "diagnostic_grasp_frame" or root["child"] != PALM_BODY
                or root["type"] != "fixed"
                or not np.allclose(root["origin_xyz_m"], 0, atol=1e-15, rtol=0)
                or not np.allclose(_rpy_rotation(root["origin_rpy_rad"]), A.T, atol=1e-15, rtol=0)
                or not np.array_equal(A @ A.T, np.eye(3)) or np.linalg.det(A) != 1):
            raise ValueError("Invalid diagnostic-to-official palm frame definition")
        maximum_rotation_error = 0.
        for j, edge in enumerate(resource["moving_joints"]):
            body = int(model.jnt_bodyid[j])
            parent_name = model.body(int(model.body_parentid[body])).name
            if (edge["name"] != self.joint_names[j] or edge["type"] != "revolute"
                    or model.body(body).name != edge["child"] or parent_name != edge["parent"]):
                raise ValueError(f"Sharpa joint/link topology differs at {edge['name']}")
            comparisons = (
                (model.body_pos[body], edge["origin_xyz_m"], tolerances["origin_position_atol_m"]),
                (model.jnt_pos[j], [0, 0, 0], tolerances["origin_position_atol_m"]),
                (model.jnt_axis[j], edge["axis"], tolerances["axis_atol"]),
                (model.jnt_range[j], edge["limits_rad"], tolerances["joint_limit_atol_rad"]),
                (self._rest_rotations[body], _rpy_rotation(edge["origin_rpy_rad"]), tolerances["origin_rotation_atol"]),
                (model.qpos0[j], 0., 1e-15),
            )
            if any(not np.allclose(actual, expected, atol=atol, rtol=0) for actual, expected, atol in comparisons):
                raise ValueError(f"Sharpa native kinematics differs from authoritative mapping at {edge['name']}")
            maximum_rotation_error = max(maximum_rotation_error,
                float(np.max(np.abs(self._rest_rotations[body] - _rpy_rotation(edge["origin_rpy_rad"])))))
        self._kinematic_validation = {"status": "passed", "moving_joint_count": 22,
            "maximum_rest_rotation_entry_difference": maximum_rotation_error,
            "tolerances": tolerances,
            "fk_source": "native MJCF transforms, after checking URDF topology/axes/origins/limits; no pose-axis conversion"}

    def _add_legacy20_points(self, points):
        model = self._model
        for finger, prefix in (("thumb", "th"), ("index", "ff"), ("middle", "mf"),
                               ("ring", "rf"), ("pinky", "lf")):
            if finger == "thumb":
                segments = (("MC", "MCP_VL"), ("PP", "DP"))
            else:
                segments = (("PP", "MP"), ("MP", "DP"))
            for segment, next_segment in segments:
                name = f"right_{finger}_{segment}"
                next_name = f"right_{finger}_{next_segment}"
                body_id = self._body_id(name)
                next_id = self._body_id(next_name)
                if model.body_parentid[next_id] != body_id:
                    raise ValueError(f"Unexpected Sharpa chain at {next_name}")
                local_point = model.body_pos[next_id] * 0.5 + [0.0, 0.0, 0.004]
                self._add_point(f"{finger}_{segment.lower()}_mid", name, local_point, points)
            distal_name = f"right_{finger}_DP"
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_d2")
            if site_id < 0 or model.site_bodyid[site_id] != self._body_id(distal_name):
                raise ValueError(f"Missing or relocated Sharpa distal site {prefix}_d2")
            distal_tip = np.array(model.site_pos[site_id], copy=True)
            self._add_point(f"{finger}_dp_mid", distal_name,
                            0.5 * distal_tip + [0.0, 0.0, 0.004], points)
            self._add_point(f"{finger}_pad", distal_name, distal_tip, points)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Missing Sharpa body: {name}")
        return body_id

    def _add_point(self, name: str, body: str, position: np.ndarray,
                   points: list[np.ndarray]) -> None:
        self.keypoint_names.append(name)
        self.keypoint_bodies.append(body)
        self._point_body_ids.append(self._body_id(body))
        points.append(np.asarray(position, dtype=np.float64))

    def keypoints(self, q: np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
        """Compute the selected landmarks once, with bounded FK workspace."""
        q = np.asarray(q, dtype=np.float64)
        single = q.ndim == 1
        if single:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != 22:
            raise ValueError(f"Expected Sharpa joint angles with shape (N,22), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("Sharpa joint angles contain NaN or infinity")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        result = np.empty((len(q), len(self.keypoint_names), 3), dtype=np.float64)
        for start in range(0, len(q), batch_size):
            result[start:start + batch_size] = self._keypoints_chunk(q[start:start + batch_size])
        return result[0] if single else result

    def _keypoints_chunk(self, q: np.ndarray) -> np.ndarray:
        model = self._model
        n = len(q)
        rotations = np.empty((model.nbody, n, 3, 3), dtype=np.float64)
        translations = np.empty((model.nbody, n, 3), dtype=np.float64)
        result = np.empty((n, len(self.keypoint_names), 3), dtype=np.float64)
        rotations[0] = np.eye(3)
        translations[0] = 0.0
        for body_id in range(1, model.nbody):
            parent = int(model.body_parentid[body_id])
            rotation = rotations[parent] @ self._rest_rotations[body_id]
            translation = (translations[parent]
                           + rotations[parent] @ model.body_pos[body_id])
            if model.body_jntnum[body_id]:
                j = int(model.body_jntadr[body_id])
                angle = q[:, model.jnt_qposadr[j]] - model.qpos0[model.jnt_qposadr[j]]
                x, y, z = model.jnt_axis[j]
                cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
                outer = np.outer(model.jnt_axis[j], model.jnt_axis[j])
                cosine = np.cos(angle)[:, None, None]
                sine = np.sin(angle)[:, None, None]
                joint_rotation = cosine * np.eye(3) + (1 - cosine) * outer + sine * cross
                moved_rotation = rotation @ joint_rotation
                anchor = model.jnt_pos[j]
                translation = translation + rotation @ anchor - moved_rotation @ anchor
                rotation = moved_rotation
            rotations[body_id] = rotation
            translations[body_id] = translation
            point_ids = self._points_by_body[body_id]
            if len(point_ids):
                result[:, point_ids] = (self.keypoint_local_positions[point_ids] @
                                        np.swapaxes(rotation, -1, -2)
                                        + translation[:, None, :])
        return result

    def joint_observability(self, delta: float = 1e-4) -> dict[str, float]:
        """Per-joint landmark RMS sensitivity (m/rad) at joint-range midpoint.

        Positive values verify each individual control changes the descriptor;
        they do not establish global injectivity of the landmark map.
        """
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be positive and finite")
        baseline = np.mean(self.joint_ranges, axis=1)
        perturbed = np.tile(baseline, (22, 1)) + np.eye(22) * delta
        difference = self.keypoints(perturbed) - self.keypoints(baseline)
        sensitivity = np.sqrt(np.sum(difference ** 2 * self.keypoint_weights[None, :, None],
                                     axis=(1, 2))) / delta
        return dict(zip(self.joint_names, sensitivity.tolist()))

    def metadata(self) -> dict:
        """JSON-serializable model identity and exact, ordered landmark definition."""
        return {
            "hand": "sharpa_v2", "xml_path": str(self.xml_path),
            "xml_sha256": self.model_sha256, "palm_frame": PALM_BODY,
            "root_to_palm": {"translation_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
            "joint_names": list(self.joint_names), "joint_units": "radian",
            "joint_ranges": self.joint_ranges.tolist(),
            "keypoint_scheme": (DEFAULT_KEYPOINT_SCHEME if self.keypoint_scheme == DEFAULT_KEYPOINT_SCHEME
                                else "sharpa_segment_landmarks_v1"),
            "selection_option": self.keypoint_scheme,
            "keypoint_count": len(self.keypoint_names),
            "feature_dimension": 12 + 3 * len(self.keypoint_names),
            "keypoint_weighting": "uniform 1/L over every listed point, including constant wrist/palm anchors",
            "landmark_definition": ({"resource": LANDMARK_RESOURCE,
                "sha256": LANDMARK_RESOURCE_SHA256,
                "source_files": self._landmark_resource["source_files"],
                "description": self._landmark_resource["description"],
                "source_keypoints": self._source_points,
                "fixed_tip_chains": self._landmark_resource["fixed_tip_chains"],
                "rigid_palm_anchor_indices": self._landmark_resource["rigid_palm_anchor_indices"],
                "official_from_grasp_rotation": self._landmark_resource["official_from_grasp_rotation"],
                "output_frame_note": "Points are in official right_hand_C_MC, not diagnostic_grasp_frame; native root poses are unchanged."}
                if self._landmark_resource else {"description": "Historical 20 segment-interior/distal-site landmarks; explicit reproduction option only."}),
            "native_kinematic_validation": self._kinematic_validation,
            "keypoint_units": "metre",
            "keypoints": [
                {"name": name, "body": body, "local_position_m": point.tolist(),
                 "weight": float(weight)}
                for name, body, point, weight in zip(
                    self.keypoint_names, self.keypoint_bodies,
                    self.keypoint_local_positions, self.keypoint_weights)
            ],
            "joint_sensitivity_m_per_rad": self.joint_observability(),
        }
