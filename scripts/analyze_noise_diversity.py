"""Measure within-condition handshape spread before and after force testing."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from dexgrasp.geometry.hand import URDFHand
from dexgrasp.test.frozen import decoded


def distances(poses, hand, tips):
    """Mean unordered-pair distance; remove all wrist motion from finger shape."""
    q = poses["q"].astype(float)
    xyz = np.stack([hand.world_keypoints(np.eye(3), np.zeros(3), j)[tips] for j in q])
    quat = Rotation.from_matrix(poses["R"]).as_quat()
    i, j = np.triu_indices(len(q), 1)
    angles = 2 * np.arccos(np.clip(abs((quat[i] * quat[j]).sum(1)), 0, 1))
    return dict(
        joint_rms_deg=float(np.degrees(pdist(q).mean() / np.sqrt(q.shape[1]))),
        fingertip_rms_mm=float(pdist(xyz.reshape(len(q), -1)).mean() * 1000 / np.sqrt(len(tips))),
        wrist_translation_mm=float(pdist(poses["t"]).mean() * 1000),
        wrist_rotation_deg=float(np.degrees(angles).mean()),
        pairs=len(i),
    )


def analyze(root, inputs, force):
    plan = json.loads((root / "plan.json").read_text())
    hand = URDFHand(Path(plan["root"]) / "hand/hand_spec.json")
    tips = [i for i, k in enumerate(hand.spec["keypoints"]) if k["link"].endswith("_fingertip")]
    assert len(tips) == 5
    outcomes = np.load(force / "outcomes.npy")
    design = json.loads((inputs / "design.json").read_text())
    count = design["directions_per_object"]
    objects = plan["objects"]
    if "object_indices" in design:
        objects = [objects[i] for i in design["object_indices"]]
    report = {}
    for i, g in enumerate("ABCD"):
        poses = decoded(inputs, g, "noise").reshape(-1, count)
        valid = outcomes[i].reshape(-1, count) == 1
        rows = []
        for obj, p, ok in zip(objects, poses, valid, strict=True):
            item = dict(object=obj, passed=int(ok.sum()), n=count)
            for key, mask in [("all", p["decoded"]), ("successful", ok)]:
                if mask.sum() >= 2:
                    item[key] = distances(p[mask], hand, tips)
            rows.append(item)
        summary = {}
        for key in ("all", "successful"):
            subset = [r[key] for r in rows if key in r]
            summary[key] = dict(
                conditions=len(subset),
                pairs=sum(r["pairs"] for r in subset),
                equal_object_mean=(
                    {k: np.mean([r[k] for r in subset]).item() for k in subset[0] if k != "pairs"}
                    if subset
                    else {}
                ),
            )
        report[g] = dict(summary=summary, per_object=rows)
    (force / "diversity.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "inputs", "force"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    analyze(args.root, args.inputs, args.force)
