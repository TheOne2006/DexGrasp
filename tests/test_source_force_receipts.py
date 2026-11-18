import copy
import unittest

from scripts.audit_source_force import physics_verified


AXES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]


class SourceReceiptTests(unittest.TestCase):
    def test_floating_rechecks_all_six_axes_and_strict_thresholds(self):
        record = dict(
            type="force_closure_6d",
            passes=True,
            directions_completed=6,
            directions_required=6,
            translation_threshold_m=0.05,
            rotation_threshold_deg=15,
            direction_results=[
                dict(
                    direction=axis,
                    passes=True,
                    completed_holds=10,
                    max_translation_error_m=0.03,
                    max_rotation_error_deg=14.99,
                )
                for axis in AXES
            ],
        )
        self.assertTrue(physics_verified(record))
        for field, invalid in (
            ("completed_holds", 9),
            ("direction", AXES[1]),
            ("max_rotation_error_deg", 15),
            ("max_translation_error_m", 0.05),
        ):
            with self.subTest(field=field):
                wrong = copy.deepcopy(record)
                wrong["direction_results"][0][field] = invalid
                self.assertFalse(physics_verified(wrong))

    def test_tabletop_requires_same_slide_snapshot_and_no_table_recontact(self):
        record = dict(
            joint_success=True,
            slide=dict(candidate_id="gt", passes=True, translation_m=0.01, rotation_deg=2),
            directions=[
                dict(
                    candidate_id="gt",
                    passes=True,
                    direction=axis,
                    start_snapshot_sha256="1" * 64,
                    table_contact_counts=[0],
                    pose_checks=[
                        dict(passes=True, translation_m=0.01, rotation_deg=3) for _ in range(10)
                    ],
                )
                for axis in AXES
            ],
        )
        self.assertTrue(physics_verified(record))
        for field, invalid in (
            ("start_snapshot_sha256", "2" * 64),
            ("table_contact_counts", [1]),
            ("candidate_id", "a different GT"),
        ):
            with self.subTest(field=field):
                wrong = copy.deepcopy(record)
                wrong["directions"][3][field] = invalid
                self.assertFalse(physics_verified(wrong))
        record["slide"]["passes"] = False
        self.assertFalse(physics_verified(record))
