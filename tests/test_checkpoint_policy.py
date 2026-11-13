import json
from pathlib import Path
import tempfile
import unittest
from dexgrasp.test.checkpoint_policy import exclusion, require_eligible


class CheckpointPolicyTests(unittest.TestCase):
    def test_boundary_alias_and_other_training_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'evaluation_policy.json').write_text(json.dumps(dict(
                schema='dexgrasp-evaluation-eligibility-v1',runs={'A':dict(max_step=40000,excluded_aliases=['last.pt'])})))
            self.assertIsNone(exclusion(root/'A/step_040000.pt'))
            with self.assertRaisesRegex(ValueError,'excluded'):
                require_eligible(root/'A/step_045000.pt')
            with self.assertRaisesRegex(ValueError,'excluded'):
                require_eligible(root/'A/last.pt')
            self.assertIsNone(exclusion(root/'A_fixed/step_050000.pt'))
            self.assertIsNone(exclusion(root/'another_experiment/A/step_050000.pt'))

    def test_ambiguous_alias_fails_instead_of_bypassing_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'evaluation_policy.json').write_text(json.dumps(dict(
                schema='dexgrasp-evaluation-eligibility-v1',runs={'B':dict(max_step=90000)})))
            with self.assertRaisesRegex(ValueError,'Cannot establish'):
                exclusion(root/'B/unknown.pt')
            self.assertIsNone(exclusion(root/'B/unknown.pt',90000))
            self.assertIsNotNone(exclusion(root/'B/unknown.pt',95000))


if __name__=='__main__':unittest.main()
