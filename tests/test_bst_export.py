"""Prior exports cannot silently mix backbones or partial recordings."""
import json
import os
import tempfile
import unittest

import numpy as np

from tools.bst.export_priors import check_existing_directory, validate_cached_file


class TestPriorExport(unittest.TestCase):
    def test_nonempty_unidentified_directory_is_rejected(self):
        identity = {"schema": 1, "backbone_sha256": "a", "window_ms": 50,
                    "recording_n_windows": 160, "split": "val"}
        with tempfile.TemporaryDirectory() as path:
            self.assertIsNone(check_existing_directory(path, identity))
            with open(os.path.join(path, "unrelated.txt"), "w") as out:
                out.write("existing data")
            with self.assertRaisesRegex(ValueError, "lacks provenance"):
                check_existing_directory(path, identity)

    def test_same_identity_can_resume_but_other_backbone_cannot(self):
        identity = {"schema": 1, "backbone_sha256": "a", "window_ms": 50,
                    "recording_n_windows": 160, "split": "val"}
        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, "prior_provenance.json"), "w") as out:
                json.dump(identity, out)
            self.assertEqual(check_existing_directory(path, identity), identity)
            with self.assertRaisesRegex(ValueError, "different backbone"):
                check_existing_directory(path, dict(identity, backbone_sha256="b"))

    def test_file_identity_duration_rows_and_completion_are_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "val_000.npz")
            payload = dict(traj=np.array([[1.0], [2.0]]), labels=np.array([0, 1]),
                           window_ms=50, n_windows=160, backbone_sha256="a", complete=True)
            np.savez(path, **payload)
            validate_cached_file(path, "a", 50, 160, [0, 1])
            with self.assertRaisesRegex(ValueError, "backbone identity"):
                validate_cached_file(path, "b", 50, 160, [0, 1])
            with self.assertRaisesRegex(ValueError, "temporal"):
                validate_cached_file(path, "a", 50, 159, [0, 1])
            with self.assertRaisesRegex(ValueError, "rows"):
                validate_cached_file(path, "a", 50, 160, [1, 0])
            np.savez(path, **dict(payload, complete=False))
            with self.assertRaisesRegex(ValueError, "marked complete"):
                validate_cached_file(path, "a", 50, 160, [0, 1])


if __name__ == "__main__":
    unittest.main()
