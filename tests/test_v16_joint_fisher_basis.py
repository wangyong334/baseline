"""Joint-template preservation contracts, not a claim of detection gains."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.covariance_revision import CovarianceRevision


def random_observations():
    torch.manual_seed(137)
    n, d = 384, 7
    y = (torch.arange(n) % 2).double()
    common = torch.randn(n, d, dtype=torch.float64)
    h1 = common + .4 * torch.randn_like(common)
    h2 = .6 * common + .5 * torch.randn_like(common)
    h1[:, 1] += 1.3 * y
    h2[:, 3] -= .8 * y
    z1 = h1[:, 0] + .7 * y
    z2 = h2[:, 0] + .4 * y
    return (z1, z2, h1, h2), y


def raw_templates(model, values, y):
    """Independent direct full-space moment calculation for score equivalence."""
    z1, z2, h1, h2 = values
    x1 = (torch.cat((h1, z1[:, None]), 1) - model.raw_mean) / model.raw_scale
    x2 = (torch.cat((h2, z2[:, None]), 1) - model.raw_mean) / model.raw_scale
    x = torch.cat((x2, x1), 1)
    class0, class1 = x[y == 0], x[y == 1]
    mu0, mu1 = class0.mean(0), class1.mean(0)
    a, b = class0 - mu0, class1 - mu1
    cov = .5 * (a.t().mm(a) / len(a) + b.t().mm(b) / len(b))
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    omega = ((1. - model.shrinkage) * cov + model.shrinkage * torch.diag(cov.diag())
             + 1e-5 * max(float(cov.diag().mean()), 1e-8) * eye)
    delta = mu1 - mu0
    d = x1.shape[1]
    joint = torch.linalg.solve(omega, delta)
    second = torch.linalg.solve(omega[:d, :d], delta[:d])
    return x1 - mu0[d:], x2 - mu0[:d], second, joint


class JointFisherBasisTests(unittest.TestCase):
    def test_three_signed_templates_reconstruct_from_three_shared_channels(self):
        values, labels = random_observations()
        model = CovarianceRevision(feature_dim=7, channels=3, hidden=4,
                                   channel_basis="joint_fisher").double()
        report = model.fit_statistics(*values, labels)
        x1, x2, second, joint = raw_templates(model, values, labels)
        p, d = model.projection, x1.shape[1]
        for x, weight in ((x2, second), (x2, joint[:d]), (x1, joint[d:])):
            raw_score = x.mv(weight)
            reconstructed = x.mm(p).mv(p.t().mv(weight))
            self.assertTrue(torch.allclose(raw_score, reconstructed, rtol=1e-10, atol=1e-10))
        raw_joint = x2.mv(joint[:d]) + x1.mv(joint[d:])
        compact_joint = x2.mm(p).mv(p.t().mv(joint[:d])) + x1.mm(p).mv(p.t().mv(joint[d:]))
        self.assertTrue(torch.allclose(raw_joint, compact_joint, rtol=1e-10, atol=1e-10))
        # The detector is signed, not a squared anomaly distance.
        self.assertGreater(float(raw_joint[labels == 1].mean()), float(raw_joint[labels == 0].mean()))
        self.assertEqual(report["channel_basis"], "joint_fisher")
        for item in report["channel_basis_template_audit"].values():
            self.assertLess(item["relative_reconstruction_error"], 1e-10)
            self.assertLess(item["training_score_max_abs_error"], 1e-9)

    def test_opposite_view_mean_shifts_survive_when_shared_fisher_loses_them(self):
        # Three strongly correlated nuisance pairs occupy the top three PCs.
        # The label signal reverses sign between views; pooled-view Fisher is
        # exactly zero. A shared-Fisher/PCA basis thus discards that signal.
        ids = torch.arange(256, dtype=torch.long)
        bits = torch.stack([1. - 2. * ((ids >> j) & 1).double() for j in range(8)], 1)
        base = torch.stack((bits[:, 0] + .1 * bits[:, 1], bits[:, 0] - .1 * bits[:, 1],
                            bits[:, 2] + .1 * bits[:, 3], bits[:, 2] - .1 * bits[:, 3],
                            bits[:, 4] + .1 * bits[:, 5], bits[:, 4] - .1 * bits[:, 5],
                            bits[:, 6]), 1).repeat(2, 1)
        y = torch.cat((torch.zeros(256), torch.ones(256))).double()
        h1, h2 = base.clone(), base.clone()
        h1[:, 6] += 4. * (y - .5)
        h2[:, 6] -= 4. * (y - .5)
        z = bits[:, 7].repeat(2)
        values = (z, z, h1, h2)
        old = CovarianceRevision(feature_dim=7, channels=3).double()
        new = CovarianceRevision(feature_dim=7, channels=3, channel_basis="joint_fisher").double()
        old.fit_statistics(*values, y)
        new.fit_statistics(*values, y)
        raw = torch.cat((h2, z[:, None]), 1)
        old_compact = ((raw - old.raw_mean) / old.raw_scale).mm(old.projection)
        new_compact = ((raw - new.raw_mean) / new.raw_scale).mm(new.projection)
        self.assertLess(float((old_compact[y == 1].mean(0) - old_compact[y == 0].mean(0)).norm()), 1e-9)
        self.assertGreater(float((new_compact[y == 1].mean(0) - new_compact[y == 0].mean(0)).norm()), 1.)

    def test_both_bases_keep_shapes_zero_start_and_capacity_for_all_variants(self):
        values, labels = random_observations()
        parameter_counts, state_keys = [], []
        for basis in CovarianceRevision.CHANNEL_BASES:
            for variant in ("full", "no_context", "no_revision"):
                with self.subTest(basis=basis, variant=variant):
                    model = CovarianceRevision(feature_dim=7, channels=3, hidden=4,
                                               channel_basis=basis, variant=variant).double()
                    model.fit_statistics(*values, labels)
                    q = model.encode_pair(*values)
                    self.assertEqual(tuple(q.shape), (len(labels), 6))
                    self.assertTrue(bool(torch.isfinite(q).all()))
                    self.assertTrue(torch.equal(model(*values), values[1]))
                    parameter_counts.append(sum(p.numel() for p in model.parameters()))
                    state_keys.append(tuple(model.state_dict()))
                    restored = CovarianceRevision(feature_dim=7, channels=3, hidden=4,
                                                  channel_basis=basis, variant=variant).double()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    self.assertTrue(torch.equal(q, restored.encode_pair(*values)))
        self.assertEqual(len(set(parameter_counts)), 1)
        self.assertEqual(len(set(state_keys)), 1)

    def test_legacy_default_is_identical_to_explicit_shared_fisher(self):
        values, labels = random_observations()
        torch.manual_seed(9)
        default = CovarianceRevision(feature_dim=7, channels=3).double()
        torch.manual_seed(9)
        explicit = CovarianceRevision(feature_dim=7, channels=3, channel_basis="shared_fisher").double()
        default.fit_statistics(*values, labels)
        explicit.fit_statistics(*values, labels)
        for name, tensor in default.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[name]), name)

    def test_joint_basis_rejects_silent_template_truncation(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            CovarianceRevision(feature_dim=7, channels=2, channel_basis="joint_fisher")
        with self.assertRaisesRegex(ValueError, "channel_basis"):
            CovarianceRevision(channel_basis="unknown")
        # Two raw dimensions can span at most two independent template blocks.
        CovarianceRevision(feature_dim=1, channels=2, channel_basis="joint_fisher")


if __name__ == "__main__":
    unittest.main()
