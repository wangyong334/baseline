"""V9 optimisation tests (CPU, synthetic data).

    1. Module two in physical units: the estimator's constants at 50 ms are the old literals; at 25 ms the same physical
       times; the ring reference (and the forgetting averages) of a 25 ms stream is exactly half the 50 ms one for the
       same Hz/px; reference rates per window scale with the window length (x 1.0 at 50 ms)
    2. Estimator ablations (--bg-ablation): ring only without the centre component, no restarts without forgetting
    3. Calibrated commit interface (tools/commit_calibration.py): the anchored temperature keeps every thresholded
       decision; the NLL fit recovers a known temperature; calibrated alpha = uncalibrated alpha' exactly; T = 1 equals
       final_analysis' adaptive policy
    4. The 25 ms full-model config differs from V5 @ 25 ms exactly as bb_e3b_50 differs from bb_v5_50
    5. Decision A script (tools/final/plan_a.sh): LF, valid bash, only existing configs, no E3b run, no noise evals
"""
import json
import math
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from model.comod import REF_RATE, BackgroundEstimator  # noqa: E402


def stream(window_ms, seconds=3.0, step_at=1.5, h=40, w=48, seed=0):
    """Deterministic piecewise-constant counts per window for the same Hz/px at any window length (no Poisson noise)."""
    rng = np.random.RandomState(seed)
    base = rng.uniform(0.2, 1.0, size=(1, 2, h, w)) * 0.04       # counts per 50 ms window
    out = []
    n = int(round(seconds * 1000.0 / window_ms))
    for k in range(n):
        t = math.floor(k * window_ms / 50.0) * 0.05                 # piecewise constant over 50 ms blocks
        level = base * (1.0 + 0.5 * math.sin(2.0 * math.pi * t / 2.0)) * (1.6 if t >= step_at else 1.0)
        out.append(torch.from_numpy(level * window_ms / 50.0))
    return out


def run(est, counts):
    vals = []
    for c in counts:
        b = est.estimate()
        vals.append(None if b is None else b.clone())
        est.update(c)
    return vals


class PhysicalUnitTests(unittest.TestCase):
    def test_constants_at_50_and_25_ms(self):
        e50 = BackgroundEstimator()
        self.assertEqual(e50.a_s, math.exp(-1.0 / 20.0))
        self.assertEqual(e50.g_c, math.exp(-1.0 / 100.0))
        self.assertEqual(e50.trunc, 2.0)
        self.assertEqual(e50.a_f, math.exp(-1.0 / 2.0))
        self.assertEqual(e50.alarm_windows, 2)
        e25 = BackgroundEstimator(window_ms=25.0)
        self.assertAlmostEqual(e25.a_s, math.exp(-1.0 / 40.0), places=15)
        self.assertAlmostEqual(e25.g_c, math.exp(-1.0 / 200.0), places=15)
        self.assertEqual(e25.trunc, 1.0)
        self.assertAlmostEqual(e25.a_f, math.exp(-1.0 / 4.0), places=15)
        self.assertEqual(e25.alarm_windows, 4)

    def test_ring_reference_is_window_length_free(self):
        # centre component and forgetting off: the bias-corrected ring EMA is exactly linear in the counts
        e50 = BackgroundEstimator(center=False, forget=False)
        e25 = BackgroundEstimator(center=False, forget=False, window_ms=25.0)
        v50, v25 = run(e50, stream(50.0)), run(e25, stream(25.0))
        for k in range(1, len(v50)):
            np.testing.assert_allclose(v25[2 * k].numpy() * 2.0, v50[k].numpy(), rtol=1e-9, atol=1e-15)

    def test_full_estimator_close_across_window_lengths(self):
        # with the centre component and forgetting on: the centre LCB differs by (1 + g_25) / 2 and a restart can
        # fall 25 ms apart, so compare before the step and once the restart has settled
        e50, e25 = BackgroundEstimator(), BackgroundEstimator(window_ms=25.0)
        v50, v25 = run(e50, stream(50.0)), run(e25, stream(25.0))
        for k in list(range(1, 30)) + [len(v50) - 1]:
            np.testing.assert_allclose(v25[2 * k].numpy() * 2.0, v50[k].numpy(), rtol=0.03, atol=1e-6)
        self.assertGreater(e50.resets, 0)
        self.assertGreater(e25.resets, 0)

    def test_reference_rate_per_window(self):
        from train_stream_v1 import ref_rate_per_window
        self.assertEqual(ref_rate_per_window({"window_ms": 50}), REF_RATE)
        self.assertEqual(ref_rate_per_window({"window_ms": 50, "invariant_ref_rate": [0.00155, 0.01133]}),
                         (0.00155, 0.01133))
        np.testing.assert_allclose(ref_rate_per_window({"window_ms": 25}), np.array(REF_RATE) / 2.0, rtol=1e-15)

    def test_tracker_passes_window_and_ablation_switches(self):
        from model.noise_invariant import BackgroundTracker

        class Seq(object):
            def window_index(self, k):
                return np.zeros(0, dtype=np.int64)

        cfg = {"pad_height": 16, "pad_width": 16, "height": 16, "width_px": 16, "window_ms": 25,
               "invariant_bg_center": False, "invariant_bg_forget": False}
        tr = BackgroundTracker(Seq(), cfg, torch.device("cpu"))
        self.assertFalse(tr.estimator.center)
        self.assertFalse(tr.estimator.forget)
        self.assertEqual(tr.estimator.alarm_windows, 4)
        self.assertAlmostEqual(tr.estimator.eps_b, 0.5 * REF_RATE[0] / 2.0, places=18)
        tr50 = BackgroundTracker(Seq(), dict(cfg, window_ms=50, invariant_bg_center=True, invariant_bg_forget=True),
                                 torch.device("cpu"))
        self.assertTrue(tr50.estimator.center and tr50.estimator.forget)
        self.assertEqual(tr50.estimator.eps_b, 0.5 * REF_RATE[0])


class EstimatorAblationTests(unittest.TestCase):
    def test_no_center_returns_ring(self):
        counts = stream(50.0, seconds=1.0)
        full, ring = BackgroundEstimator(), BackgroundEstimator(center=False)
        hot = [c.clone() for c in counts]
        for c in hot:
            c[0, 0, 5, 5] += 3.0                                     # an isolated hot pixel
        vf, vr = run(full, hot), run(ring, hot)
        self.assertGreater(float(vf[-1][0, 0, 5, 5]), float(vr[-1][0, 0, 5, 5]))   # the centre component takes over
        self.assertTrue(torch.equal(ring.estimate(), ring.As / ring.Ws))

    def test_no_forget_never_restarts(self):
        counts = stream(50.0, seconds=3.0)
        jump = [c * (6.0 if k >= 30 else 1.0) for k, c in enumerate(counts)]
        on, off = BackgroundEstimator(), BackgroundEstimator(forget=False)
        run(on, jump)
        run(off, jump)
        self.assertGreater(on.resets, 0)
        self.assertEqual(off.resets, 0)

    def test_eval_flag_choices(self):
        import train_stream_v1
        self.assertEqual(train_stream_v1.BG_ABLATIONS, ("none", "no_center", "no_forget", "ring_only"))


def synthetic_recs(n_rec=3, n=4000, depth=4, temperature=1.0, seed=0, log_w=2.0):
    """Recordings whose per-lag logits follow the anchored model with a known temperature."""
    from final_analysis import THRESHOLD, _logit
    rng = np.random.RandomState(seed)
    tau = _logit(THRESHOLD) - log_w
    recs = []
    for r in range(n_rec):
        z = tau + rng.normal(0.0, 3.0, size=n)                       # lag-0 logit
        drift = np.cumsum(rng.normal(0.0, 0.6, size=(n, depth)), 1)
        traj = np.concatenate([z[:, None], z[:, None] + drift], 1).astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-(tau + (traj[:, -1] - tau) / temperature)))
        pos = rng.rand(n) < p
        traj[-3:, -1] = np.nan                                       # flushed events near the end
        t = np.sort(rng.randint(0, 2000, size=n))
        tid = np.where(pos, 1 + (t // 400), 0).astype(np.float64)
        recs.append({"name": "rec_%d.npz" % r, "x": rng.randint(0, 64, size=n), "y": rng.randint(0, 48, size=n),
                     "t": t, "tid": tid, "pos": pos, "traj": traj, "log_w": log_w})
    return recs, tau


class CommitCalibrationTests(unittest.TestCase):
    def test_anchored_temperature_keeps_decisions(self):
        from commit_calibration import anchored
        z = np.linspace(-8, 8, 1001)
        z = z[np.abs(z + 1.2) > 1e-9]                                # away from float rounding at the threshold itself
        for t in (0.3, 1.0, 2.5):
            np.testing.assert_array_equal(anchored(z, -1.2, t) >= -1.2, z >= -1.2)

    def test_fit_recovers_temperature(self):
        from commit_calibration import fit_temperature
        recs, tau = synthetic_recs(n_rec=4, n=20000, depth=0, temperature=2.0, seed=1)
        self.assertAlmostEqual(fit_temperature(recs, tau)["temperature"], 2.0, delta=0.15)

    def test_calibrated_alpha_equals_uncalibrated_alpha_prime(self):
        from commit_calibration import anchored, equivalent_alpha
        from final_analysis import lag_adaptive
        recs, tau = synthetic_recs(depth=6, seed=2)
        for t in (0.5, 1.7):
            for a in (0.05, 0.2):
                for rec in recs:
                    lag_c, src_c = lag_adaptive(anchored(rec["traj"], tau, t), tau, a)
                    lag_u, src_u = lag_adaptive(rec["traj"], tau, equivalent_alpha(a, t))
                    agree = (lag_c == lag_u).mean()
                    self.assertGreater(agree, 0.999)                 # float rounding at the bound only

    def test_unit_temperature_equals_final_analysis(self):
        from commit_calibration import evaluate
        from final_analysis import analyse
        recs, tau = synthetic_recs(depth=4, seed=3)
        ours = evaluate(recs, set(), 50.0, 1.0, alphas=(0.05,), full=False)["policies"]
        ref = analyse(recs, set(), 50.0, alphas=(0.05,), full=False)["policies"]
        for name in ("zero", "fixed_4", "adaptive_0.05"):
            self.assertEqual(ours[name]["iou"], ref[name]["iou"], name)
            self.assertEqual(ours[name]["latency_ms"], ref[name]["latency_ms"], name)
        self.assertEqual(ours["adaptive_cal_0.05"]["iou"], ref["adaptive_0.05"]["iou"])


class CalibratedAnalysisTests(unittest.TestCase):
    """final_analysis --temperature (alpha with its calibrated meaning) and the report helpers (latency convention)."""

    def test_unit_temperature_reproduces_uncalibrated_policies(self):
        from final_analysis import analyse
        recs, _ = synthetic_recs(depth=4, seed=4)
        plain = analyse(recs, set(), 50.0, alphas=(0.05, 0.2), full=False)
        cal = analyse(recs, set(), 50.0, alphas=(0.05, 0.2), full=False, temperature=1.0)
        self.assertIsNone(plain["temperature"])
        self.assertFalse(any("_cal_" in n for n in plain["policies"]))
        same = lambda x, y: json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)  # noqa: E731 (NaN-safe)
        for n, r in plain["policies"].items():
            self.assertTrue(same(cal["policies"][n], r), n)                    # every old policy unchanged
        for a in ("0.05", "0.2"):
            for fam in ("adaptive", "slice"):
                self.assertTrue(same(cal["policies"]["%s_cal_%s" % (fam, a)], plain["policies"]["%s_%s" % (fam, a)]))

    def test_temperature_maps_alpha(self):
        from commit_calibration import equivalent_alpha
        from final_analysis import policy_lag
        recs, tau = synthetic_recs(depth=6, seed=5)
        for rec in recs:
            win = rec["t"] // 50
            lag_c, _ = policy_lag("adaptive_cal_0.05", rec["traj"], tau, win, temperature=1.6)
            lag_u, _ = policy_lag("adaptive_%r" % equivalent_alpha(0.05, 1.6), rec["traj"], tau, win)
            self.assertGreater((lag_c == lag_u).mean(), 0.999)
        with self.assertRaises(ValueError):
            policy_lag("adaptive_cal_0.05", recs[0]["traj"], tau, recs[0]["t"] // 50)   # needs a temperature

    def test_algorithmic_latency_removes_compute(self):
        from utils.stream_metrics import algorithmic_latency
        carry = {"latency": {"latency_median_ms": 60.594, "latency_mean_ms": 113.428},
                 "timing": {"net_ms_mean": 10.594, "total_ms_mean": 19.7}}
        self.assertAlmostEqual(algorithmic_latency(carry), 50.0, places=9)
        self.assertAlmostEqual(algorithmic_latency(carry, "latency_mean_ms"), 102.834, places=9)
        self.assertIsNone(algorithmic_latency({"latency": {}}))
        self.assertEqual(algorithmic_latency({"latency": {"latency_median_ms": 40.0}}), 40.0)    # no timing recorded

    def test_paper_tables_prefer_calibrated_policy(self):
        import paper_tables
        a = {"policies": {"adaptive_0.05": {}, "adaptive_cal_0.05": {}, "zero": {}}}
        self.assertEqual(paper_tables.pick_policy(a, "adaptive_0.05"), "adaptive_cal_0.05")
        self.assertEqual(paper_tables.pick_policy(a, "zero"), "zero")
        self.assertEqual(paper_tables.pick_policy({"policies": {"adaptive_0.05": {}}}, "adaptive_0.05"), "adaptive_0.05")

    def test_cal_analysis_script_is_lf(self):
        with open(os.path.join(ROOT, "tools", "final", "cal_analysis.sh"), "rb") as fh:
            body = fh.read()
        self.assertNotIn(b"\r", body)
        self.assertIn(b"--temperature", body)


class FullModel25msConfigTests(unittest.TestCase):
    def load(self, rel):
        from utils.stream_common import load_flat_config
        return load_flat_config(os.path.join(ROOT, rel))

    def test_e3b_25_differs_from_v5_25_like_50ms(self):
        v50, e50 = self.load("configs/final/bb_v5_50.yaml"), self.load("configs/final/bb_e3b_50.yaml")
        v25, e25 = self.load("configs/final/bb_v5_25.yaml"), self.load("configs/final/bb_e3b_25.yaml")
        keys50 = {k for k in set(v50) | set(e50) if v50.get(k) != e50.get(k)}
        keys25 = {k for k in set(v25) | set(e25) if v25.get(k) != e25.get(k)}
        self.assertEqual(keys50, keys25)
        self.assertEqual(e25["save_root"], "log/final/bb_e3b_25_seed37")
        self.assertTrue(e25["invariant_input"])
        r = e50["window_ms"] / e25["window_ms"]
        self.assertEqual(e25["noise_aug_prob"], e50["noise_aug_prob"])
        self.assertEqual(e25["noise_aug_rate_min"] * r, e50["noise_aug_rate_min"])          # same Hz/px
        self.assertEqual(e25["noise_aug_rate_max"] * r, e50["noise_aug_rate_max"])

    def test_queue_script_exists_and_is_lf(self):
        with open(os.path.join(ROOT, "tools", "final", "queue_25ms_full.sh"), "rb") as fh:
            body = fh.read()
        self.assertNotIn(b"\r", body)
        self.assertIn(b"bb_e3b_25", body)



class PlanATests(unittest.TestCase):
    """Decision A (11-04 evening): tools/final/plan_a.sh replaces the E3b parts of the running queues."""

    def test_script_is_lf_valid_and_uses_existing_configs(self):
        import re
        import shutil
        import subprocess
        path = os.path.join(ROOT, "tools", "final", "plan_a.sh")
        with open(path, "rb") as fh:
            body = fh.read().decode("utf-8")
        self.assertNotIn("\r", body)
        for name in sorted(set(re.findall(r"name=([a-z0-9_]+);", body)) | {"bb_v5_50", "bb_v5_25", "rcl_feature_evf"}):
            self.assertTrue(os.path.exists(os.path.join(ROOT, "configs", "final", name + ".yaml")), name)
        self.assertTrue(os.path.exists(os.path.join(ROOT, re.search(r"ref=(configs/\S+\.jsonl)", body).group(1))))
        self.assertNotIn("bb_e3b_50 ", body.replace("bb_e3b_50.yaml --mode train", ""))    # only stopped, never run
        self.assertIn('HZ=""', body)                                                         # no noise evals
        bash = shutil.which("bash")
        if bash:
            self.assertEqual(subprocess.call([bash, "-n", path]), 0)


if __name__ == "__main__":
    unittest.main()
