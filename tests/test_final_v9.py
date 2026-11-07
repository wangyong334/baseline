"""V9-final integration tests (CPU, synthetic data; FINAL_PLAN_20261104.md).

    1. Trajectories: ChangepointLayer(trace=...) does not change any output; the per-lag trajectory reproduces the
       fixed-D commit (incl. flush), --commit-lag k, and the adaptive commit (lag and value per event) offline
       (tools/final_analysis.py policies); window-shared release / simple commit follow their definitions
    2. B2 look-ahead head (model/lookahead_head.py): starts at V5, every event committed once, chunk-split invariant,
       causal, a reset clears only that stream, past-only reads nothing after arrival, event loss reaches every
       parameter, adaptive arrival commit follows RCL's bound
    3. Recording groups (shot-noise rule); checkpoint selection (tools/select_checkpoint.py): guarded (default), plain
       best val, and a stopped bit-exact replica selected on its reference run's full record
    4. stream_groups (ported from v6-3): grouped training = ungrouped; the dense NISI background map is sliced per stream
    5. Noise in physical units: Hz/px -> load L for 50 / 25 ms windows
    6. B1 persistence counts against a direct loop
    7. End to end: backbone (save_epochs) -> selection -> RCL and B2 heads -> eval with trajectories and noise
       -> final_analysis -> B1, on synthetic recordings
"""
import contextlib
import copy
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from test_changepoint_layer import MIXTURE, make_events, make_layer, run_eval, chunk_slice  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def trace_run(layer, ev, z0, feat, n_windows, cuts, batch=1):
    """run_eval with a trace: returns (emitted {key: (z, lag, status)}, trajectory [N, D + 1])."""
    import train_track_head as TH
    state = layer.init_state(batch, torch.device("cpu"))
    emitted, trace = {}, []
    bounds = [0] + list(cuts) + [n_windows]
    with torch.no_grad():
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            events, z, f, _, idx = chunk_slice(ev, z0, feat, lo, hi)
            _, state, aux = layer.forward_chunk(events, z, f, state, hi - lo, keys=idx, trace=trace)
            for d in (aux, ):
                if d["delayed_z"] is not None:
                    for k, zz, lag, st in zip(d["delayed_key"].tolist(), d["delayed_z"].tolist(),
                                              d["delayed_lag"].tolist(), d["delayed_status"].tolist()):
                        emitted[k] = (zz, lag, st)
        fl = layer.flush(state)
        if fl["delayed_z"] is not None:
            for k, zz, lag, st in zip(fl["delayed_key"].tolist(), fl["delayed_z"].tolist(), fl["delayed_lag"].tolist(),
                                      fl["delayed_status"].tolist()):
                emitted[k] = (zz, lag, st)
    return emitted, TH.trajectory_array(len(z0), trace, layer.defer_max)


def perturbed_mixture(D=4, seed=3):
    layer = make_layer(defer_max=D, **dict(MIXTURE, raw_residual=False, expert_input="score"))
    torch.manual_seed(seed)
    with torch.no_grad():
        layer.fuse_out.weight.add_(0.5 * torch.randn_like(layer.fuse_out.weight))
        layer.fuse_out.bias.fill_(0.2)
        layer.att_foot.add_(0.3 * torch.randn_like(layer.att_foot))
        layer.pi_logit.copy_(0.3 * torch.randn_like(layer.pi_logit))
        for p in layer.member.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return layer


class TrajectoryTests(unittest.TestCase):
    def test_trace_is_passive_and_reproduces_every_policy(self):
        import final_analysis as FA
        D, T = 4, 14
        layer = perturbed_mixture(D)
        ev, z0, feat = make_events(T, seed=5)
        plain = run_eval(layer, ev, z0, feat, T, cuts=(6,))
        traced, traj = trace_run(layer, ev, z0, feat, T, cuts=(6,))
        self.assertEqual(set(plain), set(traced))
        for k in plain:
            self.assertEqual(plain[k], traced[k])                     # the trace changes nothing
        self.assertTrue(np.isfinite(traj[:, 0]).all())
        lag_d = FA.lag_fixed(traj, D)
        for k, (z, lag, st) in plain.items():                        # fixed D (with flush) = column min(D, avail)
            self.assertEqual(lag, int(lag_d[k]))
            self.assertAlmostEqual(z - layer.log_w, float(traj[k, lag_d[k]]), places=5)
        for k_lag in (1, 2, 3):                                      # --commit-lag k
            lk = copy.deepcopy(layer)
            lk.commit_lag = k_lag
            out = run_eval(lk, ev, z0, feat, T, cuts=(6,))
            lag_k = FA.lag_fixed(traj, k_lag)
            for k, (z, lag, st) in out.items():
                self.assertEqual(lag, int(lag_k[k]))
                self.assertAlmostEqual(z - layer.log_w, float(traj[k, lag_k[k]]), places=5)
        for alpha in (0.05, 0.2, 0.4):                               # adaptive: lag and value per event
            la = copy.deepcopy(layer)
            la.commit, la.alpha = "adaptive", alpha
            out = run_eval(la, ev, z0, feat, T, cuts=(6,))
            lag_a, src = FA.lag_adaptive(traj, layer.tau_logit, alpha)
            self.assertEqual(set(out), set(range(len(z0))))
            for k, (z, lag, st) in out.items():
                self.assertEqual(lag, int(lag_a[k]), (alpha, k))
                self.assertAlmostEqual(z - layer.log_w, float(traj[k, lag_a[k]]), places=5)
                self.assertEqual(int(src[k]) == 0, lag == 0 and st == 0)       # arrival = confirmed at lag 0
            self.assertTrue((src == 0).any() and (src >= 2).any())

    def test_window_shared_and_simple(self):
        import final_analysis as FA
        traj = np.array([[3.0, 3.0, 3.0], [-1.0, 2.0, 2.5], [-1.0, -1.0, np.nan], [0.5, 0.6, 0.7]], np.float32)
        win = np.array([0, 0, 1, 1])
        tau = 0.0
        lag, src = FA.lag_adaptive(traj, tau, 0.2)                  # hi = logit(0.8) = 1.386, lo = -1.386
        self.assertEqual(lag.tolist(), [0, 1, 1, 2])                # arrival, mid-wait, flushed at avail, deadline
        self.assertEqual(src.tolist(), [0, 1, 3, 2])
        self.assertEqual(FA.lag_window_shared(lag, win).tolist(), [1, 1, 2, 2])
        self.assertEqual(FA.lag_simple(traj, tau, 0.2, 2).tolist(), [0, 2, 1, 2])
        self.assertEqual(FA.lag_fixed(traj, 2).tolist(), [2, 2, 1, 2])
        self.assertEqual(FA.lag_slice(traj, tau, 0.2, win).tolist(), [1, 1, 1, 2])   # all confident at 1 / never

    def test_fn_categories_match_error_budget(self):
        import error_budget
        import final_analysis as FA
        rng = np.random.RandomState(0)
        n = 4000
        t = np.sort(rng.randint(0, 2000, n))
        win = t // 50
        tid = np.where(rng.rand(n) < 0.3, rng.randint(1, 4, n), 0).astype(np.float64)
        pos = tid > 0
        prob = np.where(pos, rng.rand(n), rng.rand(n) * 0.95).astype(np.float32)
        pred = prob >= 0.9
        cat = FA.fn_categories(win, pos, pred, tid)
        ref = error_budget.analyse_arrays("x", rng.randint(0, 40, n), rng.randint(0, 30, n), t, pos, prob, tid, 0.9,
                                          h=30, w=40)[0]["fn_cat"]
        for i, name in enumerate(FA.FN_CATS, start=1):
            self.assertEqual(int((cat == i).sum()), ref[name], name)


class LookaheadTests(unittest.TestCase):
    def make(self, D=3, seed=0, perturb=True, **kw):
        from model.lookahead_head import LookaheadHead
        torch.manual_seed(seed)
        head = LookaheadHead(4, 16, 24, pos_weight=30.0, defer_max=D, cell=4, **kw)
        if perturb:
            with torch.no_grad():
                for p in head.parameters():
                    p.add_(0.3 * torch.randn_like(p))
        return head

    def drive(self, head, ev, z0, feat, T, cuts=(), batch=1, keep=None):
        state = head.init_state(batch, torch.device("cpu"))
        emitted, trace = {}, []
        bounds = [0] + list(cuts) + [T]
        with torch.no_grad():
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                events, z, f, _, idx = chunk_slice(ev, z0, feat, lo, hi)
                kp = None if keep is None else torch.from_numpy(keep[lo:hi]).float()
                _, state, aux = head.forward_chunk(events, z, f, state, hi - lo, keep=kp, keys=idx, trace=trace)
                for d in (aux,):
                    if d["delayed_z"] is not None:
                        for k, zz, lag, st in zip(d["delayed_key"].tolist(), d["delayed_z"].tolist(),
                                                  d["delayed_lag"].tolist(), d["delayed_status"].tolist()):
                            self.assertNotIn(k, emitted)
                            emitted[k] = (zz, lag, st)
            fl = head.flush(state)
            if fl["delayed_z"] is not None:
                for k, zz, lag, st in zip(fl["delayed_key"].tolist(), fl["delayed_z"].tolist(),
                                          fl["delayed_lag"].tolist(), fl["delayed_status"].tolist()):
                    self.assertNotIn(k, emitted)
                    emitted[k] = (zz, lag, st)
        return emitted, trace

    def test_starts_at_v5_and_commits_once(self):
        head = self.make(perturb=False)
        ev, z0, feat = make_events(10, batch=2, seed=1)
        out, _ = self.drive(head, ev, z0, feat, 10, cuts=(4,), batch=2)
        self.assertEqual(set(out), set(range(len(z0))))
        for k, (z, lag, st) in out.items():
            self.assertAlmostEqual(z, float(z0[k]), places=4)

    def test_chunks_causality_reset_and_past(self):
        head = self.make()
        T = 12
        ev, z0, feat = make_events(T, batch=2, seed=2)
        one, _ = self.drive(head, ev, z0, feat, T, cuts=(), batch=2)
        two, _ = self.drive(head, ev, z0, feat, T, cuts=(3, 7), batch=2)
        for k in one:
            self.assertAlmostEqual(one[k][0], two[k][0], places=4)
            self.assertEqual(one[k][1:], two[k][1:])
        later = z0.copy()
        later[ev["t"] >= 8] += 4.0                                   # only windows >= 8 change
        three, _ = self.drive(head, ev, later, feat, T, cuts=(), batch=2)
        for k in one:
            if int(ev["t"][k]) + one[k][1] < 8:
                self.assertAlmostEqual(one[k][0], three[k][0], places=5)
        past = copy.deepcopy(head)
        past.future_stats = "past"
        p1, _ = self.drive(past, ev, z0, feat, T, batch=2)
        p2, _ = self.drive(past, ev, later, feat, T, batch=2)
        for k in p1:                                                 # past-only: nothing after the event's window
            if int(ev["t"][k]) < 8:
                self.assertAlmostEqual(p1[k][0], p2[k][0], places=5)
        keep = np.ones((T, 2), dtype=np.float32)
        keep[6, 1] = 0.0                                             # stream 1 starts a new recording at window 6
        r1, _ = self.drive(head, ev, z0, feat, T, batch=2, keep=keep)
        for k in one:
            if int(ev["b"][k]) == 0:
                self.assertAlmostEqual(one[k][0], r1[k][0], places=5)
        alt = z0.copy()
        alt[(ev["b"] == 1) & (ev["t"] < 6)] -= 3.0                   # earlier recording of stream 1 must not matter
        r2, _ = self.drive(head, ev, alt, feat, T, batch=2, keep=keep)
        for k in r1:
            if int(ev["b"][k]) == 1 and int(ev["t"][k]) >= 6:
                self.assertAlmostEqual(r1[k][0], r2[k][0], places=5)

    def test_gradients_and_adaptive_arrival(self):
        head = self.make()
        T = 8
        ev, z0, feat = make_events(T, seed=4)
        events = {k: torch.from_numpy(ev[k]) for k in ("t", "b", "x", "y", "p", "t_local", "bin")}
        _, _, aux = head.forward_chunk(events, torch.from_numpy(z0), torch.from_numpy(feat), None, T,
                                       labels=torch.from_numpy(ev["label"]))
        self.assertGreater(aux["evt_n"], 0)
        (aux["evt_sum"] / aux["evt_n"]).backward()
        missing = [n for n, p in head.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        self.assertEqual(missing, [])
        ad = self.make(commit="adaptive", defer_alpha=0.05)
        out, _ = self.drive(ad, ev, z0, feat, T)
        _, hi = ad.commit_bounds()
        for k, (z, lag, st) in out.items():                         # flushed last-window events also have lag 0
            self.assertEqual(lag == 0 and st == 0, float(z0[k]) - ad.log_w >= hi)


class LookaheadOccupancyTests(unittest.TestCase):
    def test_occupancy_variant(self):
        """occupancy_aux: same outputs as plain B2 with the same weights; the auxiliary loss needs events['occ'], is
        counted in exist_sum / exist_n and trains the encoder and recurrences but not the readout."""
        from model.lookahead_head import LookaheadHead
        from test_changepoint_layer import occ_from_events
        torch.manual_seed(0)
        plain = LookaheadHead(4, 16, 24, pos_weight=30.0, defer_max=3, cell=4)
        torch.manual_seed(0)
        occh = LookaheadHead(4, 16, 24, pos_weight=30.0, defer_max=3, cell=4, occupancy_aux=True)
        self.assertIsNone(plain.occupancy_spec)
        self.assertEqual(occh.occupancy_spec, (4, 4, 6))
        for (n1, p1), (n2, p2) in zip(plain.named_parameters(), occh.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), n1)
        T = 9
        ev, z0, feat = make_events(T, batch=2, seed=6)
        drv = LookaheadTests()
        a, _ = drv.drive(plain, ev, z0, feat, T, batch=2)
        b, _ = drv.drive(occh, ev, z0, feat, T, batch=2)
        for k in a:
            self.assertAlmostEqual(a[k][0], b[k][0], places=6)
        events = {k: torch.from_numpy(ev[k]) for k in ("t", "b", "x", "y", "p", "t_local", "bin")}
        labels = torch.from_numpy(ev["label"])
        with self.assertRaises(ValueError):
            occh.forward_chunk(events, torch.from_numpy(z0), torch.from_numpy(feat), None, T, labels=labels)
        events["occ"] = torch.from_numpy(occ_from_events(ev, T, 2))
        _, _, aux = occh.forward_chunk(events, torch.from_numpy(z0), torch.from_numpy(feat), None, T, labels=labels)
        self.assertGreater(aux["exist_n"], 0)
        (aux["exist_sum"] / aux["exist_n"]).backward()
        got = {n for n, p in occh.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
        self.assertTrue({"occ_head.weight", "obs_bias"} <= got)
        self.assertTrue(any(n.startswith("enc.") for n in got) and any(n.startswith("bwd.") for n in got)
                        and any(n.startswith("fwd.") for n in got))
        self.assertFalse(any(n.startswith("read_") for n in got))
        _, _, aux0 = plain.forward_chunk(events, torch.from_numpy(z0), torch.from_numpy(feat), None, T, labels=labels)
        self.assertEqual(aux0["exist_n"], 0)                          # plain B2: event loss only
        init = occh.init_occupancy_prior(1e-3)                        # prior-probability bias init (Lin et al. 2017)
        self.assertAlmostEqual(float(occh.occ_head.bias), math.log(1e-3 / (1 - 1e-3)), places=5)
        self.assertAlmostEqual(init["occupancy_rate"], 1e-3)


class GroupsAndSelectionTests(unittest.TestCase):
    def write_split(self, root, split, specs):
        (root / split).mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(0)
        for name, (n, on, tmax) in specs.items():
            t = np.sort(rng.randint(0, tmax, n))
            p = (rng.rand(n) < on).astype(np.float64)
            loc = np.stack([rng.randint(0, 346, n), rng.randint(0, 260, n), t], 1).astype(np.float64)
            norm = np.column_stack([loc, p, np.zeros(n), np.zeros(n)])
            np.savez(root / split / name, ev_loc=loc, evs_norm=norm)

    def test_shot_noise_rule(self):
        from utils.recording_groups import dense_names, is_dense
        self.assertTrue(is_dense(0.54, 0.539))
        self.assertFalse(is_dense(0.39, 0.994))                      # hot-pixel / leakage: ON-dominated
        self.assertFalse(is_dense(0.09, 0.61))                       # balanced but sparse
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            px = 260 * 346
            self.write_split(root, "val", {"a.npz": (int(0.6 * px * 2), 0.5, 2000), "b.npz": (int(0.6 * px * 2), 0.95, 2000),
                                           "c.npz": (int(0.1 * px * 2), 0.5, 2000)})
            self.assertEqual(dense_names(str(root), ("val",))["val"], ["a.npz"])

    def test_guarded_selection(self):
        import select_checkpoint as SC
        dense = ["d.npz"]

        def rec(epoch, n_tp, d_tp):                                 # normal: 1000 targets; dense: 100 targets
            return {"epoch": epoch, "seed": 37, "val_sequences": {"n.npz": [n_tp, 1000 - n_tp, 1000 - n_tp],
                                                                  "d.npz": [d_tp, 100 - d_tp, 100 - d_tp]},
                    "val_iou": (n_tp + d_tp) / float(n_tp + d_tp + 2 * (1100 - n_tp - d_tp))}
        records = [rec(0, 900, 10), rec(1, 950, 30), rec(2, 930, 95), rec(3, 948, 40)]
        res = SC.select(SC.epoch_table(records), dense, delta=0.005, rule="guarded")
        self.assertEqual(res["best_val_epoch"], 2)                  # dense spike with a normal drop
        self.assertEqual(res["epoch"], 3)                           # guarded: best val among normal >= best - 0.005
        self.assertEqual(res["candidates"], [1, 3])
        legacy = SC.select(SC.epoch_table([{"epoch": 0, "val_iou": 0.5}, {"epoch": 1, "val_iou": 0.6}]), dense,
                           rule="guarded")
        self.assertEqual(legacy["epoch"], 1)
        with tempfile.TemporaryDirectory() as d:                    # frozen run without metrics.jsonl (the 10-23 V5)
            run = Path(d) / "bb_v5_frozen_seed37"
            run.mkdir()
            (run / "best_val_iou_seed37.pt").write_bytes(b"frozen-weights")
            with contextlib.redirect_stdout(io.StringIO()):
                SC.main(["--run", str(run), "--data-root", str(Path(d) / "no-data"), "--copy"])
            self.assertEqual((run / "selected_seed37.pt").read_bytes(), b"frozen-weights")
            sel = json.loads((run / "selection.json").read_text(encoding="utf-8"))
            self.assertTrue(sel["frozen"])
            self.assertEqual(sel["seed"], 37)
        bad = copy.deepcopy(records)
        bad[0]["val_iou"] = 0.1
        with self.assertRaises(RuntimeError):                       # counts must reproduce the official IoU
            SC.select(SC.epoch_table(bad), dense, rule="guarded")
        self.assertEqual(SC.select(SC.epoch_table(records), dense)["epoch"], 3)            # default = guarded
        self.assertEqual(SC.select(SC.epoch_table(records), rule="plain")["epoch"], 2)     # plain: the dense spike

    def test_plain_and_replica_selection(self):
        import select_checkpoint as SC
        tie = SC.select(SC.epoch_table([{"epoch": e, "val_iou": v} for e, v in ((0, 0.5), (1, 0.7), (2, 0.7))]),
                        rule="plain")
        self.assertEqual(tie["epoch"], 1)                         # first epoch reaching the best = trainer's best_val_iou
        ref = [{"epoch": e, "val_iou": v, "seed": 37} for e, v in enumerate((0.3, 0.6, 0.8, 0.7, 0.75))]
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            run, refp = Path(d) / "bb_v5_25_seed37", Path(d) / "reference.jsonl"
            (run / "epochs").mkdir(parents=True)
            refp.write_text("".join(json.dumps(r) + "\n" for r in ref), encoding="utf-8")
            for e in range(4):                                     # stopped after epoch 3, bit-identical so far
                (run / "epochs" / ("epoch_%03d.pt" % e)).write_bytes(b"w%d" % e)
            (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref[:4]), encoding="utf-8")
            SC.main(["--run", str(run), "--reference-metrics", str(refp), "--copy"])
            sel = json.loads((run / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual((sel["epoch"], sel["replica_epochs_compared"], sel["reference_epochs"]), (2, 4, 5))
            self.assertEqual((run / "selected_seed37.pt").read_bytes(), b"w2")
            late = [dict(r, val_iou=0.9) if r["epoch"] == 4 else r for r in ref]   # best epoch not trained yet
            refp.write_text("".join(json.dumps(r) + "\n" for r in late), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                SC.main(["--run", str(run), "--reference-metrics", str(refp)])
            bad = [dict(r, val_iou=r["val_iou"] + 1e-12) if r["epoch"] == 1 else r for r in ref]
            refp.write_text("".join(json.dumps(r) + "\n" for r in bad), encoding="utf-8")
            with self.assertRaises(RuntimeError):                  # not a bit-exact replica
                SC.main(["--run", str(run), "--reference-metrics", str(refp)])
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                SC.main(["--run", str(run), "--rule", "guarded"])  # guarded needs the data root

    def test_v5_w25_reference_record(self):
        import select_checkpoint as SC
        ref = SC.read_records(os.path.join(ROOT, "configs", "final", "reference", "bb_v5_25_seed37_v63_record.jsonl"))
        self.assertEqual(len(ref), 50)
        self.assertEqual(SC.select(SC.epoch_table(ref), rule="plain")["epoch"], 18)   # the 10-25 run's best_val_iou


class BackboneChoiceTests(unittest.TestCase):
    def test_rule_and_branches(self):
        import select_backbone as SB

        def summ(val, normal, noise, fpg, msg=2.0):
            return {"val_iou": val, "normal": normal, "dense": 0.5, "noise_mean_iou": noise, "fp_growth": fpg,
                    "msg_growth": msg}
        table = {"v5": summ(0.867, 0.926, 0.20, 50.0, 7.4), "e1": summ(0.884, 0.925, 0.55, 3.5, 3.5),
                 "e3a": summ(0.885, 0.918, 0.45, 3.4, 2.4), "e3b": summ(0.886, 0.927, 0.57, 1.8, 3.3)}
        win, cands, _ = SB.choose(table)
        self.assertNotIn("e3a", cands)                              # normal 0.918 < 0.926 - 0.005
        self.assertEqual(win, "e3b")                                 # within 0.003 of e1: more robust (0.57 vs 0.55)
        table["e3b"]["noise_mean_iou"] = 0.555
        self.assertEqual(SB.choose(table)[0], "e3b")                 # noise within 0.01: smaller FP growth
        self.assertTrue(SB.claim_branch(table).startswith("B"))
        table["e3b"]["noise_mean_iou"] = 0.60
        self.assertTrue(SB.claim_branch(table).startswith("A"))
        table["e3b"].update(noise_mean_iou=0.50, fp_growth=4.0)
        self.assertTrue(SB.claim_branch(table).startswith("C"))
        table["e1"]["val_iou"] = 0.890                               # clear clean winner: no tie-break
        self.assertEqual(SB.choose(table)[0], "e1")


class SummaryTests(unittest.TestCase):
    def test_summary_tables(self):
        """tools/final_summary.py lists real runs once (symlinked names as markers) and reads the eval / analysis files."""
        import final_summary as FS
        with tempfile.TemporaryDirectory() as d:
            log, run = Path(d) / "log", Path(d) / "run"
            bb, head = log / "bb_e3b_50_seed37", log / "rcl_feature_50_on_e3b_seed37"
            for p in (bb, head, run / "analysis"):
                p.mkdir(parents=True)
            carry = {"iou": 0.88, "acc": 0.96, "pd": 0.95, "fa": 1e-5, "messages_mean": 7e4,
                     "per_sequence": {"a.npz": [80, 10, 10], "b.npz": [10, 5, 5]}, "latency": {"latency_median_ms": 58.0}}
            for split in ("val", "test"):
                (bb / ("eval_%s_selected_seed37.json" % split)).write_text(json.dumps({"carry": carry}), encoding="utf-8")
            (bb / "selection.json").write_text(json.dumps({"epoch": 40, "best_val_epoch": 24}), encoding="utf-8")
            track = {"iou": 0.89, "acc": 0.97, "tp": 1, "fp": 1, "fn": 1}
            ev = {"base": dict(track, iou=0.87), "track": track, "per_sequence": {"track": {"a.npz": [80, 9, 9]}},
                  "base_checkpoint": str(bb / "selected_seed37.pt")}
            for split in ("val", "test"):
                (head / ("eval_%s_selected_seed37.json" % split)).write_text(json.dumps(ev), encoding="utf-8")
            pol = {"iou": 0.89, "first_beat_recall": 0.7, "latency_ms": {"target_mean": 64.0, "target_median": 28.0},
                   "flips": {"rescued": {"same": 3, "onset": 1, "gap": 1, "never": 0}, "suppressed": 2, "fp_removed": 9,
                             "fp_added": 4}}
            (run / "analysis" / "rcl_feature_50_on_e3b_seed37_test.json").write_text(
                json.dumps({"D": 10, "policies": {"zero": pol, "fixed_10": pol, "adaptive_0.05": pol}}), encoding="utf-8")
            (run / "backbone_choice_seed37.json").write_text(json.dumps(
                {"winner": "e3b", "winner_run": str(bb), "claim_branch": "A: ...", "log": ["x"]}), encoding="utf-8")
            try:
                os.symlink("bb_e3b_50_seed37", str(log / "bb_m2_50_seed37"))
                linked = True
            except (OSError, NotImplementedError):                 # no symlink privilege on this Windows account
                linked = False
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                FS.main(["--log-root", str(log), "--run-root", str(run)])
            text = out.getvalue()
            self.assertIn("| bb_e3b_50", text)
            self.assertIn("40（24）", text)
            self.assertIn("0.8700 → 0.8900", text)
            self.assertIn("64.0 / 28.0", text)
            self.assertIn("5 / 2 / 9 / 4", text)
            self.assertIn("选定：**e3b**", text)
            self.assertEqual(text.count("| bb_e3b_50"), 1)
            if linked:
                self.assertIn("← bb_m2_50_seed37", text)


class StreamGroupTests(unittest.TestCase):
    def test_groups_match_ungrouped_training(self):
        import train_stream_v1 as T
        from test_stream_comm import POS_WEIGHT, budget_cfg
        from test_stream_layerwise import Q99, synthetic_sequence
        from test_stream_tif import make_tif_net
        from utils.comm_budget import CommBudget
        data = [synthetic_sequence(seed) for seed in (0, 1, 2, 3, 4)]
        lengths = [seq.n_windows for seq in data]
        plan = [(0, [(0, 12)]), (1, [(1, 11)]), (0, [(2, 12), (3, 12)]), (3, [(4, 12)])]
        base = make_tif_net()
        cpu = torch.device("cpu")
        runs = {}
        for groups in (1, 2, 3):
            net = copy.deepcopy(base)
            opt = torch.optim.SGD(net.parameters(), lr=0.05)
            cfg = budget_cfg(1e-3, streams=4, accumulate=2, stream_groups=groups)
            comm = CommBudget.from_config(cfg, net)
            comm.mu = 0.5
            out = T.train_streams(net, data, lengths, opt, cfg, Q99, POS_WEIGHT, cpu, None, plan=plan, comm=comm)
            runs[groups] = (net, out, comm.mu)
        net1, out1, mu1 = runs[1]
        for groups in (2, 3):
            net, out, mu = runs[groups]
            self.assertEqual((out["steps"], out["events"]), (out1["steps"], out1["events"]))
            self.assertAlmostEqual(out["loss_sum"], out1["loss_sum"], delta=1e-9 * abs(out1["loss_sum"]))
            self.assertAlmostEqual(mu, mu1, places=12)
            for (name, pa), (_, pb) in zip(net1.named_parameters(), net.named_parameters()):
                self.assertTrue(torch.allclose(pa, pb, rtol=0, atol=1e-10), name)
        self.assertEqual(T.stream_groups(4, 3), [(0, 1), (1, 3), (3, 4)])
        with self.assertRaises(ValueError):
            T.stream_groups(4, 5)

    def test_select_streams_slices_background_map(self):
        import train_stream_v1 as T
        x = torch.randn(3, 4, 2, 5, 6)
        ev = {"b": torch.tensor([0, 1, 2, 3, 3]), "t": torch.tensor([0, 1, 2, 0, 1]), "x": torch.arange(5),
              "y": torch.arange(5), "p": torch.ones(5), "t_local": torch.zeros(5), "bg": torch.randn(3, 4, 2, 5, 6)}
        lab = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        keep, valid = torch.ones(3, 4), torch.ones(3, 4)
        xg, eg, lg, kg, vg = T.select_streams(x, ev, lab, keep, valid, 2, 4)
        self.assertEqual(tuple(eg["bg"].shape), (3, 2, 2, 5, 6))
        self.assertTrue(torch.equal(eg["bg"], ev["bg"][:, 2:4]))
        self.assertEqual(eg["b"].tolist(), [0, 1, 1])
        self.assertEqual(lg.tolist(), [0.0, 1.0, 1.0])


class NoiseUnitTests(unittest.TestCase):
    def test_hz_to_load(self):
        import train_stream_v1 as T
        import train_track_head as TH
        self.assertAlmostEqual(T.noise_load_from_hz(0.8, 50), 1.0)
        self.assertAlmostEqual(T.noise_load_from_hz(4.0, 50), 5.0)
        self.assertAlmostEqual(T.noise_load_from_hz(0.8, 25), 0.5)
        self.assertIsNone(TH.noise_settings(SimpleNamespace(eval_noise_hz=None), {"window_ms": 50}))
        ns = TH.noise_settings(SimpleNamespace(eval_noise_hz=2.4, eval_noise_seed=3), {"window_ms": 25})
        self.assertEqual((ns["eval_noise_seed"], ns["eval_noise_hz"]), (3, 2.4))
        self.assertAlmostEqual(ns["eval_noise_load"], 1.5)


class B1Tests(unittest.TestCase):
    def test_persistence_matches_loop(self):
        import b1_delayed_confirm as B1
        rng = np.random.RandomState(1)
        n = 600
        t = np.sort(rng.randint(0, 1000, n))
        rec = {"t": t, "x": rng.randint(0, 40, n), "y": rng.randint(0, 24, n)}
        prob = rng.rand(n).astype(np.float32)
        for r in (0, 1, 2):
            got = B1.persistence(rec, prob, 50, r, (1, 3, 5))
            win, cy, cx = t // 50, rec["y"] // 4, rec["x"] // 4
            T = int(win.max()) + 1
            for d in (1, 3, 5):
                want = np.zeros(n, dtype=np.int64)
                for e in range(n):
                    for j in range(int(win[e]) + 1, min(int(win[e]) + d, T - 1) + 1):
                        m = (win == j) & (prob >= 0.9) & (np.abs(cy - cy[e]) <= r) & (np.abs(cx - cx[e]) <= r)
                        want[e] += int(m.any())
                self.assertEqual(got[d].tolist(), want.tolist(), (r, d))


FINAL_SOURCES = {
    "bb_v5_50": "configs/evisseg_stream_tif_comm_peak.yaml",
    "bb_e1_50": "configs/evisseg_stream_tif_comm_peak_noiseaug.yaml",
    "bb_e3a_50": "configs/evisseg_stream_tif_comm_peak_invariant.yaml",
    "bb_e3b_50": "configs/evisseg_stream_tif_comm_peak_invariant_noiseaug.yaml",
    "bb_v5_evf": "configs/evisseg_stream_tif_comm_peak_evflying.yaml",
    "bb_e3b_evf": "configs/evisseg_stream_tif_comm_peak_invariant_noiseaug_evflying.yaml",
    "rcl_feature_50": "configs/changepoint_v41_insample_feature.yaml",
    "rcl_feature_causal_50": "configs/changepoint_v41_insample_feature_causal.yaml",
    "rcl_score_50": "configs/changepoint_v41_insample.yaml",
    "rcl_feature_evf": "configs/changepoint_v41_evflying_feature.yaml",
}
CLOCK_KEYS = {"window_ms", "n_windows", "tbptt_k", "eval_chunk", "calib_windows", "comm_budget_messages",
              "rolling_window", "segments", "timing_warmup_windows", "save_root", "stats_path"}


class FinalConfigTests(unittest.TestCase):
    def load(self, rel):
        from utils.stream_common import load_flat_config
        return load_flat_config(os.path.join(ROOT, rel))

    def test_final_configs_equal_their_sources(self):
        for name, src in FINAL_SOURCES.items():
            a, b = self.load(src), self.load("configs/final/%s.yaml" % name)
            self.assertEqual(set(b), set(a) | {"save_epochs"}, name)
            allowed = {"save_root"} | ({"invariant_sensor_mask"} if name == "bb_e3b_evf" else set())
            self.assertEqual({k for k in a if a[k] != b[k]}, allowed, name)
            self.assertEqual(b["save_root"], "log/final/%s_seed37" % name)
            self.assertTrue(b["save_epochs"])
        self.assertFalse(self.load("configs/final/bb_e3b_evf.yaml")["invariant_sensor_mask"])

    def test_25ms_configs_keep_physical_quantities(self):
        a, b = self.load("configs/final/bb_v5_50.yaml"), self.load("configs/final/bb_v5_25.yaml")
        self.assertEqual(set(b), set(a) | {"stream_groups"})
        self.assertEqual({k for k in a if a[k] != b[k]}, CLOCK_KEYS)
        r = a["window_ms"] // b["window_ms"]
        self.assertEqual(r, 2)
        self.assertEqual(a["window_ms"] * a["n_windows"], b["window_ms"] * b["n_windows"])
        for key in ("tbptt_k", "eval_chunk", "calib_windows", "rolling_window", "timing_warmup_windows"):
            self.assertEqual(a[key] * r, b[key], key)
        self.assertEqual(a["comm_budget_messages"] / r, b["comm_budget_messages"])
        self.assertTrue(os.path.exists(os.path.join(ROOT, b["stats_path"])))
        h50, h25 = self.load("configs/final/rcl_feature_50.yaml"), self.load("configs/final/rcl_feature_25.yaml")
        b50, b25 = self.load("configs/final/b2_lookahead_50.yaml"), self.load("configs/final/b2_lookahead_25.yaml")
        self.assertEqual(set(b25), set(b50))
        self.assertEqual({k for k in b50 if b50[k] != b25[k]}, {k for k in h50 if h50[k] != h25[k]})
        for key in ("defer_max", "tbptt_k", "eval_chunk", "rolling_window", "segments", "timing_warmup_windows",
                    "base_checkpoint"):                         # B2 @ 25 ms = RCL @ 25 ms's clock
            self.assertEqual(b25[key], h25[key], key)
        self.assertEqual(b25["save_root"], "log/final/b2_lookahead_25_seed37")
        self.assertEqual(set(h25), set(h50))
        self.assertEqual({k for k in h50 if h50[k] != h25[k]},
                         {"defer_max", "tbptt_k", "eval_chunk", "rolling_window", "segments", "timing_warmup_windows",
                          "base_checkpoint", "save_root"})
        self.assertEqual(h50["defer_max"] * 50, h25["defer_max"] * 25)          # 500 ms future budget
        self.assertEqual(h50["tbptt_k"] * 50, h25["tbptt_k"] * 25)
        self.assertEqual(h25["segments"], b["segments"])

    def test_b2_occupancy_config(self):
        a, b = self.load("configs/final/b2_lookahead_50.yaml"), self.load("configs/final/b2_lookahead_occ_50.yaml")
        self.assertEqual(set(b), set(a) | {"occupancy_aux"})
        self.assertEqual({k for k in a if a[k] != b[k]}, {"lambda_exist", "save_root"})
        self.assertTrue(b["occupancy_aux"])
        self.assertEqual(b["lambda_exist"], self.load("configs/final/rcl_score_50.yaml")["lambda_exist"])

    def test_b2_matches_rcl_protocol(self):
        rcl, b2 = self.load("configs/final/rcl_score_50.yaml"), self.load("configs/final/b2_lookahead_50.yaml")
        for key in ("root", "input_device", "cell", "defer_max", "defer_alpha", "commit", "seed", "epochs", "lr",
                    "lr_end", "streams", "tbptt_k", "stream_layout", "grad_clip", "lambda_event", "now_loss_weight",
                    "lag_loss_weight", "deterministic", "eval_chunk", "save_epochs", "calibrate_existence", "threshold",
                    "pd_detT", "correct_thresh", "rolling_window", "segments", "timing_warmup_windows"):
            self.assertEqual(rcl[key], b2[key], key)
        self.assertEqual(rcl["expert_input"], "score")
        self.assertEqual(b2["head_kind"], "lookahead")
        self.assertEqual((b2["lambda_exist"], b2["lambda_dynamics"], b2["dynamics_prefit_steps"]), (0.0, 0.0, 0))

    def test_v5lif_swaps_only_the_neuron(self):
        a, b = self.load("configs/final/bb_v5_50.yaml"), self.load("configs/final/bb_v5lif_50.yaml")
        comm = {"comm_budget_messages", "comm_dual_lr", "comm_budget_mode", "comm_peak_alpha"}
        self.assertEqual(set(b), set(a) - comm - {"tif"})          # the budget loss is tif-only
        self.assertEqual({k for k in b if a[k] != b[k]}, {"neuron", "save_root"})
        self.assertEqual((a["neuron"], b["neuron"]), ("tif", "lif"))
        self.assertEqual(b["save_root"], "log/final/bb_v5lif_50_seed37")

    def test_window_sweep_configs(self):
        """Slice-length sweep (11-06): BIR / LIF at 25 / 50 / 100 / 200 / 400 ms change only per-window quantities."""
        a = self.load("configs/final/bb_v5_50.yaml")
        comm = {"comm_budget_messages", "comm_dual_lr", "comm_budget_mode", "comm_peak_alpha"}
        for ms in (100, 200, 400):
            b = self.load("configs/final/bb_v5_%d.yaml" % ms)
            r = ms // a["window_ms"]
            self.assertEqual(set(b), set(a), ms)
            self.assertEqual({k for k in a if a[k] != b[k]}, CLOCK_KEYS, ms)
            self.assertEqual(a["window_ms"] * a["n_windows"], b["window_ms"] * b["n_windows"])   # 8 s
            for key in ("tbptt_k", "eval_chunk", "calib_windows"):                            # exact: same time span
                self.assertEqual(a[key], b[key] * r, (ms, key))
            for key in ("rolling_window", "timing_warmup_windows"):                            # diagnostics only: ceil
                self.assertEqual(-(-a[key] // r), b[key], (ms, key))
            self.assertEqual(a["comm_budget_messages"] * r, b["comm_budget_messages"])        # same budget per second
            self.assertEqual([[s // r, (e + 1) // r - 1] for s, e in a["segments"]], b["segments"])
            self.assertEqual(b["save_root"], "log/final/bb_v5_%d_seed37" % ms)
            self.assertTrue(os.path.exists(os.path.join(ROOT, b["stats_path"])), b["stats_path"])
        for ms in (25, 100, 200, 400):
            bir, lif = self.load("configs/final/bb_v5_%d.yaml" % ms), self.load("configs/final/bb_v5lif_%d.yaml" % ms)
            self.assertEqual(set(lif), set(bir) - comm - {"tif"}, ms)
            self.assertEqual({k for k in lif if bir[k] != lif[k]}, {"neuron", "save_root"}, ms)
            self.assertEqual((bir["neuron"], lif["neuron"]), ("tif", "lif"))
            self.assertEqual(lif["save_root"], "log/final/bb_v5lif_%d_seed37" % ms)


def write_synthetic_dataset(root):
    """Small EV-UAV-format splits (train 3 / val 2 / test 1 sequences) + stats.json under root."""
    from test_stream_layerwise import Q99, synthetic_sequence
    pos = neg = 0
    for split, seeds in (("train", (3, 4, 5)), ("val", (6, 8)), ("test", (7,))):
        (root / split).mkdir()
        for seed in seeds:
            seq = synthetic_sequence(seed=seed, n=3000)
            loc = np.stack((seq.x, seq.y, seq.t), axis=1)
            norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
            np.savez(root / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
            if split == "train":
                pos += int(seq.label.sum())
                neg += seq.n_events - int(seq.label.sum())
    stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0))
    (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


def small_backbone_config(name, root, save_root):
    """A final backbone config shrunk to the synthetic data (2 epochs, tiny network)."""
    from test_stream_layerwise import H as SH, SMALL, W as SW
    from utils.stream_common import load_flat_config
    cfg = load_flat_config(os.path.join(ROOT, "configs", "final", name + ".yaml"))
    cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(save_root),
               channels=list(SMALL), height=SH, width_px=SW, pad_height=SH, pad_width=SW,
               n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
               calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
               segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
               execution="layer", input_device="gpu", streams=2, stream_groups=2)
    return cfg


class EndToEndTests(unittest.TestCase):
    def test_pipeline(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
            import yaml
        except ImportError as error:
            self.skipTest("完整训练入口需要 cv2/pandas/tqdm/yaml: %s" % error)
        import b1_delayed_confirm as B1
        import final_analysis as FA
        import select_checkpoint as SC
        import train_stream_v1 as T
        import train_track_head as TH
        from test_stream_layerwise import H as SH, W as SW
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            write_synthetic_dataset(root)
            base_dir = root / "base"
            cfg = small_backbone_config("bb_e3b_50", root, base_dir)
            self.assertTrue(cfg["save_epochs"])
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            records = [json.loads(x) for x in (base_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(set(records[0]["val_sequences"]), {"seq_6.npz", "seq_8.npz"})
            for r in records:                                        # per-sequence counts reproduce the official IoU
                c = np.sum(list(r["val_sequences"].values()), 0)
                self.assertAlmostEqual(c[0] / float(c.sum()), r["val_iou"], places=6)
            self.assertTrue((base_dir / "epochs" / "epoch_001.pt").exists())
            SC.main(["--run", str(base_dir), "--data-root", str(root), "--copy", "--height", str(SH),
                     "--width", str(SW)])
            sel = json.loads((base_dir / "selection.json").read_text(encoding="utf-8"))
            base_ckpt = str(base_dir / "selected_seed37.pt")
            self.assertTrue(Path(base_ckpt).exists())
            self.assertIn(sel["epoch"], (0, 1))
            for name in ("rcl_feature_50", "b2_lookahead_50", "b2_lookahead_occ_50"):
                head_cfg = load_flat_config(os.path.join(ROOT, "configs", "final", name + ".yaml"))
                head_cfg.update(root=str(root), base_checkpoint=base_ckpt, save_root=str(root / name), epochs=2,
                                streams=2, tbptt_k=4, eval_chunk=4, segments=[[0, 3], [4, 7], [8, 11]],
                                timing_warmup_windows=1, defer_max=3)
                cfg_path = root / (name + ".yaml")
                cfg_path.write_text(yaml.safe_dump({"ALL": head_cfg}), encoding="utf-8")
                TH.main(["--config", str(cfg_path), "--mode", "train", "--device", "cpu"])
                SC.main(["--run", str(root / name), "--data-root", str(root), "--copy", "--height", str(SH),
                         "--width", str(SW)])
                ckpt = root / name / "selected_seed37.pt"
                self.assertTrue(ckpt.exists())
                traj_dir = root / ("traj_" + name)
                TH.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                         "--checkpoint", str(ckpt), "--trajectory-dir", str(traj_dir)])
                TH.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "val",
                         "--checkpoint", str(ckpt), "--trajectory-dir", str(root / ("trajval_" + name))])
                TH.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                         "--checkpoint", str(ckpt), "--commit", "adaptive", "--tag", "adaptive"])
                adaptive = json.loads((root / name / "eval_test_selected_seed37_adaptive.json").read_text(
                    encoding="utf-8"))
                res = json.loads((root / name / "eval_test_selected_seed37.json").read_text(encoding="utf-8"))
                TH.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                         "--checkpoint", str(ckpt), "--eval-noise-hz", "4", "--tag", "noise"])
                noisy = json.loads((root / name / "eval_test_selected_seed37_noise_noise4hz_seed0.json").read_text(
                    encoding="utf-8"))
                self.assertEqual(noisy["noise"]["eval_noise_hz"], 4.0)
                if name == "rcl_feature_50":                         # the head sees the backbone eval's noise realisation
                    net, tap, base, q99, _, _ = TH.load_base(dict(head_cfg), torch.device("cpu"))
                    data = TH.split_dataset(base, "test")
                    noise = TH.noise_settings(SimpleNamespace(eval_noise_hz=4.0, eval_noise_seed=0), base)
                    seq = TH.BaseOutputs(net, tap, base, q99, torch.device("cpu"), data, noise=noise).get(0)[0]
                    clean = data[0]
                    ref = T.eval_noise(clean, dict(base, eval_noise_load=T.noise_load_from_hz(4.0, base["window_ms"]),
                                                   eval_noise_seed=0), 0)
                    self.assertGreater(seq.n_events, clean.n_events)
                    self.assertTrue(np.array_equal(seq.t, ref.t) and np.array_equal(seq.x, ref.x))
                    self.assertEqual(int(seq.label[clean.n_events:].sum()), 0)
                recs = FA.load_split(str(traj_dir), str(root), "test")
                out = FA.analyse(recs, set(), float(head_cfg.get("window_ms", 50)), sensor=(SH, SW))
                pol = out["policies"]
                self.assertEqual(pol["fixed_3"]["tp"], res["track"]["tp"])
                self.assertEqual(pol["fixed_3"]["fp"], res["track"]["fp"])
                self.assertEqual(pol["zero"]["tp"], res["base"]["tp"])
                self.assertAlmostEqual(pol["fixed_3"]["pd"], res["track"]["pd"], places=6)
                self.assertEqual((pol["adaptive_0.05"]["tp"], pol["adaptive_0.05"]["fp"]),
                                 (adaptive["track"]["tp"], adaptive["track"]["fp"]))   # offline adaptive = official
                self.assertEqual(len(out["info_arrival"]), 4)
                if name == "rcl_feature_50":
                    B1.main(["--val-traj", str(root / ("trajval_" + name)), "--test-traj", str(traj_dir),
                             "--data-root", str(root), "--window-ms", "50", "--out", str(root / "b1.json"),
                             "--height", str(SH), "--width", str(SW)])
                    b1 = json.loads((root / "b1.json").read_text(encoding="utf-8"))
                    self.assertEqual(set(b1["chosen_on_val"]), {"1", "3", "5", "10"})

    def test_lif_backbone_with_b2(self):
        """The LIF cell of the 2x2: bb_v5lif_50 trains without the budget loss and carries a B2 head (fixed D + past)."""
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
            import yaml
        except ImportError as error:
            self.skipTest("完整训练入口需要 cv2/pandas/tqdm/yaml: %s" % error)
        import select_checkpoint as SC
        import train_stream_v1 as T
        import train_track_head as TH
        from test_stream_layerwise import H as SH, W as SW
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            write_synthetic_dataset(root)
            base_dir = root / "base"
            cfg = small_backbone_config("bb_v5lif_50", root, base_dir)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            SC.main(["--run", str(base_dir), "--data-root", str(root), "--copy", "--height", str(SH),
                     "--width", str(SW)])
            base_ckpt = str(base_dir / "selected_seed37.pt")
            saved = torch.load(base_ckpt, map_location="cpu")["config"]
            self.assertEqual(saved["neuron"], "lif")
            self.assertIsNone(saved.get("comm_budget_messages"))
            head_cfg = load_flat_config(os.path.join(ROOT, "configs", "final", "b2_lookahead_50.yaml"))
            head_cfg.update(root=str(root), base_checkpoint=base_ckpt, save_root=str(root / "b2"), epochs=2,
                            streams=2, tbptt_k=4, eval_chunk=4, segments=[[0, 3], [4, 7], [8, 11]],
                            timing_warmup_windows=1, defer_max=3)
            cfg_path = root / "b2.yaml"
            cfg_path.write_text(yaml.safe_dump({"ALL": head_cfg}), encoding="utf-8")
            TH.main(["--config", str(cfg_path), "--mode", "train", "--device", "cpu"])
            SC.main(["--run", str(root / "b2"), "--data-root", str(root), "--copy", "--height", str(SH),
                     "--width", str(SW)])
            ckpt = str(root / "b2" / "selected_seed37.pt")
            common = ["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                      "--checkpoint", ckpt]
            TH.main(common)
            TH.main(common + ["--future-stats", "past", "--tag", "past"])
            res = json.loads((root / "b2" / "eval_test_selected_seed37.json").read_text(encoding="utf-8"))
            past = json.loads((root / "b2" / "eval_test_selected_seed37_past.json").read_text(encoding="utf-8"))
            self.assertEqual(res["base"], past["base"])              # same frozen LIF backbone in both evals
            for r in (res, past):
                self.assertTrue(0.0 <= r["track"]["iou"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
