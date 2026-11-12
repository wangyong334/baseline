"""V15 delayed supervision, record boundaries, online outputs and metrics."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from dataset import stream_windows as sw
from dataset.stream_source import make_window_source
from model.observation_revision import ObservationRevision
from tools.v15 import train as T
from test_v15_backbone import make_net, H, W
from utils.stream_metrics import FastRoc, window_confusion


def sequence(name="synthetic.npz", counts=(4, 0, 5, 3), seed=3):
    times = np.concatenate([50 * k + 1 + 3 * np.arange(n) for k, n in enumerate(counts)]).astype(np.int64)
    index = np.arange(len(times), dtype=np.int64)
    x, y, p = (index * 5) % W, (index * 3) % H, (index % 2).astype(np.int8)
    labels = (index % 3 == 0).astype(np.float32)
    perm = np.random.RandomState(seed).permutation(len(times))
    x, y, times, p, labels = [value[perm] for value in (x, y, times, p, labels)]
    order, bounds = sw.split_windows(times, 50, len(counts))
    bins, local = sw.time_bin_and_local(times, 50, 5)
    return sw.StreamSequence(name, x, y, times, p, labels, labels.astype(np.float64),
                             bins, local, order, bounds)


def config(**kwargs):
    cfg = dict(time_bins=5, pad_height=H, pad_width=W, height=H, width_px=W,
               input_clip=3.0, window_ms=50, n_windows=4, input_device="gpu",
               state_mode="carry", streams=2, tbptt_k=3, revision_pair_batch=2,
               revision_source="observation", threshold=0.5, timing_warmup_windows=0,
               grad_clip=10.0, log_every_chunks=100, pd_detT=50, correct_thresh=0.0001,
               comm_budget_messages=157700, channels=[2, 4, 4, 4], readout_hidden=5)
    cfg.update(kwargs)
    return cfg


def revision(nonzero=False):
    head = ObservationRevision(channels=2, width=4, heads=2, patch=2, radius=1, pair_batch=2)
    if nonzero:
        with torch.no_grad():
            head.output.weight.copy_(torch.linspace(-0.3, 0.4, head.output.weight.numel())
                                      .reshape_as(head.output.weight))
    return head


class V15TrainingTests(unittest.TestCase):
    def test_unknown_or_probe_training_scope_is_not_formal(self):
        for scope in (None, {}, {"max_chunks": None},
                      {"limit_records": None, "max_chunks": 3}):
            self.assertFalse(T.training_is_complete_scope({"_training_scope": scope}))
        self.assertFalse(T.training_is_complete_scope({}))
        self.assertTrue(T.training_is_complete_scope({"_training_scope":
                         {"limit_records": None, "max_chunks": None}}))

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.device = torch.device("cpu")
        self.q99 = np.ones(12, dtype=np.float32)

    def test_chunk_pairs_commit_each_window_once_without_crossing_records(self):
        data = [sequence("record_%d.npz" % i, seed=i + 1) for i in range(3)]
        cfg = config()
        plan = [(0, [(0, 4), (1, 4)]), (1, [(2, 4)])]
        sources = T.StreamSources(data, cfg, self.q99, self.device, torch.float32)
        pending, finalized, input_events, output_events, empty_future = {}, [], 0, 0, 0
        saw_crosschunk = False
        for chunk in range(3):
            pieces = T.stream_pieces(plan, chunk * 3, (chunk + 1) * 3)
            _, ev, lab, _ = T.assemble_streams(sources, pieces, 3, cfg, self.device, torch.float32)
            features = torch.zeros(3, 2, 2, H, W, requires_grad=True)
            compact = torch.zeros(3, 2, 4, H // 2, W // 2, requires_grad=True)
            with torch.no_grad():
                for b, stream in enumerate(pieces):
                    for rec, lo, hi, offset, _ in stream:
                        for window in range(lo, hi):
                            pos = offset + window - lo
                            features[pos, b].fill_(rec * 10 + window)
                            compact[pos, b].fill_(rec * 10 + window + 0.5)
            targets, counts = T.chunk_targets(features, compact, ev, lab, sources, pieces, pending, chunk)
            input_events += counts["input_events"]
            output_events += counts["final_events"]
            empty_future += counts["valid_empty_future_windows"]
            for frame, new, cross in targets:
                finalized.append((frame["record"], frame["window"]))
                if new is None:
                    self.assertEqual(frame["window"], 3)
                else:
                    self.assertEqual(float(new.flatten()[0]), frame["record"] * 10 + frame["window"] + 1.5)
                if cross:
                    saw_crosschunk = True
                    self.assertFalse(frame["feature"].requires_grad)
                    self.assertFalse(frame["observation"].requires_grad)
            pending = {k: T._detach_frame(v) for k, v in pending.items()}
            for frame in pending.values():
                self.assertIsNone(frame["feature"].grad_fn)
                self.assertIsNone(frame["feature"]._base)
                self.assertEqual(frame["feature"].storage().size(), frame["feature"].numel())
        self.assertEqual(sorted(finalized), [(r, w) for r in range(3) for w in range(4)])
        self.assertEqual(input_events, sum(s.n_events for s in data))
        self.assertEqual(output_events, input_events)
        self.assertEqual(empty_future, 3)
        self.assertTrue(saw_crosschunk)
        self.assertEqual(pending, {})

    def test_bad_record_boundary_and_missing_pending_are_rejected(self):
        sources = type("Sources", (), {"bounds": {0: np.array([0, 1, 2]), 1: np.array([0, 1, 2])}})()
        ev = {"x": torch.zeros(1, dtype=torch.long), "y": torch.zeros(1, dtype=torch.long),
              "p": torch.zeros(1), "t_local": torch.zeros(1)}
        feature, compact, label = torch.zeros(1, 1, 2, H, W), torch.zeros(1, 1, 4, H // 2, W // 2), torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "recording boundary"):
            T.chunk_targets(feature, compact, ev, label, sources,
                            [[(1, 0, 1, 0, True)]], {0: {}}, 0)
        with self.assertRaisesRegex(RuntimeError, "disappeared"):
            T.chunk_targets(feature, compact, ev, label, sources,
                            [[(0, 1, 2, 0, False)]], {}, 1)

    def test_loss_supervision_mass_matches_original_events_not_number_of_heads(self):
        net = make_net().float()
        head = revision()
        seq = sequence(counts=(2, 3))
        cfg = config(n_windows=2, streams=1)
        sources = T.StreamSources([seq], cfg, self.q99, self.device, torch.float32)
        pieces = [[(0, 0, 2, 0, True)]]
        x, ev, lab, keep = T.assemble_streams(sources, pieces, 2, cfg, self.device, torch.float32)
        _, _, info = net.forward_chunk(x, ev, None, keep=keep, return_features=True, skip_readout=True)
        compact = T.encode_observations(head, info, cfg)
        current = net.read_features(info["features"], ev)
        targets, _ = T.chunk_targets(info["features"], compact, ev, lab, sources, pieces, {}, 0)
        raw, mass, detail = T.classification_loss(net, head, targets, 3.0, 2, current, lab)
        expected = F.binary_cross_entropy_with_logits(current, lab, pos_weight=current.new_tensor([3.0]), reduction="sum")
        self.assertEqual(mass, float(seq.n_events))
        self.assertAlmostEqual(float(raw), float(expected), places=6)
        self.assertEqual(detail["causal_events"], seq.n_events)
        self.assertEqual(detail["final_events"], seq.n_events)
        # Tails reuse existing logits: paired events are the only extra readout.
        with mock.patch.object(net, "read_features", wraps=net.read_features) as read:
            T.classification_loss(net, head, targets, 3.0, 2, current, lab)
        self.assertEqual(read.call_count, 1)
        _, causal_mass, causal_detail = T.classification_loss(net, None, [], 3.0, 2, current, lab)
        self.assertEqual(causal_mass, float(seq.n_events))
        self.assertEqual(causal_detail["final_events"], seq.n_events)

    def test_pair_final_loss_reaches_previous_and_future_backbone_inputs(self):
        net = make_net().float()
        head = revision(nonzero=True)
        old_x = torch.rand(1, 12, H, W, requires_grad=True)
        new_x = torch.rand(1, 12, H, W, requires_grad=True)
        ev = {"b": torch.zeros(3, dtype=torch.long), "x": torch.tensor([3, 5, 7]),
              "y": torch.tensor([2, 4, 6]), "p": torch.tensor([0., 1., 0.]),
              "t_local": torch.tensor([0.1, 0.3, 0.6])}
        _, states, old = net(old_x, ev, None, return_features=True, skip_readout=True)
        _, _, new = net(new_x, ev, states, return_features=True, skip_readout=True)
        old_e = head.encode_observation(old["observation"])
        new_e = head.encode_observation(new["observation"])
        logits = net.read_features(head.revise(old["features"], old_e, new_e), ev)
        F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
        for name, value in (("old input", old_x), ("future input", new_x)):
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
            self.assertGreater(float(value.grad.abs().sum()), 0.0, name)

    def test_training_k1_keeps_immediate_backbone_gradient_and_closes_supervision(self):
        net = make_net().float()
        # This uncalibrated three-event toy can have hard-boundary zero
        # gradients along the supervised pixels' early-layer paths. Boundary
        # surrogates isolate loss connectivity; production BIR is unchanged.
        for block in net.blocks():
            block.neuron.surrogate = True
        head = revision(nonzero=True)
        cfg = config(tbptt_k=1, streams=1, n_windows=3)
        data = [sequence(counts=(3, 0, 2))]
        optimizer = torch.optim.SGD(list(net.parameters()) + list(head.parameters()), lr=1e-4)
        before = net.enc1.conv.weight.detach().clone()
        first_grad = []
        original_step = optimizer.step

        def step(*args, **kwargs):
            if not first_grad:
                first_grad.append(float(net.enc1.conv.weight.grad.abs().sum()))
            return original_step(*args, **kwargs)

        with mock.patch.object(optimizer, "step", side_effect=step):
            result = T.train_epoch(net, head, data, optimizer, cfg, self.q99, 2.0, self.device,
                                   np.random.RandomState(1), plan=[(0, [(0, 3)])])
        self.assertEqual(result["input_events"], 5)
        self.assertEqual(result["final_events"], 5)
        self.assertEqual(result["supervision_mass"], 5.0)
        self.assertEqual(result["pending_windows_at_truncation"], 0)
        self.assertGreater(result["crosschunk_events"], 0)
        # At update one no future block exists; this gradient must come from
        # immediate causal supervision, not a later correction branch.
        self.assertGreater(first_grad[0], 0.0)
        self.assertGreater(float((before - net.enc1.conv.weight).abs().sum()), 0.0)

    def test_default_bir_first_window_receives_causal_gradient_before_future_arrives(self):
        net, head = make_net().float(), revision(nonzero=True)
        cfg = config(tbptt_k=1, streams=1, n_windows=3)
        data = [sequence(counts=(3, 0, 2))]
        sources = T.StreamSources(data, cfg, self.q99, self.device, torch.float32)
        pieces = [[(0, 0, 1, 0, True)]]
        x, ev, lab, keep = T.assemble_streams(sources, pieces, 1, cfg, self.device, torch.float32)
        _, _, info = net.forward_chunk(x, ev, None, keep=keep,
                                       return_features=True, skip_readout=True)
        info["features"].retain_grad()
        current = net.read_features(info["features"], ev)
        targets, _ = T.chunk_targets(info["features"], T.encode_observations(head, info, cfg),
                                     ev, lab, sources, pieces, {}, 0)
        self.assertEqual(targets, [])
        raw, mass, _ = T.classification_loss(net, head, targets, 2.0, 2, current, lab)
        self.assertEqual(mass, 0.5 * lab.numel())
        (raw / mass).backward()
        self.assertIsNotNone(info["features"].grad)
        self.assertGreater(float(info["features"].grad.abs().sum()), 0.0)
        self.assertTrue(all(not block.neuron.surrogate for block in net.blocks()))

    def test_training_releases_previous_chunk_features_before_next_forward(self):
        net = make_net().float()
        head = revision(nonzero=True)
        cfg = config(tbptt_k=2, streams=1, n_windows=4)
        data = [sequence(counts=(2, 2, 2, 2))]
        optimizer = torch.optim.SGD(list(net.parameters()) + list(head.parameters()), lr=1e-4)
        original = net.forward_chunk
        references, observed = [], []

        def forward(*args, **kwargs):
            if references:
                observed.append(references[-1]() is None)
            result = original(*args, **kwargs)
            references.append(weakref.ref(result[2]["features"]))
            return result

        with mock.patch.object(net, "forward_chunk", side_effect=forward):
            T.train_epoch(net, head, data, optimizer, cfg, self.q99, 2.0, self.device,
                           np.random.RandomState(1), plan=[(0, [(0, 4)])])
        self.assertTrue(observed)
        self.assertTrue(all(observed), "Prior chunk feature is still strongly referenced")

    def test_pending_is_frozen_history_across_optimizer_updates(self):
        """A cached feature is an online historical state, not a parameter view."""
        parameter = torch.nn.Parameter(torch.ones(2, H, W))
        source = parameter * 3.0
        frame = {"feature": source, "observation": source[:, ::2, ::2],
                 "events": {"x": torch.tensor([1]), "y": torch.tensor([2]),
                            "p": torch.tensor([1.]), "t_local": torch.tensor([0.4])},
                 "labels": torch.ones(1), "record": 0, "window": 0}
        cached = T._detach_frame(frame)
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        source.sum().backward()
        optimizer.step()
        # Recomputing after the update would change history. TBPTT deliberately
        # keeps the original value, just as it retains the detached membrane.
        self.assertFalse(torch.equal(parameter * 3.0, cached["feature"]))
        self.assertTrue(torch.equal(cached["feature"], torch.full_like(parameter, 3.0)))
        self.assertIsNone(cached["feature"].grad_fn)
        self.assertIsNone(cached["observation"].grad_fn)
        with torch.no_grad():
            source.zero_()
        self.assertTrue(torch.equal(cached["feature"], torch.full_like(parameter, 3.0)))

    def test_statistics_reject_stale_sources_and_nonfinite_normalizers(self):
        cfg = config(pos_weight_cap=10)
        keys = ("window_ms", "n_windows", "time_bins", "height", "width_px", "pad_height", "pad_width")
        with tempfile.TemporaryDirectory() as folder:
            cfg["root"] = folder
            record = Path(folder) / "train_000.npz"
            record.write_bytes(b"source-record")
            dataset = SimpleNamespace(paths=[record])
            stats = {"q99": [1.] * 12, "train_pos": 3, "train_neg": 12,
                     "pos_weight_cap": 10, "pos_weight": 4.,
                     "config_used": {key: cfg[key] for key in keys},
                     "provenance": {"statistics_split": "train", "root": str(Path(folder).resolve()),
                                    "records": T.file_inventory([record])}}
            with mock.patch.object(T, "records", return_value=dataset):
                T.validate_statistics(stats, cfg)
                for replacement in (float("nan"), float("inf"), 0., -1.):
                    invalid = copy.deepcopy(stats)
                    invalid["q99"][0] = replacement
                    with self.subTest(q99=replacement), self.assertRaisesRegex(ValueError, "q99"):
                        T.validate_statistics(invalid, cfg)
                invalid = copy.deepcopy(stats)
                invalid.pop("provenance")
                with self.assertRaisesRegex(ValueError, "provenance"):
                    T.validate_statistics(invalid, cfg)
                record.write_bytes(b"changed-source-record")
                with self.assertRaisesRegex(ValueError, "provenance"):
                    T.validate_statistics(stats, cfg)

    def test_resume_rejects_changed_validation_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "output"
            train_file, val_file = Path(folder) / "train.npz", Path(folder) / "val.npz"
            train_file.write_bytes(b"train")
            val_file.write_bytes(b"validation")
            cfg = {"seed": 37, "deterministic": True}
            saved = dict(cfg, _training_scope={"limit_records": None, "max_chunks": None},
                         _source_sha256={name: T.sha256(T.ROOT / name) for name in T.SOURCE_FILES},
                         _training_inventory=T.file_inventory([train_file]),
                         _validation_inventory=T.file_inventory([val_file]))
            val_file.write_bytes(b"changed-validation")
            args = SimpleNamespace(out=str(out), device="cpu", resume=True,
                                   limit_records=None, max_chunks=None)
            datasets = {"train": SimpleNamespace(paths=[train_file]),
                        "val": SimpleNamespace(paths=[val_file])}
            with mock.patch.object(T, "records", side_effect=lambda c, split, limit: datasets[split]), \
                 mock.patch.object(T, "load_checkpoint", return_value={"design": T.DESIGN, "config": saved}), \
                 mock.patch.object(T, "make_models") as build:
                with self.assertRaisesRegex(ValueError, "Resume config/source/data differs"):
                    T.train(cfg, args)
                build.assert_not_called()

    def test_evaluation_rejects_nan_and_restores_train_modes_on_failure(self):
        net, head = make_net().float().train(), revision(nonzero=True).eval()
        seq = sequence()
        original = net.read_features

        def corrupt(*args, **kwargs):
            return original(*args, **kwargs) * float("nan")

        with mock.patch.object(net, "read_features", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "Nonfinite .* predictions"):
                T.evaluate(net, head, [seq], config(), self.q99, self.device)
        self.assertTrue(net.training)
        self.assertFalse(head.training)
        with self.assertRaisesRegex(ValueError, "no recordings"):
            T.evaluate(net, head, [], config(), self.q99, self.device)
        self.assertTrue(net.training)
        self.assertFalse(head.training)

    def test_config_rejects_sensor_outside_canvas(self):
        cfg = T.load_flat_config(str(T.ROOT / "configs/v15/observation.yaml"))
        T.validate_config(cfg)
        for dimensions in ((0, 1), (cfg["pad_height"] + 1, 1), (1, cfg["pad_width"] + 1)):
            invalid = dict(cfg, height=dimensions[0], width_px=dimensions[1])
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, "Sensor dimensions"):
                T.validate_config(invalid)

    def test_online_inference_reads_every_event_once_and_preserves_original_order(self):
        net = make_net().float().eval()
        head = revision().eval()
        seq, cfg = sequence(), config()
        with mock.patch.object(net, "read_features", wraps=net.read_features) as read, \
                mock.patch.object(head, "revise", wraps=head.revise) as revise:
            prob, timing, detail = T.infer_record(net, head, seq, cfg, self.q99, self.device,
                                                  modes=("final",), step=True)
        self.assertEqual(detail["event_readouts"], seq.n_events)
        self.assertEqual(detail["counts"]["final_events"], seq.n_events)
        self.assertEqual(detail["counts"]["valid_empty_future_windows"], 1)
        self.assertEqual(revise.call_count, 2)  # Nonempty old windows 0 and 2.
        self.assertEqual(sum(call[0][1]["x"].numel() for call in read.call_args_list), seq.n_events)
        self.assertEqual(len(timing["all_step_ms"]), seq.n_windows)
        self.assertFalse(timing["includes_auxiliary_causal_readout"])
        source = make_window_source(seq, cfg, self.q99, self.device, torch.float32)
        expected = np.empty(seq.n_events, dtype=np.float32)
        states = None
        with torch.no_grad():
            for k in range(seq.n_windows):
                x, ev, _, ids = source.window(k)
                logits, states, _ = net(x, ev, states)
                expected[ids] = torch.sigmoid(logits).numpy()
        self.assertTrue(np.array_equal(prob["final"], expected))
        np.testing.assert_array_equal(detail["release_window"], np.minimum(seq.t // 50 + 1, 3))
        np.testing.assert_array_equal(detail["output_time_ms"], (np.minimum(seq.t // 50 + 1, 3) + 1) * 50)

    def test_causal_inference_has_no_encoder_or_revision_and_no_extra_delay(self):
        net = make_net().float().eval()
        seq, cfg = sequence(), config(revision_source="causal")
        with mock.patch.object(T.ObservationRevision, "encode_observation", side_effect=AssertionError("encoder used")), \
                mock.patch.object(T.ObservationRevision, "revise", side_effect=AssertionError("revision used")):
            _, _, detail = T.infer_record(net, None, seq, cfg, self.q99, self.device, modes=("final",))
        self.assertEqual(detail["event_readouts"], seq.n_events)
        np.testing.assert_array_equal(detail["release_window"], seq.t // 50)

    def test_evaluation_keeps_original_four_metric_definitions_and_record_reset(self):
        net = make_net().float().eval()
        head = revision(nonzero=True).eval()
        # Produce both true/false positives and false negatives instead of
        # accepting an uninformative all-background metric comparison.
        with torch.no_grad():
            for parameter in net.readout.parameters():
                parameter.zero_()
            net.readout[0].weight[0, 2] = 1.0  # Two feature channels, then polarity.
            net.readout[2].weight[0, 0] = 10.0
            net.readout[2].bias.fill_(-5.0)
        cfg = config()
        data = [sequence("first.npz", seed=1), sequence("second.npz", seed=2)]
        expected_count = np.zeros(3, dtype=np.int64)
        roc = FastRoc(cfg["pd_detT"], cfg["correct_thresh"], H, W)
        for seq in data:
            prob, _, _ = T.infer_record(net, head, seq, cfg, self.q99, self.device, modes=("final",))
            expected_count += np.asarray(window_confusion(prob["final"], seq.label, np.float32(cfg["threshold"]))[:3])
            roc.update(seq.t, prob["final"], seq.target_id, seq.label, seq.x, seq.y, cfg["threshold"])
        result = T.evaluate(net, head, data, cfg, self.q99, self.device, full=True)
        expected = T.metric_row(expected_count, roc)
        np.testing.assert_array_equal(expected_count, [4, 8, 4])
        self.assertAlmostEqual(expected["iou"], 0.25)
        self.assertAlmostEqual(expected["acc"], 0.5)
        self.assertGreater(expected["pd"], 0.0)
        self.assertGreater(expected["fa"], 0.0)
        for metric in ("iou", "acc", "pd", "fa"):
            self.assertAlmostEqual(result["metrics"][metric], expected[metric], places=12, msg=metric)
        self.assertEqual(result["events"], sum(s.n_events for s in data))
        self.assertEqual(result["counts"]["tail_windows"], len(data))
        self.assertEqual(result["timing"]["execution"], "step")
        self.assertIn("positive-event recall", result["metric_definitions"]["acc"])
        self.assertFalse(net.training)
        self.assertFalse(head.training)


if __name__ == "__main__":
    unittest.main()
