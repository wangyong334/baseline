"""enc1 当前观测亚阈值残差、纵向穿透与事件级门控跳连的回归测试。"""
import copy
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.stream_source import TorchWindowSource
from model.evspsegnet_stream import EvSpSegNetStream, LAYER_NAMES, count_parameters, estimate_operations
from model.lif2d_stream import state_tensors
from test_stream_layerwise import CFG, H, W, Q99, SMALL, synthetic_sequence
import train_stream_v1 as T


def events(n=1, with_time=False):
    result = {"b": torch.zeros(n, dtype=torch.long), "y": torch.full((n,), 3, dtype=torch.long),
              "x": torch.full((n,), 4, dtype=torch.long), "p": torch.ones(n),
              "t_local": torch.full((n,), 0.25)}
    if with_time:
        result["t"] = torch.zeros(n, dtype=torch.long)
    return result


def make_net(vertical=True, event_skip=True, residual_extra=None):
    torch.manual_seed(37)
    residual = {"vertical": vertical, "event_skip": event_skip, "local_kernel": 3}
    if residual_extra:
        residual.update(residual_extra)
    return EvSpSegNetStream(
        channels=SMALL, v_threshold=0.05, dt_ms=50.0, tau_init_ms=200.0,
        tau_min_ms=50.0, tau_max_ms=2000.0,
        dual_memory={"layers": ["enc2", "enc3", "enc4"]},
        subthreshold_residual=residual).double()


class ObservationResidualTests(unittest.TestCase):
    def test_zero_current_never_replays_old_membrane(self):
        model = make_net(vertical=True, event_skip=False)
        with torch.no_grad():
            model.enc1.conv.weight.fill_(0.0001)
        first = torch.ones(1, 12, H, W, dtype=torch.float64)
        zero = torch.zeros_like(first)
        with torch.no_grad():
            _, state, info1 = model(first, events(), None, collect=True)
            _, _, info2 = model(zero, events(), state, collect=True)
        self.assertGreater(float(info1["subthreshold_residual"].abs().sum()), 0.0)
        self.assertGreater(float(info2["u_pre"][0].abs().sum()), 0.0)
        self.assertEqual(float(info2["subthreshold_residual"].abs().sum()), 0.0)

    def test_vertical_path_changes_enc2_without_changing_enc1_state(self):
        vertical = make_net(vertical=True, event_skip=False)
        baseline = make_net(vertical=False, event_skip=False)
        with torch.no_grad():
            for model in (vertical, baseline):
                model.enc1.conv.weight.fill_(0.0001)
                model.enc2.conv.weight.fill_(0.1)
            x = torch.ones(1, 12, H, W, dtype=torch.float64)
            _, sv, iv = vertical(x, events(), None, collect=True)
            _, sb, ib = baseline(x, events(), None, collect=True)
        self.assertTrue(torch.equal(sv[0], sb[0]))
        self.assertEqual(float(ib["current"][1].abs().sum()), 0.0)
        self.assertGreater(float(iv["current"][1].abs().sum()), 0.0)

    def test_event_skip_requires_deep_support_and_does_not_write_state(self):
        model = make_net(vertical=False, event_skip=True)
        with torch.no_grad():
            model.residual_project.weight.copy_(torch.eye(SMALL[0], dtype=torch.float64))
        u = torch.randn(1, SMALL[0], H, W, dtype=torch.float64)
        residual = torch.randn_like(u)
        deep_zero = torch.zeros_like(u)
        plain, gate0, correction0 = model._event_features(u, residual, deep_zero, events())
        expected = u[events()["b"], :, events()["y"], events()["x"]]
        self.assertTrue(torch.equal(plain, expected))
        self.assertEqual(float(gate0.detach().sum()), 0.0)
        self.assertEqual(float(correction0.detach().abs().sum()), 0.0)
        deep = torch.zeros_like(u)
        deep[:, :, 3, 4] = 1.0
        changed, gate1, correction1 = model._event_features(u, residual, deep, events())
        self.assertAlmostEqual(float(gate1.detach().item()), 0.5, places=12)
        self.assertTrue(torch.equal(changed, expected + correction1))
        self.assertGreater(float(correction1.detach().abs().sum()), 0.0)

    def test_all_off_is_exact_base_network(self):
        hybrid = make_net(vertical=True, event_skip=True)
        base = make_net(vertical=False, event_skip=False)
        hybrid.set_subthreshold_ablation("all_off")
        x = torch.randn(1, 12, H, W, dtype=torch.float64)
        with torch.no_grad():
            a, sa, _ = hybrid(x, events(4), None)
            b, sb, _ = base(x, events(4), None)
        self.assertTrue(torch.equal(a, b))
        for left, right in zip(state_tensors(sa), state_tensors(sb)):
            self.assertTrue(torch.equal(left, right))


class ResidualExecutionTests(unittest.TestCase):
    def test_competitive_vertical_starts_as_exact_baseline(self):
        competitive = make_net(vertical=True, event_skip=False,
                               residual_extra={"vertical_mode": "competitive"})
        baseline = make_net(vertical=False, event_skip=False)
        x = torch.randn(1, 12, H, W, dtype=torch.float64)
        with torch.no_grad():
            a, sa, info = competitive(x, events(4), None, collect=True)
            b, sb, _ = baseline(x, events(4), None)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(float(info["subthreshold_vertical_signal"].abs().sum()), 0.0)
        for left, right in zip(state_tensors(sa), state_tensors(sb)):
            self.assertTrue(torch.equal(left, right))

    def test_competitive_vertical_selects_local_peaks_and_is_bounded(self):
        model = make_net(vertical=True, event_skip=False,
                         residual_extra={"vertical_mode": "competitive",
                                         "vertical_peak_kernel": 3})
        with torch.no_grad():
            model.residual_vertical_scale_raw.fill_(1.0)
        residual = torch.zeros(1, SMALL[0], H, W, dtype=torch.float64)
        residual[0, 0, 3, 3] = 0.8
        residual[0, 0, 3, 4] = 0.7  # 超阈但不是局部峰值
        residual[0, 1, 6, 6] = -0.9
        signal = model._vertical_residual(residual)
        scale = 0.5 * np.tanh(1.0)
        self.assertAlmostEqual(float(signal[0, 0, 3, 3].detach()), scale, places=12)
        self.assertEqual(float(signal[0, 0, 3, 4].detach()), 0.0)
        self.assertAlmostEqual(float(signal[0, 1, 6, 6].detach()), -scale, places=12)
        self.assertEqual(int(signal.ne(0).sum()), 2)

    def test_competitive_vertical_parameters_receive_gradients(self):
        model = make_net(vertical=True, event_skip=False,
                         residual_extra={"vertical_mode": "competitive"})
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        out = T.train_sequence(model, synthetic_sequence(seed=17), optimizer,
                               dict(CFG, execution="layer", input_device="gpu"),
                               Q99, 3.0, torch.device("cpu"), np.random.RandomState(6))
        self.assertEqual(out["missing_grads"], [])
        self.assertGreater(out["grad_norms"]["residual_vertical_scale_raw"], 0.0)

    def test_selective_dualpath_starts_as_exact_baseline(self):
        selective = make_net(vertical=True, event_skip=True,
                             residual_extra={"vertical_mode": "competitive",
                                             "event_skip_mode": "selective"})
        baseline = make_net(vertical=False, event_skip=False)
        x = torch.randn(1, 12, H, W, dtype=torch.float64)
        with torch.no_grad():
            a, sa, info = selective(x, events(4), None, collect=True)
            b, sb, _ = baseline(x, events(4), None)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(float(info["subthreshold_vertical_signal"].abs().sum()), 0.0)
        self.assertEqual(float(info["event_residual_correction"].abs().sum()), 0.0)
        for left, right in zip(state_tensors(sa), state_tensors(sb)):
            self.assertTrue(torch.equal(left, right))

    def test_selective_horizontal_is_sparse_and_bounded(self):
        model = make_net(vertical=True, event_skip=True,
                         residual_extra={"vertical_mode": "competitive",
                                         "event_skip_mode": "selective",
                                         "event_correction_max": 0.25})
        with torch.no_grad():
            model.residual_project.weight.copy_(10.0 * torch.eye(SMALL[0], dtype=torch.float64))
        u = torch.zeros(1, SMALL[0], H, W, dtype=torch.float64)
        residual = torch.zeros_like(u)
        residual[0, 0, 3, 4] = 0.9
        residual[0, 0, 3, 5] = 0.8  # 超阈但不是局部峰值
        deep = torch.zeros_like(u)
        deep[0, :, 3, 4] = 1.0
        deep[0, :, 3, 5] = 0.9
        ev = events(2)
        ev["x"] = torch.tensor([4, 5], dtype=torch.long)
        _, gate, correction = model._event_features(u, residual, deep, ev)
        self.assertGreater(float(gate[0].detach()), 0.0)
        self.assertEqual(float(gate[1].detach()), 0.0)
        self.assertGreater(float(correction[0].detach().abs().sum()), 0.0)
        self.assertEqual(float(correction[1].detach().abs().sum()), 0.0)
        self.assertLessEqual(float(correction.detach().abs().max()), 0.25)

    def test_selective_dualpath_parameters_are_connected(self):
        model = make_net(vertical=True, event_skip=True,
                         residual_extra={"vertical_mode": "competitive",
                                         "event_skip_mode": "selective"})
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        out = T.train_sequence(model, synthetic_sequence(seed=29), optimizer,
                               dict(CFG, execution="layer", input_device="gpu"),
                               Q99, 3.0, torch.device("cpu"), np.random.RandomState(8))
        self.assertEqual(out["missing_grads"], [])
        self.assertIn("residual_event_threshold_raw", out["grad_norms"])
        self.assertGreater(out["grad_norms"]["residual_project"], 0.0)

    def test_training_updates_new_projection_and_all_parameters_are_connected(self):
        model = make_net(vertical=True, event_skip=True)
        before = model.residual_project.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        out = T.train_sequence(model, synthetic_sequence(seed=9), optimizer,
                               dict(CFG, execution="layer", input_device="gpu"),
                               Q99, 3.0, torch.device("cpu"), np.random.RandomState(4))
        self.assertEqual(out["missing_grads"], [])
        self.assertGreater(out["grad_norms"]["residual_project"], 0.0)
        self.assertFalse(torch.equal(before, model.residual_project.weight))

    def test_step_and_layer_match_forward_states_logits_and_gradients(self):
        source = TorchWindowSource(synthetic_sequence(seed=13), CFG, Q99, torch.device("cpu"), torch.float64)
        step = make_net(vertical=True, event_skip=True,
                        residual_extra={"vertical_mode": "competitive",
                                        "event_skip_mode": "selective"})
        with torch.no_grad():
            step.residual_vertical_scale_raw.fill_(0.4)
            step.residual_project.weight.copy_(0.1 * torch.eye(SMALL[0], dtype=torch.float64))
        layer = copy.deepcopy(step)
        xs, event_parts = [], []
        step_logits, state_step = [], None
        for t in range(4):
            x, ev, _, _ = source.window(t)
            xs.append(x)
            out, state_step, _ = step(x, ev, state_step)
            step_logits.append(out)
            part = dict(ev)
            part["t"] = torch.full_like(ev["b"], t)
            event_parts.append(part)
        chunk_events = {key: torch.cat([part[key] for part in event_parts]) for key in event_parts[0]}
        chunk_logits, state_layer, _ = layer.forward_chunk(torch.stack(xs), chunk_events, None)
        step_logits = torch.cat(step_logits)
        self.assertLess(float((step_logits - chunk_logits).detach().abs().max()), 1e-10)
        for left, right in zip(state_tensors(state_step), state_tensors(state_layer)):
            self.assertLess(float((left - right).detach().abs().max()), 1e-10)
        weights = torch.linspace(0.1, 1.0, int(step_logits.numel()), dtype=torch.float64)
        (step_logits * weights).sum().backward()
        (chunk_logits * weights).sum().backward()
        for (name_a, pa), (name_b, pb) in zip(step.named_parameters(), layer.named_parameters()):
            self.assertEqual(name_a, name_b)
            if pa.grad is None:
                self.assertIsNone(pb.grad, name_a)
            else:
                scale = max(float(pa.grad.abs().max()), 1.0)
                self.assertLess(float((pa.grad - pb.grad).abs().max()) / scale, 1e-9, name_a)

    def test_parameter_and_operation_accounting(self):
        base = make_net(vertical=False, event_skip=False)
        vertical = make_net(vertical=True, event_skip=False)
        hybrid = make_net(vertical=True, event_skip=True)
        competitive = make_net(vertical=True, event_skip=False,
                               residual_extra={"vertical_mode": "competitive"})
        selective = make_net(vertical=True, event_skip=True,
                             residual_extra={"vertical_mode": "competitive",
                                             "event_skip_mode": "selective"})
        self.assertEqual(count_parameters(vertical)["total"], count_parameters(base)["total"])
        expected_parameters = SMALL[0] + 1 + SMALL[0] ** 2
        self.assertEqual(count_parameters(hybrid)["total"] - count_parameters(base)["total"],
                         expected_parameters)
        self.assertEqual(count_parameters(competitive)["total"] - count_parameters(base)["total"],
                         2 * SMALL[0])
        self.assertEqual(count_parameters(selective)["total"] - count_parameters(base)["total"],
                         2 * SMALL[0] + expected_parameters + 1)
        rates = dict.fromkeys(LAYER_NAMES, 0.1)
        rates["subthreshold_vertical"] = 0.02
        base_ops = estimate_operations(base, H, W, rates, 5)
        vertical_ops = estimate_operations(vertical, H, W, rates, 5)
        hybrid_ops = estimate_operations(hybrid, H, W, rates, 5)
        competitive_ops = estimate_operations(competitive, H, W, rates, 5)
        selective_ops = estimate_operations(selective, H, W, rates, 5)
        base_enc2 = next(row for row in base_ops["per_layer"] if row["layer"] == "enc2")
        vertical_enc2 = next(row for row in vertical_ops["per_layer"] if row["layer"] == "enc2")
        self.assertEqual(base_enc2["mac"], 0.0)
        self.assertEqual(vertical_enc2["mac"], vertical_enc2["dense"])
        competitive_enc2 = next(row for row in competitive_ops["per_layer"]
                                if row["layer"] == "enc2")
        self.assertEqual(competitive_enc2["mac"], 0.0)
        self.assertAlmostEqual(competitive_enc2["sop"], 0.12 * competitive_enc2["dense"])
        self.assertEqual(competitive_ops["subthreshold_residual"]["parameters"], 2 * SMALL[0])
        self.assertEqual(hybrid_ops["subthreshold_residual"]["parameters"], expected_parameters)
        self.assertEqual(selective_ops["subthreshold_residual"]["parameters"],
                         2 * SMALL[0] + expected_parameters + 1)
        self.assertEqual(selective_ops["subthreshold_residual"]["event_skip_mode"], "selective")
        self.assertTrue(any(row["layer"] == "event_residual_skip" for row in hybrid_ops["per_layer"]))
        hybrid.set_subthreshold_ablation("all_off")
        ablated_ops = estimate_operations(hybrid, H, W, rates, 5)
        ablated_enc2 = next(row for row in ablated_ops["per_layer"] if row["layer"] == "enc2")
        self.assertEqual(ablated_enc2["mac"], 0.0)
        self.assertFalse(ablated_ops["subthreshold_residual"]["vertical_enabled"])
        self.assertFalse(ablated_ops["subthreshold_residual"]["event_skip_enabled"])
        self.assertFalse(any(row["layer"] == "event_residual_skip"
                             for row in ablated_ops["per_layer"]))


if __name__ == "__main__":
    unittest.main()
