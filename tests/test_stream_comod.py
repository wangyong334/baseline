"""model/comod.py（协同调制推理期插件）与 tools/diagnose_tif.py --plugin 的单元测试（CPU，不需要数据集）。

    1. 背景期望输入与蒙特卡洛一致；由输入反推计数在未截断处精确
    2. 背景估计：均匀泊松下收敛；环形挖空使中心的目标不进入参照；阶跃时突变遗忘生效
    3. 插件：参照为 0 时三种模式与原网络逐位相同；detach 后恢复；参照非 0 时与手算公式一致
    4. 入口：同一个 checkpoint 以两个名称加载，一个挂插件（online / oracle），JSON 合法，未挂插件的 run 不变
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import comod as C  # noqa: E402
from test_stream_layerwise import H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from tools import diagnose_tif as D  # noqa: E402
from train_stream_v1 import build_net  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402


class InputTests(unittest.TestCase):
    def test_expected_input_matches_monte_carlo(self):
        rng = np.random.RandomState(0)
        q99 = np.linspace(0.6, 1.4, 12)
        for rate in (0.01, 0.3, 2.0):
            b = torch.full((1, 2, 1, 1), rate, dtype=torch.float64)
            ex = C.expected_input(b, q99, 5, 3.0)[0, :, 0, 0].numpy()
            for c in (0, 1, 2, 7):
                r = rate if c < 2 else rate / 5.0
                samples = np.clip(np.log1p(rng.poisson(r, 200000)) / q99[c], 0, 3.0)
                self.assertAlmostEqual(ex[c], samples.mean(), delta=4 * samples.std() / np.sqrt(2e5) + 1e-9)

    def test_counts_from_input_inverts_transform(self):
        counts = torch.tensor([0., 1., 2., 5., 9.], dtype=torch.float64).view(1, 1, 1, 5).repeat(1, 2, 1, 1)
        q = np.array([0.8, 1.2] + [1.0] * 10)
        x = torch.zeros(1, 12, 1, 5, dtype=torch.float64)
        x[:, :2] = torch.clamp(torch.log1p(counts) / torch.tensor(q[:2]).view(1, 2, 1, 1), 0, 3.0)
        self.assertTrue(torch.equal(C.counts_from_input(x, q), counts))


class EstimatorTests(unittest.TestCase):
    def run_stream(self, rate_fn, steps, target=None, forget=True):
        torch.manual_seed(0)
        est = C.BackgroundEstimator(tau_s=10.0, forget=forget, eps_b=1e-3)
        out = []
        for t in range(steps):
            lam = rate_fn(t).clone()
            if target is not None:
                lam[..., 30:35, 30:35] += target
            out.append(est.estimate())
            est.update(torch.poisson(lam))
        return out, est

    def test_uniform_converges_and_center_target_excluded(self):
        field = torch.full((1, 2, 64, 64), 0.1, dtype=torch.float64)
        out, est = self.run_stream(lambda t: field, 60, target=2.0)
        b = out[-1]
        far = b[..., 0:16, 0:16].mean().item()
        self.assertAlmostEqual(far, 0.1, delta=0.01)
        ring = est.As / est.Ws
        self.assertLess(ring[..., 32, 32].mean().item(), 0.1 * 1.3)  # 环形参照挖空中心：静止源不进入自己的参照
        # 持续 3 s、每窗 2 个事件的静止源由中心分量接管（热像素处理；也是悬停目标的可辨识性界）
        self.assertGreater(b[..., 32, 32].mean().item(), 1.0)
        self.assertLess(out[3][..., 32, 32].mean().item(), 0.5)       # 出现后最初几窗还没有接管

    def test_step_change_forgetting(self):
        lo = torch.full((1, 2, 44, 44), 0.005, dtype=torch.float64)
        hi = torch.full((1, 2, 44, 44), 0.2, dtype=torch.float64)
        rate = lambda t: hi if t >= 20 else lo  # noqa: E731
        with_f, est = self.run_stream(rate, 30, forget=True)
        without, _ = self.run_stream(rate, 30, forget=False)
        self.assertGreater(est.resets, 0)
        self.assertGreater(with_f[26].mean().item(), without[26].mean().item())
        self.assertAlmostEqual(with_f[29].mean().item(), 0.2, delta=0.05)


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.q99 = Q99.astype(np.float64)
        torch.manual_seed(1)
        self.x = torch.rand(1, 12, H, W, dtype=torch.float64) * (torch.rand(1, 12, H, W) < 0.2).double()

    def forward_enc1(self, net, steps=4):
        state, outs = None, []
        for _ in range(steps):
            out, state, memory = net.enc1(self.x, state)[:3]
            outs.append(memory)
        return torch.stack(outs)

    def test_zero_reference_is_identity_and_detach_restores(self):
        base = make_tif_net(quantize=True).eval()
        ref = self.forward_enc1(base)
        for mode in C.MODES:
            net = make_tif_net(quantize=True).eval()
            plugin = C.CoModPlugin(net, mode, "online", self.q99, 5, 3.0, dwell_ms=175.0)
            plugin.reference = torch.zeros(1, SMALL[0], H, W, dtype=torch.float64)
            self.assertTrue(torch.equal(self.forward_enc1(net), ref), mode)
            plugin.detach()
            plugin.reference = torch.ones(1, SMALL[0], H, W, dtype=torch.float64)
            self.assertTrue(torch.equal(self.forward_enc1(net), ref), mode)

    def test_nonzero_reference_matches_formula(self):
        net = make_tif_net(quantize=False).eval()
        neuron = net.enc1.neuron
        current = net.enc1._currents(self.x)[0].detach()
        Dref = torch.full_like(current, 0.03)
        lam = neuron.beta().view(1, -1, 1, 1).double()
        k = neuron.k().view(1, -1, 1, 1).double()
        cap = (neuron.theta() * neuron.levels).view(1, -1, 1, 1).double()
        tau = -50.0 / torch.log(lam)
        for mode, dwell in (("sub", 175.0), ("nocap", 175.0), ("comod", 60.0)):
            plugin = C.CoModPlugin(net, mode, "online", self.q99, 5, 3.0, dwell_ms=dwell)
            plugin.reference = Dref
            g = k / (k + Dref)
            if mode == "comod":
                g = torch.maximum(g, (tau / dwell).clamp(max=1.0))
            lam_eff = lam.expand_as(Dref) if mode == "sub" else lam ** g
            z = current - k - Dref
            m1 = torch.minimum(torch.relu((1 - lam_eff) * z), cap)
            m2 = torch.minimum(torch.relu(lam_eff * m1 + (1 - lam_eff) * z), cap)
            out = self.forward_enc1(net, steps=2)
            self.assertTrue(torch.allclose(out[0], m1, atol=1e-12), mode)
            self.assertTrue(torch.allclose(out[1], m2, atol=1e-12), mode)
            plugin.detach()


class DiagnoseEntryTests(unittest.TestCase):
    def test_main_with_plugins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "val").mkdir()
            for seed in (6, 7):
                seq = synthetic_sequence(seed=seed)
                loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                np.savez(root / "val" / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
            base = load_flat_config("configs/evisseg_stream_tif_notransport.yaml")
            base.update(root=str(root), channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                        n_windows=WINDOWS, threshold=0.5)
            cfg = dict(base, tif=dict(base["tif"], k_init=1e-3))
            torch.manual_seed(3)
            net = build_net(cfg, torch.device("cpu"))
            for block in net.blocks():
                block.gain.set_gain(torch.full((block.out_ch,), 4.0))
            net.gain_calibrated.fill_(True)
            path = root / "tif.pt"
            torch.save({"config": cfg, "model": net.state_dict(), "stats": {"q99": Q99.tolist()}, "epoch": 1}, path)
            yaml_path = root / "eval.yaml"
            yaml_path.write_text(json.dumps({"DATA": {"root": str(root)}, "EVAL": {
                key: base[key] for key in D.EVAL_KEYS}}), encoding="utf-8")
            outs = {}
            for tag, extra in (("plain", []), ("plug", ["--run", "c=%s" % path, "--run", "o=%s" % path,
                                                        "--plugin", "c=comod:online", "--plugin", "o=sub:oracle"])):
                out = root / ("diag_%s.json" % tag)
                argv = ["--config", str(yaml_path), "--split", "val", "--device", "cpu", "--out", str(out),
                        "--run", "v5=%s" % path] + extra
                with contextlib.redirect_stdout(io.StringIO()):
                    D.main(argv)
                outs[tag] = json.loads(out.read_text(encoding="utf-8"))
            plain, plug = outs["plain"]["runs"]["v5"], outs["plug"]["runs"]
            self.assertEqual(plain["totals"], plug["v5"]["totals"])                 # 未挂插件的 run 不受影响
            self.assertIsNone(plug["v5"]["plugin"])
            self.assertEqual(plug["c"]["plugin"]["mode"], "comod")
            self.assertEqual(plug["o"]["plugin"]["background"], "oracle")
            self.assertIn("comod", plug["c"]["sequences"][0])
            with self.assertRaises(ValueError):
                D.parse_plugins(["x=comod:online"], ["v5"])


if __name__ == "__main__":
    unittest.main()
