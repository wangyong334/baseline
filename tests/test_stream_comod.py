"""model/comod.py（噪声校准推理期插件）与 tools/diagnose_tif.py --plugin 的单元测试（CPU，不需要数据集）。

    1. 输入矩：12 通道的均值、方差与"整窗 × 时间箱"协方差与蒙特卡洛一致；enc1 电流的背景均值、方差与蒙特卡洛一致；
       由输入反推计数在未截断处精确
    2. 背景估计：均匀泊松下收敛；环形挖空与孤立点源门控使静止目标不进入参照、热像素被中心分量接管；突变遗忘生效
    3. 插件：背景不高于参照时与原网络逐位相同（ntt 在浮点误差内）；detach 后恢复；校正时与手算公式一致
    4. 入口：同一个 checkpoint 以多个名称加载，部分挂插件（online / oracle），JSON 合法，未挂插件的 run 不变
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import comod as C  # noqa: E402
from test_stream_layerwise import H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from tools import diagnose_tif as D  # noqa: E402
from train_stream_v1 import build_net  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402


def encode(on_bins, off_bins, q99, clip=3.0):
    """按 stream_windows 的通道布局，把每箱计数 [..., T] 编成 12 通道 clip(log1p(n)/q99)。"""
    T = on_bins.shape[-1]
    chans = [on_bins.sum(-1), off_bins.sum(-1)]
    for t in range(T):
        chans += [on_bins[..., t], off_bins[..., t]]
    x = np.stack(chans, -1).astype(np.float64)
    return np.clip(np.log1p(x) / q99, 0, clip)


class MomentTests(unittest.TestCase):
    def test_input_moments_match_monte_carlo(self):
        rng = np.random.RandomState(0)
        q99 = np.linspace(0.6, 1.4, 12)
        n = 400000
        for rate_on, rate_off in ((0.02, 0.3), (1.5, 0.004)):
            b = torch.tensor([rate_on, rate_off], dtype=torch.float64).view(1, 2, 1, 1)
            m, v, c = (t[0, :, 0, 0].numpy() for t in C.input_moments(b, q99, 5, 3.0))
            x = encode(rng.poisson(rate_on / 5, (n, 5)), rng.poisson(rate_off / 5, (n, 5)), q99)
            se = x.std(0) / np.sqrt(n)
            np.testing.assert_allclose(m, x.mean(0), atol=float(5 * se.max() + 1e-9))
            # 方差的蒙特卡洛相对误差约 1/sqrt(非零样本数)：按各通道的期望事件数给容差
            rates = np.array([rate_on, rate_off] + [rate_on / 5, rate_off / 5] * 5)
            rtol = 4.0 / np.sqrt(n * np.minimum(rates, 1.0)) + 0.01
            self.assertTrue(np.all(np.abs(v - x.var(0)) <= rtol * x.var(0) + 1e-9), (v, x.var(0)))
            for t in range(5):
                for p in (0, 1):
                    emp = np.cov(x[:, p], x[:, 2 + 2 * t + p])[0, 1]
                    tol = rtol[2 + 2 * t + p] * abs(emp) + 1e-6
                    self.assertAlmostEqual(c[2 * t + p], emp, delta=tol)

    def test_current_moments_match_monte_carlo(self):
        rng = np.random.RandomState(1)
        net = make_tif_net(quantize=True).eval()
        plugin = C.CoModPlugin(net, "comod", "online", Q99.astype(np.float64), 5, 3.0, dwell_ms=175.0)
        rate = np.array([0.05, 0.02])
        b = torch.tensor(rate, dtype=torch.float64).view(1, 2, 1, 1).expand(1, 2, 9, 9).contiguous()
        mu, var = plugin.moments(b)
        samples = []
        for _ in range(20000):
            x = encode(rng.poisson(rate[0] / 5, (9, 9, 5)), rng.poisson(rate[1] / 5, (9, 9, 5)), Q99.astype(np.float64))
            xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                samples.append(net.enc1._currents(xt)[0][0, :, 4, 4].numpy())
        s = np.array(samples)
        np.testing.assert_allclose(mu[0, :, 4, 4].numpy(), s.mean(0), atol=float(5 * s.std(0).max() / np.sqrt(len(s))))
        np.testing.assert_allclose(var[0, :, 4, 4].numpy(), s.var(0), rtol=0.06)

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
        self.assertAlmostEqual(b[..., 0:16, 0:16].mean().item(), 0.1, delta=0.01)
        ring = est.As / est.Ws
        self.assertLess(ring[..., 32, 32].mean().item(), 0.1 * 1.3)  # 环形参照挖空中心：静止源不进入自己的参照
        self.assertLess(b[..., 32, 32].mean().item(), 0.1 * 1.3)     # 5x5 的静止目标不是孤立点源，中心分量不接管

    def test_isolated_point_source_gate(self):
        """热像素（孤立单像素）由中心分量接管；同样强度、静止的 5x5 目标不会被当成热像素。"""
        hot = torch.full((1, 2, 64, 64), 0.01, dtype=torch.float64)
        hot[..., 10, 10] = 1.0
        torch.manual_seed(2)
        est = C.BackgroundEstimator(tau_s=10.0, forget=False)
        for _ in range(80):
            lam = hot.clone()
            lam[..., 40:45, 40:45] += 1.0
            est.update(torch.poisson(lam))
        b = est.estimate()
        self.assertGreater(b[..., 10, 10].mean().item(), 0.5)
        self.assertLess(b[..., 42, 42].mean().item(), 0.05)

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

    def set_rate(self, plugin, rate):
        """让插件按给定的均匀背景率校正（绕过在线估计）。"""
        plugin.background = "oracle"
        plugin.oracle_b = torch.tensor(rate, dtype=torch.float64).view(1, 2, 1, 1).expand(1, 2, H, W).contiguous()
        plugin.set_window(self.x)

    def test_reference_level_is_identity_and_detach_restores(self):
        base = make_tif_net(quantize=True).eval()
        ref = self.forward_enc1(base)
        for mode in C.MODES:
            net = make_tif_net(quantize=True).eval()
            plugin = C.CoModPlugin(net, mode, "online", self.q99, 5, 3.0, dwell_ms=175.0)
            self.set_rate(plugin, (0.001, 0.0))                       # 低于参照：不校正
            out = self.forward_enc1(net)
            if mode == "ntt":
                self.assertTrue(torch.allclose(out, ref, atol=1e-12), mode)
            else:
                self.assertTrue(torch.equal(out, ref), mode)
            plugin.detach()
            self.set_rate(plugin, (0.3, 0.3))
            self.assertTrue(torch.equal(self.forward_enc1(net), ref), mode)

    def test_correction_matches_formula(self):
        net = make_tif_net(quantize=False).eval()
        neuron = net.enc1.neuron
        current = net.enc1._currents(self.x)[0].detach()
        lam = neuron.beta().view(1, -1, 1, 1).double()
        k = neuron.k().view(1, -1, 1, 1).double()
        cap = (neuron.theta() * neuron.levels).view(1, -1, 1, 1).double()
        tau = -50.0 / torch.log(lam)
        for mode, dwell in (("sub", 175.0), ("nocap", 175.0), ("comod", 60.0), ("ntt", 60.0), ("ntt", 1e6)):
            plugin = C.CoModPlugin(net, mode, "online", self.q99, 5, 3.0, dwell_ms=dwell)
            self.set_rate(plugin, (0.2, 0.1))
            dmu, rho, mu_ref, mu_eff = plugin.state
            self.assertTrue(bool((rho >= 1).all()))
            if mode == "sub":
                s = torch.ones_like(rho)
            elif mode == "nocap":
                s = rho
            else:
                s = torch.minimum(rho, (dwell / tau).clamp(min=1.0))
            lam_eff = lam ** (1.0 / s)
            I = mu_ref + torch.sqrt(s / rho) * (current - mu_eff) if mode == "ntt" else current - dmu
            z = I - k
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
                                                        "--plugin", "c=ntt:online", "--plugin", "o=sub:oracle",
                                                        "--ref-rate", "0.002,0.0001"])):
                out = root / ("diag_%s.json" % tag)
                argv = ["--config", str(yaml_path), "--split", "val", "--device", "cpu", "--out", str(out),
                        "--run", "v5=%s" % path] + extra
                with contextlib.redirect_stdout(io.StringIO()):
                    D.main(argv)
                outs[tag] = json.loads(out.read_text(encoding="utf-8"))
            plain, plug = outs["plain"]["runs"]["v5"], outs["plug"]["runs"]
            self.assertEqual(plain["totals"], plug["v5"]["totals"])                 # 未挂插件的 run 不受影响
            self.assertIsNone(plug["v5"]["plugin"])
            self.assertEqual(plug["c"]["plugin"]["mode"], "ntt")
            self.assertEqual(plug["o"]["plugin"]["background"], "oracle")
            self.assertIn("comod", plug["c"]["sequences"][0])
            with self.assertRaises(ValueError):
                D.parse_plugins(["x=comod:online"], ["v5"])


if __name__ == "__main__":
    unittest.main()
