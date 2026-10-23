"""通信预算训练（utils/comm_budget.py：每窗峰值带宽约束，SOP 按合并解码器口径；Stream-SNN-Innovations.md 12.13~12.21、12.32）。

CPU、float64 小网络，不需要数据集。
    1. CommBudget.from_config：没有预算时为 None（关闭）；只接受 TIF 量化网络 + trainer=streams + comm_accounting: merged；
       平均预算 mean 已删除，写了会报错；新建合并口径网络不改变随机状态
    2. 合并口径：enc1~enc3 的系数与原形式相同，enc4/dec3/dec2 的系数是原形式的 4 倍（ConvT 由 2x2 变为 4x4），dec1 为 0；
       原形式网络的 |Δq| 乘合并口径系数 = 换算成合并形式后评估的 SOP（严格相等）
    3. window_rates：前向值与评估口径一致——片段首窗与上一片段末状态比较、新录像（keep = 0）首窗与 0 比较，
       并与逐窗统计 TIF 消息的 stop_stats 一致；反向按 |Δx| 求导（没越界的变化也有梯度）；空窗不计入
    4. 峰值约束：只罚超出预算的窗；对偶按违约比例；评估的逐窗 |Δq| 与单窗 SOP 分布（可传入合并口径系数）
    5. train_streams：约束不紧（mu 保持 0）时与关闭逐位相同；超出时 mu 增大、参数轨迹改变、无漏梯度
    6. 配置文件：与基础版只差 comm_* 五项与 save_root，预算为 V1 合并后的 val SOP
"""
import copy
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_stream_v1 as T  # noqa: E402
from dataset.stream_source import TorchWindowSource  # noqa: E402
from model.evspsegnet_stream import (LAYER_NAMES, EvSpSegNetStream, estimate_operations,  # noqa: E402
                                     merge_decoder_state_dict)
from test_stream_layerwise import CFG, H, Q99, SMALL, W, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from utils.comm_budget import (CommBudget, merged_sop_coefficients, merged_twin, sop_coefficients,  # noqa: E402
                               window_rates, window_sop_summary)

CPU = torch.device("cpu")
POS_WEIGHT = 4.0


def stream_cfg(**kw):
    base = dict(CFG, execution="layer", tbptt_k=4, streams=2, accumulate=1, trainer="streams")
    base.update(kw)
    return base


def budget_cfg(budget, **kw):
    """开启通信预算的最小配置（合并口径）。"""
    return stream_cfg(comm_budget_sop=budget, comm_accounting="merged", **kw)


def mean_rates(rates, valid=None):
    """逐窗率 [T,B] 在有效窗上的平均（评估口径的每窗平均 |Δq|）。"""
    weight = torch.ones_like(rates[0]) if valid is None else valid.to(rates[0].dtype)
    return [(r * weight).sum() / weight.sum() for r in rates]


class CommBudgetConfigTests(unittest.TestCase):
    def test_off_and_validation(self):
        net = make_tif_net()
        self.assertIsNone(CommBudget.from_config(stream_cfg(), net))
        self.assertIsNone(CommBudget.from_config(stream_cfg(comm_budget_sop=None), net))
        comm = CommBudget.from_config(budget_cfg(1e5, comm_dual_lr=0.02, comm_peak_alpha=0.1), net)
        self.assertEqual((comm.budget, comm.dual_lr, comm.alpha, comm.mu), (1e5, 0.02, 0.1, 0.0))
        for bad in (dict(comm_budget_sop=0.0), dict(comm_dual_lr=-1.0), dict(trainer="sequence"),
                    dict(comm_budget_mode="mean"), dict(comm_accounting=None), dict(comm_accounting="as_built"),
                    dict(comm_peak_alpha=0.0), dict(comm_peak_alpha=1.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                CommBudget.from_config(dict(budget_cfg(1e5), **bad), net)
        with self.assertRaises(ValueError):
            CommBudget.from_config(budget_cfg(1e5), EvSpSegNetStream(channels=SMALL, neuron="lif"))
        with self.assertRaises(ValueError):
            CommBudget.from_config(budget_cfg(1e5), make_tif_net(quantize=False))

    def test_random_state_untouched(self):
        # 求合并口径系数要新建一个网络；它不能消耗随机数，否则开启通信预算会改变后续训练的随机序列
        net = make_tif_net()
        torch.manual_seed(123)
        before = torch.get_rng_state()
        CommBudget.from_config(budget_cfg(1e5), net)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class MergedAccountingTests(unittest.TestCase):
    def test_coefficients_relative_to_original(self):
        net = make_tif_net()
        _, original = sop_coefficients(net, H, W)
        base, merged = merged_sop_coefficients(net, H, W)
        self.assertEqual(base, 0.0)
        for i in range(3):                                                  # enc1~enc3：后接卷积与跳连，形状不变
            self.assertAlmostEqual(merged[i], original[i], delta=1e-9 * original[i])
        for i in range(3, 6):                                               # enc4/dec3/dec2：后接 ConvT，2x2 -> 4x4
            self.assertAlmostEqual(merged[i], 4.0 * original[i], delta=1e-9 * original[i])
        self.assertEqual((merged[6], original[6]), (0.0, 0.0))              # dec1 只接读出（MAC）
        twin = merged_twin(net)
        self.assertTrue(twin.merged_decoder)
        self.assertIs(merged_twin(make_tif_net(merged=True)).merged_decoder, True)
        comm = CommBudget.from_config(budget_cfg(1e5), net)
        self.assertEqual(comm.coef, [float(c) for c in merged])

    def test_matches_evaluation_of_converted_network(self):
        # 原形式网络的 |Δq| 乘合并口径系数 = 换算成合并形式后（函数严格相同）评估得到的 SOP
        net = make_tif_net().eval()
        merged = make_tif_net(merged=True).eval()
        merged.load_state_dict(merge_decoder_state_dict(net.state_dict()))
        seq = synthetic_sequence(0)
        source = TorchWindowSource(seq, CFG, Q99, CPU, torch.float64)
        stats = []
        with torch.no_grad():
            for model in (net, merged):
                model.start_tif_stats()
                states = None
                for k in range(12):
                    x, events, _, _ = source.window(k)
                    _, states, _ = model(x, events, states)
                stats.append(model.stop_tif_stats())
        rates = [{n: s[n]["message_abs_rate"] for n in LAYER_NAMES} for s in stats]
        for n in LAYER_NAMES:
            self.assertAlmostEqual(rates[0][n], rates[1][n], places=12, msg=n)
        self.assertGreater(sum(rates[0].values()), 0.0)
        comm = CommBudget.from_config(budget_cfg(1e5), net)
        expected = estimate_operations(merged, H, W, rates[1], 0.0)["sop"]
        self.assertAlmostEqual(comm.sop(rates[0]), expected, delta=1e-9 * expected)
        self.assertGreater(expected, estimate_operations(net, H, W, rates[0], 0.0)["sop"])


class WindowRateTests(unittest.TestCase):
    def test_exact_counting_and_gradient(self):
        net = make_tif_net()
        thetas = [b.neuron.theta().detach().double() for b in net.blocks()]
        g = torch.Generator().manual_seed(1)
        raw, memories, prev_states = [], [], []
        for block, theta in zip(net.blocks(), thetas):
            c = block.out_ch
            x = torch.rand(5, 2, c, 3, 3, generator=g, dtype=torch.float64) * 4      # 电平坐标
            x0 = torch.rand(2, c, 3, 3, generator=g, dtype=torch.float64) * 4        # 上一片段末窗
            raw.append((x0, x))
            memories.append((x * theta.view(1, 1, -1, 1, 1)).requires_grad_(True))
            prev_states.append(x0 * theta.view(1, -1, 1, 1))
        keep = torch.ones(5, 2, dtype=torch.float64)
        keep[0, 0] = 0.0                                                    # 流 0 在片段首窗换录像
        keep[3, 1] = 0.0                                                    # 流 1 在第 3 窗换录像
        rates = mean_rates(window_rates(net, memories, prev_states, keep))
        for rate, (x0, x) in zip(rates, raw):
            prev = torch.cat([x0.unsqueeze(0), x[:-1]], 0) * keep.view(5, 2, 1, 1, 1)
            expected = float((torch.round(x) - torch.round(prev)).abs().sum()) / x.numel()
            self.assertAlmostEqual(float(rate), expected, places=12)
        sum(rates).backward()
        for memory, (x0, x), theta in zip(memories, raw, thetas):
            # 反向应等于 |Δx| 总变差（同一套首窗 / 换录像规则）的梯度：没越界的变化也有梯度
            leaf = memory.detach().clone().requires_grad_(True)
            xs = leaf / theta.view(1, 1, -1, 1, 1)
            prev = torch.cat([x0.unsqueeze(0), xs[:-1]], 0) * keep.view(5, 2, 1, 1, 1)
            ((xs - prev).abs().sum() / x.numel()).backward()
            self.assertLess(float((memory.grad - leaf.grad).abs().max()), 1e-12)
            moved = (torch.round(x) == torch.round(torch.cat([x0.unsqueeze(0), x[:-1]], 0))) & (memory.grad != 0)
            self.assertTrue(bool(moved.any()))                                # 没有消息的位置也被惩罚
        none_prev = mean_rates(window_rates(net, [m.detach() for m in memories], None, None))
        for rate, (_, x) in zip(none_prev, raw):                            # 没有上一状态 = 从 0 开始
            prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
            expected = float((torch.round(x) - torch.round(prev)).abs().sum()) / x.numel()
            self.assertAlmostEqual(float(rate), expected, places=12)

    def test_equals_evaluation_statistics(self):
        # 同一条流分 3 个片段：逐窗率的加权和 = 逐窗 stop_stats 的 |Δq| 率（评估口径）
        net = make_tif_net().eval()
        seq = synthetic_sequence(0)
        source = TorchWindowSource(seq, CFG, Q99, CPU, torch.float64)
        with torch.no_grad():
            net.start_tif_stats()
            states = None
            for k in range(12):
                x, events, _, _ = source.window(k)
                _, states, _ = net(x, events, states)
            stats = net.stop_tif_stats()
            states, sums = None, [0.0] * 7
            for lo, hi in ((0, 3), (3, 8), (8, 12)):
                x, events, _, _ = source.chunk(lo, hi)
                prev = states
                _, states, info = net.forward_chunk(x, events, states, collect=True)
                for i, r in enumerate(window_rates(net, info["u_pre"], prev, None)):
                    sums[i] += float(r.sum())
        for i, name in enumerate(LAYER_NAMES):
            self.assertAlmostEqual(sums[i] / 12.0, stats[name]["message_abs_rate"], places=10, msg=name)

    def test_empty_windows_excluded(self):
        # 空窗（valid = 0）不计入片段统计与惩罚；全部为空时返回 None
        net = make_tif_net()
        g = torch.Generator().manual_seed(3)
        memories = []
        for block in net.blocks():
            theta = block.neuron.theta().detach().double().view(1, 1, -1, 1, 1)
            memories.append(torch.rand(4, 2, block.out_ch, 2, 2, generator=g, dtype=torch.float64) * 3 * theta)
        valid = torch.ones(4, 2, dtype=torch.float64)
        valid[:2, 1] = 0.0                                                  # 流 1 前 2 窗为空（错开起点）
        keep = torch.ones(4, 2, dtype=torch.float64)
        keep[2, 1] = 0.0                                                    # 流 1 在第 2 窗开始录像
        comm = CommBudget.from_config(budget_cfg(1e5), net)
        per_window = comm.sop(window_rates(net, memories, None, keep))
        comm.budget = float(per_window[valid > 0].median())
        comm.mu = 1.0
        loss = comm.chunk_loss(net, memories, None, keep, 1, valid)
        expected = (torch.relu(per_window / comm.budget - 1.0) * valid).sum() / valid.sum()
        self.assertAlmostEqual(float(loss), float(expected), places=10)
        mean_sop, count, over = comm._block[-1]
        self.assertEqual(count, 6.0)
        self.assertAlmostEqual(mean_sop, float(per_window[valid > 0].mean()), places=6)
        self.assertEqual(over, float((per_window[valid > 0] > comm.budget).sum()))
        self.assertIsNone(comm.chunk_loss(net, memories, None, keep, 1, torch.zeros(4, 2, dtype=torch.float64)))
        self.assertTrue(torch.equal(T.stream_valid([[(0, 0, 4, 0, True)], [(1, 0, 2, 2, True)]], 4, CPU,
                                                   torch.float64), torch.tensor([[1.0, 0], [1, 0], [1, 1], [1, 1]],
                                                                                dtype=torch.float64)))


class PeakConstraintTests(unittest.TestCase):
    def make_memories(self, net, scale):
        g = torch.Generator().manual_seed(7)
        memories = []
        for block in net.blocks():
            theta = block.neuron.theta().detach().double().view(1, 1, -1, 1, 1)
            x = torch.rand(4, 2, block.out_ch, 3, 3, generator=g, dtype=torch.float64) * scale
            memories.append((x * theta).requires_grad_(True))
        return memories

    def test_only_windows_over_budget_are_penalized(self):
        net = make_tif_net()
        memories = self.make_memories(net, 4.0)
        comm = CommBudget.from_config(budget_cfg(1.0), net)
        per_window = comm.sop([r.detach() for r in window_rates(net, memories, None, None)])
        comm.budget = float(per_window.median())                           # 约一半的窗超出预算
        comm.mu = 1.0
        loss = comm.chunk_loss(net, memories, None, None, 1)
        expected = torch.relu(per_window / comm.budget - 1.0).mean()
        self.assertAlmostEqual(float(loss), float(expected), places=10)
        loss.backward()
        over = per_window > comm.budget
        for memory in memories:
            g = memory.grad.flatten(2).abs().sum(2)                        # [T, B]
            # 未超出预算的窗：只有作为下一窗的"上一窗"时才可能有梯度；最后一窗未超出时一定为 0
            if not bool(over[-1].any()):
                self.assertEqual(float(g[-1].sum()), 0.0)
        comm.end_update()
        violation = float(over.double().mean())
        self.assertAlmostEqual(comm.mu, max(0.0, 1.0 + comm.dual_lr * (violation - 0.05)))
        s = comm.summary()
        self.assertEqual((s["mode"], s["accounting"]), ("peak", "merged"))
        self.assertAlmostEqual(s["violation_mean"], violation)
        self.assertIsNotNone(s["sop_window_p95"])

    def test_dual_update_and_validation(self):
        comm = CommBudget(100.0, 0.1, [1.0] * 7, alpha=0.1)
        comm._block = [(80.0, 10.0, 3.0)]                                  # 违约 30% > alpha 10%
        comm.end_update()
        self.assertAlmostEqual(comm.mu, 0.02)
        comm._block = [(50.0, 10.0, 0.0)]                                  # 无违约：每次回落 0.1 * 0.1 = 0.01
        comm.end_update()
        self.assertAlmostEqual(comm.mu, 0.01)
        for _ in range(2):
            comm._block = [(50.0, 10.0, 0.0)]
            comm.end_update()
        self.assertEqual(comm.mu, 0.0)                                     # 不小于 0
        s = comm.summary()
        self.assertAlmostEqual(s["over_budget_frac"], 0.25)
        self.assertAlmostEqual(s["violation_mean"], 0.075)
        net = make_tif_net()
        mem = [torch.zeros(3, 1, b.out_ch, 2, 2, dtype=torch.float64) for b in net.blocks()]
        self.assertIsNone(CommBudget(100.0, 0.1, [1.0] * 7).chunk_loss(net, mem, None, None, 1))   # mu = 0
        for bad in (dict(alpha=0.0), dict(alpha=1.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                CommBudget(100.0, 0.1, [1.0] * 7, **bad)
        with self.assertRaises(TypeError):
            CommBudget(100.0, 0.1, [1.0] * 7, mode="mean")                 # mean 已删除

    def test_evaluation_window_distribution(self):
        # 评估口径：逐窗 |Δq| 率（stop_stats 的 per_window_rate）-> 单窗 SOP 分布；均值等于按平均率算的 SOP
        net = make_tif_net().eval()
        seq = synthetic_sequence(0)
        source = TorchWindowSource(seq, CFG, Q99, CPU, torch.float64)
        with torch.no_grad():
            net.start_tif_stats(per_window=True)
            states = None
            for k in range(12):
                x, events, _, _ = source.window(k)
                _, states, _ = net(x, events, states)
            stats = net.stop_tif_stats()
        for name in LAYER_NAMES:
            self.assertEqual(len(stats[name]["per_window_rate"]), 12)
            self.assertAlmostEqual(float(np.mean(stats[name]["per_window_rate"])), stats[name]["message_abs_rate"],
                                   places=12)
        for coefficients in (None, merged_sop_coefficients(net, H, W)):
            summary = window_sop_summary(net, stats, H, W, coefficients)
            base, coef = sop_coefficients(net, H, W) if coefficients is None else coefficients
            mean_sop = base + sum(c * stats[n]["message_abs_rate"] for c, n in zip(coef, LAYER_NAMES))
            self.assertAlmostEqual(summary["mean"], mean_sop, delta=1e-6 * max(mean_sop, 1.0))
            self.assertLessEqual(summary["p95"], summary["max"])
            self.assertEqual(summary["windows"], 12)
        self.assertGreater(window_sop_summary(net, stats, H, W, merged_sop_coefficients(net, H, W))["mean"],
                           window_sop_summary(net, stats, H, W)["mean"])
        with torch.no_grad():
            net.start_tif_stats()
            net(source.window(0)[0], source.window(0)[1], None)
            self.assertNotIn("per_window_rate", net.stop_tif_stats()["enc1"])      # 默认不记逐窗

    def test_train_streams(self):
        data = [synthetic_sequence(seed) for seed in (0, 1, 2)]
        lengths = [seq.n_windows for seq in data]
        plan = [(0, [(0, 12)]), (1, [(1, 12), (2, 12)])]
        net_a = make_tif_net()
        net_b, net_c = copy.deepcopy(net_a), copy.deepcopy(net_a)
        opts = [torch.optim.SGD(n.parameters(), lr=0.05) for n in (net_a, net_b, net_c)]
        out_a = T.train_streams(net_a, data, lengths, opts[0], stream_cfg(), Q99, POS_WEIGHT, CPU, None, plan=plan)
        slack = CommBudget.from_config(budget_cfg(1e30), net_b)
        out_b = T.train_streams(net_b, data, lengths, opts[1], stream_cfg(), Q99, POS_WEIGHT, CPU, None, plan=plan,
                                comm=slack)
        self.assertEqual(slack.mu, 0.0)
        self.assertEqual(out_a["loss_sum"], out_b["loss_sum"])
        for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
            self.assertTrue(torch.equal(pa, pb), name)                      # 约束不紧时与关闭逐位相同
        self.assertNotIn("comm", out_a)
        self.assertEqual(out_b["comm"]["violation_mean"], 0.0)
        self.assertGreater(out_b["comm"]["sop_est_mean"], 0.0)
        tight = CommBudget.from_config(budget_cfg(1.0), net_c)
        tight.mu = 0.5
        out = T.train_streams(net_c, data, lengths, opts[2], stream_cfg(), Q99, POS_WEIGHT, CPU, None, plan=plan,
                              comm=tight)
        self.assertGreater(tight.mu, 0.5)
        self.assertEqual((out["comm"]["mu_start"], out["comm"]["mu_end"]), (0.5, tight.mu))
        self.assertEqual(out["comm"]["violation_mean"], 1.0)
        self.assertTrue(any(not torch.equal(pa, pc) for pa, pc in zip(net_a.parameters(), net_c.parameters())))
        self.assertEqual(out["missing_grads"], [])


class CommConfigTests(unittest.TestCase):
    def test_peak_config(self):
        from utils.stream_common import load_flat_config
        a = load_flat_config("configs/evisseg_stream_tif_notransport.yaml")
        b = load_flat_config("configs/evisseg_stream_tif_comm_peak.yaml")
        comm_keys = {"comm_budget_sop", "comm_dual_lr", "comm_budget_mode", "comm_accounting", "comm_peak_alpha"}
        self.assertEqual(set(b) - set(a), comm_keys)
        for key in a:
            if key != "save_root":
                self.assertEqual(a[key], b[key], key)
        self.assertNotEqual(a["save_root"], b["save_root"])
        self.assertEqual(b["save_root"], "log/stream_tif_commpeak_merged_seed37")
        self.assertIsInstance(b["comm_budget_sop"], float)
        self.assertEqual((b["comm_budget_sop"], b["comm_dual_lr"], b["comm_budget_mode"], b["comm_accounting"],
                          b["comm_peak_alpha"]), (42960000.0, 0.01, "peak", "merged", 0.05))
        comm = CommBudget.from_config(dict(budget_cfg(1.0), **{k: b[k] for k in comm_keys}), make_tif_net())
        self.assertEqual((comm.budget, comm.alpha), (42960000.0, 0.05))

    def test_every_budget_config_is_valid(self):
        # 现存配置中凡是开启通信预算的，都必须写明合并口径（旧口径配置已归档到 configs/_removed_comm_20261023/）
        import glob
        from utils.stream_common import load_flat_config
        found = 0
        for path in glob.glob("configs/evisseg_stream*.yaml"):
            cfg = load_flat_config(path)
            if cfg.get("comm_budget_sop") is not None:
                found += 1
                self.assertEqual((cfg.get("comm_accounting"), cfg.get("comm_budget_mode", "peak")),
                                 ("merged", "peak"), path)
        self.assertGreaterEqual(found, 1)


if __name__ == "__main__":
    unittest.main()
