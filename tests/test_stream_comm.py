"""通信预算训练（utils/comm_budget.py，损失函数层，Stream-SNN-Innovations.md 12.13~12.15）的单元测试。

CPU、float64 小网络，不需要数据集。
    1. CommBudget.from_config：没有预算时为 None（关闭）；只接受 TIF 量化网络 + trainer=streams；
       SOP 估计 = estimate_operations 的 SOP（线性分解精确）
    2. message_rates：前向值与评估口径一致——片段首窗与上一片段末状态比较、新录像（keep = 0）首窗与 0 比较，
       并与逐窗统计 TIF 消息的 stop_stats 一致；反向按 |Δx| 求导（没越界的变化也有梯度）
    3. 对偶上升：超预算时 mu 增大、低于预算时减小且不小于 0；mu = 0 时 chunk_loss 返回 None
    4. train_streams：预算充足（mu 保持 0）时与关闭逐位相同；超预算时 mu 增大、参数轨迹改变；冒烟式 mu > 0 无漏梯度
    5. 配置文件：与基础版只差通信预算两项与 save_root
"""
import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_stream_v1 as T  # noqa: E402
from dataset.stream_source import TorchWindowSource  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream, estimate_operations  # noqa: E402
from test_stream_layerwise import CFG, H, Q99, SMALL, W, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from utils.comm_budget import CommBudget, message_rates  # noqa: E402

CPU = torch.device("cpu")
POS_WEIGHT = 4.0


def stream_cfg(**kw):
    base = dict(CFG, execution="layer", tbptt_k=4, streams=2, accumulate=1, trainer="streams")
    base.update(kw)
    return base


class CommBudgetConfigTests(unittest.TestCase):
    def test_off_and_validation(self):
        net = make_tif_net()
        self.assertIsNone(CommBudget.from_config(stream_cfg(), net))
        self.assertIsNone(CommBudget.from_config(stream_cfg(comm_budget_sop=None), net))
        comm = CommBudget.from_config(stream_cfg(comm_budget_sop=1e5, comm_dual_lr=0.02), net)
        self.assertEqual((comm.budget, comm.dual_lr, comm.mu), (1e5, 0.02, 0.0))
        for bad in (dict(comm_budget_sop=0.0), dict(comm_budget_sop=1e5, comm_dual_lr=-1.0),
                    dict(comm_budget_sop=1e5, trainer="sequence")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                CommBudget.from_config(stream_cfg(**bad), net)
        with self.assertRaises(ValueError):
            CommBudget.from_config(stream_cfg(comm_budget_sop=1e5), EvSpSegNetStream(channels=SMALL, neuron="lif"))
        with self.assertRaises(ValueError):
            CommBudget.from_config(stream_cfg(comm_budget_sop=1e5), make_tif_net(quantize=False))

    def test_sop_matches_estimate_operations(self):
        net = make_tif_net()
        comm = CommBudget.from_config(stream_cfg(comm_budget_sop=1e5), net)
        g = torch.Generator().manual_seed(0)
        for _ in range(3):
            rates = {name: float(torch.rand(1, generator=g)) for name in LAYER_NAMES}
            sop = estimate_operations(net, H, W, rates, 0.0)["sop"]
            self.assertAlmostEqual(comm.sop(rates), sop, delta=1e-6 * max(sop, 1.0))
        self.assertGreater(min(comm.coef[:6]), 0.0)                    # enc1..dec2 的消息都有下游突触


class MessageRateTests(unittest.TestCase):
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
        rates = message_rates(net, memories, prev_states, keep)
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
        none_prev = message_rates(net, [m.detach() for m in memories], None, None)
        for rate, (_, x) in zip(none_prev, raw):                            # 没有上一状态 = 从 0 开始
            prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
            expected = float((torch.round(x) - torch.round(prev)).abs().sum()) / x.numel()
            self.assertAlmostEqual(float(rate), expected, places=12)

    def test_equals_evaluation_statistics(self):
        # 同一条流分 3 个片段：message_rates 的加权和 = 逐窗 stop_stats 的 |Δq| 率（评估口径）
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
                rates = message_rates(net, info["u_pre"], prev, None)
                for i, r in enumerate(rates):
                    sums[i] += float(r) * (hi - lo)
        for i, name in enumerate(LAYER_NAMES):
            self.assertAlmostEqual(sums[i] / 12.0, stats[name]["message_abs_rate"], places=10, msg=name)


    def test_empty_windows_excluded(self):
        # 空窗（valid = 0）不计入：结果等于只在有效窗上手算；全部为空时返回 None
        net = make_tif_net()
        g = torch.Generator().manual_seed(3)
        memories, raw = [], []
        for block in net.blocks():
            theta = block.neuron.theta().detach().double().view(1, 1, -1, 1, 1)
            x = torch.rand(4, 2, block.out_ch, 2, 2, generator=g, dtype=torch.float64) * 3
            raw.append(x)
            memories.append(x * theta)
        valid = torch.ones(4, 2, dtype=torch.float64)
        valid[:2, 1] = 0.0                                                  # 流 1 前 2 窗为空（错开起点）
        keep = torch.ones(4, 2, dtype=torch.float64)
        keep[2, 1] = 0.0                                                    # 流 1 在第 2 窗开始录像
        rates = message_rates(net, memories, None, keep, valid)
        for rate, x in zip(rates, raw):
            prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0) * keep.view(4, 2, 1, 1, 1)
            term = (torch.round(x) - torch.round(prev)).abs() * valid.view(4, 2, 1, 1, 1)
            self.assertAlmostEqual(float(rate), float(term.sum()) / (float(valid.sum()) * x[0, 0].numel()), places=12)
        self.assertIsNone(message_rates(net, memories, None, keep, torch.zeros(4, 2, dtype=torch.float64)))
        self.assertTrue(torch.equal(T.stream_valid([[(0, 0, 4, 0, True)], [(1, 0, 2, 2, True)]], 4, CPU,
                                                   torch.float64), torch.tensor([[1.0, 0], [1, 0], [1, 1], [1, 1]],
                                                                                dtype=torch.float64)))


class DualAscentTests(unittest.TestCase):
    def test_dual_update(self):
        comm = CommBudget(100.0, 0.1, [1.0] * 7)
        comm._block = [(150.0, 1.0), (250.0, 1.0)]                          # 平均 200 = 预算的 2 倍
        comm.end_update()
        self.assertAlmostEqual(comm.mu, 0.1)
        comm._block = [(300.0, 1.0), (0.0, 3.0)]                            # 按有效窗加权：平均 75 < 预算
        comm.end_update()
        self.assertAlmostEqual(comm.mu, 0.075)
        comm._block = [(0.0, 1.0)]
        comm.end_update()
        self.assertEqual(comm.mu, 0.0)                                     # 不小于 0
        s = comm.summary()
        self.assertEqual(s["mu_end"], 0.0)
        self.assertAlmostEqual(s["over_budget_frac"], 1.0 / 3.0)
        net = make_tif_net()
        mem = [torch.zeros(3, 1, b.out_ch, 2, 2, dtype=torch.float64) for b in net.blocks()]
        self.assertIsNone(CommBudget(100.0, 0.1, [1.0] * 7).chunk_loss(net, mem, None, None, 1))


class TrainStreamsCommTests(unittest.TestCase):
    def run_pair(self, comm):
        data = [synthetic_sequence(seed) for seed in (0, 1, 2)]
        lengths = [seq.n_windows for seq in data]
        plan = [(0, [(0, 12)]), (1, [(1, 12), (2, 12)])]
        net_a = make_tif_net()
        net_b = copy.deepcopy(net_a)
        opt_a = torch.optim.SGD(net_a.parameters(), lr=0.05)
        opt_b = torch.optim.SGD(net_b.parameters(), lr=0.05)
        out_a = T.train_streams(net_a, data, lengths, opt_a, stream_cfg(), Q99, POS_WEIGHT, CPU, None, plan=plan)
        out_b = T.train_streams(net_b, data, lengths, opt_b, stream_cfg(), Q99, POS_WEIGHT, CPU, None, plan=plan,
                                comm=comm)
        return net_a, net_b, out_a, out_b

    def test_inactive_budget_is_bitwise_identical(self):
        comm = CommBudget.from_config(stream_cfg(comm_budget_sop=1e30), make_tif_net())
        net_a, net_b, out_a, out_b = self.run_pair(comm)
        self.assertEqual(comm.mu, 0.0)
        self.assertEqual(out_a["loss_sum"], out_b["loss_sum"])
        for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
            self.assertTrue(torch.equal(pa, pb), name)
        self.assertNotIn("comm", out_a)
        self.assertEqual(out_b["comm"]["over_budget_frac"], 0.0)
        self.assertGreater(out_b["comm"]["sop_est_mean"], 0.0)

    def test_over_budget_raises_mu_and_changes_training(self):
        comm = CommBudget.from_config(stream_cfg(comm_budget_sop=1.0, comm_dual_lr=0.01), make_tif_net())
        comm.mu = 0.5
        net_a, net_b, out_a, out_b = self.run_pair(comm)
        self.assertGreater(comm.mu, 0.5)
        self.assertEqual((out_b["comm"]["mu_start"], out_b["comm"]["mu_end"]), (0.5, comm.mu))
        self.assertEqual(out_b["comm"]["over_budget_frac"], 1.0)
        self.assertTrue(any(not torch.equal(pa, pb) for pa, pb in zip(net_a.parameters(), net_b.parameters())))
        self.assertEqual(out_b["missing_grads"], [])


class CommConfigTests(unittest.TestCase):
    def test_config_differs_only_in_budget(self):
        from utils.stream_common import load_flat_config
        a = load_flat_config("configs/evisseg_stream_tif_notransport.yaml")
        b = load_flat_config("configs/evisseg_stream_tif_comm.yaml")
        self.assertEqual(set(b) - set(a), {"comm_budget_sop", "comm_dual_lr"})
        for key in a:
            if key != "save_root":
                self.assertEqual(a[key], b[key], key)
        self.assertNotEqual(a["save_root"], b["save_root"])
        self.assertIsInstance(b["comm_budget_sop"], float)
        self.assertEqual((b["comm_budget_sop"], b["comm_dual_lr"]), (17150000.0, 0.01))


if __name__ == "__main__":
    unittest.main()
