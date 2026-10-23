"""通信预算训练（utils/comm_budget.py：每窗消息数的峰值约束；Stream-SNN-Innovations.md 12.13~12.21、12.35、12.36）。

CPU、float64 小网络，不需要数据集。
    1. CommBudget.from_config：没有预算时为 None（关闭）；只接受 TIF 量化网络 + trainer=streams；
       平均预算 mean、旧键 comm_budget_sop / comm_accounting 都报错
    2. 网络声明的发送层（message_layers）与 forward 的实际连接一致：逐层扰动输出，看是否影响其他层或 logits
       （LIF / TIF、原形式 / 合并解码器四种网络）
    3. 消息数：取自实际张量形状；每窗消息数 = 前向逐窗统计的发送层 |Δq| 之和；与解码器写法无关；
       LIF 的平均消息数 = 发送层脉冲数之和（预算的来源）
    4. window_rates：前向值与评估口径一致——片段首窗与上一片段末状态比较、新录像（keep = 0）首窗与 0 比较，
       并与逐窗统计 TIF 消息的 stop_stats 一致；反向按 |Δx| 求导（没越界的变化也有梯度）；空窗不计入
    5. 峰值约束：只罚超出预算的窗；对偶按违约比例；不发消息的层不受约束
    6. 评估：messages_per_window 与训练同口径；sop_per_window 按网络自身结构计（能耗只在评估环节）
    7. train_streams：约束不紧（mu 保持 0）时与关闭逐位相同；超出时 mu 增大、参数轨迹改变、无漏梯度
    8. 配置文件：与基础版只差 comm_* 四项与 save_root，预算为 V1 的 val 每窗平均脉冲数
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
from model.evspsegnet_stream import (LAYER_NAMES, MESSAGE_LAYERS, EvSpSegNetStream,  # noqa: E402
                                     estimate_operations, merge_decoder_state_dict, window_sop_summary)
from test_stream_layerwise import CFG, H, Q99, SMALL, W, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from utils.comm_budget import (CommBudget, mean_messages, window_message_summary, window_messages,  # noqa: E402
                               window_rates)

CPU = torch.device("cpu")
POS_WEIGHT = 4.0
SCALE = {"enc1": 1, "enc2": 2, "enc3": 4, "enc4": 8, "dec3": 4, "dec2": 2, "dec1": 1}   # 只用于构造测试输入


def stream_cfg(**kw):
    base = dict(CFG, execution="layer", tbptt_k=4, streams=2, accumulate=1, trainer="streams")
    base.update(kw)
    return base


def budget_cfg(budget, **kw):
    """开启通信预算的最小配置。"""
    return stream_cfg(comm_budget_messages=budget, **kw)


def make_memories(net, scale, steps=4, streams=2, seed=7, grad=True):
    """各层按真实分辨率构造的记忆 M（电平坐标 x 在 [0, scale) 内均匀分布）。"""
    g = torch.Generator().manual_seed(seed)
    memories = []
    for name, block in zip(LAYER_NAMES, net.blocks()):
        theta = block.neuron.theta().detach().double().view(1, 1, -1, 1, 1)
        x = torch.rand(steps, streams, block.out_ch, H // SCALE[name], W // SCALE[name], generator=g,
                       dtype=torch.float64) * scale
        memory = x * theta
        memories.append(memory.requires_grad_(True) if grad else memory)
    return memories


def run_stats(model, source, windows=12, per_window=False):
    """带状态连续前向若干窗，返回 TIF 逐层统计。"""
    with torch.no_grad():
        model.start_tif_stats(per_window=per_window)
        states = None
        for k in range(windows):
            x, events, _, _ = source.window(k)
            _, states, _ = model(x, events, states)
        return model.stop_tif_stats()


def make_lif_net(merged=False):
    torch.manual_seed(5)
    net = EvSpSegNetStream(channels=SMALL, neuron="lif", merged_decoder=merged, v_threshold=0.05).double()
    for block in net.blocks():
        block.gain.set_gain(torch.full((block.out_ch,), 20.0, dtype=torch.float64))
    return net


class CommBudgetConfigTests(unittest.TestCase):
    def test_off_and_validation(self):
        net = make_tif_net()
        self.assertIsNone(CommBudget.from_config(stream_cfg(), net))
        self.assertIsNone(CommBudget.from_config(stream_cfg(comm_budget_messages=None), net))
        comm = CommBudget.from_config(budget_cfg(1e3, comm_dual_lr=0.02, comm_peak_alpha=0.1), net)
        self.assertEqual((comm.budget, comm.dual_lr, comm.alpha, comm.mu), (1e3, 0.02, 0.1, 0.0))
        for bad in (dict(comm_budget_messages=0.0), dict(comm_dual_lr=-1.0), dict(trainer="sequence"),
                    dict(comm_budget_mode="mean"), dict(comm_peak_alpha=0.0), dict(comm_peak_alpha=1.0),
                    dict(comm_budget_sop=1e5), dict(comm_accounting="merged")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                CommBudget.from_config(dict(budget_cfg(1e3), **bad), net)
        with self.assertRaises(ValueError):                                 # 旧配置（只有 SOP 预算）不会被静默关闭
            CommBudget.from_config(stream_cfg(comm_budget_sop=1e5), net)
        with self.assertRaises(ValueError):
            CommBudget.from_config(budget_cfg(1e3), EvSpSegNetStream(channels=SMALL, neuron="lif"))
        with self.assertRaises(ValueError):
            CommBudget.from_config(budget_cfg(1e3), make_tif_net(quantize=False))


class MessageLayerDeclarationTests(unittest.TestCase):
    def test_declaration_matches_forward(self):
        # 逐层把该层的输出加一个扰动：若其他层的 u_pre 或 logits 随之改变，说明它的输出被接收（发出消息）
        source = TorchWindowSource(synthetic_sequence(0), CFG, Q99, CPU, torch.float64)
        x, events, _, _ = source.window(3)
        for net in (make_tif_net(), make_tif_net(merged=True), make_lif_net(), make_lif_net(merged=True)):
            net.eval()
            with torch.no_grad():
                base_logits, _, base = net(x, events, None, collect=True)
            consumed = []
            for index, (name, block) in enumerate(zip(LAYER_NAMES, net.blocks())):
                def bump(module, inputs, output):
                    return (output[0] + 1.0,) + tuple(output[1:])
                handle = block.register_forward_hook(bump)
                try:
                    with torch.no_grad():
                        logits, _, info = net(x, events, None, collect=True)
                finally:
                    handle.remove()
                changed = not torch.equal(logits, base_logits) or any(
                    not torch.equal(a, b) for j, (a, b) in enumerate(zip(info["u_pre"], base["u_pre"])) if j != index)
                if changed:
                    consumed.append(name)
            self.assertEqual(tuple(consumed), net.message_layers(), (net.neuron_kind, net.merged_decoder))
        self.assertEqual(MESSAGE_LAYERS, tuple(n for n in LAYER_NAMES if n != "dec1"))


class MessageCountTests(unittest.TestCase):
    def test_messages_match_forward_and_are_decoder_independent(self):
        # 原形式与换算后的合并形式（函数严格相同）：逐层 |Δq| 与神经元数相同，每窗消息数相同
        net = make_tif_net().eval()
        merged = make_tif_net(merged=True).eval()
        merged.load_state_dict(merge_decoder_state_dict(net.state_dict()))
        source = TorchWindowSource(synthetic_sequence(0), CFG, Q99, CPU, torch.float64)
        stats = [run_stats(model, source, per_window=True) for model in (net, merged)]
        for name, block in zip(LAYER_NAMES, net.blocks()):
            self.assertEqual(stats[0][name]["neurons"], block.out_ch * (H // SCALE[name]) * (W // SCALE[name]))
            self.assertEqual(stats[0][name]["neurons"], stats[1][name]["neurons"])
            self.assertAlmostEqual(stats[0][name]["message_abs_rate"], stats[1][name]["message_abs_rate"],
                                   places=12, msg=name)
            np.testing.assert_allclose(stats[0][name]["per_window_rate"], stats[1][name]["per_window_rate"],
                                       rtol=0, atol=1e-12)
        direct = sum(stats[0][n]["message_abs_rate"] * stats[0][n]["neurons"] for n in MESSAGE_LAYERS)
        self.assertGreater(direct, 0.0)
        self.assertAlmostEqual(mean_messages(net, None, stats[0]), direct, delta=1e-9 * direct)
        summaries = [window_message_summary(model, s) for model, s in zip((net, merged), stats)]
        self.assertAlmostEqual(summaries[0]["mean"], direct, delta=1e-9 * direct)
        self.assertEqual(summaries[0], summaries[1])
        self.assertIsNone(window_message_summary(net, run_stats(net, source, windows=2)))     # 没有逐窗统计
        self.assertIsNone(mean_messages(make_tif_net(quantize=False), None, stats[0]))

    def test_window_messages_equals_layer_sums(self):
        net = make_tif_net()
        memories = make_memories(net, 4.0, grad=False)
        keep = torch.ones(4, 2, dtype=torch.float64)
        keep[2, 1] = 0.0
        rates = window_rates(net, memories, None, keep)
        expected = sum(r * float(m[0, 0].numel()) for n, r, m in zip(LAYER_NAMES, rates, memories) if n != "dec1")
        torch.testing.assert_allclose(window_messages(net, memories, None, keep), expected, rtol=0, atol=1e-9)

    def test_lif_mean_messages(self):
        # LIF：平均消息数 = 发送层每窗脉冲数之和（同结构 LIF 的这个值就是通信预算）
        net = make_lif_net().eval()
        source = TorchWindowSource(synthetic_sequence(0), CFG, Q99, CPU, torch.float64)
        monitor = T.LayerMonitor(net.v_threshold)
        counts, states = 0.0, None
        with torch.no_grad():
            for k in range(12):
                x, events, _, _ = source.window(k)
                _, states, info = net(x, events, states, collect=True)
                monitor.update(info)
                counts += sum(float((s > 0).sum()) for n, s in zip(LAYER_NAMES, info["spikes"]) if n != "dec1")
        layers = monitor.summary()
        for name, block in zip(LAYER_NAMES, net.blocks()):
            self.assertEqual(layers[name]["neurons"], block.out_ch * (H // SCALE[name]) * (W // SCALE[name]))
        self.assertGreater(counts, 0.0)
        self.assertAlmostEqual(mean_messages(net, layers), counts / 12.0, delta=1e-9 * counts)
        self.assertIsNone(mean_messages(EvSpSegNetStream(channels=SMALL, neuron="relu"), layers))


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
        rates = [r.mean() for r in window_rates(net, memories, prev_states, keep)]
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
        none_prev = [r.mean() for r in window_rates(net, [m.detach() for m in memories], None, None)]
        for rate, (_, x) in zip(none_prev, raw):                            # 没有上一状态 = 从 0 开始
            prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
            expected = float((torch.round(x) - torch.round(prev)).abs().sum()) / x.numel()
            self.assertAlmostEqual(float(rate), expected, places=12)

    def test_equals_evaluation_statistics(self):
        # 同一条流分 3 个片段：逐窗率的加权和 = 逐窗 stop_stats 的 |Δq| 率；逐窗消息数之和 = 评估的消息总数
        net = make_tif_net().eval()
        source = TorchWindowSource(synthetic_sequence(0), CFG, Q99, CPU, torch.float64)
        stats = run_stats(net, source)
        with torch.no_grad():
            states, sums, messages = None, [0.0] * 7, 0.0
            for lo, hi in ((0, 3), (3, 8), (8, 12)):
                x, events, _, _ = source.chunk(lo, hi)
                prev = states
                _, states, info = net.forward_chunk(x, events, states, collect=True)
                for i, r in enumerate(window_rates(net, info["u_pre"], prev, None)):
                    sums[i] += float(r.sum())
                messages += float(window_messages(net, info["u_pre"], prev, None).sum())
        for i, name in enumerate(LAYER_NAMES):
            self.assertAlmostEqual(sums[i] / 12.0, stats[name]["message_abs_rate"], places=10, msg=name)
        self.assertAlmostEqual(messages / 12.0, mean_messages(net, None, stats), delta=1e-9 * messages)

    def test_empty_windows_excluded(self):
        # 空窗（valid = 0）不计入片段统计与惩罚；全部为空时返回 None
        net = make_tif_net()
        memories = make_memories(net, 3.0, seed=3, grad=False)
        valid = torch.ones(4, 2, dtype=torch.float64)
        valid[:2, 1] = 0.0                                                  # 流 1 前 2 窗为空（错开起点）
        keep = torch.ones(4, 2, dtype=torch.float64)
        keep[2, 1] = 0.0                                                    # 流 1 在第 2 窗开始录像
        comm = CommBudget.from_config(budget_cfg(1e3), net)
        per_window = window_messages(net, memories, None, keep)
        comm.budget = float(per_window[valid > 0].median())
        comm.mu = 1.0
        loss = comm.chunk_loss(net, memories, None, keep, 1, valid)
        expected = (torch.relu(per_window / comm.budget - 1.0) * valid).sum() / valid.sum()
        self.assertAlmostEqual(float(loss), float(expected), places=10)
        mean_msg, count, over = comm._block[-1]
        self.assertEqual(count, 6.0)
        self.assertAlmostEqual(mean_msg, float(per_window[valid > 0].mean()), places=6)
        self.assertEqual(over, float((per_window[valid > 0] > comm.budget).sum()))
        self.assertIsNone(comm.chunk_loss(net, memories, None, keep, 1, torch.zeros(4, 2, dtype=torch.float64)))
        self.assertTrue(torch.equal(T.stream_valid([[(0, 0, 4, 0, True)], [(1, 0, 2, 2, True)]], 4, CPU,
                                                   torch.float64), torch.tensor([[1.0, 0], [1, 0], [1, 1], [1, 1]],
                                                                                dtype=torch.float64)))


class PeakConstraintTests(unittest.TestCase):
    def test_only_windows_over_budget_are_penalized(self):
        net = make_tif_net()
        memories = make_memories(net, 4.0)
        comm = CommBudget.from_config(budget_cfg(1.0), net)
        per_window = window_messages(net, [m.detach() for m in memories], None, None)
        comm.budget = float(per_window.median())                           # 约一半的窗超出预算
        comm.mu = 1.0
        loss = comm.chunk_loss(net, memories, None, None, 1)
        expected = torch.relu(per_window / comm.budget - 1.0).mean()
        self.assertAlmostEqual(float(loss), float(expected), places=10)
        loss.backward()
        over = per_window > comm.budget
        for memory in memories[:-1]:
            g = memory.grad.flatten(2).abs().sum(2)                        # [T, B]
            # 未超出预算的窗：只有作为下一窗的"上一窗"时才可能有梯度；最后一窗未超出时一定为 0
            if not bool(over[-1].any()):
                self.assertEqual(float(g[-1].sum()), 0.0)
        self.assertIsNone(memories[-1].grad)                               # dec1 不发消息，不参与约束
        comm.end_update()
        violation = float(over.double().mean())
        self.assertAlmostEqual(comm.mu, max(0.0, 1.0 + comm.dual_lr * (violation - 0.05)))
        s = comm.summary()
        self.assertEqual((s["mode"], s["unit"]), ("peak", "messages"))
        self.assertAlmostEqual(s["violation_mean"], violation)
        self.assertIsNotNone(s["messages_window_p95"])

    def test_dual_update_and_validation(self):
        comm = CommBudget(100.0, 0.1, alpha=0.1)
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
        self.assertIsNone(CommBudget.from_config(budget_cfg(1e3), net).chunk_loss(
            net, make_memories(net, 4.0, grad=False), None, None, 1))       # mu = 0 时不参与反传
        for bad in (dict(alpha=0.0), dict(alpha=1.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                CommBudget(100.0, 0.1, **bad)
        with self.assertRaises(TypeError):
            CommBudget(100.0, 0.1, mode="mean")                            # mean 已删除

    def test_evaluation_window_distribution(self):
        # 评估：每窗消息数（与训练同口径）与单窗 SOP（按网络自身结构，只用于报告）的分布
        net = make_tif_net().eval()
        source = TorchWindowSource(synthetic_sequence(0), CFG, Q99, CPU, torch.float64)
        stats = run_stats(net, source, per_window=True)
        for name in LAYER_NAMES:
            self.assertEqual(len(stats[name]["per_window_rate"]), 12)
            self.assertAlmostEqual(float(np.mean(stats[name]["per_window_rate"])), stats[name]["message_abs_rate"],
                                   places=12)
        msg = window_message_summary(net, stats)
        self.assertAlmostEqual(msg["mean"], mean_messages(net, None, stats), delta=1e-9 * msg["mean"])
        rates = {n: stats[n]["message_abs_rate"] for n in LAYER_NAMES}
        sop = window_sop_summary(net, stats, H, W)
        self.assertAlmostEqual(sop["mean"], estimate_operations(net, H, W, rates, 0.0)["sop"],
                               delta=1e-6 * max(sop["mean"], 1.0))
        for summary in (msg, sop):
            self.assertLessEqual(summary["p95"], summary["max"])
            self.assertEqual(summary["windows"], 12)
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
        self.assertGreater(out_b["comm"]["messages_est_mean"], 0.0)
        tight = CommBudget.from_config(budget_cfg(1e-3), net_c)
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
        comm_keys = {"comm_budget_messages", "comm_dual_lr", "comm_budget_mode", "comm_peak_alpha"}
        self.assertEqual(set(b) - set(a), comm_keys)
        for key in a:
            if key != "save_root":
                self.assertEqual(a[key], b[key], key)
        self.assertEqual(b["save_root"], "log/stream_tif_commpeak_msg_seed37")
        self.assertIsInstance(b["comm_budget_messages"], float)
        self.assertEqual((b["comm_budget_messages"], b["comm_dual_lr"], b["comm_budget_mode"], b["comm_peak_alpha"]),
                         (157700.0, 0.01, "peak", 0.05))
        comm = CommBudget.from_config(dict(stream_cfg(), **{k: b[k] for k in comm_keys}), make_tif_net())
        self.assertEqual((comm.budget, comm.alpha), (157700.0, 0.05))

    def test_evflying_peak_config(self):
        # EV-Flying 版：与 EV-Flying 基础版只差 comm_* 四项与 save_root；预算为 V1 在 EV-Flying val 上的每窗平均脉冲数
        from utils.stream_common import load_flat_config
        a = load_flat_config("configs/evisseg_stream_tif_notransport_evflying.yaml")
        b = load_flat_config("configs/evisseg_stream_tif_comm_peak_evflying.yaml")
        e = load_flat_config("configs/evisseg_stream_tif_comm_peak.yaml")
        comm_keys = {"comm_budget_messages", "comm_dual_lr", "comm_budget_mode", "comm_peak_alpha"}
        self.assertEqual(set(b) - set(a), comm_keys)
        for key in a:
            if key != "save_root":
                self.assertEqual(a[key], b[key], key)
        self.assertEqual(b["save_root"], "log/stream_tif_commpeak_msg_evflying_x4_seed37")
        self.assertEqual(b["comm_budget_messages"], 293200.0)
        for key in comm_keys - {"comm_budget_messages"}:                # 除预算外与 EV-UAV 定型版相同
            self.assertEqual(b[key], e[key], key)

    def test_every_budget_config_is_valid(self):
        # 现存配置中凡是开启通信预算的，都按消息数约束、不带旧键（旧配置已归档到 configs/_removed_comm_20261023/）
        import glob
        from utils.stream_common import load_flat_config
        found = 0
        for path in glob.glob("configs/evisseg_stream*.yaml"):
            cfg = load_flat_config(path)
            self.assertNotIn("comm_budget_sop", cfg, path)
            self.assertNotIn("comm_accounting", cfg, path)
            if cfg.get("comm_budget_messages") is not None:
                found += 1
                self.assertEqual(cfg.get("comm_budget_mode", "peak"), "peak", path)
        self.assertGreaterEqual(found, 1)


if __name__ == "__main__":
    unittest.main()
