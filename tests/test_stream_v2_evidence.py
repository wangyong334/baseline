"""流式 SNN V2（方案一）的单元测试：CUSUM 判决层、证据前端、网络与训练/推理流程。

全部在 CPU 上运行，不需要数据集：
    1. shift2d、单像素证据的期望（泊松/负二项下 E0[exp(e)] = 1）、各假设的证据按平移后的强度场计算、
       CUSUM 与逐起点暴力计算一致（静止与移动假设）
    2. 管道强度记忆的递推；足迹聚合 sum / lme 的公式；延迟读出 TubeReadout 与暴力计算一致；
       纯背景下带复位的告警率不超过理论上界，低估背景时明显超标；
       像素完全相关的背景（逐像素仍是泊松）下 sum 聚合超标、lme 聚合满足（评审给出的反例）
    2b. 位置级告警评估（首次告警延迟、虚警连通域）；按实际发布窗计算的首次检出延迟
    3. 前端时间矩递推与逐事件直接计算一致（跨片段）；背景只用过去；偶极方向与极性排列一致；事件来源计数正确
    4. 网络 forward 与 forward_chunk 等价；train_sequence 使全部参数得到梯度；run_sequence 的各读出覆盖全部事件
误差一律用 (a - b).abs().max() 比较（torch 1.9 的 assert_close 默认比较步长）。
"""
import math
import unittest

import numpy as np
import torch

from dataset import stream_windows as sw
from dataset.stream_features import FEATURE_GROUPS, EventChunkSource, EvidenceFrontEnd
from model.evidence_neuron import LOG_ZERO, DriftCUSUM, TubeReadout, pixel_evidence, shift2d, velocity_grid
from model.evidence_snn import EvidenceSNN

H, W, WINDOWS = 24, 32, 12
TAUS = [20.0, 100.0, 500.0]
CFG = dict(pad_height=H, pad_width=W, window_ms=50, fe_taus_ms=TAUS, tbptt_k=4, grad_clip=1.0,
           loss_mark_weight=1.0, loss_intensity_weight=1.0, threshold=0.5, readout_delays=[1, 3], eval_chunk=5, state_mode="carry")


def max_error(a, b):
    """两个张量的最大绝对误差（空张量记为 0）。"""
    return float((a.detach() - b.detach()).abs().max()) if a.numel() else 0.0


def synthetic_sequence(seed=0, n_noise=700):
    """合成序列：均匀噪声 + 一个向右移动、后沿 ON 前沿 OFF 的小目标；文件顺序打乱（检验下标回填）。"""
    rng = np.random.RandomState(seed)
    t = rng.randint(0, 50 * WINDOWS, n_noise)
    x, y, p = rng.randint(0, W, n_noise), rng.randint(0, H, n_noise), rng.randint(0, 2, n_noise)
    label = np.zeros(n_noise, dtype=np.float32)
    tt = rng.randint(0, 50 * WINDOWS, 400)
    cx = 3 + tt * 0.03                                         # 0.03 像素/ms 向右
    dx = rng.randint(-2, 3, 400)
    tx = np.clip(np.round(cx + dx), 0, W - 1).astype(np.int64)
    ty = np.clip(10 + rng.randint(-1, 2, 400), 0, H - 1)
    tp = (dx < 0).astype(np.int64)                             # 后沿（左侧）ON，前沿（右侧）OFF
    t = np.concatenate([t, tt]); x = np.concatenate([x, tx]); y = np.concatenate([y, ty])
    p = np.concatenate([p, tp]); label = np.concatenate([label, np.ones(400, dtype=np.float32)])
    perm = rng.permutation(t.size)
    t, x, y, p, label = t[perm].astype(np.int64), x[perm].astype(np.int64), y[perm].astype(np.int64), \
        p[perm].astype(np.int8), label[perm]
    order, bounds = sw.split_windows(t, 50, WINDOWS)
    bins, local = sw.time_bin_and_local(t, 50, 5)
    return sw.StreamSequence("synthetic.npz", x, y, t, p, label, label.astype(np.float64), bins, local, order, bounds)


def make_frontend():
    return EvidenceFrontEnd(TAUS, 50.0, [100.0], dipole_radius=2, bg_smooth_radius=2).double()


# ---------------------------------------------------------------------------
# 1–2. CUSUM 判决层
# ---------------------------------------------------------------------------


class ShiftAndEvidenceTests(unittest.TestCase):
    def test_shift2d_matches_definition(self):
        x = torch.arange(5 * 7, dtype=torch.float64).view(1, 5, 7)
        for sy, sx in ((0, 0), (1, 0), (-2, 3), (4, -6), (5, 0), (0, -7)):
            out = shift2d(x, sy, sx, fill=-1.0)
            for yy in range(5):
                for xx in range(7):
                    sy_, sx_ = yy - sy, xx - sx
                    want = x[0, sy_, sx_] if (0 <= sy_ < 5 and 0 <= sx_ < 7) else -1.0
                    self.assertEqual(float(out[0, yy, xx]), float(want))

    def test_poisson_evidence_has_unit_expectation_under_noise(self):
        # E0[exp(N log r - mu0(r-1))] = exp((mu - mu0)(r-1))：mu0 等于真实均值时为 1，偏大时 < 1
        n = torch.arange(0, 80, dtype=torch.float64)
        for mu, mu0 in ((0.05, 0.05), (0.8, 0.8), (0.05, 0.1)):
            log_pmf = n * math.log(mu) - mu - torch.lgamma(n + 1)
            log_g = torch.full_like(n, math.log(0.7))
            e, _ = pixel_evidence(n, torch.full_like(n, mu0), log_g)
            expectation = float(torch.exp(log_pmf + e).sum())
            r = 1.0 + 0.7 / mu0
            self.assertAlmostEqual(expectation, math.exp((mu - mu0) * (r - 1.0)), places=9)
            self.assertLessEqual(expectation, 1.0 + 1e-9)

    def test_negbin_evidence_has_unit_expectation_under_negbin_noise(self):
        mu, kappa = 0.3, 0.8
        n = torch.arange(0, 400, dtype=torch.float64)
        p_success = kappa / (kappa + mu)
        log_pmf = (torch.lgamma(n + kappa) - torch.lgamma(n + 1) - math.lgamma(kappa)
                   + kappa * math.log(p_success) + n * math.log(1 - p_success))
        e, _ = pixel_evidence(n, torch.full_like(n, mu), torch.full_like(n, math.log(0.5)), "negbin", kappa)
        self.assertAlmostEqual(float(torch.exp(log_pmf + e).sum()), 1.0, places=6)


def path_offset(velocity, k):
    """速度假设在第 k 窗的累计整数位移（与 DriftCUSUM.offsets 相同的取整）。"""
    return int(math.floor(velocity[0] * k + 0.5)), int(math.floor(velocity[1] * k + 0.5))


def brute_force_cusum(ells, velocity, k, y, x):
    """沿速度假设的管道从 (y,x,k) 往回走，取所有起点的证据和的最大值（至少为 0）。ells[m] 为第 m 窗该假设的证据图。"""
    best, run = 0.0, 0.0
    end = path_offset(velocity, k)
    for m in range(k, -1, -1):
        here = path_offset(velocity, m)
        py, px = y - (end[0] - here[0]), x - (end[1] - here[1])
        if not (0 <= py < ells[m].shape[0] and 0 <= px < ells[m].shape[1]):
            break
        run += float(ells[m][py, px])
        best = max(best, run)
    return best


class CusumTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.steps, self.h, self.w = 9, 8, 10
        self.counts = torch.poisson(torch.full((self.steps, 1, 1, self.h, self.w), 0.3, dtype=torch.float64))
        self.mu0 = torch.full_like(self.counts, 0.3)
        self.log_g = torch.randn(self.steps, 1, 1, self.h, self.w, dtype=torch.float64) - 1.0

    def run_cusum(self, cusum):
        """逐窗运行；第 k 窗用第 k-1 窗的强度场（第 0 窗没有）。返回每窗之后的状态列表。"""
        state = cusum.init_state(1, self.h, self.w, "cpu", torch.float64)
        states = []
        for k in range(self.steps):
            state, _, _ = cusum.step(state, self.counts[k], self.mu0[k], self.log_g[k - 1] if k > 0 else None)
            states.append(state)
        return states

    def test_hypothesis_evidence_uses_shifted_intensity(self):
        velocities = [(0.0, 0.0), (1.0, -2.0), (-0.5, 1.5)]
        pool = lambda z: torch.nn.functional.avg_pool2d(z, 3, 1, 1, count_include_pad=True)  # noqa: E731
        for aggregate in ("sum", "mean", "lme"):
            cusum = DriftCUSUM(velocities, footprint=3, aggregate=aggregate)
            states = self.run_cusum(cusum)
            k = 4
            for v, (sy, sx) in enumerate(cusum.step_shifts(k)):
                shifted = shift2d(self.log_g[k - 1][:, 0], sy, sx, fill=LOG_ZERO).unsqueeze(1)
                e, _ = pixel_evidence(self.counts[k], self.mu0[k], shifted)
                lme = torch.log(torch.nn.functional.avg_pool2d(
                    torch.nn.functional.pad(torch.exp(e), (1, 1, 1, 1), value=1.0), 3, 1))   # 画面外按证据 0
                want = {"sum": pool(e) * 9.0, "mean": pool(e), "lme": lme}[aggregate]
                self.assertLess(max_error(states[k]["ell"][:, v:v + 1], want), 1e-9, aggregate)
            self.assertLess(float(states[0]["ell"].abs().max()), 1e-9)    # 第 0 窗没有强度场，证据为 0

    def test_track_memory_carries_intensity_along_tube(self):
        rho = 0.6
        velocities = [(0.0, 1.0), (1.0, 0.0)]
        cusum = DriftCUSUM(velocities, footprint=1, track_decay=rho)
        states = self.run_cusum(cusum)
        for k in (2, 5):
            shifts = cusum.step_shifts(k)
            for v, (sy, sx) in enumerate(shifts):
                fresh = shift2d(torch.exp(self.log_g[k - 1][:, 0]), sy, sx)
                carried = rho * shift2d(states[k - 1]["G"][:, v], sy, sx)
                self.assertLess(max_error(states[k]["G"][:, v], torch.maximum(fresh, carried)), 1e-12)
        # 网络不再给出强度时，管道仍按 rho 衰减地携带原来的强度（离开目标后持续产生负证据）
        state = cusum.init_state(1, self.h, self.w, "cpu", torch.float64)
        g = torch.zeros(1, 1, self.h, self.w, dtype=torch.float64)
        g[0, 0, 4, 2] = 2.0
        state, _, _ = cusum.step(state, torch.zeros_like(g), torch.full_like(g, 0.3), None)
        state, _, _ = cusum.step(state, torch.zeros_like(g), torch.full_like(g, 0.3), torch.log(g + 1e-30))
        state, _, _ = cusum.step(state, torch.zeros_like(g), torch.full_like(g, 0.3), None)
        self.assertAlmostEqual(float(state["G"].max()), 2.0 * rho, places=12)
        self.assertLess(float(state["ell"].min()), -1.0)

    def test_matches_brute_force_for_static_and_moving_hypotheses(self):
        for velocity in ((0.0, 0.0), (0.0, 1.0), (-1.0, 0.5), (2.0, -1.5)):
            cusum = DriftCUSUM([velocity], footprint=3)
            states = self.run_cusum(cusum)
            ells = [st["ell"][0, 0] for st in states]
            for k in (0, 3, self.steps - 1):
                C, C_pre = states[k]["C"][0, 0], states[k]["C_pre"][0, 0]
                self.assertLess(max_error(C, torch.relu(C_pre)), 1e-12)
                for y in range(self.h):
                    for x in range(self.w):
                        self.assertAlmostEqual(float(C[y, x]), brute_force_cusum(ells, velocity, k, y, x), places=9)

    def test_tube_readout_matches_brute_force(self):
        velocities = velocity_grid([-1.0, 0.0, 1.5])
        cusum = DriftCUSUM(velocities, footprint=1)
        delays = [1, 3]
        readout = TubeReadout(cusum, delays)
        y = torch.tensor([0, 3, 7, 4], dtype=torch.long)
        x = torch.tensor([0, 5, 9, 1], dtype=torch.long)
        b = torch.zeros(4, dtype=torch.long)
        state = cusum.init_state(1, self.h, self.w, "cpu", torch.float64)
        states, got = [], {}
        for k in range(self.steps):
            state, _, _ = cusum.step(state, self.counts[k], self.mu0[k], self.log_g[k - 1] if k > 0 else None)
            states.append(state)
            for key, d, scores, published in readout.step(state, k, b, y, x, k):
                self.assertEqual(published, key + d)
                got[(key, d)] = scores
        for key, d, scores, published in readout.flush():
            self.assertEqual(published, self.steps - 1)
            got[(key, d)] = scores
        self.assertEqual(sorted(got), sorted((k, d) for k in range(self.steps) for d in delays))
        last = self.steps - 1
        for (k, d), scores in got.items():
            for i in range(4):
                sums = []
                for v, vel in enumerate(velocities):
                    run, start = 0.0, path_offset(vel, k)
                    for m in range(k + 1, min(k + d, last) + 1):
                        now = path_offset(vel, m)
                        py, px = int(y[i]) + now[0] - start[0], int(x[i]) + now[1] - start[1]
                        if 0 <= py < self.h and 0 <= px < self.w:
                            run += float(states[m]["ell"][0, v, py, px])
                    sums.append(run)
                want = math.log(sum(math.exp(t) for t in sums) / len(sums))
                self.assertAlmostEqual(float(scores[i]), want, places=9)

    def test_false_alarm_rate_respects_bound_and_detects_underestimated_background(self):
        # 纯背景；预测只用过去（满足可预测条件）：一个"追着上一窗噪声跑"的对手、一个常数强度。
        # 单假设时每位置每窗的虚警率上界为 1/(e^theta - 1)。背景被低估到 0.2 倍时，
        # 常数 g = mu - mu0 让每步证据的期望为正（mu*log(1+g/mu0) - g > 0），告警率应明显超标。
        gen = torch.Generator().manual_seed(3)
        h = w = 24
        steps, mu, theta = 400, 0.05, 4.0
        bound = 1.0 / (math.exp(theta) - 1.0)

        def alarm_rate(mu0, predictor):
            cusum = DriftCUSUM([(0.0, 1.0)], footprint=3, aggregate="sum")        # 背景独立，sum 也成立
            state = cusum.init_state(1, h, w, "cpu", torch.float64)
            prev = torch.zeros(1, 1, h, w, dtype=torch.float64)
            alarms = 0
            for _ in range(steps):
                counts = torch.poisson(torch.full((1, 1, h, w), mu, dtype=torch.float64), generator=gen)
                state, _, alarm = cusum.step(state, counts, torch.full_like(counts, mu0), predictor(prev),
                                             theta=theta, reset_on_alarm=True)
                alarms += int(alarm.sum())
                prev = counts
            return alarms / float(steps * h * w)

        chase = lambda prev: torch.log(0.5 * prev + 1e-3)                  # noqa: E731
        constant = lambda prev: torch.full_like(prev, math.log(0.6 * mu))  # noqa: E731
        self.assertLessEqual(alarm_rate(mu, chase), bound)
        self.assertLessEqual(alarm_rate(mu, constant), bound)
        under = lambda prev: torch.full_like(prev, math.log(0.8 * mu))     # noqa: E731
        self.assertGreater(alarm_rate(0.2 * mu, under), 3 * bound)


    def test_correlated_background_breaks_sum_but_not_lme(self):
        # 每个 3x3 块内的像素取同一个泊松计数：逐像素仍严格是 Poisson(mu)、mu0 完全正确、预测只用过去，
        # 但足迹内像素完全相关。sum 聚合要求独立，应明显超标；lme 对任意相关都成立，应满足上界。
        # 参数取 g = mu(r-1) 使逐像素证据恰好是鞅（E0[exp(e)] = exp(mu(r-1) - g) = 1），超标只能来自相关性
        gen = torch.Generator().manual_seed(5)
        steps, mu, theta, size = 300, 0.1, 6.0, 24
        bound = 1.0 / (math.exp(theta) - 1.0)

        def alarm_rate(aggregate):
            cusum = DriftCUSUM([(0.0, 0.0)], footprint=3, aggregate=aggregate)
            state = cusum.init_state(1, size, size, "cpu", torch.float64)
            log_g = torch.full((1, 1, size, size), math.log(0.4), dtype=torch.float64)       # r = 5
            alarms = 0
            for _ in range(steps):
                block = torch.poisson(torch.full((1, 1, size // 3, size // 3), mu, dtype=torch.float64), generator=gen)
                counts = block.repeat_interleave(3, 2).repeat_interleave(3, 3)
                state, _, alarm = cusum.step(state, counts, torch.full_like(counts, mu), log_g, theta=theta,
                                             reset_on_alarm=True)
                alarms += int(alarm.sum())
            return alarms / float(steps * size * size)

        self.assertGreater(alarm_rate("sum"), 3 * bound)
        self.assertLessEqual(alarm_rate("lme"), bound)
        self.assertLessEqual(alarm_rate("mean"), bound)


class AlarmAndLatencyTests(unittest.TestCase):
    def test_alarm_evaluator_counts_true_and_false_alarms(self):
        from utils.alarm_metrics import AlarmEvaluator
        # 一个目标：第 2、3 窗在 (y=10, x=10..12) 附近有事件，首个事件 t=105 ms
        t = np.array([105, 120, 160, 5], dtype=np.int64)
        x = np.array([10, 11, 12, 30], dtype=np.int64)
        y = np.array([10, 10, 10, 20], dtype=np.int64)
        label = np.array([1, 1, 1, 0], dtype=np.float32)
        tid = np.array([1, 1, 1, 0], dtype=np.float64)
        order, bounds = sw.split_windows(t, 50, 5)
        bins, local = sw.time_bin_and_local(t, 50, 5)
        seq = sw.StreamSequence("toy.npz", x, y, t, np.ones(4, np.int8), label, tid, bins, local, order, bounds)
        ev = AlarmEvaluator([5.0], 24, 32, 50.0, 2, hypotheses=9)
        ev.begin_sequence(seq, 5)
        alarm = np.zeros((24, 32), dtype=bool)
        alarm[0:2, 0:2] = True                     # 第 1 窗：远离目标，一个虚警连通域
        ev.update(5.0, 1, alarm.copy())
        alarm[:] = False
        alarm[11, 12] = True                       # 第 3 窗：离目标事件 1 像素，真告警
        alarm[20, 25:27] = True                    # 同窗另一个虚警
        ev.update(5.0, 3, alarm.copy())
        ev.end_sequence()
        row = ev.summary()["5.0"]
        self.assertEqual(row["n_targets"], 1)
        self.assertEqual(row["n_detected"], 1)
        self.assertAlmostEqual(row["latency_median_ms"], 4 * 50.0 - 105.0)
        self.assertEqual(row["false_components"], 2)
        self.assertAlmostEqual(row["false_alarm_rate"], 2.0 / (5 * 24 * 32))

    def test_latency_uses_actual_publish_window(self):
        from utils.stream_metrics import first_detection_latencies_published
        t = np.array([60, 70, 260], dtype=np.int64)
        label = np.ones(3, dtype=np.float32)
        tid = np.ones(3, dtype=np.float64)
        prob = np.array([0.95, 0.2, 0.99], dtype=np.float32)
        publish = np.array([0, 3, 4, 5, 5, 5])     # 第 1 窗的结果在第 3 窗末发布（延迟 2 窗）
        rec = first_detection_latencies_published(t, label, tid, prob, 50, 0.9, 1e-4, publish)[0]
        self.assertEqual(rec["detect_window"], 1)
        self.assertEqual(rec["publish_window"], 3)
        self.assertAlmostEqual(rec["latency_ms"], 4 * 50.0 - 60.0)


# ---------------------------------------------------------------------------
# 3. 证据前端与事件来源
# ---------------------------------------------------------------------------


class FrontEndTests(unittest.TestCase):
    def setUp(self):
        self.seq = synthetic_sequence()
        self.source = EventChunkSource(self.seq, CFG, "cpu", torch.float64)
        self.frontend = make_frontend()

    def test_source_counts_match_numpy(self):
        chunk = self.source.chunk(2, 7)
        for t, k in enumerate(range(2, 7)):
            idx = self.seq.window_index(k)
            for pol_index, pol in ((0, 1), (1, 0)):
                sel = idx[self.seq.p[idx] == pol]
                want = np.zeros((H, W))
                np.add.at(want, (self.seq.y[sel], self.seq.x[sel]), 1)
                self.assertEqual(float(np.abs(chunk["counts"][t, 0, pol_index].numpy() - want).max()), 0.0)
            want_t = np.zeros((H, W))
            np.add.at(want_t, (self.seq.y[idx], self.seq.x[idx]), self.seq.label[idx])
            self.assertEqual(float(np.abs(chunk["target_counts"][t, 0, 0].numpy() - want_t).max()), 0.0)

    def test_moment_recursion_matches_direct_sum_across_chunks(self):
        state = self.frontend.init_state(1, H, W, "cpu", torch.float64)
        for start, end in ((0, 3), (3, 8), (8, WINDOWS)):
            state, _, _, _ = self.frontend.run_chunk(state, self.source.chunk(start, end))
        t_end = 50.0 * WINDOWS
        for j, tau in enumerate(TAUS):
            for pol_index, pol in ((0, 1), (1, 0)):
                sel = self.seq.p == pol
                age = t_end - self.seq.t[sel].astype(np.float64)
                a = np.zeros((H, W)); b = np.zeros((H, W))
                np.add.at(a, (self.seq.y[sel], self.seq.x[sel]), np.exp(-age / tau))
                np.add.at(b, (self.seq.y[sel], self.seq.x[sel]), np.exp(-age / tau) * age)
                self.assertLess(np.abs(state["A"][0, 2 * j + pol_index].numpy() - a).max(), 1e-9)
                self.assertLess(np.abs(state["B"][0, 2 * j + pol_index].numpy() - b).max(), 1e-7)

    def test_background_uses_only_past_windows(self):
        chunk = self.source.chunk(0, WINDOWS)
        state = self.frontend.init_state(1, H, W, "cpu", torch.float64)
        _, _, mu_a, _ = self.frontend.run_chunk(state, chunk)
        altered = dict(chunk)
        altered["counts"] = chunk["counts"].clone()
        altered["counts"][6] += 5.0                                   # 只改第 6 窗
        _, _, mu_b, _ = self.frontend.run_chunk(self.frontend.init_state(1, H, W, "cpu", torch.float64), altered)
        self.assertEqual(max_error(mu_a[:7], mu_b[:7]), 0.0)
        self.assertGreater(max_error(mu_a[7:], mu_b[7:]), 0.0)
        self.assertGreaterEqual(float(mu_a.min()), self.frontend.bg_floor)

    def test_dipole_points_from_off_to_on_side(self):
        # 目标向右运动：ON 在左（后沿）、OFF 在右（前沿）-> ON 质心 - OFF 质心 的 x 分量为负
        frontend = make_frontend()
        A_pos = torch.zeros(1, 1, 9, 9, dtype=torch.float64)
        A_neg = torch.zeros_like(A_pos)
        A_pos[0, 0, 4, 2:4] = 3.0
        A_neg[0, 0, 4, 5:7] = 3.0
        d = frontend.dipole(A_pos, A_neg)
        self.assertLess(float(d[0, 0, 4, 4]), -0.5)
        self.assertLess(abs(float(d[0, 1, 4, 4])), 1e-9)
        self.assertEqual(float(frontend.dipole(A_pos, torch.zeros_like(A_neg)).abs().max()), 0.0)

    def test_feature_group_selection_and_constant_background(self):
        # 消融开关：features 选择特征组（通道数与顺序随之变化，其余通道的数值不变）；bg_mode=constant 关掉背景归一化
        chunk = self.source.chunk(0, 4)
        full = make_frontend()
        state = full.init_state(1, H, W, "cpu", torch.float64)
        _, feats_full, mu_full, _ = full.run_chunk(state, chunk)
        names_full = full.feature_names()
        for subset in (("count",), ("count", "dipole"), ("ratio", "age"), FEATURE_GROUPS):
            fe = EvidenceFrontEnd(TAUS, 50.0, [100.0], dipole_radius=2, bg_smooth_radius=2, features=subset).double()
            _, feats, mu, _ = fe.run_chunk(fe.init_state(1, H, W, "cpu", torch.float64), chunk)
            names = fe.feature_names()
            self.assertEqual(int(feats.shape[2]), fe.n_features)
            self.assertEqual(len(names), fe.n_features)
            self.assertEqual(max_error(mu, mu_full), 0.0)                    # 背景估计与特征选择无关
            for i, name in enumerate(names):                                  # 选中的通道数值与完整版逐位相同
                self.assertEqual(max_error(feats[:, :, i], feats_full[:, :, names_full.index(name)]), 0.0, name)
        const = EvidenceFrontEnd(TAUS, 50.0, [100.0], dipole_radius=2, bg_smooth_radius=2,
                                 bg_prior=0.02, bg_mode="constant").double()
        _, _, mu_const, _ = const.run_chunk(const.init_state(1, H, W, "cpu", torch.float64), chunk)
        self.assertEqual(float(mu_const.min()), 0.02)
        self.assertEqual(float(mu_const.max()), 0.02)
        self.assertRaises(ValueError, EvidenceFrontEnd, TAUS, 50.0, [], features=("count", "unknown"))

    def test_features_are_zero_without_events_and_finite(self):
        state = self.frontend.init_state(1, H, W, "cpu", torch.float64)
        empty = {key: torch.zeros_like(value) for key, value in self.source.chunk(0, 2).items()
                 if key in ("counts", "moment_a", "moment_b")}
        _, feats, _, _ = self.frontend.run_chunk(state, empty)
        self.assertEqual(float(feats.abs().max()), 0.0)
        _, feats, _, _ = self.frontend.run_chunk(state, self.source.chunk(0, WINDOWS))
        self.assertTrue(bool(torch.isfinite(feats).all()))
        self.assertEqual(int(feats.shape[2]), self.frontend.n_features)


# ---------------------------------------------------------------------------
# 4. 网络、训练与推理流程
# ---------------------------------------------------------------------------


def make_model(frontend, seed=4):
    torch.manual_seed(seed)
    model = EvidenceSNN(frontend.n_features, channels=(4, 4, 4, 4), head_hidden=6).double()
    for block in model.blocks():
        block.neuron.v_threshold = 0.05                       # 小网络调低阈值，保证各层都有发放
    return model


class ModelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.seq = synthetic_sequence(seed=1)
        self.frontend = make_frontend()
        self.model = make_model(self.frontend)
        source = EventChunkSource(self.seq, CFG, "cpu", torch.float64)
        state = self.frontend.init_state(1, H, W, "cpu", torch.float64)
        _, self.feats, _, _ = self.frontend.run_chunk(state, source.chunk(0, WINDOWS))

    def test_forward_chunk_matches_per_window_forward(self):
        mark_c, log_g_c, states_c, _ = self.model.forward_chunk(self.feats, None)
        states = None
        for k in range(WINDOWS):
            mark, log_g, states, _ = self.model(self.feats[k], states)
            self.assertLess(max_error(mark, mark_c[k]), 1e-10)
            self.assertLess(max_error(log_g, log_g_c[k]), 1e-10)
        for a, b in zip(states, states_c):
            self.assertLess(max_error(a, b), 1e-10)
        self.assertLessEqual(float(log_g_c.detach().max()), self.model.log_g_max)

    def test_train_sequence_updates_all_parameters(self):
        from train_stream_v2 import train_sequence
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in self.model.parameters()]
        out = train_sequence(self.model, self.frontend, self.seq, optimizer, CFG, torch.device("cpu"),
                             np.random.RandomState(0))
        self.assertEqual(out["missing_grads"], [])
        self.assertTrue(math.isfinite(out["loss_sum"]) and out["mark_sum"] > 0 and out["intensity_sum"] != 0)
        changed = sum(int(max_error(a, b) > 0) for a, b in zip(before, self.model.parameters()))
        self.assertEqual(changed, len(before))

    def test_run_sequence_readouts_cover_all_events(self):
        from train_stream_v2 import run_sequence
        cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.8)
        self.model.eval()
        with torch.no_grad():
            probs, extra, confusion = run_sequence(self.model, self.frontend, cusum, self.seq, CFG,
                                                   torch.device("cpu"), "carry")
        self.assertEqual(sorted(probs), ["fused_d1", "fused_d3", "net"])
        self.assertEqual(sorted(extra), ["evidence_d1", "evidence_d3", "logit_net", "publish_d1", "publish_d3"])
        self.assertEqual(extra["publish_d3"].tolist(), [min(k + 3, WINDOWS - 1) for k in range(WINDOWS)])
        fused = 1.0 / (1.0 + np.exp(-(extra["logit_net"] + extra["evidence_d1"])))
        self.assertLess(float(np.abs(fused - probs["fused_d1"]).max()), 1e-5)
        for name, p in probs.items():
            self.assertEqual(p.shape[0], self.seq.n_events)
            self.assertTrue(np.all((p >= 0) & (p <= 1)), name)
        self.assertEqual(int(confusion[:, 3].sum()), int(self.seq.label.sum()))
        # 逐窗推理（eval_chunk=1）与片段推理结果相同
        cfg_step = dict(CFG, eval_chunk=1)
        with torch.no_grad():
            probs_step, _, _ = run_sequence(self.model, self.frontend, cusum, self.seq, cfg_step,
                                            torch.device("cpu"), "carry")
        for name in probs:
            self.assertLess(float(np.abs(probs[name] - probs_step[name]).max()), 1e-6, name)


if __name__ == "__main__":
    unittest.main()
