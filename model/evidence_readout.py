"""V2/V3 判决层在冻结 V5 骨干上的接线（v6-2：检验"之前的方案在 V5 上是否还有效"）。

来源（除接口外逐式照搬，不改公式与常数）：
    EvidenceHeads   v3 分支 model/evidence_snn.py 的两个逐像素头：读 dec1 的 u_pre（V5 读出头读的同一张图），
                    1x1 卷积 -> mark（本窗该像素上的事件是目标的 logit）与 log_g（本窗目标强度，log 每像素每窗目标事件数）；
                    初值：最后一层权重 x0.1，偏置取先验；log_g 用平滑上界 m - softplus(m - raw)
    BackgroundRate  v3 分支 dataset/stream_features.EvidenceFrontEnd 的背景强度 mu0（只用过去的计数）：
                    mu = (S + n0*prior) / (W + n0)，快速项 500 ms 取 15x15 邻域平均，慢速项 5 s 逐像素，
                    mu0 = max(floor, 慢速, 快速的邻域平均)。只给判决层当背景基准，不作为网络输入（V5 的输入不变）
    EvidenceRun     v3 分支 train_stream_v2.run_sequence 里判决层的逐窗调用顺序：
                    mu0 用第 k 窗之前的计数 -> DriftCUSUM.step(本窗计数, mu0, 上一窗 log_g) -> 背景更新 ->
                    TubeReadout（V2-1 的 fused_d）与 PublishUnits（V3 的 pub，锚定证据链）推进
读出：
    net       sigmoid(mark)，零等待
    fused_d   sigmoid(mark + w * F(d))，F 为之后 d 窗沿 49 条速度管道的证据（V2-1）
    pub       V3 双边界发布（年龄 0..D 择机发布，期限强制），概率按 utils/publish_eval.published_probability 换算
位置级告警不接（只报四个指标）。本模块没有判决层的可学习参数；只有两个头可训练。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.stream_windows import refill_by_index
from model.evidence_neuron import DriftCUSUM, TubeReadout, velocity_grid
from model.publish_readout import PublishRule, PublishUnits

DEC1 = 6            # LAYER_NAMES 中 dec1 的位置（u_pre 列表下标）


class EvidenceHeads(nn.Module):
    """u7 [N,C1,H,W] -> (mark [N,1,H,W], log_g [N,1,H,W])，与 v3 的 EvidenceSNN 头相同。"""

    def __init__(self, c1, hidden=16, mark_prior=0.03, intensity_prior=2e-4, log_g_max=8.0):
        super(EvidenceHeads, self).__init__()
        if not (0.0 < float(mark_prior) < 1.0 and float(intensity_prior) > 0.0):
            raise ValueError("mark_prior 必须在 (0,1) 内，intensity_prior 必须为正")
        if int(hidden) > 0:
            last = nn.Conv2d(int(hidden), 2, 1, bias=True)
            self.head = nn.Sequential(nn.Conv2d(int(c1), int(hidden), 1, bias=True), nn.ReLU(), last)
        else:
            last = nn.Conv2d(int(c1), 2, 1, bias=True)
            self.head = last
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.copy_(torch.tensor([math.log(mark_prior / (1.0 - mark_prior)), math.log(intensity_prior)]))
        self.log_g_max = float(log_g_max)

    def forward(self, u7):
        out = self.head(u7)
        mark = out[:, 0:1]
        log_g = self.log_g_max - F.softplus(self.log_g_max - out[:, 1:2])
        return mark, log_g


class BackgroundRate(object):
    """只用过去计数估计的背景强度 mu0（每像素每窗期望事件数），公式与 v3 的 EvidenceFrontEnd.background 相同。"""

    def __init__(self, window_ms, fast_ms=500.0, slow_ms=5000.0, prior=0.01, prior_windows=2.0, floor=1e-3,
                 smooth_radius=7):
        if min(float(fast_ms), float(slow_ms), float(prior), float(prior_windows), float(floor)) <= 0:
            raise ValueError("背景的时间常数、先验、伪窗数、下限都必须为正")
        self.beta_fast = math.exp(-float(window_ms) / float(fast_ms))
        self.beta_slow = math.exp(-float(window_ms) / float(slow_ms))
        self.prior, self.prior_windows = float(prior), float(prior_windows)
        self.floor, self.smooth_radius = float(floor), int(smooth_radius)

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        zeros = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
        return {"S_fast": zeros, "S_slow": zeros, "W_fast": 0.0, "W_slow": 0.0}

    def background(self, state):
        n0, prior = self.prior_windows, self.prior
        fast = (state["S_fast"] + n0 * prior) / (state["W_fast"] + n0)
        slow = (state["S_slow"] + n0 * prior) / (state["W_slow"] + n0)
        r = self.smooth_radius
        if r > 0:
            fast = F.avg_pool2d(fast, 2 * r + 1, stride=1, padding=r, count_include_pad=False)
        return torch.clamp(torch.maximum(fast, slow), min=self.floor)

    def update(self, state, total):
        return {"S_fast": self.beta_fast * state["S_fast"] + total,
                "S_slow": self.beta_slow * state["S_slow"] + total,
                "W_fast": self.beta_fast * state["W_fast"] + 1.0,
                "W_slow": self.beta_slow * state["W_slow"] + 1.0}


def pixel_counts(y, x, batch_index, batch, height, width, dtype, weight=None):
    """事件在画布上的计数 [B,1,H,W]（index_put_ 累加，确定性模式可用）。weight 给出时按权累加（目标计数用标签）。"""
    out = torch.zeros(batch * height * width, dtype=dtype, device=y.device)
    if y.numel():
        values = torch.ones_like(y, dtype=dtype) if weight is None else weight.to(dtype)
        out.index_put_(((batch_index * height + y) * width + x,), values, accumulate=True)
    return out.view(batch, 1, height, width)


def build_cusum(cfg):
    """与 v3 train_stream_v2.build_cusum 相同（告警复位半径不接，固定 0）。"""
    tau = float(cfg.get("cusum_track_tau_ms", 0.0))
    decay = math.exp(-float(cfg["window_ms"]) / tau) if tau > 0 else 0.0
    return DriftCUSUM(velocity_grid(cfg["cusum_axis_velocities"]), int(cfg["cusum_footprint"]),
                      cfg["cusum_compensator"], cfg.get("cusum_nb_kappa"), cfg.get("cusum_aggregate", "lme"), decay,
                      float(cfg.get("cusum_memory_gain", 1.0)), float(cfg.get("cusum_gate_eps", 0.0)), 0)


def build_background(cfg):
    return BackgroundRate(cfg["window_ms"], cfg["bg_fast_ms"], cfg["bg_slow_ms"], cfg["bg_prior"],
                          cfg["bg_prior_windows"], cfg["bg_floor"], cfg["bg_smooth_radius"])


def publish_theta(cfg):
    """最终阈值 θ（logit）= logit(threshold)，与其他读出同一工作点。"""
    p = float(cfg["threshold"])
    return math.log(p / (1.0 - p))


def build_rule(cfg):
    return PublishRule(publish_theta(cfg), float(cfg["publish_upper"]), float(cfg["publish_lower"]),
                       int(cfg["publish_deadline"]), cfg["publish_collapse"], cfg.get("publish_gate"),
                       float(cfg["fusion_weight"]))


def published_probability(z, label, theta, threshold):
    """与 v3 utils/publish_eval.published_probability 相同：sigmoid(z - θ + logit(threshold))，再夹到发布标签一侧。"""
    thr = np.float32(threshold)
    shift = math.log(float(threshold) / (1.0 - float(threshold))) - float(theta)
    prob = (1.0 / (1.0 + np.exp(-(np.asarray(z, np.float64) + shift)))).astype(np.float32)
    below = np.nextafter(thr, np.float32(0.0))
    return np.where(label, np.maximum(prob, thr), np.minimum(prob, below)).astype(np.float32)


class EvidenceRun(object):
    """一条序列上的判决层（B = 1）：每窗 step，序列末 outputs。"""

    def __init__(self, cfg, height, width, device, dtype=torch.float32):
        self.cfg = cfg
        self.height, self.width = int(height), int(width)
        self.delays = [int(d) for d in cfg["readout_delays"]]
        self.weight = float(cfg["fusion_weight"])
        self.cusum = build_cusum(cfg)
        self.background = build_background(cfg)
        self.rule = build_rule(cfg)
        self.anchor = bool(cfg["publish_anchor"])
        self.c_state = self.cusum.init_state(1, self.height, self.width, device, dtype)
        self.bg_state = self.background.init_state(1, self.height, self.width, device, dtype)
        self.readout = TubeReadout(self.cusum, self.delays)
        self.units = PublishUnits(self.cusum, self.rule, anchor=self.anchor)
        self.prev_log_g = None
        self.window_info = {}
        self.parts = {name: ([], []) for name in ["net"] + ["fused_d%d" % d for d in self.delays]}
        self.pub = {name: ([], []) for name in ("z", "label", "age", "reason")}

    def step(self, k, log_g, mark_logits, y, x, idx):
        """第 k 窗。log_g [1,1,H,W] 本窗强度场；mark_logits [n] 本窗事件的 mark logit；y/x [n] 画布坐标；idx 原始下标。"""
        k = int(k)
        dtype = log_g.dtype
        b = torch.zeros_like(y)
        counts = pixel_counts(y, x, b, 1, self.height, self.width, dtype)
        mu0 = self.background.background(self.bg_state)
        self.c_state, _, _ = self.cusum.step(self.c_state, counts, mu0, self.prev_log_g)
        self.bg_state = self.background.update(self.bg_state, counts)
        idx = np.asarray(idx, dtype=np.int64)
        self.window_info[k] = (idx, mark_logits)
        self.parts["net"][0].append(idx)
        self.parts["net"][1].append(torch.sigmoid(mark_logits).float().cpu().numpy())
        self._collect(self.readout.step(self.c_state, k, b, y, x, k))
        support = self.cusum.support_field(counts) if self.anchor else None
        self._collect_pub(self.units.step(self.c_state, k, b, y, x, k, mark_logits, support))
        self.prev_log_g = log_g

    def _collect(self, results):
        for key, delay, scores, _ in results:
            idx, logit = self.window_info[key]
            self.parts["fused_d%d" % delay][0].append(idx)
            self.parts["fused_d%d" % delay][1].append(
                torch.sigmoid(logit + self.weight * scores).float().cpu().numpy())

    def _collect_pub(self, records):
        for key, pos, label, z, _, reason, age in records:
            idx = self.window_info[key][0][pos.cpu().numpy()]
            n = idx.shape[0]
            for name, value in (("z", z.detach().to("cpu", torch.float64).numpy()), ("label", label.cpu().numpy()),
                                ("age", np.full(n, age, np.int64)), ("reason", np.full(n, reason, np.int64))):
                self.pub[name][0].append(idx)
                self.pub[name][1].append(value)

    def outputs(self, n_events):
        """序列结束：flush 后回填为文件顺序。返回 (probs {net, fused_d*, pub}, extra {age_pub, reason_pub, z_pub})。"""
        self._collect(self.readout.flush())
        self._collect_pub(self.units.flush())
        probs = {name: refill_by_index(n_events, *parts) for name, parts in self.parts.items()}
        z = refill_by_index(n_events, *self.pub["z"], dtype=np.float64)
        label = refill_by_index(n_events, *self.pub["label"], dtype=np.uint8).astype(bool)
        probs["pub"] = published_probability(z, label, self.rule.theta, float(self.cfg["threshold"]))
        extra = {"age_pub": refill_by_index(n_events, *self.pub["age"], dtype=np.int64),
                 "reason_pub": refill_by_index(n_events, *self.pub["reason"], dtype=np.int64),
                 "z_pub": z.astype(np.float32)}
        return probs, extra
