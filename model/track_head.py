"""模块三：目标级航迹头 v2（Track Head；Stream-SNN-Innovations.md 13.60～13.61）。

问题：流式逐像素记忆没有"目标"这个单位。证据钉在像素上、起落同速，目标离开后旧位置继续报目标（残影 / 拖尾误报），
      目标弱的几窗断掉（断裂 / 断续与第一拍漏检）。
原理：贝叶斯多目标滤波 = 卡尔曼形状态 + 伯努利存在概率（对标 CNSF 2026、Residual Kalman 2026、MOTIP 2025）；
      可选的一窗固定滞后判决 = RTS 平滑（暂定输出 + 修正）。
接口：只读骨干（E3a）的逐事件 logit z0 与 dec1 特征，不回写骨干（避免输出回写造成的自激；骨干照常逐层并行）。
每窗一次（B 条流并行，每条流 N 个航迹槽）：
    1. 预测   匀速模型：每个坐标轴的状态为 [位置, 速度]，两轴共用 2x2 协方差 P；过程噪声 q 学出来
    2. 量测   证据图 = 每格（cell x cell 像素）事件概率之和，再取 3x3 格之和；候选 = 证据图的局部峰。
              每条航迹在验证门（gate_sigma 倍标准差）内按"证据加权的似然" log(质量) - d^2 / (2S) 选一个峰
              （带幅度信息的最近邻），几条航迹抢同一峰时存在概率高的优先；量测位置 = 概率加权质心，
              量测特征 = 质量、平均概率、事件数、极性、空间扩散、概率加权的平均 dec1 特征
    3. 更新   卡尔曼增益 K = P H^T / (H P H^T + R)；量测噪声 R 由量测特征预测（网络给出"这个量测多可靠"）
    4. 存在   GRU 累积航迹级证据（有无量测、量测特征、归一化新息、不确定度、速度、年龄、连续漏检）-> 存在概率
    5. 出生   所有验证门之外的局部峰（质量 >= birth_floor），附带"到现有航迹当前位置、近期轨迹的距离"，
              由存在监督学会不让拖尾（落在近期轨迹上的峰）长成新目标；同一窗内两次出生至少隔开一个量测邻域
              （3x3 格之外）；没有空槽时替换存在概率最低（< recycle_exist）的航迹
    6. 消亡   存在概率 < death_exist，或连续漏检 > max_coast 窗，或位置离开画布
    7. 修正   最终 logit = z0 + Δ(z0, 事件特征, 极性, 窗内时间, 事件与航迹的关系)；Δ 的末层零初始化，起步与 z0 逐位相同。
              事件与航迹的关系：到最近航迹的归一化距离、沿迹 / 横迹坐标（残影在目标身后）、到近期轨迹的距离、
              航迹的存在概率、速度、本窗有无量测、年龄
    8. 滞后判决（decision_lag = 1 时）：第 k 窗结束后，用 RTS 平滑得到的第 k-1 窗航迹位置、第 k 窗的存在概率与量测、
              事件所在处第 k 窗的证据（持续性）重判第 k-1 窗的事件；零等待的输出照常给出
    9. 目标停时判决（defer_max > 0，v4 系列；existence = ipda）：每个事件即时给暂定输出；所属目标的状态还没被证据确定
              （存在未定 / 那一窗缺量测 / 事件发生时目标还没出生、之后被新生目标回溯认领）的事件挂起，
              目标的存在对数几率 L 越过 +-A（A = log((1 - alpha) / alpha)）、缺口后重新捕获、丢失或等满 defer_max 窗时终判。
              证据修正（defer_revision = evidence，v4e）：z_final = z_now + m x kappa[原因] x (L(终判) - L_ref)，
              L_ref = 事件那一窗的 L（暂定输出已用过的证据）；孤儿事件当时没有目标，L_ref = 学出的新生目标先验对数几率 l0。
              m 为校准的归属概率：目标在等待期间新增的存在证据按归属分给事件，证据不变则不修正，不属于目标的事件不被修正
context = "none" 是对照组：航迹照常运行，但修正项看不到航迹（检验增益来自航迹，而不是"再训一个读出头"）。
训练监督（标签里的 target_id）：逐窗把航迹与标注目标一一匹配（距离 <= max(match_min_px, 2.5 x 目标扩散)），
    匹配上的航迹存在 = 1、位置回归到目标质心；其余存在 = 0。
实现注意：只用 torch 1.9 可用、确定性模式下允许的算子（累加用 index_put_(accumulate=True)，取值用高级索引，不用 gather /
    scatter_add_）；距离用 sqrt(平方和 + eps)，避免重合点处梯度为 NaN。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAS_SUMS = 8          # 每格累加量：p, p^2, 1, p*pol, p*x, p*y, p*x^2, p*y^2（其后为 p*feat）
BIG = 1e6
TRACK_KEYS = ("active", "pos", "vel", "P", "h", "ex", "pd", "beta", "age", "miss", "has", "hist", "hist_ok")
SURV_MAX_LOGIT = math.log(0.999 / 0.001)   # survival p_S <= 0.999: existence log-odds stay bounded
PEND_KEYS = ("b", "x", "y", "p", "t_local", "z0", "feat", "ctx", "key")
CONTEXTS = ("track", "none")
EXISTENCE = ("gru", "ipda")
CTX_DIM, LAG_DIM = 11, 7
# Object-level deferred decision (defer_max > 0): entries of events whose object state at their own window is not yet
# settled by evidence. Status at release: 0 confirmed, 1 rejected, 2 reacquired, 3 lost, 4 forced (deadline).
# Reason for waiting: 0 existence undecided, 1 object coasting (no reliable measurement), 2 orphan claimed by a newborn.
DEFER_KEYS = ("b", "x", "y", "p", "t_local", "z0", "feat", "ctx", "key", "znow", "lc", "lag", "slot", "age_off",
              "reason", "L_then", "m_then", "beta_then", "fx", "fy", "miss_then")
DEFER_LONG = ("b", "x", "y", "lag", "slot", "age_off", "reason")
FIN_DIM = 15
REV_DIM = 13          # gated revision: status 5 + reason 3 + L_now + L_then + lag + beta_then + z_now
REVISIONS = ("free", "gated", "evidence")
STATUS_NAMES = ("confirmed", "rejected", "reacquired", "lost", "forced")


def _q_cat(a, b):
    if a is None or int(a["b"].shape[0]) == 0:
        return b
    if b is None or int(b["b"].shape[0]) == 0:
        return a
    return {k: torch.cat([a[k], b[k]]) for k in a}


def _q_take(q, mask):
    return None if q is None else {k: v[mask] for k, v in q.items()}


def _logit(p):
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))


def _zero_last(mlp):
    nn.init.zeros_(mlp[2].weight)
    nn.init.zeros_(mlp[2].bias)
    return mlp


def _dist(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1) + 1e-6)


class TrackHead(nn.Module):
    """目标级航迹头。forward_chunk(events, z0, feat, state, steps, keep) -> (z, state, aux)。"""

    def __init__(self, feat_dim, height, width, cell=4, slots=10, hidden=32, history=21, gate_sigma=3.0,
                 gate_max_px=24.0, gate_floor=0.5, birth_floor=1.0, max_births=4, death_exist=0.02,
                 max_coast=20, match_min_px=8.0, corr_hidden=32, r_min=0.25, recycle_exist=0.5,
                 decision_lag=0, context="track", existence="gru", defer_max=0, defer_alpha=0.05,
                 defer_revision="free", pd_max=1.0, provisional_gate=False):
        super(TrackHead, self).__init__()
        if context not in CONTEXTS:
            raise ValueError("context 必须是 %s 之一" % (CONTEXTS,))
        if existence not in EXISTENCE:
            raise ValueError("existence 必须是 %s 之一" % (EXISTENCE,))
        if int(decision_lag) not in (0, 1):
            raise ValueError("decision_lag 只支持 0（零等待）或 1（一窗固定滞后）")
        self.feat_dim = int(feat_dim)
        self.height, self.width, self.cell = int(height), int(width), int(cell)
        self.gh, self.gw = -(-self.height // self.cell), -(-self.width // self.cell)
        self.slots, self.hidden, self.history = int(slots), int(hidden), int(history)
        self.gate_sigma, self.gate_max_px = float(gate_sigma), float(gate_max_px)
        self.gate_floor, self.birth_floor = float(gate_floor), float(birth_floor)
        self.max_births, self.death_exist, self.max_coast = int(max_births), float(death_exist), float(max_coast)
        self.match_min_px, self.r_min, self.recycle_exist = float(match_min_px), float(r_min), float(recycle_exist)
        self.decision_lag, self.context, self.existence = int(decision_lag), str(context), str(existence)
        self.meas_dim = 5 + self.feat_dim
        self.u_dim = self.meas_dim + 11
        self.log_q = nn.Parameter(torch.tensor(math.log(0.25)))       # 过程噪声（像素/窗^2 的加速度方差）
        self.log_v0 = nn.Parameter(torch.tensor(math.log(4.0)))       # 出生时的速度方差（(像素/窗)^2）
        self.r_net = _mlp(self.meas_dim, 16, 1)                        # 量测特征 -> 量测噪声 R（像素^2）
        nn.init.zeros_(self.r_net[2].weight)
        nn.init.constant_(self.r_net[2].bias, math.log(math.expm1(4.0)))   # 初始 R = 4 + r_min
        self.gru = nn.GRUCell(self.u_dim, self.hidden)
        self.exist = nn.Linear(self.hidden, 1)
        if self.existence == "ipda":
            # IPDA (Musicki 1994): existence and association updated together. Learned parts: detection probability
            # P_D from track memory + clutter at the predicted position; measurement likelihood ratio (target vs
            # clutter) from measurement features, normalized innovation and clutter at the measurement; survival.
            self.pd_head = nn.Linear(self.hidden + 1, 1)
            nn.init.zeros_(self.pd_head.weight)
            nn.init.constant_(self.pd_head.bias, math.log(4.0))                 # P_D = 0.8
            self.lr_net = _mlp(self.meas_dim + 2, 16, 1)
            nn.init.zeros_(self.lr_net[2].weight)
            nn.init.constant_(self.lr_net[2].bias, 2.0)                         # likelihood ratio e^2
            self.surv_logit = nn.Parameter(torch.tensor(math.log(0.995 / 0.005)))
        # ipda: the clutter term is a density (clutter count over the 3x3-cell area), so it starts at -log(area)
        none0 = -math.log(9.0 * self.cell * self.cell) if self.existence == "ipda" else 0.0
        self.none_logit = nn.Parameter(torch.tensor(none0))
        base_in = 1 + self.feat_dim + 2 + CTX_DIM
        self.corr = _zero_last(_mlp(base_in, int(corr_hidden), 1))
        if self.decision_lag:
            self.corr_lag = _zero_last(_mlp(base_in + LAG_DIM, int(corr_hidden), 1))
        # 目标级延迟决策：等待单位是目标；判决依据是校准的存在后验越过 +-A（A = log((1 - alpha) / alpha)，
        # alpha = 0 表示 A = 无穷，即所有挂起事件都等满 defer_max 窗，作固定等待的对照）
        self.defer_max, self.defer_revision = int(defer_max), str(defer_revision)
        # 检测概率上限（v4c）：P_D = pd_max * sigmoid(.)。骨干从不"必然检出"目标，一次漏检的代价至多 log(1 - pd_max)；
        # 否则 P_D 在几乎总能检出的数据上学到饱和（v3：置信航迹中位 1.00），第一次漏检就让存在概率塌掉
        self.pd_max = float(pd_max)
        if not 0.0 < self.pd_max <= 1.0:
            raise ValueError("pd_max 必须在 (0, 1] 内")
        if self.defer_max:
            if self.existence != "ipda":
                raise ValueError("延迟决策需要 existence = ipda（判决依据是校准的存在后验）")
            if self.decision_lag:
                raise ValueError("defer_max 与 decision_lag 只能选一个")
            alpha = float(defer_alpha)
            if not 0.0 <= alpha < 0.5:
                raise ValueError("defer_alpha 必须在 [0, 0.5) 内")
            self.defer_A = math.log((1.0 - alpha) / alpha) if alpha > 0 else float("inf")
            if defer_revision not in REVISIONS:
                raise ValueError("defer_revision 必须是 %s 之一" % (REVISIONS,))
            self.defer_revision = str(defer_revision)
            if self.defer_revision in ("gated", "evidence"):
                # 门控修正（v4b）：修正量 = 事件的归属概率 x 目标层的修正。归属概率是校准过的几何模型
                # （到目标的归一化距离、目标位置不确定度、本地杂波），单调性由 softplus 保证，只由归属标签监督
                self.mem_theta = nn.Parameter(torch.full((3,), math.log(math.e - 1.0)))      # 三个权重初始 1
                self.mem_bias = nn.Parameter(torch.tensor(math.log(9.0 * self.cell * self.cell)))
                if self.defer_revision == "gated":
                    self.corr_rev = _zero_last(_mlp(REV_DIM, int(corr_hidden), 1))
                else:
                    # 证据修正（v4e）：修正量 = 归属概率 x kappa[等待原因] x 暂定输出之后目标新增的存在证据。
                    # 学 3 个非负系数（softplus，初始 1；理论值：存在后验校准时为 1）和孤儿的参照 l0
                    # （新生目标真实存在的先验对数几率，初始 0）：证据不变就不修正，不属于目标的事件不被修正
                    self.kappa_raw = nn.Parameter(torch.full((3,), math.log(math.e - 1.0)))
                    self.orphan_ref = nn.Parameter(torch.zeros(()))
            else:
                self.corr_final = _zero_last(_mlp(base_in + FIN_DIM, int(corr_hidden), 1))
        self.has_delayed = bool(self.decision_lag or self.defer_max)
        self.gated = bool(self.defer_max) and self.defer_revision in ("gated", "evidence")
        # 暂定修正也按"存在 x 归属"分解（v4d）：零等待的修正量 = 事件对最近目标的归属概率 x 修正项；
        # 没有目标的事件不被修正（判决层只有一条规则：修正来自目标层的证据，按归属分给事件）
        self.provisional_gate = bool(provisional_gate)
        if self.provisional_gate and not self.gated:
            raise ValueError("provisional_gate 需要归属模型（defer_revision = gated 或 evidence）")
        self.collect_calibration = False                              # 评估时收集存在概率的校准样本
        gy = torch.arange(self.gh, dtype=torch.float32).repeat_interleave(self.gw)
        gx = torch.arange(self.gw, dtype=torch.float32).repeat(self.gh)
        half = (self.cell - 1) / 2.0
        self.register_buffer("centers", torch.stack([gx * self.cell + half, gy * self.cell + half], 1),
                             persistent=False)

    # ------------------------------------------------------------------ 状态
    def init_state(self, batch, device):
        B, N, H, D = int(batch), self.slots, self.history, self.hidden
        zeros = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        flag = lambda *shape: torch.zeros(*shape, dtype=torch.bool, device=device)  # noqa: E731
        return {"active": flag(B, N), "pos": zeros(B, N, 2), "vel": zeros(B, N, 2), "P": zeros(B, N, 2, 2),
                "h": zeros(B, N, D), "ex": zeros(B, N), "pd": zeros(B, N), "beta": zeros(B, N), "age": zeros(B, N),
                "miss": zeros(B, N), "has": flag(B, N),
                "hist": zeros(B, N, H, 2), "hist_ok": flag(B, N, H), "pending": None, "defer": None, "orphan": None}

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        out = {}
        for key, value in state.items():
            if value is None:
                out[key] = None
            elif isinstance(value, dict):
                out[key] = {k: v.detach() for k, v in value.items()}
            else:
                out[key] = value.detach()
        return out

    @staticmethod
    def _reset(state, mask):
        """mask [B] bool：这些流开始新录像，航迹全部清空，上一窗待判的事件不再重判（保留零等待输出）。"""
        out = {}
        for key in TRACK_KEYS:
            value = state[key]
            view = mask.view((-1,) + (1,) * (value.dim() - 1))
            out[key] = torch.where(view, torch.zeros_like(value), value)
        for key in ("pending", "defer", "orphan"):                    # 新录像开始：该流待判的事件保留暂定输出
            q = state.get(key)
            out[key] = None if q is None else {k: v[~mask[q["b"]]] for k, v in q.items()}
        return out

    # ------------------------------------------------------------------ 量测
    def _cells(self, ex, ey):
        return (torch.div(ey, self.cell, rounding_mode="floor") * self.gw
                + torch.div(ex, self.cell, rounding_mode="floor"))

    def _maps(self, eb, ex, ey, prob, pol, feat, batch):
        """每格累加量 -> (3x3 格之和 [B, C, G], 每格自身的质量 [B, G])。"""
        C, G = MEAS_SUMS + self.feat_dim, self.gh * self.gw
        cell = self._cells(ex, ey)
        xf, yf = ex.to(prob.dtype), ey.to(prob.dtype)
        cols = [prob, prob * prob, torch.ones_like(prob), prob * pol, prob * xf, prob * yf, prob * xf * xf,
                prob * yf * yf]
        vals = torch.cat([torch.stack(cols, 1), prob[:, None] * feat], 1)            # [Ne, C]
        chans = torch.arange(C, device=prob.device)
        flat = (eb[:, None] * C + chans[None, :]) * G + cell[:, None]
        buf = torch.zeros(batch * C * G, dtype=prob.dtype, device=prob.device)
        buf = buf.index_put((flat.reshape(-1),), vals.reshape(-1), accumulate=True)
        raw = buf.view(batch, C, self.gh, self.gw)
        sums = F.avg_pool2d(raw, 3, stride=1, padding=1, count_include_pad=True) * 9.0
        return sums.view(batch, C, G), raw[:, 0].reshape(batch, G)

    def _measure(self, s):
        """累加量 [..., C] -> (量测特征 [..., meas_dim], 质心 [..., 2])。"""
        mass = s[..., 0].clamp(min=1e-6)
        mean_p = s[..., 1] / mass
        on = s[..., 3] / mass
        cx, cy = s[..., 4] / mass, s[..., 5] / mass
        var = (s[..., 6] / mass - cx * cx) + (s[..., 7] / mass - cy * cy)
        spread = torch.sqrt(var.clamp(min=0.0) + 1e-6) / self.cell
        mfeat = s[..., MEAS_SUMS:] / mass[..., None]
        f = torch.cat([torch.log1p(s[..., 0].clamp(min=0.0))[..., None], mean_p[..., None],
                       torch.log1p(s[..., 2].clamp(min=0.0))[..., None], on[..., None], spread[..., None], mfeat], -1)
        return f, torch.stack([cx, cy], -1)

    def _noise(self, f):
        return F.softplus(self.r_net(f).squeeze(-1)) + self.r_min

    # ------------------------------------------------------------------ 一窗
    def forward_window(self, state, ev, reset_mask=None):
        """state：init_state 的字典；ev：本窗事件 b/x/y/p/t_local/z0/feat（滞后判决时另有 key）。

        返回 (z [Ne] 零等待输出, 新状态, info)；info["delayed"] 为 None 或 (上一窗事件的滞后输出, 它们的 key)。
        """
        if reset_mask is not None:
            state = self._reset(state, reset_mask)
        defer_q, orphan_q = state.get("defer"), state.get("orphan")
        if self.defer_max:                                             # 上一窗以前的条目各多等了一窗
            if defer_q is not None:
                defer_q = dict(defer_q, lag=defer_q["lag"] + 1)
            if orphan_q is not None:
                orphan_q = _q_take(dict(orphan_q, lag=orphan_q["lag"] + 1), orphan_q["lag"] + 1 < self.defer_max)
        act = state["active"]
        B, N = act.shape
        dev, cell = act.device, float(self.cell)
        z0, feat = ev["z0"], ev["feat"]
        dtype = z0.dtype
        prob = torch.sigmoid(z0)
        idx_n = torch.arange(N, device=dev)
        pos_prev, vel_prev, P_prev = state["pos"], state["vel"], state["P"]

        # 1. 预测（匀速，离散白噪声加速度模型，dt = 1 窗）
        q = torch.exp(self.log_q)
        P00 = P_prev[..., 0, 0] + 2.0 * P_prev[..., 0, 1] + P_prev[..., 1, 1] + 0.25 * q
        P01 = P_prev[..., 0, 1] + P_prev[..., 1, 1] + 0.5 * q
        P11 = P_prev[..., 1, 1] + q
        Ppred = torch.stack([torch.stack([P00, P01], -1), torch.stack([P01, P11], -1)], -2)
        actv, actm = act[..., None], act[..., None, None]
        pos_pred = torch.where(actv, pos_prev + vel_prev, pos_prev)
        vel_pred = vel_prev
        Ppred = torch.where(actm, Ppred, P_prev)

        # 2. 量测：验证门内的局部峰，按证据加权的似然选
        sums, own = self._maps(ev["b"], ev["x"], ev["y"], prob, ev["p"].to(dtype), feat, B)   # [B, C, G]
        mass = sums[:, 0]
        # 局部峰：3x3 格内的最大值。一个目标落在 2x2 格内时四个格的 3x3 和相等：按格自身的质量取事件真正所在的格，
        # 仍相同时按格号取一个
        G = mass.shape[1]
        rank = 1e-6 * (1.0 - torch.arange(G, device=dev, dtype=dtype) / float(G))
        mass_tb = mass + 1e-3 * own + rank * (1.0 + mass)
        peak_map = F.max_pool2d(mass_tb.view(B, 1, self.gh, self.gw), 3, stride=1, padding=1).view(B, -1)
        is_peak = (mass_tb >= peak_map) & (mass >= self.gate_floor)
        d2 = ((self.centers[None, None] - pos_pred[:, :, None]) ** 2).sum(-1)             # [B, N, G]
        spred = Ppred[..., 0, 0] + cell * cell
        gate_r2 = (self.gate_sigma ** 2 * spred).clamp(max=self.gate_max_px ** 2)
        in_gate = (d2 <= gate_r2[..., None]) & actv
        logm = torch.log(mass.clamp(min=1e-6))
        score = torch.where(in_gate & is_peak[:, None], logm[:, None] - 0.5 * d2 / spred[..., None],
                            torch.full_like(d2, -1e9))
        best, cidx = score.max(-1)
        has = act & (best > -1e8)
        ex_prev = state["ex"]
        same = (cidx[:, :, None] == cidx[:, None, :]) & has[:, :, None] & has[:, None, :]
        better = (ex_prev[:, None, :] > ex_prev[:, :, None]) | (
            (ex_prev[:, None, :] == ex_prev[:, :, None]) & (idx_n[None, None, :] < idx_n[None, :, None]))
        has = has & ~(same & better).any(-1)
        bidx = torch.arange(B, device=dev)[:, None].expand(B, N)
        f, zpos = self._measure(sums.permute(0, 2, 1)[bidx, cidx])

        # 3. 卡尔曼更新
        R = self._noise(f)
        S = Ppred[..., 0, 0] + R
        K = Ppred[..., :, 0] / S[..., None]                                              # [B, N, 2]：位置、速度增益
        innov = zpos - pos_pred
        d2n = (innov ** 2).sum(-1) / S
        P_upd = Ppred - K[..., :, None] * Ppred[..., 0, :][..., None, :]
        clutter = None
        if self.existence == "ipda":
            # 杂波量 = 3x3 格内骨干判为背景的事件数之和 sum(1 - p)；不用任何数据集常数
            clutter = (sums[:, 2] - sums[:, 0]).clamp(min=0.0)                            # [B, G]
            px = pos_pred[..., 0].detach().round().clamp(0, self.width - 1).long()
            py = pos_pred[..., 1].detach().round().clamp(0, self.height - 1).long()
            c_pred = clutter[bidx, self._cells(px, py)]
            c_meas = clutter[bidx, cidx]
            pd_logit = self.pd_head(torch.cat([state["h"], torch.log1p(c_pred)[..., None]], -1)).squeeze(-1)
            log_lr = self.lr_net(torch.cat([f, torch.log1p(d2n)[..., None], torch.log1p(c_meas)[..., None]],
                                           -1)).squeeze(-1)
            if self.pd_max < 1.0:
                log_pd = math.log(self.pd_max) + F.logsigmoid(pd_logit)
                log_miss = torch.logaddexp(torch.full_like(pd_logit, math.log(1.0 - self.pd_max)),
                                           math.log(self.pd_max) + F.logsigmoid(-pd_logit))   # log(1 - P_D)
            else:
                log_pd, log_miss = F.logsigmoid(pd_logit), F.logsigmoid(-pd_logit)
            log_det = torch.logaddexp(log_miss, log_pd + log_lr)                          # log(1 - P_D + P_D l)
            beta = torch.where(has, torch.exp(log_pd + log_lr - log_det), torch.zeros_like(log_det))
            bv, bm = beta[..., None], beta[..., None, None]
            pos = pos_pred + bv * K[..., 0:1] * innov                                     # PDAF：按关联概率更新
            vel = vel_pred + bv * K[..., 1:2] * innov
            spread = (beta * (1.0 - beta) * (innov ** 2).mean(-1))[..., None, None] * (K[..., :, None] * K[..., None, :])
            P = bm * P_upd + (1.0 - bm) * Ppred + spread
            pd_now = torch.exp(log_pd) if self.pd_max < 1.0 else torch.sigmoid(pd_logit)
        else:
            hv = has[..., None]
            pos = torch.where(hv, pos_pred + K[..., 0:1] * innov, pos_pred)
            vel = torch.where(hv, vel_pred + K[..., 1:2] * innov, vel_pred)
            P = torch.where(has[..., None, None], P_upd, Ppred)
            pd_now = torch.zeros_like(state["ex"])
            beta = has.to(dtype)
        hf = beta if self.existence == "ipda" else has.to(dtype)       # ipda：航迹记忆记的是软检测（关联概率）
        zero = torch.zeros_like(hf)
        miss = torch.where(act, torch.where(has, zero, state["miss"] + 1.0), zero)
        age = torch.where(act, state["age"] + 1.0, zero)

        # 4. 航迹级证据积累 -> 存在
        speed = torch.sqrt((vel ** 2).sum(-1) + 1e-6) / cell
        u = torch.cat([hf[..., None], f * hf[..., None], (torch.log1p(d2n) * hf)[..., None],
                       (torch.log(S) * hf)[..., None], torch.log(Ppred[..., 0, 0].clamp(min=1e-6))[..., None],
                       (speed.clamp(max=4.0) / 4.0)[..., None], (torch.log1p(age) / 3.0)[..., None],
                       (miss / self.max_coast)[..., None], torch.zeros(B, N, 4, device=dev, dtype=dtype)], -1)
        h_new = self.gru(u.reshape(B * N, self.u_dim), state["h"].reshape(B * N, self.hidden)).view(B, N, self.hidden)
        h = torch.where(actv, h_new, state["h"])
        hist, hist_ok = state["hist"], state["hist_ok"]

        # 5. 出生：所有验证门之外的局部峰；没有空槽时回收存在概率最低的航迹
        cand = is_peak & (mass >= self.birth_floor) & ~in_gate.any(1)
        k = min(self.max_births, mass.shape[1])
        cv, ci = torch.where(cand, mass, torch.full_like(mass, -1.0)).topk(k, dim=1)
        cv_l, ci_l = cv.cpu().tolist(), ci.cpu().tolist()
        act_l, ex_l = act.cpu().tolist(), ex_prev.cpu().tolist()
        bb, ss, cc, recycled = [], [], [], 0
        for b in range(B):
            free = [n for n in range(N) if not act_l[b][n]]
            weak = sorted((n for n in range(N) if act_l[b][n] and ex_l[b][n] < self.recycle_exist),
                          key=lambda n: ex_l[b][n])
            taken = []                                            # 本窗已出生的格：3x3 量测邻域内不再出生第二条
            for value, c in zip(cv_l[b], ci_l[b]):
                if value < self.birth_floor:
                    break
                gy, gx = divmod(int(c), self.gw)
                if any(abs(gy - ty) <= 2 and abs(gx - tx) <= 2 for ty, tx in taken):
                    continue
                if free:
                    slot = free.pop(0)
                elif weak:
                    slot = weak.pop(0)
                    recycled += 1
                else:
                    break
                taken.append((gy, gx))
                bb.append(b)
                ss.append(slot)
                cc.append(c)
        born = torch.zeros(B, N, dtype=torch.bool, device=dev)
        if bb:
            bt = torch.tensor(bb, dtype=torch.long, device=dev)
            st = torch.tensor(ss, dtype=torch.long, device=dev)
            ct = torch.tensor(cc, dtype=torch.long, device=dev)
            f_b, z_b = self._measure(sums.permute(0, 2, 1)[bt, ct])
            scale = 8.0 * cell
            act_s = act[bt]
            dcur = torch.where(act_s, _dist(z_b[:, None], pos[bt]), torch.full_like(act_s, BIG, dtype=dtype))
            dcur = dcur.amin(1)
            hok = hist_ok[bt] & act_s[..., None]
            dh = torch.where(hok, _dist(z_b[:, None, None], hist[bt]), torch.full_like(hok, BIG, dtype=dtype))
            dh = dh.amin(2).amin(1)
            R_b = self._noise(f_b)
            M = int(bt.shape[0])
            ones, zeros1 = torch.ones(M, 1, device=dev, dtype=dtype), torch.zeros(M, 1, device=dev, dtype=dtype)
            logR = torch.log(R_b)[:, None]
            u_b = torch.cat([ones, f_b, zeros1, logR, logR, zeros1, zeros1, zeros1, ones,
                             (dcur / scale).clamp(max=1.0)[:, None], (dh / scale).clamp(max=1.0)[:, None],
                             (dh < 2.0 * cell).to(dtype)[:, None]], -1)
            h_b = self.gru(u_b, torch.zeros(M, self.hidden, device=dev, dtype=dtype))
            v0 = torch.exp(self.log_v0).expand(M)
            P_b = torch.stack([torch.stack([R_b, torch.zeros_like(R_b)], -1),
                               torch.stack([torch.zeros_like(R_b), v0], -1)], -2)
            pos = pos.index_put((bt, st), z_b)
            vel = vel.index_put((bt, st), torch.zeros(M, 2, device=dev, dtype=dtype))
            P = P.index_put((bt, st), P_b)
            h = h.index_put((bt, st), h_b)
            born = born.index_put((bt, st), torch.ones(M, dtype=torch.bool, device=dev))
            age = age.index_put((bt, st), torch.zeros(M, device=dev, dtype=dtype))
            miss = miss.index_put((bt, st), torch.zeros(M, device=dev, dtype=dtype))
            has = has & ~born
            hist_ok = hist_ok & ~born[..., None]
        live = act | born

        # 6. 存在概率；7. 消亡
        extra = {}
        if self.existence == "ipda":
            # 伯努利递推（对数几率）：预测 r -> p_S r；有量测乘 (1 - P_D + P_D l)，漏检乘 (1 - P_D)；新生航迹由出生编码给出
            r = ex_prev.clamp(1e-6, 1.0 - 1e-6)
            rp = (torch.sigmoid(self.surv_logit.clamp(0.0, SURV_MAX_LOGIT)) * r).clamp(max=1.0 - 1e-6)
            L_pred = torch.log(rp) - torch.log1p(-rp)
            L_post = L_pred + torch.where(has, log_det, log_miss)
            ex_logit = torch.where(born, self.exist(h).squeeze(-1), L_post)
            # 量测来源的后验对数几率（目标存在且量测来自它）：log(r P_D l) - log(1 - r P_D)，供校准监督
            rpd = (rp * torch.exp(log_pd)).clamp(max=1.0 - 1e-6)
            extra = {"pd_logit": (log_pd - log_miss) if self.pd_max < 1.0 else pd_logit, "org_logit": torch.log(rpd) + log_lr - torch.log1p(-rpd), "zpos": zpos,
                     "prior": act & ~born}
        else:
            ex_logit = self.exist(h).squeeze(-1)
        ex = torch.sigmoid(ex_logit)
        m = self.gate_max_px
        outside = (pos[..., 0] < -m) | (pos[..., 0] > self.width + m) | (pos[..., 1] < -m) | (pos[..., 1] > self.height + m)
        die = live & ((ex < self.death_exist) | (miss > self.max_coast) | outside)
        alive = live & ~die
        info = {"ex_logit": ex_logit, "live": live, "pos": pos, "born": born, "has": has,
                "births": len(bb), "recycled": recycled, "delayed": None}
        info.update(extra)

        # 8. 滞后判决：用第 k 窗的信息重判第 k-1 窗的事件（在清零消亡航迹之前，用到它们本窗的后验）
        pend = state.get("pending")
        if self.decision_lag and pend is not None and int(pend["z0"].shape[0]):
            info["delayed"] = self._delayed(pend, act & ~born, pos_prev, P_prev, pos_pred, vel_pred, Ppred,
                                            pos, vel, ex, has, sums)

        # 延迟决策的释放读消亡清零之前的状态：因存在概率塌掉而删除的航迹算"否定"，不算"丢失"
        pre = {"active": live, "pos": pos, "vel": vel, "P": P, "ex": ex, "beta": beta, "age": age, "miss": miss} \
            if self.defer_max else None
        av, am = alive[..., None], alive[..., None, None]
        pos, vel = torch.where(av, pos, torch.zeros_like(pos)), torch.where(av, vel, torch.zeros_like(vel))
        P, h = torch.where(am, P, torch.zeros_like(P)), torch.where(av, h, torch.zeros_like(h))
        ex = torch.where(alive, ex, torch.zeros_like(ex))
        pd_now = torch.where(alive & ~born, pd_now, torch.zeros_like(pd_now))
        beta = torch.where(alive & has, beta, torch.zeros_like(beta))
        age, miss = torch.where(alive, age, zero), torch.where(alive, miss, zero)
        hist = torch.cat([pos[:, :, None], hist[:, :, :-1]], 2)
        hist_ok = torch.cat([alive[:, :, None], hist_ok[:, :, :-1]], 2) & alive[..., None]
        new_state = {"active": alive, "pos": pos, "vel": vel, "P": P, "h": h, "ex": ex, "pd": pd_now, "beta": beta,
                     "age": age, "miss": miss, "has": has & alive, "hist": hist, "hist_ok": hist_ok, "pending": None,
                     "defer": None, "orphan": None}

        # 9. 本窗事件的零等待修正
        if clutter is not None:
            ev = dict(ev, clutter=clutter[ev["b"], self._cells(ev["x"], ev["y"])])        # 事件所在处的杂波量
        assoc = None
        if int(z0.shape[0]):
            ctx, assoc = self._context(ev, new_state, with_assoc=True)
            if self.provisional_gate:
                m_now = torch.sigmoid(self._membership_logit(assoc["d2"], assoc["spred"], assoc["lc"])).detach()
                m_now = torch.where(assoc["none"], torch.zeros_like(m_now), m_now)
                z = z0 + m_now * self._delta(self.corr, z0, feat, ev, ctx)
            else:
                z = self._fuse(self.corr, z0, feat, ev, ctx)
        else:
            ctx = z0.new_zeros((0, CTX_DIM))
            z = z0
        if self.gated:
            info["assoc"] = assoc                                      # 归属概率的校准监督用
        if self.defer_max:                                             # 10. 目标级延迟决策
            out = self._defer_step(defer_q, orphan_q, pre, new_state, born & alive, ev, z, ctx, assoc)
            new_state["defer"], new_state["orphan"] = out[0], out[1]
            if out[2] is not None:
                info["delayed"] = (out[2], out[3])
                info["delayed_lag"], info["delayed_status"] = out[4], out[5]
        if self.decision_lag:
            key = ev["key"] if "key" in ev else torch.zeros_like(z0)
            new_state["pending"] = {"b": ev["b"], "x": ev["x"], "y": ev["y"], "p": ev["p"], "t_local": ev["t_local"],
                                    "z0": z0, "feat": feat, "ctx": ctx, "key": key}
        return z, new_state, info

    def _delta(self, mlp, z0, feat, ev, ctx, extra=None):
        parts = [z0.clamp(-15.0, 15.0)[:, None] / 5.0, feat, ev["p"].to(z0.dtype)[:, None],
                 ev["t_local"].to(z0.dtype)[:, None], ctx]
        if extra is not None:
            parts.append(extra)
        return mlp(torch.cat(parts, 1)).squeeze(-1)

    def _fuse(self, mlp, z0, feat, ev, ctx, extra=None):
        return z0 + self._delta(mlp, z0, feat, ev, ctx, extra)

    def _nearest(self, exy, pos_e, vel_e, act_e, delta2):
        """最近航迹（按归一化距离）的沿迹 / 横迹坐标与下标；没有航迹的事件沿迹 0、横迹 1。"""
        scale = 8.0 * float(self.cell)
        j = torch.where(act_e, delta2, torch.full_like(delta2, BIG)).argmin(1)
        none = ~act_e.any(1)
        rows = torch.arange(int(exy.shape[0]), device=exy.device)
        rel = exy - pos_e[rows, j]
        v = vel_e[rows, j]
        vn = torch.sqrt((v ** 2).sum(1) + 1e-6) + 0.5
        along = ((rel * v).sum(1) / vn / scale).clamp(-1.0, 1.0)
        cross = ((rel[:, 0] * v[:, 1] - rel[:, 1] * v[:, 0]).abs() / vn / scale).clamp(max=1.0)
        along = torch.where(none, torch.zeros_like(along), along)
        cross = torch.where(none, torch.ones_like(cross), cross)
        return j, rows, none, along, cross

    def _context(self, ev, st, with_assoc=False):
        """本窗事件与航迹的关系 [Ne, CTX_DIM]（距离按 8 格归一并截到 1；with_assoc 时另返回最近航迹的归属信息）：
        0 无航迹权重 1 加权存在 2 加权归一化距离 3 加权速度 4 到最近航迹当前位置的距离 5 到近期轨迹（2～20 窗前）的距离
        6 本流活航迹数占比 7 沿迹坐标（负 = 在航迹身后） 8 横迹距离 9 最近航迹本窗有无量测 10 最近航迹年龄。"""
        N, cell = self.slots, float(self.cell)
        eb = ev["b"]
        dtype = ev["z0"].dtype
        exy = torch.stack([ev["x"], ev["y"]], 1).to(dtype)
        act_e, ex_e = st["active"][eb], st["ex"][eb]
        pos_e, vel_e = st["pos"][eb], st["vel"][eb]
        d = _dist(exy[:, None], pos_e)                                                   # [Ne, N]
        delta2 = d * d / (st["P"][eb][..., 0, 0] + cell * cell)
        speed = torch.sqrt((vel_e ** 2).sum(-1) + 1e-6) / cell
        if self.existence == "ipda":
            # PDA 形式的事件关联：航迹 r N(e; x, S)（含 1/S 归一化）对比事件所在处的杂波密度
            spred_e = st["P"][eb][..., 0, 0] + cell * cell
            logit_w = torch.where(act_e, -0.5 * delta2 - torch.log(spred_e) + torch.log(ex_e.clamp(min=1e-6)),
                                  torch.full_like(d, -1e4))
            none = (self.none_logit + torch.log1p(ev["clutter"].to(dtype)))[:, None]
        else:
            logit_w = torch.where(act_e, -0.5 * delta2 + torch.log(ex_e.clamp(min=1e-6)), torch.full_like(d, -1e4))
            none = self.none_logit.expand(int(d.shape[0]), 1)
        w = torch.softmax(torch.cat([logit_w, none], 1), 1)
        wt, wn = w[:, :N], w[:, N]
        scale = 8.0 * cell
        dcur = torch.where(act_e, d, torch.full_like(d, BIG)).amin(1)
        hist = st["hist"][eb][:, :, 2:]                                                   # 2～20 窗之前的位置
        hok = st["hist_ok"][eb][:, :, 2:]
        dh = _dist(exy[:, None, None], hist)
        tail = torch.where(hok, dh, torch.full_like(dh, BIG)).amin(2).amin(1)
        j, rows, none, along, cross = self._nearest(exy, pos_e, vel_e, act_e, delta2)
        det = st["beta"] if self.existence == "ipda" else st["has"].to(dtype)            # ipda：软检测
        has_j = torch.where(none, torch.zeros_like(along), det[eb][rows, j])
        age_j = torch.where(none, torch.zeros_like(along), (torch.log1p(st["age"][eb][rows, j]) / 3.0).clamp(max=1.0))
        ctx = torch.stack([wn, (wt * ex_e).sum(1), (wt * delta2.clamp(max=25.0) / 25.0).sum(1),
                           (wt * speed.clamp(max=4.0) / 4.0).sum(1), (dcur / scale).clamp(max=1.0),
                           (tail / scale).clamp(max=1.0), act_e.to(dtype).sum(1) / float(N), along, cross,
                           has_j, age_j], 1)
        if self.context == "none":
            ctx = torch.zeros_like(ctx)
        if not with_assoc:
            return ctx
        spred_j = st["P"][eb][rows, j][:, 0, 0] + cell * cell
        lc = torch.log1p(ev["clutter"].to(dtype)) if "clutter" in ev else torch.zeros_like(along)
        return ctx, {"j": j, "none": none, "d2": delta2[rows, j], "spred": spred_j, "lc": lc}

    # ------------------------------------------------------------------ 目标级延迟决策
    def _membership_logit(self, d2, S, lc):
        """假如目标存在、事件属于它的对数几率：随归一化距离 d2、目标位置方差 S、本地杂波 lc = log1p(杂波量) 单调下降。"""
        w = F.softplus(self.mem_theta)
        return -0.5 * w[0] * d2 - w[1] * torch.log(S) - w[2] * lc + self.mem_bias

    def _defer_step(self, dq, oq, pre, st, newborn, ev, znow, ctx, assoc):
        """一窗的延迟决策。返回 (挂起队列, 孤儿队列, 终判 logit, key, 等待窗数, 状态码)；没有终判时后四项为 None。

        pre：本窗消亡清零之前的航迹状态（释放用）；st：本窗结束时的航迹状态（认领、分类用）。
        1. 释放：挂起条目所属目标越过 +-A（确认 / 否定）、缺口后重新捕获、丢失（槽位换成新目标或已不在）或等满 defer_max 窗；
        2. 回溯认领：本窗新生的目标认领前几窗附近的孤儿事件（第一拍：事件发生时目标还没有出生）；
        3. 本窗事件分类：落在某目标验证门内、且该目标存在未定或本窗没有可靠量测的挂起；不在任何目标门内的进孤儿队列。
        """
        A, D, cell = self.defer_A, self.defer_max, float(self.cell)
        dtype = znow.dtype
        exl = _logit(st["ex"])
        outs = [None, None, None, None]
        if dq is not None and int(dq["b"].shape[0]):
            b, j = dq["b"], dq["slot"]
            same = pre["active"][b, j] & (pre["age"][b, j].long() + dq["age_off"] == dq["lag"])
            L, beta = _logit(pre["ex"])[b, j], pre["beta"][b, j]
            coast = dq["reason"] == 1
            settled = [same & (L >= A) & ~coast, same & (L <= -A), same & coast & (L >= A) & (beta >= 0.5), ~same]
            status = torch.full_like(b, -1)
            for code in (3, 2, 1, 0):                                  # 丢失优先，其次重新捕获、否定、确认
                status = torch.where(settled[code] & (status < 0), torch.full_like(status, code), status)
            status = torch.where((status < 0) & (dq["lag"] >= D), torch.full_like(status, 4), status)
            rel = status >= 0
            if bool(rel.any()):
                r = _q_take(dq, rel)
                outs = [self._final(r, status[rel], pre, _logit(pre["ex"])), r["key"], r["lag"], status[rel]]
            dq = _q_take(dq, ~rel)
        if oq is not None and int(oq["b"].shape[0]) and bool(newborn.any()):
            nb_b, nb_j = torch.nonzero(newborn, as_tuple=True)
            pos_nb = st["pos"][nb_b, nb_j]
            var_nb = st["P"][nb_b, nb_j][:, 0, 0] + cell * cell
            lag = oq["lag"].to(dtype)
            oxy = torch.stack([oq["x"], oq["y"]], 1).to(dtype)
            d2 = ((oxy[:, None] - pos_nb[None]) ** 2).sum(-1) / (var_nb[None] + torch.exp(self.log_v0).detach()
                                                                  * lag[:, None] ** 2)
            d2 = torch.where(oq["b"][:, None] == nb_b[None], d2, torch.full_like(d2, BIG))
            best, idx = d2.min(1)
            claim = best <= self.gate_sigma ** 2
            if bool(claim.any()):
                c = _q_take(oq, claim)
                ci = idx[claim]
                cb, cj = nb_b[ci], nb_j[ci]
                zf = torch.zeros_like(c["znow"])
                c.update(slot=cj, age_off=c["lag"].clone(), reason=torch.full_like(c["lag"], 2), L_then=exl[cb, cj],
                         m_then=zf, beta_then=zf, fx=zf, fy=zf, miss_then=zf)
                dq = _q_cat(dq, c)
                oq = _q_take(oq, ~claim)
        if int(znow.shape[0]):
            b, j = ev["b"], assoc["j"]
            L, ones = exl[b, j], torch.ones_like(znow)
            meas = torch.where(newborn[b, j], ones, st["beta"][b, j])           # 新生航迹由本窗的量测出生
            gated = ~assoc["none"] & (assoc["d2"] <= self.gate_sigma ** 2)
            pend = gated & ((L.abs() < A) | ((meas < 0.5) & (L > -A)))
            none_e = self.none_logit.detach() + assoc["lc"]
            if self.gated:
                m_then = torch.sigmoid(self._membership_logit(assoc["d2"], assoc["spred"], assoc["lc"])).detach()
            else:
                m_then = torch.sigmoid(-0.5 * assoc["d2"] - torch.log(assoc["spred"]) - none_e)   # 假如目标存在、事件属于它
            base = {"b": b, "x": ev["x"], "y": ev["y"], "p": ev["p"].to(dtype), "t_local": ev["t_local"].to(dtype),
                    "z0": ev["z0"], "feat": ev["feat"], "ctx": ctx, "key": ev["key"].to(dtype), "znow": znow,
                    "lc": assoc["lc"], "lag": torch.zeros_like(b)}
            pos_j = st["pos"][b, j]
            full = dict(base, slot=j, age_off=-st["age"][b, j].long(), reason=(meas < 0.5).long(), L_then=L,
                        m_then=m_then, beta_then=meas, fx=pos_j[:, 0], fy=pos_j[:, 1], miss_then=st["miss"][b, j])
            if bool(pend.any()):
                dq = _q_cat(dq, _q_take(full, pend))
            orphan = ~gated
            if bool(orphan.any()):
                oq = _q_cat(oq, _q_take(full, orphan))
        return (dq, oq) + tuple(outs)

    def _final(self, r, status, st, exl):
        """挂起事件的终判：暂定 logit + 修正（读目标此刻的状态、事件当时的条件归属、平滑后的位置）。"""
        cell, dtype = float(self.cell), r["znow"].dtype
        b, j = r["b"], r["slot"]
        lost = status == 3
        lagf = r["lag"].to(dtype)
        pos, vel = st["pos"][b, j], st["vel"][b, j]
        spred = torch.where(lost, torch.full_like(lagf, 4.0 * cell * cell), st["P"][b, j][:, 0, 0] + cell * cell)
        fwd = torch.stack([r["fx"], r["fy"]], 1)
        back = pos - vel * lagf[:, None]                               # 用目标此刻的状态回推到事件那一窗
        w_b = (r["miss_then"] / (r["miss_then"] + lagf).clamp(min=1.0))[:, None]   # 缺口：前后两次检测之间插值
        coast = (r["reason"] == 1)[:, None]
        orphan = (r["reason"] == 2)[:, None]
        xs = torch.where(orphan, back, torch.where(coast, (1.0 - w_b) * fwd + w_b * back, fwd))
        xs = torch.where(lost[:, None] & ~orphan, fwd, xs)
        exy = torch.stack([r["x"], r["y"]], 1).to(dtype)
        d2s = ((exy - xs) ** 2).sum(1) / spred
        m_s = torch.sigmoid(-0.5 * d2s - torch.log(spred) - (self.none_logit + r["lc"]))
        m_s = torch.where(lost & orphan[:, 0], torch.zeros_like(m_s), m_s)
        L_now = torch.where(lost, torch.zeros_like(lagf), exl[b, j])
        if self.gated:
            # 门控修正：修正量 = 归属概率（平滑后的位置；回推 / 插值的不确定度随等待窗数增大）x 目标层的修正。
            # 修正项只看目标层的量和事件此刻的 logit，不看外观特征，所以不能把目标的存在证据分给不属于它的事件
            grow = torch.where(r["reason"] >= 1, lagf ** 2 * st["P"][b, j][:, 1, 1], torch.zeros_like(lagf))
            S_s = spred + grow
            d2g = ((exy - xs) ** 2).sum(1) / S_s
            m = torch.sigmoid(self._membership_logit(d2g, S_s, r["lc"])).detach()
            m = torch.where(lost & orphan[:, 0], torch.zeros_like(m), m)
            if self.defer_revision == "evidence":
                # 证据修正（v4e）：z_final = z_now + m x kappa[原因] x (L(终判) - L_ref)，只加暂定输出没见过的那部分证据。
                # L_ref：原因 0 / 1 取事件那一窗的存在对数几率（暂定输出已经用过）；孤儿（原因 2）事件发生时还没有目标，
                # 暂定输出没用过这个目标的任何证据，按贝叶斯组合应加"后验 - 先验"：L_ref = l0（新生目标的先验对数几率，学出来）。
                # 归属：原因 0 用事件那一窗的值（与暂定门控是同一个）；缺口 / 孤儿用插值 / 回推后的位置重算（上面的 m）。
                # 丢失（槽位已换成别的目标）不知道原目标的结局：增量记 0。存在与归属只由各自的校准监督训练（detach）
                m_ev = torch.where(r["reason"] == 0, r["m_then"], m).detach()
                ref = torch.where(r["reason"] == 2, self.orphan_ref.to(dtype).expand_as(lagf),
                                  r["L_then"].to(dtype).detach())
                gain = torch.where(lost, torch.zeros_like(lagf), L_now.detach() - ref)
                return r["znow"] + m_ev * F.softplus(self.kappa_raw)[r["reason"]] * gain
            rev = torch.cat([F.one_hot(status, 5).to(dtype), F.one_hot(r["reason"], 3).to(dtype),
                             (L_now / 3.0).clamp(-2.0, 2.0)[:, None], (r["L_then"] / 3.0).clamp(-2.0, 2.0)[:, None],
                             (lagf / float(self.defer_max))[:, None], r["beta_then"][:, None],
                             (r["znow"] / 5.0).clamp(-3.0, 3.0)[:, None]], 1)
            return r["znow"] + m * self.corr_rev(rev).squeeze(-1)
        fin = torch.cat([F.one_hot(status, 5).to(dtype), F.one_hot(r["reason"], 3).to(dtype),
                         (L_now / 3.0).clamp(-2.0, 2.0)[:, None], (r["L_then"] / 3.0).clamp(-2.0, 2.0)[:, None],
                         (lagf / float(self.defer_max))[:, None], r["m_then"][:, None], m_s[:, None],
                         (torch.sqrt(d2s + 1e-6) / 3.0).clamp(max=1.0)[:, None], r["beta_then"][:, None]], 1)
        parts = [r["z0"].clamp(-15.0, 15.0)[:, None] / 5.0, r["feat"], r["p"][:, None], r["t_local"][:, None],
                 r["ctx"], fin]
        return r["znow"] + self.corr_final(torch.cat(parts, 1)).squeeze(-1)

    def _delayed(self, pend, valid, pos_prev, P_prev, pos_pred, vel_pred, Ppred, pos_post, vel_post, ex, has, sums):
        """一窗固定滞后：RTS 平滑出第 k-1 窗的航迹位置，再重判第 k-1 窗的事件。valid [B, N]：第 k-1 窗就在、本窗没有重生的槽。"""
        cell = float(self.cell)
        pos_s = rts_position(pos_prev, P_prev, pos_pred, vel_pred, Ppred, pos_post, vel_post)
        eb = pend["b"]
        dtype = pend["z0"].dtype
        exy = torch.stack([pend["x"], pend["y"]], 1).to(dtype)
        v_e = valid[eb]
        ps = pos_s[eb]
        d = _dist(exy[:, None], ps)
        delta2 = d * d / (P_prev[eb][..., 0, 0] + cell * cell)
        j, rows, none, along, cross = self._nearest(exy, ps, vel_post[eb], v_e, delta2)
        zero = torch.zeros(int(exy.shape[0]), device=exy.device, dtype=dtype)
        ex_j = torch.where(none, zero, ex[eb][rows, j])
        has_j = torch.where(none, zero, has[eb][rows, j].to(dtype))
        dl_j = torch.where(none, torch.ones_like(zero), delta2[rows, j].clamp(max=25.0) / 25.0)
        mk = sums[eb, 0, self._cells(pend["x"], pend["y"])]                              # 事件所在处第 k 窗的证据
        lag = torch.stack([ex_j, dl_j, along, cross, has_j, (torch.log1p(mk) / 3.0).clamp(max=1.0),
                           v_e.to(dtype).sum(1) / float(self.slots)], 1)
        if self.context == "none":
            lag = torch.zeros_like(lag)
        z_d = self._fuse(self.corr_lag, pend["z0"], pend["feat"], pend, pend["ctx"], lag)
        return z_d, pend["key"]

    # ------------------------------------------------------------------ 片段
    def forward_chunk(self, events, z0, feat, state, steps, keep=None, labels=None, tid=None, log=None, keys=None):
        """events：片段事件（t 为片段内窗号、b 为流号）；z0 [Ne]、feat [Ne, F] 来自骨干。

        keep [T, B]（或 None）：keep[t, b] = 0 表示第 b 条流在第 t 窗开始一段新录像，先清空该流航迹。
        labels / tid 给出时计算航迹监督（存在 BCE 之和、质心 L1 之和及其个数）。log 为列表时逐窗记录航迹（可视化）。
        keys（滞后判决时用）：与事件一一对应的标识（训练传标签、评估传事件在文件中的下标），随滞后输出一起返回。
        返回 (z [Ne] 零等待输出，顺序与 events 相同, 片段结束时的状态, aux)；
        aux["delayed_z"] / aux["delayed_key"]：本片段内完成重判的事件（含上一片段末窗的事件），没有时为 None。
        """
        if state is None:
            batch = int(keep.shape[1]) if keep is not None else int(events["b"].max()) + 1 if events["b"].numel() else 1
            state = self.init_state(batch, z0.device)
        aux = {"exist_sum": z0.new_zeros(()), "exist_n": 0, "pos_sum": z0.new_zeros(()), "pos_n": 0,
               "pd_sum": z0.new_zeros(()), "pd_n": 0, "org_sum": z0.new_zeros(()), "org_n": 0,
               "mem_sum": z0.new_zeros(()), "mem_n": 0,
               "births": 0, "recycled": 0, "live": 0, "slot_windows": 0}
        idx_parts, z_parts, dz, dk, dl, ds = [], [], [], [], [], []
        for t in range(int(steps)):
            idx = torch.nonzero(events["t"] == t, as_tuple=False).flatten()
            ev = {key: events[key][idx] for key in ("b", "x", "y", "p", "t_local")}
            ev["z0"], ev["feat"] = z0[idx], feat[idx]
            if self.has_delayed:
                ev["key"] = keys[idx] if keys is not None else idx.to(z0.dtype)
            reset = (keep[t] == 0) if keep is not None else None
            z_t, state, info = self.forward_window(state, ev, reset)
            idx_parts.append(idx)
            z_parts.append(z_t)
            if info["delayed"] is not None:
                dz.append(info["delayed"][0])
                dk.append(info["delayed"][1])
                if "delayed_lag" in info:
                    dl.append(info["delayed_lag"])
                    ds.append(info["delayed_status"])
            aux["births"] += info["births"]
            aux["recycled"] += info["recycled"]
            aux["live"] += int(state["active"].sum())
            aux["slot_windows"] += int(state["active"].shape[0])
            if labels is not None:
                self._supervise(info, ev, labels[idx], tid[idx], aux)
            if log is not None:
                log.append({key: state[key].detach().cpu().numpy() for key in ("active", "pos", "vel", "ex", "pd",
                                                                                "beta")})
        if idx_parts and sum(int(i.shape[0]) for i in idx_parts):
            z = z0.index_put((torch.cat(idx_parts),), torch.cat(z_parts))
        else:
            z = z0
        aux["delayed_z"] = torch.cat(dz) if dz else None
        aux["delayed_key"] = torch.cat(dk) if dk else None
        aux["delayed_lag"] = torch.cat(dl) if dl else None             # 延迟决策：每个终判事件等了几窗、按什么状态终判
        aux["delayed_status"] = torch.cat(ds) if ds else None
        return z, state, aux

    # ------------------------------------------------------------------ 监督
    def _supervise(self, info, ev, label, tid, aux):
        live, pos, ex_logit = info["live"], info["pos"], info["ex_logit"]
        target = torch.zeros_like(ex_logit)
        pos_t = torch.zeros_like(pos)
        pmask = torch.zeros_like(live)
        gb, gc, gr = window_targets(ev["b"], ev["x"], ev["y"], label, tid, self.match_min_px)
        ipda = self.existence == "ipda"
        det_t, det_m = torch.zeros_like(ex_logit), torch.zeros_like(live)
        if gb.numel() and bool(live.any()):
            dist = _dist(gc[:, None], pos[gb])                                            # [Gt, N]
            ok = live[gb] & (dist <= gr[:, None])
            dist_np = torch.where(ok, dist, torch.full_like(dist, np.inf)).detach().cpu().numpy()
            pairs = greedy_match(dist_np, gb.cpu().numpy())
            if pairs:
                rows = torch.tensor([r for r, _ in pairs], dtype=torch.long, device=pos.device)
                cols = torch.tensor([c for _, c in pairs], dtype=torch.long, device=pos.device)
                bsel = gb[rows]
                target = target.index_put((bsel, cols), torch.ones_like(cols, dtype=target.dtype))
                pos_t = pos_t.index_put((bsel, cols), gc[rows])
                pmask = pmask.index_put((bsel, cols), torch.ones_like(cols, dtype=torch.bool))
                if ipda:
                    # 检测概率的标签：匹配上的目标本窗有没有产生量测（它的事件在骨干上的证据和 >= gate_floor）
                    mass = target_mass(ev["b"], label, tid, torch.sigmoid(ev["z0"]).detach())
                    det_t = det_t.index_put((bsel, cols), (mass[rows] >= self.gate_floor).to(det_t.dtype))
                    det_m = det_m.index_put((bsel, cols), torch.ones_like(cols, dtype=torch.bool))
        if ipda:
            # 校准监督（分离原理：估计器输出校准的信念）：P_D 对"目标有没有产生量测"，量测来源概率对"量测是否落在目标上"
            prior = info["prior"]
            m = det_m & prior
            if bool(m.any()):
                aux["pd_sum"] = aux["pd_sum"] + F.binary_cross_entropy_with_logits(
                    info["pd_logit"][m], det_t[m], reduction="sum")
                aux["pd_n"] += int(m.sum())
            hm = info["has"] & prior
            if bool(hm.any()):
                B, N = hm.shape
                org = torch.zeros(B * N, device=pos.device)
                if gb.numel():
                    near = (_dist(gc[:, None], info["zpos"][gb]) <= gr[:, None]).to(org.dtype)   # [Gt, N]
                    flat = (gb[:, None] * N + torch.arange(N, device=pos.device)[None, :]).reshape(-1)
                    org = org.index_put((flat,), near.reshape(-1).detach(), accumulate=True)
                org = (org.view(B, N) > 0).to(ex_logit.dtype)
                aux["org_sum"] = aux["org_sum"] + F.binary_cross_entropy_with_logits(
                    info["org_logit"][hm], org[hm], reduction="sum")
                aux["org_n"] += int(hm.sum())
        if self.gated and info.get("assoc") is not None and int(label.shape[0]):
            # 归属概率的校准监督：匹配上标注目标的航迹，其验证门内的事件是不是目标事件
            a = info["assoc"]
            near = ~a["none"] & (a["d2"] <= self.gate_sigma ** 2) & (target[ev["b"], a["j"]] > 0.5)
            if bool(near.any()):
                logit = self._membership_logit(a["d2"][near].detach(), a["spred"][near].detach(), a["lc"][near].detach())
                aux["mem_sum"] = aux["mem_sum"] + F.binary_cross_entropy_with_logits(
                    logit, (label[near] > 0.5).to(logit.dtype), reduction="sum")
                aux["mem_n"] += int(near.sum())
        if self.collect_calibration and bool(live.any()):
            aux.setdefault("calib", []).append((ex_logit[live].detach(), target[live].detach()))
        if bool(live.any()):
            aux["exist_sum"] = aux["exist_sum"] + F.binary_cross_entropy_with_logits(
                ex_logit[live], target[live], reduction="sum")
            aux["exist_n"] += int(live.sum())
        if bool(pmask.any()):
            aux["pos_sum"] = aux["pos_sum"] + F.smooth_l1_loss((pos[pmask] - pos_t[pmask]) / 4.0,
                                                               torch.zeros_like(pos[pmask]), reduction="sum")
            aux["pos_n"] += int(pmask.sum())


def rts_position(pos_prev, P_prev, pos_pred, vel_pred, Ppred, pos_post, vel_post):
    """一步 RTS 平滑的位置：x(k-1|k) = x(k-1|k-1) + C (x(k|k) - x(k|k-1))，C = P(k-1|k-1) F^T P(k|k-1)^-1，取位置行。

    每个坐标轴的状态为 [位置, 速度]，两轴共用 2x2 协方差；F = [[1, 1], [0, 1]]。没有量测时 x(k|k) = x(k|k-1)，位置不变。
    """
    a, b_ = P_prev[..., 0, 0], P_prev[..., 0, 1]
    p, q_, r = Ppred[..., 0, 0], Ppred[..., 0, 1], Ppred[..., 1, 1]
    det = (p * r - q_ * q_).clamp(min=1e-9)
    A00, A01 = a + b_, b_                                                                 # P(k-1|k-1) F^T 的位置行
    C00 = (A00 * r - A01 * q_) / det
    C01 = (-A00 * q_ + A01 * p) / det
    return pos_prev + C00[..., None] * (pos_post - pos_pred) + C01[..., None] * (vel_post - vel_pred)


def window_targets(eb, ex, ey, label, tid, match_min_px):
    """本窗标注目标：(所在流 [Gt], 质心 [Gt, 2], 匹配半径 [Gt])。目标 = 同一流里 target_id 相同的标注事件。"""
    sel = (label > 0.5) & (tid > 0)
    dev = ex.device
    if not bool(sel.any()):
        empty = torch.zeros(0, device=dev)
        return torch.zeros(0, dtype=torch.long, device=dev), empty.view(0, 2), empty
    b = eb[sel]
    key = b * 1000003 + tid[sel].to(torch.long)
    uniq, inv = torch.unique(key, return_inverse=True)
    G = int(uniq.shape[0])
    x, y = ex[sel].to(torch.float32), ey[sel].to(torch.float32)
    acc = torch.zeros(G * 5, device=dev)
    vals = torch.stack([torch.ones_like(x), x, y, x * x, y * y], 1)
    flat = inv[:, None] * 5 + torch.arange(5, device=dev)[None, :]
    acc = acc.index_put((flat.reshape(-1),), vals.reshape(-1), accumulate=True).view(G, 5)
    n = acc[:, 0]
    cx, cy = acc[:, 1] / n, acc[:, 2] / n
    var = (acc[:, 3] / n - cx * cx) + (acc[:, 4] / n - cy * cy)
    radius = (2.5 * torch.sqrt(var.clamp(min=0.0))).clamp(min=float(match_min_px))
    return torch.div(uniq, 1000003, rounding_mode="floor"), torch.stack([cx, cy], 1), radius


def target_mass(eb, label, tid, prob):
    """本窗每个标注目标在骨干上的证据和 sum(p)，顺序与 window_targets 相同（按 流号 x 1000003 + target_id 排序）。"""
    sel = (label > 0.5) & (tid > 0)
    if not bool(sel.any()):
        return torch.zeros(0, device=eb.device, dtype=prob.dtype)
    key = eb[sel] * 1000003 + tid[sel].to(torch.long)
    uniq, inv = torch.unique(key, return_inverse=True)
    acc = torch.zeros(int(uniq.shape[0]), device=eb.device, dtype=prob.dtype)
    return acc.index_put((inv,), prob[sel], accumulate=True)


def greedy_match(dist, stream):
    """按距离从小到大一一匹配（每个标注目标、每个航迹槽最多一次；只在同一流内）。dist [Gt, N]，不可配对为 inf。"""
    dist = np.array(dist, dtype=np.float64, copy=True)
    pairs = []
    while dist.size and np.isfinite(dist).any():
        r, c = np.unravel_index(np.argmin(dist), dist.shape)
        pairs.append((int(r), int(c)))
        dist[r, :] = np.inf
        dist[stream == stream[r], c] = np.inf
    return pairs
