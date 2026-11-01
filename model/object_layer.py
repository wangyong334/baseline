"""模块三 v5：目标神经元层（Object-Neuron Layer；Stream-SNN-Innovations.md 13.67）。

问题：逐像素的流式记忆里没有"目标"这个单位。每窗证据弱的目标，零等待判决只能猜；固定等待又把延迟浪费在已经确定的事件上。
v2～v4f 的航迹头把卡尔曼 / IPDA 写成网络外的程序（格子、局部峰、门、出生 / 消亡规则、手工上下文、按 EV-UAV 定的常数），
不能泛化（10-31 用户否定）。本层把同一套数学写成网络的结构，其中的未知量全部由网络学：

    1. 事件 = 各目标与杂波的泊松点过程（扩展目标量测模型，Gilholm & Salmond 2005）：每个事件直接是一次观测，不提取"量测"。
    2. 关联（PDA）：事件属于谁 = softmax(log 强度) -> 事件对"N 个目标槽 + 1 个背景槽"的注意力。
       目标槽的 logit = log(存在概率 x 期望事件数) + log N(事件位置; 预测位置, 不确定度) + 外观相似度；
       背景槽的 logit = 杂波强度（由事件特征学出，初始为"骨干 logit 越高越不像杂波"）。
    3. 预测 / 校正（卡尔曼；SAVi 的预测器 / 校正器）：槽的潜在状态经预测网络外推，位置按匀速 + 学出的漂移外推；
       注意力加权的质心作为量测，增益按卡尔曼公式由学出的协方差（过程噪声、目标展布）计算；EM 迭代 em_iters 次。
    4. 目标神经元（伯努利存在的对数几率递推 = Deneve 2008 的贝叶斯脉冲神经元）：膜电位 L = 存在对数几率。
       预测：L <- logit(p_S sigmoid(L))；一窗的证据：Lambda = w_L x sum_e log(1 + 后验几率_e / 先验) - 期望事件数，
       L <- L + log(1 - P_D + P_D e^Lambda)，P_D <= 1 - alpha（Cromwell：一窗没有事件，L 至多降 log(1 / alpha)）。
       w_L 的理论值为 1（事件独立时）；学出的值反映事件之间的相关程度。
    5. Wald 序贯检验 = 神经元的发放阈值 +-A（A = log((1 - alpha) / alpha)）：越过 +A 发"确认"，越过 -A 发"否定"并释放槽。
    6. 出生：空闲槽每窗落在"骨干认为最像目标、又没被现有目标解释"的事件上（依次放置，每放一个就按学出的新生展布把附近的证据
       算作已解释），先验存在对数几率 l0 学出来（初始 -A）。
    7. 判决时间：事件的最大后验归属是一个未判定（|L| < A）的目标时挂起；该目标的神经元发放（或等满 defer_max 窗）时终判。
       所有事件都有零等待输出 z_now = z0 + 注意力加权的读出（读出末层零初始化，起步与骨干逐位相同）；
       终判 = 用同一个读出，按所属目标在判决时刻的状态重读这一项：z_final = z_now + m x (u(目标此刻) - u(目标当时))。
       读出有界、没有新参数；不属于这个目标的事件（m -> 0）不受影响。
       分离原理：读出看到的存在概率与归属不回传梯度，存在、归属只由各自的校准损失训练（判决不改动估计）；
       端到端时目标层的估计部分读骨干输出也不回传梯度，骨干只从逐事件损失学习（一个目标的存在证据对上千个事件求和，
       回传会让目标层的估计损失主导骨干：实测其梯度是像素损失的约 27 倍）。
手设量只有：槽数（容量）、alpha、defer_max（延迟预算）、网络宽度、EM 迭代次数。长度单位 = 骨干的总步长（结构量，由网络读出）。
训练（与推理同一套规则）：逐事件 BCE（暂定 + 终判，由 train_track_head.py 计算）；槽与标注目标按"归属 dice + 存在"的代价做
    匈牙利匹配，上一窗匹配上的槽在目标仍在时保持同一身份（MOTR）；存在 BCE（神经元电位校准，Wald 错判界的前提）、
    归属交叉熵（注意力后验校准：目标事件归匹配的槽，其余事件归背景）、位置 Smooth-L1（以骨干步长为单位）。
接口与 model/track_head.TrackHead 相同（init_state / detach_state / forward_chunk），训练、评估共用 train_track_head.py。
实现注意：只用 torch 1.9 可用、确定性模式下允许的算子（累加用 index_put(accumulate=True)，取值用高级索引）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is part of the training environment
    linear_sum_assignment = None

EPS = 1e-6
L_CLAMP = 30.0                      # smooth numerical bound on the membrane (log-odds); far beyond +-A
OBJECT_KEYS = ("slots", "hidden", "attn_hidden", "em_iters", "defer_max", "defer_alpha", "clutter_source",
               "continuous_time", "object_context", "decision_rule", "separation")
SEPARATIONS = ("full", "selective")
DECISION_RULES = ("sprt", "fixed")
SLOT_KEYS = ("active", "gen", "mu", "v", "P", "h", "L", "pd", "n", "tid")
STATUS_NAMES = ("confirmed", "rejected", "reacquired", "lost", "forced")


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))


def _zero(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _const(layer, value):
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, float(value))
    return layer


def _inv_softplus(x):
    return math.log(math.expm1(float(x)))


def _logit(p):
    p = p.clamp(EPS, 1.0 - EPS)
    return torch.log(p) - torch.log1p(-p)


def _q_cat(a, b):
    if a is None or int(a["b"].shape[0]) == 0:
        return b
    if b is None or int(b["b"].shape[0]) == 0:
        return a
    return {k: torch.cat([a[k], b[k]]) for k in a}


def _q_take(q, mask):
    return None if q is None else {k: v[mask] for k, v in q.items()}


def _assign(cost):
    """最小代价一一匹配（行 = 槽，列 = 目标）。"""
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return list(rows), list(cols)
    cost = np.array(cost, dtype=np.float64, copy=True)                 # fallback: greedy on cost
    rows, cols = [], []
    while cost.size and np.isfinite(cost).any() and len(rows) < min(cost.shape):
        r, c = np.unravel_index(np.argmin(cost), cost.shape)
        rows.append(int(r))
        cols.append(int(c))
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return rows, cols


def backbone_stride(net):
    """骨干的总步长 = 解码器里步长 2 的上采样层数的 2 次幂（EvSpSegNetStream 为 8）。"""
    ups = sum(1 for m in net.modules() if isinstance(m, nn.ConvTranspose2d) and int(m.stride[0]) == 2)
    return float(2 ** ups) if ups else 1.0


class ObjectLayer(nn.Module):
    """目标神经元层。forward_chunk(events, z0, feat, state, steps, keep, labels, tid, log, keys) -> (z, state, aux)。"""

    def __init__(self, feat_dim, height, width, slots=16, hidden=32, attn_hidden=32, em_iters=2, defer_max=10,
                 defer_alpha=0.05, length_unit=8.0, clutter_source="learned", continuous_time=False, ref_rate=None,
                 object_context=True, decision_rule="sprt", separation="full"):
        """clutter_source：learned = 杂波强度只由事件特征学出（v1）；background = 杂波的空间强度取模块二的本地背景率
        （每像素每窗、按极性，下限为模块二的参照率），事件特征只学"像不像目标"的标记似然比（v2）。
        continuous_time：True 时每个事件按自己的时间戳与目标在那一刻的位置比较（窗内匀速），校正步用加权最小二乘
        一窗内同时量出位置与速度（v2）；False 为 v1（事件与窗中心位置比较，只量位置）。
        消融对照：object_context = False 时读出不看任何目标槽（z = z0 + 背景项，不挂起），检验增益来自目标而不是多了一个读出头；
        decision_rule = fixed 时挂在目标上的事件一律等满 defer_max 窗（目标被否定、槽被释放时提前终判），作 SPRT 的固定等待对照。
        separation（端到端时与骨干的耦合）：full = 目标层的估计都不回传骨干（v2）；selective = 关联（哪些事件属于同一目标）
        回传骨干，让骨干学会可聚成目标的事件特征（归属损失按事件平均，量级与像素损失相当），存在证据仍不回传。"""
        super(ObjectLayer, self).__init__()
        if clutter_source not in ("learned", "background"):
            raise ValueError("clutter_source 必须是 learned 或 background")
        self.clutter_source, self.continuous_time = str(clutter_source), bool(continuous_time)
        if decision_rule not in DECISION_RULES:
            raise ValueError("decision_rule 必须是 %s 之一" % (DECISION_RULES,))
        self.object_context, self.decision_rule = bool(object_context), str(decision_rule)
        if separation not in SEPARATIONS:
            raise ValueError("separation 必须是 %s 之一" % (SEPARATIONS,))
        self.separation = str(separation)
        self.feat_dim, self.height, self.width = int(feat_dim), int(height), int(width)
        self.slots, self.hidden, self.attn_hidden = int(slots), int(hidden), int(attn_hidden)
        self.em_iters, self.unit = int(em_iters), float(length_unit)
        if self.em_iters < 1:
            raise ValueError("em_iters 至少为 1")
        alpha = float(defer_alpha)
        if not 0.0 < alpha < 0.5:
            raise ValueError("defer_alpha 必须在 (0, 0.5) 内")
        self.alpha = alpha
        self.defer_A = math.log((1.0 - alpha) / alpha)
        self.pd_max = 1.0 - alpha                                      # Cromwell: one empty window costs <= log(1/alpha)
        self.defer_max = int(defer_max)
        if self.defer_max < 1:
            raise ValueError("defer_max 至少为 1 窗")
        # interface attributes read by train_track_head.py
        self.context, self.decision_lag, self.defer_revision = "object", 0, "object"
        self.has_delayed = True
        self.collect_calibration = False
        D, Hd = self.hidden, self.attn_hidden
        self.g_dim = g = 3 + self.feat_dim                             # [z0 / 5, dec1 feature, polarity, t_local]
        # association = PDA posterior as attention over (background, slots)
        self.event_key = nn.Linear(g, Hd)
        self.slot_query = _zero(nn.Linear(D, Hd))                     # appearance term starts at 0
        self.clutter = _mlp(g, Hd, 1)
        _zero(self.clutter[2])
        self.clutter_gain = nn.Parameter(torch.tensor(1.0))            # starts as "higher backbone logit = less clutter"
        if self.clutter_source == "background":
            # spatial clutter intensity = module 2's local background rate; this bias only shifts the mark term
            self.clutter_bias = nn.Parameter(torch.zeros(()))
            if ref_rate is None:
                from model.comod import REF_RATE
                ref_rate = REF_RATE
            self.register_buffer("ref_rate", torch.tensor([float(r) for r in ref_rate]), persistent=False)
        else:
            self.clutter_bias = nn.Parameter(torch.tensor(-math.log(2.0 * math.pi)))   # even odds at a slot centre
        # slot dynamics (predictor / corrector)
        self.pred = _mlp(D, D, D)
        _zero(self.pred[2])
        self.drift = _zero(nn.Linear(D, 2))                            # learned motion beyond constant velocity
        self.q_head = _const(nn.Linear(D, 1), _inv_softplus(0.25))     # process noise (units^2)
        self.extent_head = _const(nn.Linear(D, 1), _inv_softplus(1.0))  # target extent variance (units^2)
        self.rate_head = _const(nn.Linear(D, 1), _inv_softplus(1.0))   # expected events per window
        self.pd_head = _const(nn.Linear(D, 1), math.log(4.0))
        self.surv_head = _const(nn.Linear(D, 1), math.log(0.995 / 0.005))
        self.u_dim = 1 + g + 2 + 1
        self.gru = nn.GRUCell(self.u_dim, D)
        # birth
        self.h_init = nn.Parameter(torch.zeros(D))
        self.birth_logit = nn.Parameter(torch.tensor(-self.defer_A))  # newborn prior l0, starts at -A
        self.log_s0 = nn.Parameter(torch.zeros(()))                     # newborn position variance (units^2)
        self.log_v0 = nn.Parameter(torch.zeros(()))                     # newborn velocity variance (units^2/window^2)
        # evidence scale with theoretical value 1 (independent events)
        self.log_llr_scale = nn.Parameter(torch.zeros(()))
        # read-out (last layers zero: z_now == z0 at initialisation)
        self.read_slot = nn.Linear(D + 1, Hd)
        self.read_event = nn.Linear(g, Hd)
        self.read_out = _zero(nn.Linear(Hd, 1))
        self.read_bg = _mlp(g, Hd, 1)
        _zero(self.read_bg[2])

    # ------------------------------------------------------------------ state
    def init_state(self, batch, device):
        B, N, D = int(batch), self.slots, self.hidden
        zeros = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        return {"active": torch.zeros(B, N, dtype=torch.bool, device=device),
                "gen": torch.zeros(B, N, dtype=torch.long, device=device),
                "mu": zeros(B, N, 2), "v": zeros(B, N, 2), "P": zeros(B, N, 2, 2), "h": zeros(B, N, D),
                "L": zeros(B, N), "pd": zeros(B, N), "n": zeros(B, N), "tid": zeros(B, N), "defer": None}

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
        """mask [B] bool：这些流开始新录像，目标槽全部清空，该流挂起的事件保留零等待输出、不再终判。"""
        out = {}
        for key in SLOT_KEYS:
            value = state[key]
            view = mask.view((-1,) + (1,) * (value.dim() - 1))
            out[key] = torch.where(view, torch.zeros_like(value), value)
        q = state.get("defer")
        out["defer"] = None if q is None else {k: v[~mask[q["b"]]] for k, v in q.items()}
        return out

    def evidence_summary(self):
        """论文里报的系数：新生先验 l0、似然比尺度 w_L（理论 1）、杂波对骨干 logit 的增益。"""
        return {"orphan_ref": float(self.birth_logit.detach()),
                "llr_scale": float(torch.exp(self.log_llr_scale.detach())),
                "clutter_gain": float(self.clutter_gain.detach())}

    def init_rate(self, mean_events):
        """期望事件数的初值 = 训练集上每个目标每窗的平均事件数（相当于先做一次最大似然估计；之后由网络继续学）。"""
        with torch.no_grad():
            self.rate_head.bias.fill_(_inv_softplus(max(float(mean_events), 1.0)))

    # ------------------------------------------------------------------ pieces
    def _event_features(self, ev):
        z0 = ev["z0"]
        return torch.cat([z0.clamp(-15.0, 15.0)[:, None] / 5.0, ev["feat"], ev["p"].to(z0.dtype)[:, None],
                          ev["t_local"].to(z0.dtype)[:, None]], 1)

    def _slot_sum(self, eb, vals, batch):
        """vals [E, N, C] -> 每个流每个槽对本流事件求和 [B, N, C]。"""
        E, N, C = vals.shape
        if E == 0:
            return vals.new_zeros((batch, N, C))
        base = (eb[:, None] * N + torch.arange(N, device=eb.device)[None, :]) * C          # [E, N]
        flat = (base[:, :, None] + torch.arange(C, device=eb.device)[None, None, :]).reshape(-1)
        buf = vals.new_zeros(batch * N * C)
        buf = buf.index_put((flat,), vals.reshape(-1), accumulate=True)
        return buf.view(batch, N, C)

    def _assoc_logits(self, exy, eb, key, clut, mu, S, prior, valid, qry, vel=None, se=None, pv=None):
        """[E, N + 1]：第 0 列背景（杂波强度），其余为本流各槽 log(r gamma) + log N(x; mu, S) + 外观相似度。

        连续时间（vel 给出时）：事件与目标在它自己时刻的位置 mu + v s 比较（s = 窗内时间 - 1/2），
        方差 = Var(mu + v s) + 展布 = P00 + 2 s P01 + s^2 P11 + ext（pv = (P01, P11)）。"""
        center, Se = mu[eb], S[eb]
        if vel is not None:
            sc = se[:, None]
            center = center + vel[eb] * sc[..., None]
            Se = Se + 2.0 * sc * pv[0][eb] + sc * sc * pv[1][eb]
        d2 = ((exy[:, None, :] - center) ** 2).sum(-1)
        psi = (qry[eb] * key[:, None, :]).sum(-1) / math.sqrt(float(self.attn_hidden))
        a = prior[eb] - 0.5 * d2 / Se - torch.log(2.0 * math.pi * Se) + psi
        a = torch.where(valid[eb], a, torch.full_like(a, -1e4))
        return torch.cat([clut[:, None], a], 1)

    @torch.no_grad()
    def _birth(self, exy, eb, prob, key, clut, log_rest, free):
        """依次出生（贝叶斯的逐个解释）：每条流每步在"骨干认为最像目标、又没被解释"的事件上试放一个新目标，
        算它本窗的存在证据；没被 SPRT 立即否定（L > -A）才接受，并把它按后验存在概率解释掉的强度计入，再试下一个。
        同一目标上的第二个候选证据为负（事件已被第一个解释），自然被拒绝；不设任何阈值。

        log_rest [E]：每个事件已有的强度（杂波 + 现有目标），log 空间。返回接受的 (流, 槽, 事件) 列表。"""
        B, N = free.shape
        E = int(exy.shape[0])
        dev, dtype = exy.device, exy.dtype
        A = self.defer_A
        h0 = self.h_init.to(dtype)
        S0 = torch.exp(self.log_s0) + F.softplus(self.extent_head(h0)).squeeze(-1) + EPS
        log_gam = torch.log(F.softplus(self.rate_head(h0)).squeeze(-1) + EPS)
        pd_logit = self.pd_head(h0).squeeze(-1)
        log_pd = math.log(self.pd_max) + F.logsigmoid(pd_logit)
        log_miss = torch.logaddexp(math.log(1.0 - self.pd_max) + torch.zeros_like(pd_logit),
                                   math.log(self.pd_max) + F.logsigmoid(-pd_logit))
        psi0 = (self.slot_query(h0)[None, :] * key).sum(-1) / math.sqrt(float(self.attn_hidden))     # [E]
        w_L, l0, gam = torch.exp(self.log_llr_scale), self.birth_logit, torch.exp(log_gam)
        supp = torch.ones(E, device=dev, dtype=dtype)                              # tried locations are not retried
        same = eb[None, :] == torch.arange(B, device=dev)[:, None]                   # [B, E]
        slot_ids = torch.arange(N, device=dev)
        free_order = torch.argsort((~free).long() * N + slot_ids[None, :], 1)      # free slots first, in index order
        n_free = free.long().sum(1)
        used = torch.zeros(B, dtype=torch.long, device=dev)
        rows = torch.arange(B, device=dev)
        records = []
        for _ in range(N):                                                           # masked loop: no host syncs
            score = prob * torch.exp(clut - log_rest) * supp                         # target-like x unexplained
            masked = torch.where(same, score[None, :].expand(B, E), torch.full((B, E), -1.0, device=dev, dtype=dtype))
            val, idx = masked.max(1)
            trial = (val > 0.0) & (used < n_free)
            seed = exy[idx]                                                          # [B, 2]
            d2 = ((exy - seed[eb]) ** 2).sum(1)
            a_c = log_gam - 0.5 * d2 / S0 - torch.log(2.0 * math.pi * S0) + psi0    # candidate log-intensity [E]
            term = torch.where(trial[eb], F.softplus(a_c - log_rest), torch.zeros_like(a_c))
            ssum = torch.zeros(B, device=dev, dtype=dtype).index_put((eb,), term, accumulate=True)
            L_post = l0 + torch.logaddexp(log_miss + torch.zeros_like(ssum), log_pd + w_L * ssum - gam)
            accept = trial & (L_post > -A)
            records.append((accept, free_order[rows, used.clamp(max=N - 1)], idx))
            log_rest = torch.where(accept[eb], torch.logaddexp(log_rest, F.logsigmoid(L_post)[eb] + a_c), log_rest)
            supp = torch.where(trial[eb], supp * (1.0 - torch.exp(-0.5 * d2 / S0)), supp)
            used = used + accept.long()
        acc = torch.stack([r[0] for r in records])                                   # [steps, B]
        slot = torch.stack([r[1] for r in records])
        ev_i = torch.stack([r[2] for r in records])
        step_i, b_i = torch.nonzero(acc, as_tuple=True)
        picks = list(zip(b_i.tolist(), slot[step_i, b_i].tolist(), ev_i[step_i, b_i].tolist()))
        return picks

    def _motion(self, vel, se, pv):
        """连续时间时给 _assoc_logits 的附加参数；否则为空。"""
        return (vel, se, pv) if self.continuous_time else ()

    def _wls_update(self, eb, exy, se, bs, B, mu0, v0, a, b, c, ext, valid):
        """连续时间校正：注意力加权的最小二乘在一窗内量出位置（在加权平均时刻 s_bar）与速度，按卡尔曼公式更新 [位置, 速度]。

        量测 z = [x_bar, v_meas]，H = [[1, s_bar], [0, 1]]，R = diag(ext / n, ext / S_ss)（S_ss = 时间的加权离差平方和）；
        事件都在同一时刻或几乎没有事件时 R 趋于无穷，自动退化为只量位置 / 不更新。返回 (mu, v, 位置新息, P, n)。"""
        sc = se[:, None]
        cols = torch.cat([bs[..., None], (bs * sc)[..., None], (bs * sc * sc)[..., None], bs[..., None] * exy[:, None, :],
                          (bs * sc)[..., None] * exy[:, None, :]], -1)                         # [E, N, 7]
        sums = self._slot_sum(eb, cols, B)
        n = sums[..., 0]
        nc = n.clamp(min=EPS)
        sbar = sums[..., 1] / nc
        Sss = (sums[..., 2] - n * sbar * sbar).clamp(min=0.0)
        xbar = sums[..., 3:5] / nc[..., None]
        Sxs = sums[..., 5:7] - (n * sbar)[..., None] * xbar
        vmeas = Sxs / Sss.clamp(min=EPS)[..., None]
        R1, R2 = ext / nc, ext / Sss.clamp(min=EPS)
        S11 = a + 2.0 * sbar * b + sbar * sbar * c + R1
        S12 = b + sbar * c
        S22 = c + R2
        det = (S11 * S22 - S12 * S12).clamp(min=EPS)
        ph0, ph1 = a + sbar * b, b + sbar * c                                                 # P H^T, first column
        K00, K01 = (ph0 * S22 - b * S12) / det, (b * S11 - ph0 * S12) / det
        K10, K11 = (ph1 * S22 - c * S12) / det, (c * S11 - ph1 * S12) / det
        ok = (valid & (n > EPS))[..., None]
        nu1 = torch.where(ok, xbar - (mu0 + sbar[..., None] * v0), torch.zeros_like(mu0))
        nu2 = torch.where(ok, vmeas - v0, torch.zeros_like(v0))
        mu = mu0 + K00[..., None] * nu1 + K01[..., None] * nu2
        v = v0 + K10[..., None] * nu1 + K11[..., None] * nu2
        g0, g1 = K00 * sbar + K01, K10 * sbar + K11                                         # (K H)[:, 1]
        q00 = (1.0 - K00) * a - g0 * b
        q01 = 0.5 * (((1.0 - K00) * b - g0 * c) + (-K10 * a + (1.0 - g1) * b))
        q11 = -K10 * b + (1.0 - g1) * c
        P = torch.stack([torch.stack([q00, q01], -1), torch.stack([q01, q11], -1)], -2)
        return mu, v, nu1, P, n

    # ------------------------------------------------------------------ one window
    def forward_window(self, state, ev, reset_mask=None, label=None, tid=None, aux=None):
        """state：init_state 的字典；ev：本窗事件 b/x/y/p/t_local/z0/feat/key。label / tid / aux 给出时计算监督。

        返回 (z [Ne] 零等待输出, 新状态, info)；info["delayed"] 为 None 或 (终判 logit, key, 等待窗数, 状态码)。
        """
        if reset_mask is not None:
            state = self._reset(state, reset_mask)
        dq = state.get("defer")
        if dq is not None:
            dq = dict(dq, lag=dq["lag"] + 1)                         # entries from earlier windows waited one more
        alive = state["active"]
        B, N = alive.shape
        dev = alive.device
        z0, feat = ev["z0"], ev["feat"]
        dtype = z0.dtype
        E = int(z0.shape[0])
        A, u = self.defer_A, self.unit
        zBN = torch.zeros(B, N, device=dev, dtype=dtype)

        # 1. predict (Bernoulli existence + Kalman / learned predictor)
        hp = state["h"] + self.pred(state["h"])
        q = F.softplus(self.q_head(hp).squeeze(-1)) + EPS
        P = state["P"]
        p00 = P[..., 0, 0] + 2.0 * P[..., 0, 1] + P[..., 1, 1] + 0.25 * q
        p01 = P[..., 0, 1] + P[..., 1, 1] + 0.5 * q
        p11 = P[..., 1, 1] + q
        mu_p = state["mu"] + state["v"] + self.drift(hp)
        v_p = state["v"]
        L_p = _logit(torch.sigmoid(self.surv_head(hp).squeeze(-1)) * torch.sigmoid(state["L"]))

        if E:
            eb = ev["b"]
            exy = torch.stack([ev["x"], ev["y"]], 1).to(dtype) / u
            g = self._event_features(ev)
            # separation (2): the object estimates (association, existence, position) read the backbone without pushing
            # gradients back into it; the backbone learns only from per-event losses (its own read-out, this layer's z)
            gd, z0d = g.detach(), z0.detach()
            ga, z0a = (g, z0) if self.separation == "selective" else (gd, z0d)          # association inputs
            key = self.event_key(ga)
            mark = self.clutter_bias - self.clutter_gain * z0a.clamp(-15.0, 15.0) + self.clutter(ga).squeeze(-1)
            if self.clutter_source == "background":
                if ev.get("bg_event") is None:
                    raise ValueError("clutter_source = background 需要每个事件的背景率 ev['bg_event']（模块二）")
                floor = self.ref_rate.to(dtype)[(ev["p"] < 0.5).long()]     # channel 0 = ON (p = 1), 1 = OFF
                clut = torch.log(torch.maximum(ev["bg_event"].to(dtype), floor) * (u * u)) + mark
            else:
                clut = mark
            se = ev["t_local"].to(dtype) - 0.5 if self.continuous_time else None

        # 2. birth: free slots seeded at the most target-like unexplained events
        s0, v0 = torch.exp(self.log_s0), torch.exp(self.log_v0)
        free = ~alive
        cand = torch.zeros_like(alive)
        gen = state["gen"]
        seed_mu = torch.zeros_like(state["mu"])
        if E and bool(free.any()):
            with torch.no_grad():
                prior_a = F.logsigmoid(L_p) + torch.log(F.softplus(self.rate_head(hp).squeeze(-1)) + EPS)
                S_a = p00 + F.softplus(self.extent_head(hp).squeeze(-1)) + EPS
                lg_a = self._assoc_logits(exy, eb, key, clut, mu_p, S_a, prior_a, alive, self.slot_query(hp),
                                          *self._motion(v_p, se, (p01, p11)))
                picks = self._birth(exy, eb, torch.sigmoid(z0), key, clut, torch.logsumexp(lg_a, 1), free)
            pb, ps, pe = [p[0] for p in picks], [p[1] for p in picks], [p[2] for p in picks]
            if pb:
                bt = torch.tensor(pb, dtype=torch.long, device=dev)
                st = torch.tensor(ps, dtype=torch.long, device=dev)
                et = torch.tensor(pe, dtype=torch.long, device=dev)
                cand = cand.index_put((bt, st), torch.ones(len(pb), dtype=torch.bool, device=dev))
                seed_mu = seed_mu.index_put((bt, st), exy[et].detach())
                gen = gen.index_put((bt, st), gen[bt, st] + 1)
        valid = alive | cand
        cv = cand[..., None]
        h0 = torch.where(cv, self.h_init.to(dtype).expand_as(hp), hp)
        mu0 = torch.where(cv, seed_mu, mu_p)
        v0t = torch.where(cv, torch.zeros_like(v_p), v_p)
        p00 = torch.where(cand, s0.to(dtype).expand_as(p00), p00)
        p01 = torch.where(cand, torch.zeros_like(p01), p01)
        p11 = torch.where(cand, v0.to(dtype).expand_as(p11), p11)
        L0 = torch.where(cand, self.birth_logit.to(dtype).expand_as(L_p), L_p)

        # 3. association (EM) and Kalman correction
        ext = F.softplus(self.extent_head(h0).squeeze(-1)) + EPS
        gam = F.softplus(self.rate_head(h0).squeeze(-1)) + EPS
        pd_logit = self.pd_head(h0).squeeze(-1)
        log_pd = math.log(self.pd_max) + F.logsigmoid(pd_logit)
        log_miss = torch.logaddexp(torch.full_like(pd_logit, math.log(1.0 - self.pd_max)),
                                   math.log(self.pd_max) + F.logsigmoid(-pd_logit))      # log(1 - P_D)
        logr = F.logsigmoid(L0)
        S = p00 + ext
        qry = self.slot_query(h0)
        mu_c, v_c, n = mu0, v0t, zBN
        innov = torch.zeros_like(mu0)
        P_post = torch.stack([torch.stack([p00, p01], -1), torch.stack([p01, p11], -1)], -2)
        if E:
            prior = logr + torch.log(gam)
            for _ in range(self.em_iters):
                log_beta = torch.log_softmax(self._assoc_logits(exy, eb, key, clut, mu_c, S, prior, valid, qry,
                                                                *self._motion(v_c, se, (p01, p11))), 1)
                bs = torch.exp(log_beta[:, 1:])
                if self.continuous_time:
                    mu_c, v_c, innov, P_post, n = self._wls_update(eb, exy, se, bs, B, mu0, v0t, p00, p01, p11, ext,
                                                                   valid)
                else:
                    sums = self._slot_sum(eb, torch.cat([bs[..., None], bs[..., None] * exy[:, None, :]], -1), B)
                    n = sums[..., 0]
                    xbar = sums[..., 1:] / n.clamp(min=EPS)[..., None]
                    Sm = p00 + ext / n.clamp(min=EPS)
                    k0, k1 = p00 / Sm, p01 / Sm
                    innov = torch.where((valid & (n > EPS))[..., None], xbar - mu0, torch.zeros_like(mu0))
                    mu_c = mu0 + k0[..., None] * innov
                    v_c = v0t + k1[..., None] * innov
                    q00, q01, q11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01
                    P_post = torch.stack([torch.stack([q00, q01], -1), torch.stack([q01, q11], -1)], -2)
            beta = torch.exp(log_beta)
            lb = log_beta[:, 1:]
            if self.separation == "selective":                       # existence evidence never trains the backbone
                lb = lb.detach()
            # 4. object neuron: each event adds log(1 + posterior odds / prior); the leak is the expected count
            log1m = torch.log1p(-torch.exp(lb).clamp(max=1.0 - EPS))
            term = torch.where(valid[eb], F.softplus(lb - log1m - logr[eb]), torch.zeros_like(lb))
            ssum = self._slot_sum(eb, term[..., None], B)[..., 0]
            gsum = self._slot_sum(eb, bs[..., None] * gd[:, None, :], B)
        else:
            ssum = zBN
            gsum = torch.zeros(B, N, self.g_dim, device=dev, dtype=dtype)
        Lam = torch.exp(self.log_llr_scale) * ssum - gam
        # smooth bound keeps gradients (a wrongly confirmed object must still be correctable by training)
        L_post = torch.where(valid, L_CLAMP * torch.tanh((L0 + torch.logaddexp(log_miss, log_pd + Lam)) / L_CLAMP), zBN)

        # 5. latent update (corrector) and zero-wait read-out
        gbar = gsum / n.clamp(min=EPS)[..., None]
        nin = innov / torch.sqrt(S)[..., None]
        uin = torch.cat([torch.log1p(n)[..., None], gbar, nin, torch.tanh(Lam / A)[..., None]], -1)
        h_new = self.gru(uin.reshape(B * N, self.u_dim), h0.reshape(B * N, self.hidden)).view(B, N, self.hidden)
        h_new = torch.where(valid[..., None], h_new, torch.zeros_like(h_new))
        if E:
            bd = beta.detach()                                       # separation: decisions do not move beliefs
            qs = self.read_slot(torch.cat([h_new, torch.sigmoid(L_post).detach()[..., None]], -1))
            qe = self.read_event(g)
            u_slot = self.read_out(F.relu(qs[eb] + qe[:, None, :])).squeeze(-1)
            u_bg = self.read_bg(g).squeeze(-1)
            if self.object_context:
                z = z0 + bd[:, 0] * u_bg + (bd[:, 1:] * u_slot).sum(1)
            else:                                                    # control: same read-out capacity, no objects
                z = z0 + u_bg
        else:
            z = z0

        # 6. release: pending events whose object neuron fired, whose slot is gone, or that reached the deadline
        delayed = None
        if dq is not None and int(dq["b"].shape[0]):
            b, j = dq["b"], dq["slot"]
            same = valid[b, j] & (gen[b, j] == dq["gen"])
            Lr = L_post[b, j]
            status = torch.full_like(b, -1)
            confirm = same & (Lr >= A) if self.decision_rule == "sprt" else torch.zeros_like(same)
            for code, cond in ((3, ~same), (0, confirm), (1, same & (Lr <= -A))):
                status = torch.where(cond & (status < 0), torch.full_like(status, code), status)
            status = torch.where((status < 0) & (dq["lag"] >= self.defer_max), torch.full_like(status, 4), status)
            rel = status >= 0
            if bool(rel.any()):
                r = _q_take(dq, rel)
                sr = status[rel]
                br, jr = r["b"], r["slot"]
                qs_now = self.read_slot(torch.cat([h_new[br, jr], torch.sigmoid(Lr[rel]).detach()[:, None]], -1))
                u_now = self.read_out(F.relu(qs_now + r["qe"])).squeeze(-1)
                u_now = torch.where(sr == 3, r["u_then"], u_now)           # lost: the object's outcome is unknown
                zf = r["znow"] + r["m_then"] * (u_now - r["u_then"])
                delayed = (zf, r["key"], r["lag"], sr)
            dq = _q_take(dq, ~rel)
        # 7. this window's events wait if their most probable object is undecided
        if E:
            amax = log_beta.argmax(1)
            kk = (amax - 1).clamp(min=0)
            Lk = L_post[eb, kk]
            if not self.object_context:
                pend = torch.zeros_like(amax, dtype=torch.bool)
            elif self.decision_rule == "fixed":                       # control: every object-attached event waits
                pend = (amax > 0) & (Lk > -A)
            else:
                pend = (amax > 0) & (Lk.abs() < A)
            if bool(pend.any()):
                rows = torch.arange(E, device=dev)
                key_e = ev["key"].to(dtype) if "key" in ev else torch.zeros_like(z0)
                entry = {"b": eb, "slot": kk, "gen": gen[eb, kk], "L_then": Lk.detach(), "m_then": bd[rows, amax],
                         "u_then": u_slot[rows, kk], "qe": qe, "znow": z, "key": key_e, "lag": torch.zeros_like(eb)}
                dq = _q_cat(dq, _q_take(entry, pend))

        # 8. supervision (training): Hungarian matching of slots and labelled objects
        tid_new = torch.zeros_like(state["tid"])
        if aux is not None and label is not None:
            tid_new = self._supervise(ev, label, tid, log_beta if E else None, L_post, mu_c, P_post, valid,
                                      state["tid"] * alive.to(dtype), aux)

        # 9. rejected objects release their slot
        alive_new = valid & (L_post > -A)
        av, am = alive_new[..., None], alive_new[..., None, None]
        new_state = {"active": alive_new, "gen": gen,
                     "mu": torch.where(av, mu_c, torch.zeros_like(mu_c)),
                     "v": torch.where(av, v_c, torch.zeros_like(v_c)),
                     "P": torch.where(am, P_post, torch.zeros_like(P_post)),
                     "h": torch.where(av, h_new, torch.zeros_like(h_new)),
                     "L": torch.where(alive_new, L_post, zBN),
                     "pd": torch.where(alive_new, torch.exp(log_pd), zBN),
                     "n": torch.where(alive_new, n, zBN),
                     "tid": torch.where(alive_new, tid_new, zBN), "defer": dq}
        info = {"births": int((cand & alive_new).sum()), "delayed": delayed}
        return z, new_state, info

    def _supervise(self, ev, label, tid, log_beta, L_post, mu, P_post, valid, prev_tid, aux):
        """存在 BCE、归属交叉熵、位置高斯负对数似然；返回本窗匹配上的目标号 [B, N]（未匹配为 0）。"""
        B, N = valid.shape
        dev, dtype = L_post.device, L_post.dtype
        target = torch.zeros_like(L_post)
        tid_new = torch.zeros_like(L_post)
        E = 0 if log_beta is None else int(log_beta.shape[0])
        sel = (label > 0.5) & (tid > 0) if E else None
        cls = torch.zeros(E, dtype=torch.long, device=dev)
        if E and bool(sel.any()):
            eb = ev["b"]
            keyg = eb[sel] * 1000003 + tid[sel].to(torch.long)
            uniq, inv = torch.unique(keyg, return_inverse=True)
            G = int(uniq.shape[0])
            gb = torch.div(uniq, 1000003, rounding_mode="floor")
            gt = (uniq - gb * 1000003).to(dtype)
            ev_idx = torch.nonzero(sel, as_tuple=False).flatten()
            M = torch.zeros(E, G, device=dev, dtype=dtype).index_put((ev_idx, inv), torch.ones_like(inv, dtype=dtype))
            size = M.sum(0)
            exy = torch.stack([ev["x"], ev["y"]], 1).to(dtype) / self.unit
            cen = (M.t() @ exy) / size[:, None]
            bs = torch.exp(log_beta[:, 1:])
            inter = bs.t() @ M                                                               # [N, G]
            tot = self._slot_sum(eb, bs[..., None], B)[..., 0]                               # [B, N]
            dice = 2.0 * inter / (tot[gb].t() + size[None, :] + EPS)
            cost = (1.0 - dice) + (1.0 - torch.sigmoid(L_post)[gb].t())
            cost_np, dice_np = cost.detach().cpu().numpy(), dice.detach().cpu().numpy()
            valid_np, prev_np = valid.cpu().numpy(), prev_tid.detach().cpu().numpy()
            gb_np, gt_np = gb.cpu().numpy(), gt.cpu().numpy()
            pairs = []
            for b in range(B):
                gs = [x for x in range(G) if gb_np[x] == b]
                ks = [k for k in range(N) if valid_np[b, k]]
                if not gs or not ks:
                    continue
                used_k, used_g = set(), set()
                for x in gs:                                 # identity persistence: keep last window's assignment
                    for k in ks:
                        if k not in used_k and prev_np[b, k] == gt_np[x] and dice_np[k, x] > 0.0:
                            pairs.append((b, k, x))
                            used_k.add(k)
                            used_g.add(x)
                            break
                rk = [k for k in ks if k not in used_k]
                rg = [x for x in gs if x not in used_g]
                if rk and rg:
                    sub = cost_np[np.ix_(rk, rg)].astype(np.float64)
                    ok = dice_np[np.ix_(rk, rg)] > 0.0
                    sub[~ok] = 1e6
                    rows, cols = _assign(sub)
                    for r, c in zip(rows, cols):
                        if ok[r, c]:
                            pairs.append((b, rk[r], rg[c]))
            if pairs:
                pb = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
                pk = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev)
                pg = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=dev)
                target = target.index_put((pb, pk), torch.ones(len(pairs), device=dev, dtype=dtype))
                tid_new = tid_new.index_put((pb, pk), gt[pg])
                aux["pos_sum"] = aux["pos_sum"] + F.smooth_l1_loss(mu[pb, pk], cen[pg], reduction="sum")
                aux["pos_n"] += len(pairs)
                slot_of = torch.full((G,), -1, dtype=torch.long, device=dev)
                slot_of = slot_of.index_put((pg,), pk)
                ev_slot = slot_of[inv]
                cls = cls.index_put((ev_idx,), torch.where(ev_slot >= 0, ev_slot + 1, torch.zeros_like(ev_slot)))
        if E:
            rows = torch.arange(E, device=dev)
            nll = -log_beta[rows, cls]
            pos = (label > 0.5) & (tid > 0)
            loss = nll[~pos].mean() if bool((~pos).any()) else nll.new_zeros(())
            if bool(pos.any()):
                loss = loss + nll[pos].mean()
            aux["mem_sum"] = aux["mem_sum"] + loss
            aux["mem_n"] += 1
        if bool(valid.any()):
            aux["exist_sum"] = aux["exist_sum"] + F.binary_cross_entropy_with_logits(
                L_post[valid], target[valid], reduction="sum")
            aux["exist_n"] += int(valid.sum())
            if self.collect_calibration:
                aux.setdefault("calib", []).append((L_post[valid].detach(), target[valid].detach()))
        return tid_new

    # ------------------------------------------------------------------ chunk
    def forward_chunk(self, events, z0, feat, state, steps, keep=None, labels=None, tid=None, log=None, keys=None):
        """events：片段事件（t 为片段内窗号、b 为流号）；z0 [Ne]、feat [Ne, F] 来自骨干。接口与 TrackHead.forward_chunk 相同。"""
        if state is None:
            batch = int(keep.shape[1]) if keep is not None else int(events["b"].max()) + 1 if events["b"].numel() else 1
            state = self.init_state(batch, z0.device)
        aux = {"exist_sum": z0.new_zeros(()), "exist_n": 0, "pos_sum": z0.new_zeros(()), "pos_n": 0,
               "pd_sum": z0.new_zeros(()), "pd_n": 0, "org_sum": z0.new_zeros(()), "org_n": 0,
               "mem_sum": z0.new_zeros(()), "mem_n": 0, "births": 0, "recycled": 0, "live": 0, "slot_windows": 0}
        idx_parts, z_parts, dz, dk, dl, ds = [], [], [], [], [], []
        for t in range(int(steps)):
            idx = torch.nonzero(events["t"] == t, as_tuple=False).flatten()
            ev = {key: events[key][idx] for key in ("b", "x", "y", "p", "t_local")}
            ev["z0"], ev["feat"] = z0[idx], feat[idx]
            if events.get("bg_event") is not None:
                ev["bg_event"] = events["bg_event"][idx]
            ev["key"] = keys[idx] if keys is not None else idx.to(z0.dtype)
            reset = (keep[t] == 0) if keep is not None else None
            sup = labels is not None
            z_t, state, info = self.forward_window(state, ev, reset, labels[idx] if sup else None,
                                                   tid[idx] if sup else None, aux if sup else None)
            idx_parts.append(idx)
            z_parts.append(z_t)
            if info["delayed"] is not None:
                for lst, value in zip((dz, dk, dl, ds), info["delayed"]):
                    lst.append(value)
            aux["births"] += info["births"]
            aux["live"] += int(state["active"].sum())
            aux["slot_windows"] += int(state["active"].shape[0])
            if log is not None:
                act = state["active"]
                log.append({"active": act.cpu().numpy(), "pos": (state["mu"] * self.unit).detach().cpu().numpy(),
                            "vel": (state["v"] * self.unit).detach().cpu().numpy(),
                            "ex": torch.where(act, torch.sigmoid(state["L"]), torch.zeros_like(state["L"]))
                            .detach().cpu().numpy(),
                            "pd": state["pd"].detach().cpu().numpy(), "beta": state["n"].detach().cpu().numpy()})
        if idx_parts and sum(int(i.shape[0]) for i in idx_parts):
            z = z0.index_put((torch.cat(idx_parts),), torch.cat(z_parts))
        else:
            z = z0
        aux["delayed_z"] = torch.cat(dz) if dz else None
        aux["delayed_key"] = torch.cat(dk) if dk else None
        aux["delayed_lag"] = torch.cat(dl) if dl else None
        aux["delayed_status"] = torch.cat(ds) if ds else None
        return z, state, aux
