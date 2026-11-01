"""模块三 v6：目标 token 层（Wald 目标神经元的网络写法；Stream-SNN-Innovations.md 13.71）。

问题：逐像素的流式记忆里没有"目标"这个单位。每窗证据弱的目标，零等待判决只能猜；固定等待又把延迟浪费在已经确定的事件上。
v5（model/object_layer.py）把多目标滤波写成了 forward 里的程序：协方差预测、卡尔曼 / 最小二乘增益、高斯关联、杂波公式、
伯努利存在递推、依次出生 + 接受规则、释放状态机，网络只填其中二十来个小参数（11-01 用户否定：仍是手写）。
本层按通用的网络写法重写（PACT / PointEvent / ASA-SNN 的写法：状态是隐向量，原理只进结构与初始化，"信多少"交给学出的门）：

    槽 = 隐向量 token + 参考点（参考点只用来给注意力加位置偏置，DAB-DETR）；每窗位移由学出的头给出（PACT 的速度头）；
    预测 = 槽上的残差 MLP（SAVi 的预测器）；
    关联 = 事件对"背景 token + 槽"的交叉注意力，softmax 竞争（Slot Attention；竞争即 PDA 的后验归属）；
           位置项 = 学出宽度的位置调制（DAB-DETR）；背景 logit 由事件特征（与模块二的杂波率）学出；
    校正 = GRU（门 = 学出的增益，KalmanNet）：输入是注意力加权的特征均值与位置 / 时间的一、二阶矩（窗内运动的信息都在里面，
           怎么用由网络学）；参考点 += 学出的门 x 新息；槽之间自注意力（DETR 解码器：学会去重）；迭代 iters 次 = 展开的 EM；
    存在 = 每个槽一个脉冲神经元：膜电位按学出的漏电（随槽状态变化，Mamba 式选择性衰减）漏向静息电位，积分学出的证据电流，
           用 BCE 校准成存在对数几率；越过 +-A 发放（A = log((1 - alpha) / alpha)，alpha 是需求量，不学）：
           + 确认，- 否定并释放槽（Wald 的序贯检验：证据校准时错判率不超过 alpha）；
    出生 = 学出的提议：每窗在"骨干认为像目标、又没被现有槽解释"的事件里取前 proposals 个，经 MLP 生成候选 token
           （DETR 两阶段 / MOTR 的检测 query）；不设接受规则：候选与老槽受同一个神经元支配，重复的候选由匈牙利匹配
           （只有一个能匹配上）训练成否定；
    判决时间 = 事件跟随最大后验所属槽的脉冲终判（槽发放、槽被释放或等满 defer_max 窗）；终判 = 用同一个读出按槽在判决时刻的
           状态重读：z_final = z_now + m x (u(槽此刻) - u(槽当时))。
读出与分离原理同 v5：零等待 z_now = z0 + 注意力加权的读出（末层零初始化，起步与骨干逐位相同）；读出看到的归属与存在概率不回传；
separation = full 时目标层的估计都读不回传的骨干输出，selective 时归属（关联）回传骨干、存在 / 位置不回传。
手设量只有：槽数、每窗提议数、宽度、迭代次数（架构量）；alpha、defer_max（需求量）。没有数据集统计常数。
初始化：漏电 1 - 1 / defer_max（按延迟预算遗忘）、静息电位 -A（新 token 停在否定界上，要靠证据才活下来）、
        证据、门、预测、内容相似度的末层零初始化。
训练（与推理同一套规则）：逐事件 BCE（零等待 + 终判，由训练脚本计算）；槽与标注目标按"归属 dice + 存在"的代价做匈牙利匹配，
    上一窗匹配上的槽在目标仍在时保持同一身份（MOTR）；存在 BCE（膜电位校准）、归属交叉熵、位置 Smooth-L1（以骨干步长为单位）。
接口与 model/object_layer.ObjectLayer 相同（init_state / detach_state / forward_chunk），训练、评估共用 train_track_head.py 与
train_end2end.py。只用 torch 1.9 可用、确定性模式下允许的算子（按流求和用矩阵乘，取值用高级索引）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is part of the training environment
    linear_sum_assignment = None

EPS = 1e-6
NEG = -1e4
TOKEN_KEYS = ("slots", "proposals", "hidden", "attn_hidden", "iters", "defer_max", "defer_alpha", "clutter_source",
              "separation", "object_context", "decision_rule")
CLUTTER_SOURCES = ("none", "background")
SEPARATIONS = ("full", "selective")
DECISION_RULES = ("sprt", "fixed")
SLOT_KEYS = ("active", "gen", "h", "r", "u", "L", "n", "tid")
N_STATS = 7          # normalised innovation (2), log1p mass, spread, time-position covariance (2), time variance


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))


def _zero(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


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


class ObjectTokenLayer(nn.Module):
    """目标 token 层。forward_chunk(events, z0, feat, state, steps, keep, labels, tid, log, keys) -> (z, state, aux)。"""

    def __init__(self, feat_dim, height, width, slots=16, proposals=4, hidden=32, attn_hidden=32, iters=2,
                 defer_max=10, defer_alpha=0.05, length_unit=8.0, clutter_source="none", ref_rate=None,
                 separation="full", object_context=True, decision_rule="sprt"):
        """clutter_source：none = 背景 logit 只由事件特征学出；background = 另读模块二的本地背景率（每像素每窗、按极性，
        以参照率为单位）作为背景 logit 的输入。separation：见文件说明。
        消融对照（与 v5 相同）：object_context = False 时读出不看任何 token（z = z0 + 背景项，不挂起），检验增益来自目标而不是
        多了一个读出头；decision_rule = fixed 时挂在 token 上的事件一律等满 defer_max 窗（token 被否定、释放时提前终判），
        作 Wald 判决的固定等待对照。"""
        super(ObjectTokenLayer, self).__init__()
        if clutter_source not in CLUTTER_SOURCES:
            raise ValueError("clutter_source 必须是 %s 之一" % (CLUTTER_SOURCES,))
        if separation not in SEPARATIONS:
            raise ValueError("separation 必须是 %s 之一" % (SEPARATIONS,))
        if decision_rule not in DECISION_RULES:
            raise ValueError("decision_rule 必须是 %s 之一" % (DECISION_RULES,))
        self.clutter_source, self.separation = str(clutter_source), str(separation)
        self.object_context, self.decision_rule = bool(object_context), str(decision_rule)
        self.feat_dim, self.height, self.width = int(feat_dim), int(height), int(width)
        self.slots, self.proposals, self.iters = int(slots), int(proposals), int(iters)
        self.hidden, self.attn_hidden = int(hidden), int(attn_hidden)
        if self.slots < 1 or self.proposals < 1 or self.iters < 1:
            raise ValueError("slots、proposals、iters 至少为 1")
        alpha = float(defer_alpha)
        if not 0.0 < alpha < 0.5:
            raise ValueError("defer_alpha 必须在 (0, 0.5) 内")
        self.alpha = alpha
        self.defer_A = math.log((1.0 - alpha) / alpha)                 # Wald bound: a requirement, not a parameter
        self.defer_max = int(defer_max)
        if self.defer_max < 1:
            raise ValueError("defer_max 至少为 1 窗")
        self.unit = float(length_unit)
        # interface attributes read by the trainers
        self.context, self.decision_lag, self.defer_revision = "object", 0, "object"
        self.has_delayed, self.continuous_time = True, True
        self.collect_calibration = False
        D, Hd = self.hidden, self.attn_hidden
        self.g_dim = g = 3 + self.feat_dim                             # [sigmoid(z0), dec1 feature, polarity, t_local]
        self.event_norm = nn.LayerNorm(g)
        # association: cross-attention of the events over (background token, slot tokens)
        self.event_key = nn.Linear(g, Hd)
        self.event_value = nn.Linear(g, D)
        self.slot_query = _zero(nn.Linear(D, Hd))                     # content term starts at 0
        self.slot_bias = _zero(nn.Linear(D, 1))
        self.width_head = _zero(nn.Linear(D, 1))                      # log spread of the position modulation (units^2)
        self.bg_head = _mlp(g + (1 if self.clutter_source == "background" else 0), Hd, 1)
        _zero(self.bg_head[2])
        if self.clutter_source == "background":
            if ref_rate is None:
                from model.comod import REF_RATE
                ref_rate = REF_RATE
            self.register_buffer("ref_rate", torch.tensor([float(r) for r in ref_rate]), persistent=False)
        # token dynamics: predictor, per-window displacement, corrector, token mixing
        self.pred_norm = nn.LayerNorm(D)
        self.pred = _mlp(D, D, D)
        _zero(self.pred[2])
        self.step_head = _zero(nn.Linear(D, 2))                       # displacement per window (units)
        self.gru = nn.GRUCell(D + N_STATS, D)
        self.ffn_norm = nn.LayerNorm(D)
        self.ffn = _mlp(D, D, D)
        _zero(self.ffn[2])
        self.gate_head = _zero(nn.Linear(D + N_STATS, 1))             # gain on the position innovation (starts at 1/2)
        self.mix_norm = nn.LayerNorm(D)
        # token mixing also sees each token's existence probability (a confirmed neighbour vs a fresh proposal)
        self.mix_q, self.mix_k, self.mix_v = nn.Linear(D + 1, Hd), nn.Linear(D + 1, Hd), nn.Linear(D + 1, Hd)
        self.mix_pos = _mlp(2, Hd, 1)                                  # relative position bias between tokens
        self.mix_out = _zero(nn.Linear(Hd, D))
        # proposals
        self.seed_embed = _mlp(g, D, D)
        # existence neuron: leak toward the rest potential + learned evidence current, fires at +-A
        self.evid_head = _mlp(D + N_STATS, Hd, 1)
        _zero(self.evid_head[2])
        lam0 = 1.0 - 1.0 / max(self.defer_max, 2)
        self.leak_head = nn.Linear(D, 1)
        nn.init.zeros_(self.leak_head.weight)
        nn.init.constant_(self.leak_head.bias, math.log(lam0 / (1.0 - lam0)))
        self.rest = nn.Parameter(torch.tensor(-self.defer_A))
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
                "h": zeros(B, N, D), "r": zeros(B, N, 2), "u": zeros(B, N, 2), "L": zeros(B, N), "n": zeros(B, N),
                "tid": zeros(B, N), "defer": None}

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
        """mask [B] bool：这些流开始新录像，token 全部清空，该流挂起的事件保留零等待输出、不再终判。"""
        out = {}
        for key in SLOT_KEYS:
            value = state[key]
            view = mask.view((-1,) + (1,) * (value.dim() - 1))
            out[key] = torch.where(view, torch.zeros_like(value), value)
        q = state.get("defer")
        out["defer"] = None if q is None else {k: v[~mask[q["b"]]] for k, v in q.items()}
        return out

    def evidence_summary(self):
        """论文里报的量：新 token 的静息电位（先验存在对数几率，初值 -A）。"""
        return {"orphan_ref": float(self.rest.detach())}

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _raw_features(ev):
        z0 = ev["z0"]
        return torch.cat([torch.sigmoid(z0)[:, None], ev["feat"].to(z0.dtype), ev["p"].to(z0.dtype)[:, None],
                          ev["t_local"].to(z0.dtype)[:, None]], 1)

    def _clutter_feature(self, ev, dtype):
        """模块二的背景率（以参照率为单位、log1p）作为背景 logit 的一个输入 [E, 1]；clutter_source = none 时为 None。"""
        if self.clutter_source != "background":
            return None
        if ev.get("bg_event") is None:
            raise ValueError("clutter_source = background 需要每个事件的背景率 ev['bg_event']（模块二）")
        ref = self.ref_rate.to(dtype)[(ev["p"] < 0.5).long()]          # channel 0 = ON (p = 1), 1 = OFF
        return torch.log1p(ev["bg_event"].to(dtype).clamp(min=0.0) / ref)[:, None]

    def _bg_logit(self, g, cf):
        return self.bg_head(g if cf is None else torch.cat([g, cf], 1)).squeeze(-1)

    @staticmethod
    def _stream_sum(weights, vals, onehot):
        """weights [E, N]、vals [E, C]、onehot [E, B]（事件所在的流）-> 每个流每个槽对本流事件的加权和 [B, N, C]。"""
        E, N = weights.shape
        B = onehot.shape[1]
        w = (onehot[:, :, None] * weights[:, None, :]).reshape(E, B * N)
        return (w.t() @ vals).view(B, N, -1)

    def _logits(self, key, gb, exy, se, eb, onehot, h, r, u, valid):
        """[E, 1 + N]：第 0 列背景 token，其余为本流各槽：内容相似度 + 槽偏置 + 位置调制（事件与槽在该事件时刻的位置比较）。"""
        E = int(key.shape[0])
        B, N = valid.shape
        q = self.slot_query(h).reshape(B * N, -1)
        content = ((key @ q.t()).view(E, B, N) * onehot[:, :, None]).sum(1) / math.sqrt(float(self.attn_hidden))
        prec = torch.exp(-self.width_head(h).squeeze(-1))
        center = r[eb] + se[:, None, None] * u[eb]                                         # [E, N, 2]
        d2 = ((exy[:, None, :] - center) ** 2).sum(-1)
        a = content + self.slot_bias(h).squeeze(-1)[eb] - 0.5 * d2 * prec[eb]
        a = torch.where(valid[eb], a, torch.full_like(a, NEG))
        return torch.cat([gb[:, None], a], 1)

    def _aggregate(self, bs, val, exy, se, onehot, r, u, prec):
        """注意力加权的充分统计量 -> (特征均值 [B,N,D], 统计量 [B,N,N_STATS], 位置新息 [B,N,2], 注意力质量 n [B,N])。
        统计量 = [新息 / 展布、log(1 + n)、位置离散度 / 展布^2、时间-位置协方差 / 展布、时间方差]：窗内运动的信息都在里面，
        怎么用（增益、速度）由 GRU 与门学。"""
        D = val.shape[1]
        cols = torch.cat([val, torch.ones_like(se)[:, None], exy, se[:, None], (exy * exy).sum(1, keepdim=True),
                          se[:, None] * exy, (se * se)[:, None]], 1)
        sums = self._stream_sum(bs, cols, onehot)
        n = sums[..., D]
        nc = n.clamp(min=EPS)
        vbar = sums[..., :D] / nc[..., None]
        mx = sums[..., D + 1:D + 3] / nc[..., None]
        ms = sums[..., D + 3] / nc
        spread = (sums[..., D + 4] / nc - (mx * mx).sum(-1)).clamp(min=0.0)
        cov = sums[..., D + 5:D + 7] / nc[..., None] - ms[..., None] * mx
        var_s = (sums[..., D + 7] / nc - ms * ms).clamp(min=0.0)
        has = (n > EPS)[..., None]
        innov = torch.where(has, mx - (r + ms[..., None] * u), torch.zeros_like(mx))
        sd = torch.sqrt(prec)[..., None]
        stats = torch.cat([innov * sd, torch.log1p(n)[..., None], (spread * prec)[..., None], cov * sd,
                           var_s[..., None]], -1)
        return vbar, torch.where(has, stats, torch.zeros_like(stats)), innov, n

    def _token_mix(self, h, r, valid, p_exist):
        """token 之间的自注意力（相对位置偏置；输入另含各 token 此前的存在概率），学会去重；输出层零初始化。"""
        x = torch.cat([self.mix_norm(h), p_exist[..., None]], -1)
        q, k, v = self.mix_q(x), self.mix_k(x), self.mix_v(x)
        rel = r[:, :, None, :] - r[:, None, :, :]
        a = torch.einsum("bic,bjc->bij", q, k) / math.sqrt(float(self.attn_hidden)) + self.mix_pos(rel).squeeze(-1)
        a = a.masked_fill(~valid[:, None, :], NEG)
        out = self.mix_out(torch.einsum("bij,bjc->bic", torch.softmax(a, -1), v))
        return h + out * valid[..., None].to(h.dtype)

    @torch.no_grad()
    def _propose(self, score, free, onehot):
        """每条流在本流事件里 score 最高的 proposals 个上各放一个候选 token（只放进空闲槽，按槽号）。
        返回 (流, 槽, 事件) 三个下标张量。"""
        B, N = free.shape
        E = int(score.shape[0])
        k = min(self.proposals, N, E)
        empty = torch.zeros(0, dtype=torch.long, device=score.device)
        if k == 0:
            return empty, empty, empty
        mat = torch.where(onehot.t() > 0.5, score[None, :].expand(B, E), torch.full((B, E), -1e30, dtype=score.dtype,
                                                                                    device=score.device))
        val, idx = mat.topk(k, dim=1)
        slot_ids = torch.arange(N, device=score.device)
        free_order = torch.argsort((~free).long() * N + slot_ids[None, :], 1)              # free slots first
        rank = torch.arange(k, device=score.device)
        ok = (val > -1e29) & (rank[None, :] < free.long().sum(1)[:, None])
        b_i, j_i = torch.nonzero(ok, as_tuple=True)
        return b_i, free_order[b_i, j_i], idx[b_i, j_i]

    def _slot_readout(self, qs, qe, eb):
        """每个事件对本流各槽的读出 [E, N]。中间量 [E, N, attn_hidden] 不存，反向时重算（数值与直接计算相同）。"""
        def f(qs_, qe_, eb_):
            return self.read_out(F.relu(qs_[eb_] + qe_[:, None, :])).squeeze(-1)
        if torch.is_grad_enabled() and (qs.requires_grad or qe.requires_grad):
            return checkpoint(f, qs, qe, eb)
        return f(qs, qe, eb)

    # ------------------------------------------------------------------ one window
    def forward_window(self, state, ev, reset_mask=None, label=None, tid=None, aux=None):
        """state：init_state 的字典；ev：本窗事件 b/x/y/p/t_local/z0/feat/key（可选 bg_event）。label / tid / aux 给出时计算监督。

        返回 (z [Ne] 零等待输出, 新状态, info)；info["delayed"] 为 None 或 (终判 logit, key, 等待窗数, 状态码)，
        状态码：0 确认、1 否定、3 槽已不在、4 等满 defer_max 窗。
        """
        if reset_mask is not None:
            state = self._reset(state, reset_mask)
        dq = state.get("defer")
        if dq is not None:
            dq = dict(dq, lag=dq["lag"] + 1)                         # entries from earlier windows waited one more
        alive = state["active"]
        B, N = alive.shape
        dev = alive.device
        z0 = ev["z0"]
        dtype = z0.dtype
        E = int(z0.shape[0])
        A = self.defer_A
        zBN = torch.zeros(B, N, device=dev, dtype=dtype)

        # 1. predict: residual MLP on the tokens, learned displacement, leak of the existence membrane
        hp = state["h"] + self.pred(self.pred_norm(state["h"]))
        r_p = state["r"] + self.step_head(hp)
        rest = self.rest.to(dtype)
        L_p = rest + torch.sigmoid(self.leak_head(hp).squeeze(-1)) * (state["L"] - rest)

        gen = state["gen"]
        cand = torch.zeros_like(alive)
        h0, r0 = hp, r_p
        if E:
            eb = ev["b"]
            exy = torch.stack([ev["x"], ev["y"]], 1).to(dtype) / self.unit
            se = ev["t_local"].to(dtype) - 0.5
            onehot = (eb[:, None] == torch.arange(B, device=dev)[None, :]).to(dtype)
            raw = self._raw_features(ev)
            g = self.event_norm(raw)                                 # read-out path: per-event losses reach the backbone
            gd = self.event_norm(raw.detach())                       # object estimates: the backbone gets no gradient
            cf = self._clutter_feature(ev, dtype)
            key_d, gb_d, val = self.event_key(gd), self._bg_logit(gd, cf), self.event_value(gd)
            if self.separation == "selective":                       # association also trains the backbone features
                key_a, gb_a = self.event_key(g), self._bg_logit(g, cf)
            else:
                key_a, gb_a = key_d, gb_d

            # 2. proposals: the most target-like events that the existing tokens do not explain
            free = ~alive
            if bool(free.any()):
                with torch.no_grad():
                    lg = torch.log_softmax(self._logits(key_d, gb_d, exy, se, eb, onehot, hp, r_p, self.step_head(hp),
                                                        alive), 1)
                    pb, ps, pe = self._propose(F.logsigmoid(z0) + lg[:, 0], free, onehot)
                if int(pb.shape[0]):
                    cand = cand.index_put((pb, ps), torch.ones_like(pb, dtype=torch.bool))
                    gen = gen.index_put((pb, ps), gen[pb, ps] + 1)
                    cv = cand[..., None]
                    h0 = torch.where(cv, torch.zeros_like(hp).index_put((pb, ps), self.seed_embed(gd[pe])), hp)
                    r0 = torch.where(cv, torch.zeros_like(r_p).index_put((pb, ps), exy[pe].detach()), r_p)
        valid = alive | cand
        L_old = torch.where(cand, rest.expand_as(L_p), L_p)

        # 3. association + correction, unrolled iters times (Slot Attention / KalmanNet)
        h_c, r_c = h0, r0
        p_old = torch.sigmoid(L_old)                                 # fresh proposals start at sigmoid(rest)
        n = zBN
        stats = torch.zeros(B, N, N_STATS, device=dev, dtype=dtype)
        for _ in range(self.iters):
            u_c = self.step_head(h_c)
            if E:
                prec = torch.exp(-self.width_head(h_c).squeeze(-1))
                bs = torch.softmax(self._logits(key_d, gb_d, exy, se, eb, onehot, h_c, r_c, u_c, valid), 1)[:, 1:]
                vbar, stats, innov, n = self._aggregate(bs, val, exy, se, onehot, r_c, u_c, prec)
            else:                                                    # an empty window is evidence too (a miss)
                vbar = torch.zeros(B, N, self.hidden, device=dev, dtype=dtype)
                innov = torch.zeros_like(r_c)
            h_n = self.gru(torch.cat([vbar, stats], -1).reshape(B * N, -1), h_c.reshape(B * N, -1)).view(B, N, -1)
            h_n = h_n + self.ffn(self.ffn_norm(h_n))
            r_n = r_c + torch.sigmoid(self.gate_head(torch.cat([h_n, stats], -1))) * innov
            h_n = self._token_mix(h_n, r_n, valid, p_old)
            vv = valid[..., None]
            h_c, r_c = torch.where(vv, h_n, h_c), torch.where(vv, r_n, r_c)
        u_c = self.step_head(h_c)

        # 4. existence neuron: leaked membrane + learned evidence current; fires at +-A
        L_new = torch.where(valid, L_old + self.evid_head(torch.cat([h_c, stats], -1)).squeeze(-1), zBN)

        # 5. posterior association with the corrected tokens, zero-wait read-out
        if E:
            log_beta = torch.log_softmax(self._logits(key_a, gb_a, exy, se, eb, onehot, h_c, r_c, u_c, valid), 1)
            bd = torch.exp(log_beta).detach()                        # separation: decisions do not move beliefs
            qs = self.read_slot(torch.cat([h_c, torch.sigmoid(L_new).detach()[..., None]], -1))
            qe = self.read_event(g)
            u_slot = self._slot_readout(qs, qe, eb)
            u_bg = self.read_bg(g).squeeze(-1)
            if self.object_context:
                z = z0 + bd[:, 0] * u_bg + (bd[:, 1:] * u_slot).sum(1)
            else:                                                    # control: same read-out capacity, no tokens
                z = z0 + u_bg
        else:
            z = z0

        # 6. release: pending events whose token fired, whose token is gone, or that reached the deadline
        delayed = None
        if dq is not None and int(dq["b"].shape[0]):
            b, j = dq["b"], dq["slot"]
            same = valid[b, j] & (gen[b, j] == dq["gen"])
            Lr = L_new[b, j]
            status = torch.full_like(b, -1)
            confirm = same & (Lr >= A) if self.decision_rule == "sprt" else torch.zeros_like(same)
            for code, cond in ((3, ~same), (0, confirm), (1, same & (Lr <= -A))):
                status = torch.where(cond & (status < 0), torch.full_like(status, code), status)
            status = torch.where((status < 0) & (dq["lag"] >= self.defer_max), torch.full_like(status, 4), status)
            rel = status >= 0
            if bool(rel.any()):
                r = _q_take(dq, rel)
                sr = status[rel]
                qs_now = self.read_slot(torch.cat([h_c[r["b"], r["slot"]], torch.sigmoid(Lr[rel]).detach()[:, None]], -1))
                u_now = self.read_out(F.relu(qs_now + r["qe"])).squeeze(-1)
                u_now = torch.where(sr == 3, r["u_then"], u_now)           # gone: the token's outcome is unknown
                delayed = (r["znow"] + r["m_then"] * (u_now - r["u_then"]), r["key"], r["lag"], sr)
            dq = _q_take(dq, ~rel)
        # 7. this window's events wait if their most probable token is undecided
        if E:
            amax = log_beta.argmax(1)
            kk = (amax - 1).clamp(min=0)
            Lk = L_new[eb, kk]
            if not self.object_context:
                pend = torch.zeros_like(amax, dtype=torch.bool)
            elif self.decision_rule == "fixed":                       # control: every token-attached event waits
                pend = (amax > 0) & (Lk > -A)
            else:
                pend = (amax > 0) & (Lk.abs() < A)
            if bool(pend.any()):
                rows = torch.arange(E, device=dev)
                key_e = ev["key"].to(dtype) if "key" in ev else torch.zeros_like(z0)
                entry = {"b": eb, "slot": kk, "gen": gen[eb, kk], "L_then": Lk.detach(), "m_then": bd[rows, amax],
                         "u_then": u_slot[rows, kk], "qe": qe, "znow": z, "key": key_e, "lag": torch.zeros_like(eb)}
                dq = _q_cat(dq, _q_take(entry, pend))

        # 8. supervision (training): Hungarian matching of tokens and labelled objects
        tid_new = torch.zeros_like(state["tid"])
        if aux is not None and label is not None:
            tid_new = self._supervise(ev, label, tid, log_beta if E else None, L_new, r_c, valid,
                                      state["tid"] * alive.to(dtype), aux)

        # 9. rejected tokens release their slot
        alive_new = valid & (L_new > -A)
        av = alive_new[..., None]
        new_state = {"active": alive_new, "gen": gen,
                     "h": torch.where(av, h_c, torch.zeros_like(h_c)),
                     "r": torch.where(av, r_c, torch.zeros_like(r_c)),
                     "u": torch.where(av, u_c, torch.zeros_like(u_c)),
                     "L": torch.where(alive_new, L_new, zBN),
                     "n": torch.where(alive_new, n, zBN),
                     "tid": torch.where(alive_new, tid_new, zBN), "defer": dq}
        info = {"births": int((cand & alive_new).sum()), "delayed": delayed}
        return z, new_state, info

    def _supervise(self, ev, label, tid, log_beta, L_new, pos, valid, prev_tid, aux):
        """存在 BCE、归属交叉熵、位置 Smooth-L1；返回本窗匹配上的目标号 [B, N]（未匹配为 0）。"""
        B, N = valid.shape
        dev, dtype = L_new.device, L_new.dtype
        target = torch.zeros_like(L_new)
        tid_new = torch.zeros_like(L_new)
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
            bs = torch.exp(log_beta[:, 1:]).detach()
            inter = bs.t() @ M                                                               # [N, G]
            onehot = (eb[:, None] == torch.arange(B, device=dev)[None, :]).to(dtype)
            tot = (onehot.t() @ bs)                                                          # [B, N]
            dice = 2.0 * inter / (tot[gb].t() + size[None, :] + EPS)
            cost = (1.0 - dice) + (1.0 - torch.sigmoid(L_new.detach())[gb].t())
            cost_np, dice_np = cost.cpu().numpy(), dice.cpu().numpy()
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
                aux["pos_sum"] = aux["pos_sum"] + F.smooth_l1_loss(pos[pb, pk], cen[pg], reduction="sum")
                aux["pos_n"] += len(pairs)
                slot_of = torch.full((G,), -1, dtype=torch.long, device=dev)
                slot_of = slot_of.index_put((pg,), pk)
                ev_slot = slot_of[inv]
                cls = cls.index_put((ev_idx,), torch.where(ev_slot >= 0, ev_slot + 1, torch.zeros_like(ev_slot)))
        if E:
            rows = torch.arange(E, device=dev)
            nll = -log_beta[rows, cls]
            pos_ev = (label > 0.5) & (tid > 0)
            loss = nll[~pos_ev].mean() if bool((~pos_ev).any()) else nll.new_zeros(())
            if bool(pos_ev.any()):
                loss = loss + nll[pos_ev].mean()
            aux["mem_sum"] = aux["mem_sum"] + loss
            aux["mem_n"] += 1
        if bool(valid.any()):
            aux["exist_sum"] = aux["exist_sum"] + F.binary_cross_entropy_with_logits(
                L_new[valid], target[valid], reduction="sum")
            aux["exist_n"] += int(valid.sum())
            if self.collect_calibration:
                aux.setdefault("calib", []).append((L_new[valid].detach(), target[valid].detach()))
        return tid_new

    # ------------------------------------------------------------------ chunk
    def forward_chunk(self, events, z0, feat, state, steps, keep=None, labels=None, tid=None, log=None, keys=None):
        """events：片段事件（t 为片段内窗号、b 为流号）；z0 [Ne]、feat [Ne, F] 来自骨干。接口与 ObjectLayer.forward_chunk 相同。"""
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
                log.append({"active": act.cpu().numpy(), "pos": (state["r"] * self.unit).detach().cpu().numpy(),
                            "vel": (state["u"] * self.unit).detach().cpu().numpy(),
                            "ex": torch.where(act, torch.sigmoid(state["L"]), torch.zeros_like(state["L"]))
                            .detach().cpu().numpy(),
                            "beta": state["n"].detach().cpu().numpy()})
        if idx_parts and sum(int(i.shape[0]) for i in idx_parts):
            z = z0.index_put((torch.cat(idx_parts),), torch.cat(z_parts))
        else:
            z = z0
        aux["delayed_z"] = torch.cat(dz) if dz else None
        aux["delayed_key"] = torch.cat(dk) if dk else None
        aux["delayed_lag"] = torch.cat(dl) if dl else None
        aux["delayed_status"] = torch.cat(ds) if ds else None
        return z, state, aux
