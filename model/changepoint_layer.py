"""模块三 RCL：回溯式变点标注层（Retrospective Change-point Labeling；方案 RCL_V1_PLAN_20261101.md）。

v2（11-01，第一阶段诊断后的修正，CLAUDE.md 1.0.0）：
  - 证据改为线性对数似然比 l = g(X) - mu（v1 的 log(1 + g) 把证据压到 <= 2.4 nats，目标格的在场平均只有 0.06～0.10）；
  - 事件输出改为"在场 x 归属"：P = P_{s|s+D}(c_e) x m。目标事件必在占据格内，所以 P(y=1) = P(占据) P(y=1 | 占据)，
    是恒等分解（不是近似）；v1 的贝叶斯因子以 V5 的 p0 为先验锚，V5 漏掉的第一拍离阈值 7.65 nats，
    即使未来在场完美也只救得回 22%（val 1102 个中的 245 个）；
  - 训练加转移损失 L_dyn：训练时状态（占据）已知，出生 / 存活 / 位移核按"已知上一窗占据预测下一窗占据"直接拟合
    （有监督 HMM 的标准做法）；v1 只靠后验 BCE 间接学，位移核学成了均匀的（不动 0.05，标签 0.38）。
  v1 的写法保留为选项（evidence_form = log1p、fusion = bayes，configs/changepoint_v1.yaml），用于复现第一阶段。

两条通路 + 融合提交头：
  通路 A（冻结骨干，训练脚本缓存后传入）：每个事件的零等待 logit z0 与 dec1 读出特征。
  通路 B（本层，可训练）：每窗由原始事件重建与骨干同一编码的 12 通道表示 X_t，在 4x4 像素的判决格上做在场递推：
    (E1)  l_t = g(X_t) - mu（v1：log(1 + g) - mu）               证据卷积（无偏置，零输入零证据），Lam = exp(l)
    (E2)  pred_t = (1 - P) rho + p (pi * P)                    预测在场；* = 卷积（散布）
    (E3)  Z_t = (1 - P)[(1 - rho) + rho Lam] + P[(1 - p) + p (pi (x) Lam)]   按来源归一化；(x) = 相关（收集，画布外取 1）
    (E4)  P_t = (1 - P) rho Lam / Z_t + Lam p (pi * (P / Z_t))   过滤在场：同一来源的目标只能去一处
    (E5)  P_{u|t} = P_{u|u} [(1 - p)(1 - P_{u+1|t}) / (1 - pred_{u+1}) + p (pi (x) (P_{u+1|t} / pred_{u+1}))]   RTS 平滑
    (E6)  L = logit P_{s|t} - logit P_{s|s}                     未来对数似然比
  融合提交头：
    (E7)  q = P_{s|s}(c_e)；m = sigmoid(h([dec1 特征, p, t_local, logit p0]))（归属：占据格内的事件是不是目标）
    (E8)  fusion = product（v2）：P = P_{s|t}(c_e) x m
          fusion = bayes（v1）：BF = Lam (1 - m q) / ((1 - q) + q (1 - m) Lam)，
                               P = min(max(m, p0), sigmoid(logit p0 + log BF))；p0 = sigmoid(z0 - log w)
    (E10) 暂定输出 = z0（零额外等待）；commit = fixed：第 s+D 窗提交 1[P >= tau*]，tau* = sigmoid(logit 0.9 - log w)
          （w = 骨干训练用的正样本权重，P 与 tau* 都在校准尺度上）；
          commit = adaptive：P >= 1 - alpha 提交 1；暂定为 1 且 P <= alpha 提交 0；否则 s+D 提交。提交后不再改写。
  训练（E11）：L_pres（过滤与 1..D 各滞后平滑的在场 BCE，标签 = 目标轨迹占据）-> exist_sum；
              L_mem（占据格内事件的归属 BCE，同格正负事件都参与）-> mem_sum；
              L_evt（1..D 各滞后的事件输出 BCE；product 时在场与归属都回传，bayes 时只训练归属头）-> evt_sum；
              L_dyn（v2：由上一窗的标签占据预测这一窗的标签占据，只训练出生 / 存活 / 位移核）-> dyn_sum。
没有为 V7 那种外挂程序留的东西：没有目标列表、关联、编号、出生消亡规则；在场是网格上的概率场，转移是一个学出的卷积核，
前向用它卷积、后向用它相关。固定的只有递推的形式（由原理推出）、需求量 alpha 与 D、结构量（格子步长、位移集合）。
接口与 model/object_tokens.ObjectTokenLayer 相同（init_state / detach_state / forward_chunk），另有 flush（序列结束时提交）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
CLAMP = 30.0
CHANGEPOINT_KEYS = ("defer_max", "defer_alpha", "commit", "rings", "cell", "evidence_channels", "member_hidden",
                    "evidence_form", "fusion")
COMMITS = ("fixed", "adaptive")
EVIDENCE_FORMS = ("linear", "log1p")
FUSIONS = ("product", "bayes")
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
STATUS_CONFIRM, STATUS_REJECT, STATUS_FLUSH, STATUS_DEADLINE = 0, 1, 3, 4
QUEUE_KEYS = ("b", "cy", "cx", "lag", "key", "z0", "lp0", "m", "q", "y", "prov", "zlast")


def ring_offsets(rings):
    """位移集合：不动 + 每个半径（格）的 8 个方向。返回 [(dy, dx), ...]。"""
    out = [(0, 0)]
    for r in rings:
        out += [(int(r) * dy, int(r) * dx) for dy, dx in DIRECTIONS]
    return out


def logit(p):
    return torch.log(p) - torch.log1p(-p)


def _scalar_logit(p):
    return math.log(p / (1.0 - p))


def fuse(lp0, m, q, L, cap=True):
    """(E8)(E9)。lp0 = logit p0（校准尺度），m 归属概率，q 过滤在场，L 未来对数似然比 -> 校准尺度的 logit P。
    L = 0 时恒等于 lp0；对 L 单调；cap = True 时不超过 logit max(m, p0)。"""
    m = m.clamp(EPS, 1.0 - EPS)
    q = q.clamp(EPS, 1.0 - EPS)
    L = L.clamp(-CLAMP, CLAMP)
    log_bf = L + torch.log1p(-(m * q)) - torch.logaddexp(torch.log1p(-q), torch.log(q * (1.0 - m)) + L)
    z = lp0 + log_bf
    if cap:
        top = torch.maximum(m, torch.sigmoid(lp0)).clamp(EPS, 1.0 - EPS)
        z = torch.minimum(z, logit(top))
    return z


def occupancy_boxes(seq, cell):
    """(E12) 目标轨迹占据：每窗一组 [cy0, cy1, cx0, cx1]（格坐标，含两端）。有事件的窗取该目标事件外接框；
    同一目标编号首末出现之间没有事件的窗，外接框中心与半宽线性插值，不往外推。没有参数。
    目标 = label 为 1 的事件，按 target_id 分组（EV-Flying 的目标编号可以是 0，不能用编号判断是不是目标）。"""
    n = int(seq.n_windows)
    present = {}
    for k in range(n):
        idx = seq.order[int(seq.bounds[k]):int(seq.bounds[k + 1])]
        sel = seq.label[idx] > 0.5
        if not np.any(sel):
            continue
        tids = seq.target_id[idx][sel].astype(np.int64)
        xs, ys = seq.x[idx][sel].astype(np.float64), seq.y[idx][sel].astype(np.float64)
        for tid in np.unique(tids):
            mk = tids == tid
            box = (xs[mk].min(), xs[mk].max(), ys[mk].min(), ys[mk].max())
            present.setdefault(int(tid), {})[k] = box
    out = [[] for _ in range(n)]
    for wins in present.values():
        ks = sorted(wins)
        for a, b in zip(ks, ks[1:] + [None]):
            out[a].append(wins[a])
            if b is None or b - a <= 1:
                continue
            ca = np.array([(wins[a][0] + wins[a][1]) / 2, (wins[a][2] + wins[a][3]) / 2])
            ha = np.array([(wins[a][1] - wins[a][0]) / 2, (wins[a][3] - wins[a][2]) / 2])
            cb = np.array([(wins[b][0] + wins[b][1]) / 2, (wins[b][2] + wins[b][3]) / 2])
            hb = np.array([(wins[b][1] - wins[b][0]) / 2, (wins[b][3] - wins[b][2]) / 2])
            for k in range(a + 1, b):
                w = (k - a) / float(b - a)
                c, h = (1 - w) * ca + w * cb, (1 - w) * ha + w * hb
                out[k].append((c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1]))
    cells = []
    for boxes in out:
        arr = np.array([[int(y0 // cell), int(y1 // cell), int(x0 // cell), int(x1 // cell)]
                        for x0, x1, y0, y1 in boxes], dtype=np.int64).reshape(-1, 4)
        cells.append(arr)
    return cells


def occupancy_maps(cells, lo, hi, hc, wc):
    """把 occupancy_boxes 的第 lo..hi-1 窗画成 [hi-lo, hc, wc] 的 0/1 图。"""
    out = np.zeros((hi - lo, hc, wc), dtype=np.float32)
    for k in range(lo, hi):
        for cy0, cy1, cx0, cx1 in cells[k]:
            out[k - lo, max(cy0, 0):min(cy1, hc - 1) + 1, max(cx0, 0):min(cx1, wc - 1) + 1] = 1.0
    return out


def _inverse_softplus(x):
    return math.log(math.expm1(float(x)))


class ChangepointLayer(nn.Module):
    """回溯式变点标注层。forward_chunk(events, z0, feat, state, steps, keep, labels, tid, log, keys) -> (z0, state, aux)。"""

    def __init__(self, feat_dim, height, width, q99, input_clip, time_bins, pos_weight, threshold=0.9,
                 defer_max=10, defer_alpha=0.05, commit="fixed", rings=(1, 2, 4, 8), cell=4,
                 evidence_channels=8, member_hidden=32, evidence_form="linear", fusion="product"):
        super(ChangepointLayer, self).__init__()
        if commit not in COMMITS:
            raise ValueError("commit 必须是 %s 之一" % (COMMITS,))
        if evidence_form not in EVIDENCE_FORMS or fusion not in FUSIONS:
            raise ValueError("evidence_form 必须是 %s 之一，fusion 必须是 %s 之一" % (EVIDENCE_FORMS, FUSIONS))
        self.evidence_form, self.fusion = str(evidence_form), str(fusion)
        self.cell = int(cell)
        if self.cell < 1 or int(height) % self.cell or int(width) % self.cell:
            raise ValueError("画布 %dx%d 必须能被判决格 %d 整除" % (height, width, self.cell))
        if not 0.0 < float(defer_alpha) < 0.5 or int(defer_max) < 1:
            raise ValueError("需要 0 < defer_alpha < 0.5、defer_max >= 1")
        self.height, self.width = int(height), int(width)
        self.hc, self.wc = self.height // self.cell, self.width // self.cell
        self.time_bins = int(time_bins)
        self.channels = 2 + 2 * self.time_bins
        q = torch.tensor(np.asarray(q99, dtype=np.float32).reshape(-1))
        if int(q.numel()) != self.channels or bool((q <= 0).any()):
            raise ValueError("q99 必须是 %d 个正数" % self.channels)
        self.register_buffer("q99", q, persistent=False)
        self.input_clip = float(input_clip)
        self.log_w = math.log(float(pos_weight))
        self.tau_logit = _scalar_logit(float(threshold)) - self.log_w          # report threshold, calibrated scale
        self.alpha = float(defer_alpha)
        self.defer_max = int(defer_max)
        self.commit = str(commit)
        self.offsets = ring_offsets(tuple(int(r) for r in rings))
        self.radius = max(max(abs(dy), abs(dx)) for dy, dx in self.offsets)
        size = 2 * self.radius + 1
        basis = torch.zeros(len(self.offsets), size, size)
        for i, (dy, dx) in enumerate(self.offsets):
            basis[i, self.radius + dy, self.radius + dx] = 1.0
        self.register_buffer("basis", basis, persistent=False)
        # learned: evidence, silence cost, birth, survival, transition kernel, membership
        self.ev1 = nn.Conv2d(self.channels, int(evidence_channels), self.cell, stride=self.cell, bias=False)
        self.ev2 = nn.Conv2d(int(evidence_channels), 1, 3, padding=1, bias=False)
        self.mu_raw = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.rho_raw = nn.Parameter(torch.tensor(_scalar_logit(1e-3)))
        self.p_raw = nn.Parameter(torch.tensor(_scalar_logit(0.9)))
        self.pi_logit = nn.Parameter(torch.zeros(len(self.offsets)))
        self.member = nn.Sequential(nn.Linear(int(feat_dim) + 3, int(member_hidden)), nn.ReLU(),
                                    nn.Linear(int(member_hidden), 1))
        # interface attributes read by the trainers
        self.context, self.decision_lag, self.defer_revision = "changepoint", 0, "changepoint"
        self.has_delayed, self.collect_calibration = True, False
        self.defer_A = float("inf")                                               # no temperature calibration
        self.status_names = ("confirmed", "rejected", "unused", "flushed", "deadline")

    # ------------------------------------------------------------------ parameters
    def mu(self):
        return F.softplus(self.mu_raw)

    def rho(self):
        return torch.sigmoid(self.rho_raw)

    def p(self):
        return torch.sigmoid(self.p_raw)

    def kernel(self):
        """转移核 [K, K]：33 个位移上的 softmax 概率放在对应位置（前向卷积、后向相关共用）。"""
        w = torch.softmax(self.pi_logit, 0)
        return (w[:, None, None] * self.basis).sum(0)

    @property
    def occupancy_spec(self):
        return self.cell, self.hc, self.wc

    def evidence_summary(self):
        """论文里报、并与标签统计核对的学习量。"""
        with torch.no_grad():
            w = torch.softmax(self.pi_logit, 0)
            radius = torch.tensor([max(abs(dy), abs(dx)) for dy, dx in self.offsets], dtype=w.dtype, device=w.device)
            by_ring = {int(r): float(w[radius == r].sum()) for r in torch.unique(radius).tolist()}
            p = float(self.p())
            return {"survival": p, "mean_life_windows": 1.0 / max(1.0 - p, 1e-9), "birth": float(self.rho()),
                    "silence_cost": float(self.mu()), "move_prob_by_ring": by_ring}

    # ------------------------------------------------------------------ pieces
    def encode(self, ev, batch):
        """本窗事件 -> 与骨干同一编码的表示 [B, C, H, W]：X_c = clip(log1p(计数_c) / q99_c, 0, input_clip)。
        通道 0 / 1 = 整窗正 / 负极性，2 + 2b / 3 + 2b = 第 b 个时间箱的正 / 负极性（dataset/stream_windows.count_channels）。"""
        C, H, W = self.channels, self.height, self.width
        counts = torch.zeros(batch * C * H * W, device=self.q99.device, dtype=self.q99.dtype)
        if int(ev["x"].shape[0]):
            neg = (ev["p"] < 0.5).long()
            pix = ev["y"].long() * W + ev["x"].long()
            base = ev["b"].long() * (C * H * W)
            flat = torch.cat([base + neg * (H * W) + pix,
                              base + (2 + 2 * ev["bin"].long() + neg) * (H * W) + pix])
            # accumulate=True: the deterministic CUDA path in torch 1.9 (same as TorchWindowSource)
            counts = counts.index_put((flat,), torch.ones_like(flat, dtype=counts.dtype), accumulate=True)
        x = torch.log1p(counts.view(batch, C, H, W)) / self.q99.view(1, C, 1, 1)
        return x.clamp(0.0, self.input_clip)

    def evidence(self, x):
        """(E1) l = g(X) - mu（linear）或 log(1 + g(X)) - mu（log1p，v1），[B, hc, wc]；g >= 0 且 g(0) = 0。"""
        g = F.relu(self.ev2(F.relu(self.ev1(x)))).squeeze(1)
        return (g if self.evidence_form == "linear" else torch.log1p(g)) - self.mu()

    def transition_loss(self, prev, cur, k):
        """L_dyn：已知上一窗的占据 prev（标签），按 (E2) 预测这一窗的占据，与标签 cur 的逐格 BCE（未求和）。"""
        pred = ((1.0 - prev) * self.rho() + self.p() * self._scatter(prev, k)).clamp(EPS, 1.0 - EPS)
        return F.binary_cross_entropy(pred, cur, reduction="none")

    def _gather(self, f, k, fill):
        """(pi (x) f)(d) = sum_o pi_o f(d + o)；画布外取 fill。"""
        r = self.radius
        fp = F.pad(f[:, None], (r, r, r, r), value=float(fill))
        return F.conv2d(fp, k[None, None]).squeeze(1)

    def _scatter(self, f, k):
        """(pi * f)(x) = sum_o pi_o f(x - o)；画布外没有来源。"""
        r = self.radius
        fp = F.pad(f[:, None], (r, r, r, r), value=0.0)
        return F.conv2d(fp, torch.flip(k, (0, 1))[None, None]).squeeze(1)

    def forward_step(self, P, ell, k):
        """(E2)~(E4)：上一窗过滤在场 P、本窗证据 ell -> (本窗过滤在场, 本窗预测在场)。"""
        lam = torch.exp(ell.clamp(-CLAMP, CLAMP))
        rho, p = self.rho(), self.p()
        Z = (1.0 - P) * ((1.0 - rho) + rho * lam) + P * ((1.0 - p) + p * self._gather(lam, k, 1.0))
        post = (1.0 - P) * rho * lam / Z + lam * p * self._scatter(P / Z, k)
        pred = (1.0 - P) * rho + p * self._scatter(P, k)
        return post.clamp(EPS, 1.0 - EPS), pred.clamp(EPS, 1.0 - EPS)

    def rts_step(self, filt_u, pred_next, smooth_next, k):
        """(E5)：由 u+1 的平滑在场倒推 u 的平滑在场。"""
        p = self.p()
        ratio = smooth_next / pred_next
        term = (1.0 - p) * (1.0 - smooth_next) / (1.0 - pred_next) + p * self._gather(ratio, k, 1.0)
        return (filt_u * term).clamp(EPS, 1.0 - EPS)

    def membership(self, feat, p, t_local, lp0):
        x = torch.cat([feat.to(lp0.dtype), p.to(lp0.dtype)[:, None], t_local.to(lp0.dtype)[:, None], lp0[:, None]], 1)
        return torch.sigmoid(self.member(x).squeeze(-1)).clamp(EPS, 1.0 - EPS)

    # ------------------------------------------------------------------ state
    def init_state(self, batch, device):
        P = torch.zeros(batch, self.hc, self.wc, device=device, dtype=self.p_raw.dtype)
        return {"P": P, "filt": [], "pred": [], "occ": [],
                "age": torch.zeros(batch, dtype=torch.long, device=device), "queue": None}

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        q = state["queue"]
        return {"P": state["P"].detach(), "filt": [x.detach() for x in state["filt"]],
                "pred": [x.detach() for x in state["pred"]], "occ": [x.detach() for x in state["occ"]],
                "age": state["age"], "queue": None if q is None else {k: v.detach() for k, v in q.items()}}

    @staticmethod
    def _take(q, mask):
        return {k: v[mask] for k, v in q.items()}

    @staticmethod
    def _cat(a, b):
        if a is None or int(a["b"].shape[0]) == 0:
            return b
        if b is None or int(b["b"].shape[0]) == 0:
            return a
        return {k: torch.cat([a[k], b[k]]) for k in a}

    def _emit(self, out, z_rep, key, lag, status):
        if int(key.shape[0]):
            out.append((z_rep, key, lag, status))

    def _flush_entries(self, q, out):
        """序列结束 / 换录像：按最近一次估计 zlast、以 tau* 提交（状态 3）。"""
        if q is None or int(q["b"].shape[0]) == 0:
            return
        self._emit(out, q["zlast"] + self.log_w, q["key"], q["lag"], torch.full_like(q["lag"], STATUS_FLUSH))

    def flush(self, state):
        """把挂起事件全部提交（评估时每条录像结束调用），返回与 forward_chunk 的 aux 相同的四个延后输出键。"""
        out = []
        self._flush_entries(state["queue"] if state is not None else None, out)
        return self._pack(out)

    @staticmethod
    def _pack(out):
        if not out:
            return {"delayed_z": None, "delayed_key": None, "delayed_lag": None, "delayed_status": None}
        return {"delayed_z": torch.cat([o[0] for o in out]), "delayed_key": torch.cat([o[1] for o in out]),
                "delayed_lag": torch.cat([o[2] for o in out]), "delayed_status": torch.cat([o[3] for o in out])}

    # ------------------------------------------------------------------ chunk
    def forward_chunk(self, events, z0, feat, state, steps, keep=None, labels=None, tid=None, log=None, keys=None):
        """events：片段事件（t = 片段内窗号、b = 流号、x / y / p / t_local / bin；训练时另有 occ [steps, B, hc, wc]）。
        返回 (z0 原样 = 暂定输出, 新状态, aux)；aux 的延后输出是本片段内提交的事件（提交后不再改写）。"""
        train = labels is not None
        if state is None:
            batch = int(keep.shape[1]) if keep is not None else (int(events["b"].max()) + 1 if events["b"].numel() else 1)
            state = self.init_state(batch, z0.device)
        batch = int(state["P"].shape[0])
        D = self.defer_max
        occ_all = events.get("occ") if train else None
        if train and occ_all is None:
            raise ValueError("训练需要 events['occ']（目标轨迹占据，见 occupancy_boxes）")
        zero = z0.new_zeros(())
        aux = {"exist_sum": zero, "exist_n": 0, "pos_sum": zero, "pos_n": 0, "pd_sum": zero, "pd_n": 0,
               "org_sum": zero, "org_n": 0, "mem_sum": zero, "mem_n": 0, "evt_sum": zero, "evt_n": 0,
               "dyn_sum": zero, "dyn_n": 0, "births": 0, "recycled": 0, "live": 0, "slot_windows": 0}
        out = []
        k = self.kernel()
        lp0_all = z0 - self.log_w
        lo_commit, hi_commit = _scalar_logit(self.alpha), _scalar_logit(1.0 - self.alpha)
        adaptive = self.commit == "adaptive" and not train
        P, filt, pred, occ, age, q = (state["P"], list(state["filt"]), list(state["pred"]), list(state["occ"]),
                                      state["age"], state["queue"])
        for t in range(int(steps)):
            idx = torch.nonzero(events["t"] == t, as_tuple=False).flatten()
            if keep is not None and bool((keep[t] == 0).any()):
                reset = keep[t] == 0
                if q is not None:
                    gone = reset[q["b"]]
                    self._flush_entries(self._take(q, gone), out)
                    q = self._take(q, ~gone)
                P = torch.where(reset[:, None, None], torch.zeros_like(P), P)
                age = torch.where(reset, torch.zeros_like(age), age)
            ev = {key: events[key][idx] for key in ("b", "x", "y", "p", "bin")}
            # (E1)~(E4) evidence and forward
            ell = self.evidence(self.encode(ev, batch))
            P, pr = self.forward_step(P, ell, k)
            filt, pred = (filt + [P])[-(D + 1):], (pred + [pr])[-(D + 1):]
            if train:
                occ = (occ + [occ_all[t]])[-(D + 1):]
            age = (age + 1).clamp(max=D + 2)
            if train and len(occ) >= 2:                     # L_dyn: previous window must be the same recording
                valid = (age >= 2).to(P.dtype)[:, None, None]
                aux["dyn_sum"] = aux["dyn_sum"] + (self.transition_loss(occ[-2], occ[-1], k) * valid).sum()
                aux["dyn_n"] += int(valid.sum()) * self.hc * self.wc
            # (E5) RTS smoothing over the buffer (index -1 = this window)
            n = len(filt)
            smooth = [None] * n
            smooth[-1] = P
            for j in range(n - 2, -1, -1):
                smooth[j] = self.rts_step(filt[j], pred[j + 1], smooth[j + 1], k)
            if train:                                       # L_pres: filtered + every smoothing lag actually used
                tot = F.binary_cross_entropy_with_logits(logit(P), occ[-1], reduction="sum")
                cnt = P.numel()
                for lag in range(1, n):
                    valid = (lag < age).to(P.dtype)[:, None, None]
                    s = smooth[n - 1 - lag]
                    tot = tot + (F.binary_cross_entropy_with_logits(logit(s), occ[n - 1 - lag], reduction="none")
                                 * valid).sum()
                    cnt += int(valid.sum()) * self.hc * self.wc
                aux["exist_sum"] = aux["exist_sum"] + tot
                aux["exist_n"] += cnt
            # waiting events: one more window of future evidence
            if q is not None and int(q["b"].shape[0]):
                q = dict(q, lag=q["lag"] + 1)
                stack = torch.stack(smooth)                                       # [n, B, hc, wc]
                j = (n - 1 - q["lag"]).clamp(min=0)
                s_val = stack[j, q["b"], q["cy"], q["cx"]]
                if self.fusion == "product":                  # P = presence x membership (both trained by L_evt)
                    z_cal = logit((s_val * q["m"]).clamp(EPS, 1.0 - EPS))
                else:                                         # v1: Bayes factor on the V5 prior, presence detached
                    L = (logit(s_val) - logit(q["q"])).detach()
                    z_cal = fuse(q["lp0"], q["m"], q["q"], L)
                if train:
                    aux["evt_sum"] = aux["evt_sum"] + F.binary_cross_entropy_with_logits(z_cal, q["y"], reduction="sum")
                    aux["evt_n"] += int(z_cal.shape[0])
                z_cal = z_cal.detach()
                deadline = q["lag"] >= D
                if adaptive:
                    up = z_cal >= hi_commit
                    down = (q["prov"] > 0.5) & (z_cal <= lo_commit)
                    status = torch.where(up, torch.full_like(q["lag"], STATUS_CONFIRM),
                                         torch.where(down, torch.full_like(q["lag"], STATUS_REJECT),
                                                     torch.full_like(q["lag"], STATUS_DEADLINE)))
                    rel = up | down | deadline
                else:
                    status = torch.full_like(q["lag"], STATUS_DEADLINE)
                    rel = deadline
                if bool(rel.any()):
                    self._emit(out, z_cal[rel] + self.log_w, q["key"][rel], q["lag"][rel], status[rel])
                q = dict(q, zlast=z_cal)
                q = self._take(q, ~rel)
            # this window's events: provisional output = z0, then wait (or commit at arrival in adaptive mode)
            if int(idx.shape[0]):
                eb = ev["b"].long()
                cy = torch.div(ev["y"].long(), self.cell, rounding_mode="floor").clamp(max=self.hc - 1)
                cx = torch.div(ev["x"].long(), self.cell, rounding_mode="floor").clamp(max=self.wc - 1)
                lp0 = lp0_all[idx]
                m = self.membership(feat[idx], events["p"][idx], events["t_local"][idx], lp0)
                if train:
                    inside = occ[-1][eb, cy, cx] > 0.5
                    if bool(inside.any()):
                        aux["mem_sum"] = aux["mem_sum"] + F.binary_cross_entropy(
                            m[inside], labels[idx][inside].to(m.dtype), reduction="sum")
                        aux["mem_n"] += int(inside.sum())
                key = keys[idx] if keys is not None else idx          # file index (int64) or training label
                entry = {"b": eb, "cy": cy, "cx": cx, "lag": torch.zeros_like(eb), "key": key, "z0": z0[idx],
                         "lp0": lp0, "m": m, "q": P[eb, cy, cx].detach(),
                         "y": labels[idx].to(z0.dtype) if train else torch.zeros_like(lp0),
                         "prov": (lp0 >= self.tau_logit).to(z0.dtype), "zlast": lp0.detach()}
                if adaptive:
                    now = lp0 >= hi_commit
                    if bool(now.any()):
                        self._emit(out, entry["z0"][now], entry["key"][now], entry["lag"][now],
                                   torch.full_like(entry["lag"][now], STATUS_CONFIRM))
                    entry = self._take(entry, ~now)
                q = self._cat(q, entry)
            if log is not None:
                log.append({"presence": P.detach().cpu().numpy()})
            aux["live"] += int((P >= 0.5).sum())
            aux["slot_windows"] += batch
        packed = self._pack(out)
        aux.update(packed)
        new_state = {"P": P, "filt": filt, "pred": pred, "occ": occ, "age": age, "queue": q}
        return z0, new_state, aux
