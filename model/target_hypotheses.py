"""V2-2 目标假设库（V2 定稿设计 v1.0 第 7 节，式 (2)-(12)）。
设计页：https://claude.ai/artifact/UqE8gG3SjbyCr5hBBfDRSU

每个假设 h 是"这里有一个小目标在运动"的局部解释，保存：
    状态（候选 / 已确认）、生成窗号 b_h、出生锚点与先验速度、最近若干窗的质心测量 {j: (z_h(j), N_h(j))}、
    结构 S_h（以质心为原点的相对位置分布，(2*half+1)^2 网格，和为 1）、确认积累 C_h、未命中计数 M_h、各窗活动量 A_h(j)。
每窗 k 按顺序处理（式号对应设计页）：
    预测      式 (2)(5)：用截止到第 k-1 窗的测量做加权直线拟合，外推到第 k 窗，得到位置与不确定度
    关联      式 (3)：a_ih = sigma(m_i) * phi_h(x_i) / max(1, sum_h' phi_h'(x_i))；phi 是预测分布相对其峰值的比值，
              低于 gate_rel 的位置不在门内（phi 记 0）。关联只用预测，不用本窗更新后的状态
    测量      式 (4)：有效事件数 N_h(k) 与加权质心 z_h(k)；N_h(k) >= min_effective 才算有效测量
    结构      式 (7)：以本窗质心为原点的加权直方图（3x3 盒核，与强度头的 3x3 尺度一致），按
              beta = struct_rate * min(1, N / struct_full) 慢速混入 S_h。偏移相对质心计算，定位误差不会写进结构
    活动与确认 式 (8)(9)：A_h(k) = 门内 SNN 活动量之和；e = N_obs * log(1 + A_h(k-1)/B) - A_h(k-1)
              （V2-1 的泊松证据式，积分区域随假设移动，预测量 A_h(k-1) 在看到本窗之前给出）；
              C_h = max(0, C_h + e)，候选在 C_h >= confirm_theta 时确认
    结束      式 (10)：连续 miss_patience 窗没有有效测量；候选的 C_h 连续 miss_patience 窗为 0 也丢弃
    合并      式 (11)：两个假设的窗末分布重叠 sum_x min(q_a, q_b) 连续 2 窗 >= merge_overlap，保留更可信的一个
    生成      式 (12)：初判 >= birth_conf 且不在任何门内的事件按切比雪夫距离 birth_link 成簇（>= birth_min_events 个）；
              运动管道膜电位 max_v C_v >= candidate_theta 且不在任何门内的位置（按膜电位从高到低，每窗至多 max_tube_births 个）
窗末对已确认假设保存快照（位置、不确定度、结构），供回溯读出（model/attribution_readout.py）做冻结参照（式 13）。

拟合窗口：式 (5) 对第 j 窗的估计只用 [max(b_h, j-lag), min(截止窗, j+lag)] 内的测量（前后各 lag 窗的局部直线），
预测（截止 j-1）、窗末快照（截止 j）和回溯平滑（截止 j+d）是同一个过程，只是信息截止不同。

假设库变体（设计页第 10 节，update 参数）：
    separate   默认：定位不确定度单独估计，结构以质心为原点慢更新
    generic    对照：结构每窗直接取本窗分布（beta = 1），偏移相对预测位置计算，分布只用下限不确定度（不确定度并入结构范围）
    confident  对照：关联只用初判 >= confident_tau 的事件
本模块没有可学习参数，不参与训练。内部用 CPU float64（每窗只有少量事件落在门内）；
管道膜电位这类大张量在原设备上先取极大值，只把越过门槛的少数位置取回 CPU。
"""
import copy
import math

import torch

DEFAULT_PARAMS = {
    "lag": 4,                  # L：局部直线拟合向前、向后各看几窗
    "min_effective": 2.0,      # N_min：有效测量所需的最少有效事件数
    "gate_rel": 0.05,          # phi_min：门的相对阈值（相对预测分布峰值）
    "sigma0": 3.0,             # 新生假设的位置不确定度（像素）
    "sigma_floor": 0.5,        # 不确定度下限（像素）
    "template_radius": 2.0,    # r0：初始结构模板（各向同性高斯）的标准差（像素）
    "struct_half": 7,          # 结构网格半宽：相对位置 [-7, 7]^2
    "struct_rate": 0.1,        # beta_S：结构更新速率
    "struct_full": 5.0,        # N_0：结构按满速率更新所需的有效事件数
    "candidate_theta": 4.0,    # 运动管道候选门槛
    "confirm_theta": 6.0,      # theta_c：确认门槛（低于 V2-1 告警阈值 8~16）
    "birth_conf": 0.9,         # tau_b：SNN 入口的初判门槛
    "birth_min_events": 3,     # n_b：SNN 入口成簇的最少事件数
    "birth_link": 2,           # SNN 入口成簇的连接距离（切比雪夫，像素）
    "miss_patience": 4,        # P：连续多少窗没有有效测量即结束
    "merge_overlap": 0.5,      # kappa：合并的分布重叠门槛
    "max_tube_births": 4,      # 每窗由运动管道生成的候选上限
    "keep_windows": 16,        # 测量、活动量与快照保留的窗数（应 >= lag + 最大读出延迟 + 1）
    "update": "separate",      # separate | generic | confident
    "confident_tau": 0.9,      # confident 变体只用初判 >= 该值的事件关联
}
UPDATES = ("separate", "generic", "confident")
F64 = torch.float64


def pixel_mass(t, sigma):
    """N(0, sigma^2) 在以 t 为中心、宽 1 的区间上的概率质量（逐元素；对整数平移求和为 1）。"""
    s = float(sigma) * math.sqrt(2.0)
    return 0.5 * (torch.erf((t + 0.5) / s) - torch.erf((t - 0.5) / s))


def offset_grid(half, dtype=F64):
    """结构网格的一维相对位置 [-half, ..., half]。"""
    return torch.arange(-int(half), int(half) + 1, dtype=dtype)


def gaussian_template(half, radius):
    """初始结构：各向同性高斯，标准差 radius 像素，归一化为和 1。"""
    u = offset_grid(half)
    w = torch.exp(-0.5 * (u / float(radius)) ** 2)
    S = w[:, None] * w[None, :]
    return S / S.sum()


def distribution(S, center, sigma, ys, xs):
    """式 (6)：q(x) = sum_u S(u) * N_pix(x - center - u; sigma^2)，在像素 (ys, xs) 上取值（每像素概率）。

    S 的行对应 y 方向的相对位置、列对应 x 方向；N_pix 是在像素内积分的离散高斯，模糊不改变总质量。
    """
    if ys.numel() == 0:
        return torch.zeros(0, dtype=S.dtype)
    half = (int(S.shape[0]) - 1) // 2
    u = offset_grid(half, S.dtype)
    gy = pixel_mass(ys.to(S.dtype)[:, None] - float(center[0]) - u[None, :], sigma)
    gx = pixel_mass(xs.to(S.dtype)[:, None] - float(center[1]) - u[None, :], sigma)
    return torch.einsum("na,ab,nb->n", gy, S, gx)


def structure_spread(S, minimum=0.25):
    """结构在每个方向上的方差（两个方向取平均，像素^2），用作单个事件的位置噪声。"""
    half = (int(S.shape[0]) - 1) // 2
    u = offset_grid(half, S.dtype)
    py, px = S.sum(1), S.sum(0)
    my, mx = (py * u).sum(), (px * u).sum()
    var = 0.5 * ((py * (u - my) ** 2).sum() + (px * (u - mx) ** 2).sum())
    return max(float(var), float(minimum))


def line_fit(js, zs, ws, j):
    """式 (5) 的加权直线拟合 z = c + v * (j' - j)（两个方向共用权重）。

    js [m] 窗号，zs [m,2] 质心，ws [m] 权重（逆方差）。返回 (c [2], var_c, v [2])；窗号少于 2 个不同值时返回 None。
    var_c 是截距 c（即第 j 窗位置）的估计方差，外推越远越大。
    """
    if int(js.numel()) < 2:
        return None
    tau = js - float(j)
    W, T, TT = ws.sum(), (ws * tau).sum(), (ws * tau * tau).sum()
    det = W * TT - T * T
    if float(det) <= 1e-9 * max(float(W * TT), 1e-12):
        return None
    Z = (ws[:, None] * zs).sum(0)
    TZ = (ws[:, None] * tau[:, None] * zs).sum(0)
    c = (TT * Z - T * TZ) / det
    v = (W * TZ - T * Z) / det
    return c, float(TT / det), v


class Hypothesis(object):
    """一个局部目标假设（字段含义见模块说明）。"""

    def __init__(self, hid, birth, anchor, velocity, S, origin):
        self.hid = int(hid)
        self.birth = int(birth)
        self.anchor = anchor.to(F64)
        self.velocity = velocity.to(F64)
        self.S = S.to(F64)
        self.origin = origin
        self.status = "candidate"
        self.confirmed_at = None
        self.meas = {}             # j -> (z [2], N)
        self.activity = {}         # j -> A_h(j)
        self.C = 0.0
        self.zero_run = 0
        self.miss = 0
        self.support = 0           # 有效测量窗数

    @property
    def confirmed(self):
        return self.status == "confirmed"


class HypothesisBank(object):
    """稀疏目标假设库（无可学习参数）。

    用法：每条序列开头 reset()；每窗 k 在 SNN 与判决层处理完之后调用 step(k, ys, xs, probs, g, mu0, tube_C)；
    回溯读出通过 confirmed_hypotheses()、density()、snapshot_density()、activity_at() 读取状态（只读）。
    """

    def __init__(self, height, width, velocities=(), **params):
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError("未知的假设库参数: %s" % sorted(unknown))
        p = dict(DEFAULT_PARAMS)
        p.update(params)
        if p["update"] not in UPDATES:
            raise ValueError("update 必须是 %s 之一" % (UPDATES,))
        if int(p["lag"]) < 1 or int(p["struct_half"]) < 1 or int(p["miss_patience"]) < 1:
            raise ValueError("lag、struct_half、miss_patience 必须 >= 1")
        if not 0.0 < float(p["gate_rel"]) < 1.0:
            raise ValueError("gate_rel 必须在 (0, 1) 内")
        if not 0.0 < float(p["struct_rate"]) <= 1.0:
            raise ValueError("struct_rate 必须在 (0, 1] 内")
        if float(p["sigma0"]) <= 0 or float(p["sigma_floor"]) <= 0:
            raise ValueError("sigma0、sigma_floor 必须 > 0")
        self.p = p
        self.height, self.width = int(height), int(width)
        vel = [(float(vy), float(vx)) for vy, vx in velocities]
        self.velocities = torch.tensor(vel, dtype=F64) if vel else torch.zeros(0, 2, dtype=F64)
        self.template = gaussian_template(p["struct_half"], p["template_radius"])
        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self):
        """新序列开始：清空全部假设、快照与统计。"""
        self.alive = []
        self.next_id = 0
        self.snapshots = {}            # k -> {hid: (center [2], sigma, S)}
        self.overlap_run = {}          # (hid_a, hid_b) -> 连续重叠窗数
        self.k = -1
        self.stats = {"births_snn": 0, "births_tube": 0, "confirmed": 0, "merged": 0, "ended": 0,
                      "confirm_delays": []}

    def confirmed_hypotheses(self):
        return [h for h in self.alive if h.confirmed]

    def find(self, hid):
        for h in self.alive:
            if h.hid == int(hid):
                return h
        return None

    def activity_at(self, h, j):
        """第 j 窗的活动量 A_h(j)；该窗没有记录时取 j 之前最近一窗的值（都没有时为 0）。"""
        if j in h.activity:
            return h.activity[j]
        earlier = [jj for jj in h.activity if jj <= j]
        return h.activity[max(earlier)] if earlier else 0.0

    # ------------------------------------------------------------------ 估计
    def _sigma_for(self, sigma):
        """generic 变体把不确定度并入结构范围：分布只用下限不确定度。"""
        return float(self.p["sigma_floor"]) if self.p["update"] == "generic" else float(sigma)

    def estimate(self, h, j, cutoff):
        """式 (5)：用截止到第 cutoff 窗的测量估计第 j 窗的目标位置，返回 (center [2], sigma)。"""
        p = self.p
        lag, sig0, floor = int(p["lag"]), float(p["sigma0"]), float(p["sigma_floor"])
        lo, hi = max(h.birth, j - lag), min(int(cutoff), j + lag)
        js = sorted(jj for jj in h.meas if lo <= jj <= hi)
        spread = structure_spread(h.S)
        if len(js) >= 2:
            zs = torch.stack([h.meas[jj][0] for jj in js])
            ws = torch.tensor([h.meas[jj][1] for jj in js], dtype=F64) / spread
            fit = line_fit(torch.tensor(js, dtype=F64), zs, ws, float(j))
            if fit is not None:
                c, var_c, _ = fit
                return c, min(math.sqrt(var_c + floor * floor), 3.0 * sig0)
        known = [jj for jj in h.meas if jj <= int(cutoff)]
        if known:                                   # 不足两个可用测量：从最近的测量按先验速度外推
            jl = min(known, key=lambda jj: (abs(jj - j), -jj))
            z, n = h.meas[jl]
            c = z + h.velocity * float(j - jl)
            sigma = math.sqrt(spread / max(float(n), 1e-6) + floor * floor) if jl == j else sig0
            return c, min(sigma, 3.0 * sig0)
        return h.anchor + h.velocity * float(j - h.birth), sig0

    def _box(self, center, sigma):
        """覆盖结构与 3 sigma 模糊的像素方框，返回 (ys, xs, (y0, y1, x0, x1))；方框完全在画面外时返回 None。"""
        r = int(self.p["struct_half"]) + int(math.ceil(3.0 * float(sigma))) + 1
        cy, cx = int(round(float(center[0]))), int(round(float(center[1])))
        y0, y1 = max(0, cy - r), min(self.height - 1, cy + r)
        x0, x1 = max(0, cx - r), min(self.width - 1, cx + r)
        if y0 > y1 or x0 > x1:
            return None
        ay, ax = torch.arange(y0, y1 + 1), torch.arange(x0, x1 + 1)
        ys = ay[:, None].expand(len(ay), len(ax)).reshape(-1)
        xs = ax[None, :].expand(len(ay), len(ax)).reshape(-1)
        return ys, xs, (y0, y1, x0, x1)

    def field(self, h, j, cutoff, S=None):
        """第 j 窗、信息截止第 cutoff 窗时假设 h 的分布：返回 (center, sigma, box) 与方框内的 q 与峰值。

        返回 None（方框在画面外）或 dict(center, sigma, box=(ys, xs, bounds), q, qmax)。
        """
        center, sigma = self.estimate(h, j, cutoff)
        sigma = self._sigma_for(sigma)
        box = self._box(center, sigma)
        if box is None:
            return None
        q = distribution(h.S if S is None else S, center, sigma, box[0], box[1])
        return {"center": center, "sigma": sigma, "box": box, "q": q, "qmax": float(q.max()) if q.numel() else 0.0}

    def density(self, h, j, cutoff, ys, xs):
        """事件 (ys, xs) 处的 q_h(x; j | cutoff) 与 phi（相对峰值；方框外为 0）。ys、xs 为 CPU long。"""
        n = int(ys.numel())
        q, phi = torch.zeros(n, dtype=F64), torch.zeros(n, dtype=F64)
        fld = self.field(h, j, cutoff)
        if fld is None or fld["qmax"] <= 0 or n == 0:
            return q, phi
        y0, y1, x0, x1 = fld["box"][2]
        inside = (ys >= y0) & (ys <= y1) & (xs >= x0) & (xs <= x1)
        idx = inside.nonzero().view(-1)
        if idx.numel():
            qe = distribution(h.S, fld["center"], fld["sigma"], ys[idx], xs[idx])
            q[idx] = qe
            phi[idx] = qe / fld["qmax"]
        return q, phi

    def state(self, h, j, cutoff):
        """信息截止第 cutoff 窗时对第 j 窗的 (center, sigma, S)：式 (6) 所需的全部量（sigma 已按变体处理）。"""
        center, sigma = self.estimate(h, j, cutoff)
        return center, self._sigma_for(sigma), h.S

    def snapshot_state(self, k, hid):
        """第 k 窗末快照 (center, sigma, S)；该窗末 hid 尚未确认（没有快照）时返回 None。"""
        return self.snapshots.get(int(k), {}).get(int(hid))

    def snapshot_density(self, k, hid, ys, xs):
        """第 k 窗末快照（冻结参照）在 (ys, xs) 处的 q；该窗末 hid 尚未确认（无快照）时返回 None。"""
        snap = self.snapshot_state(k, hid)
        if snap is None:
            return None
        center, sigma, S = snap
        return distribution(S, center, sigma, ys, xs)

    # ------------------------------------------------------------------ 更新
    def _update_structure(self, h, ys, xs, a, origin):
        """式 (7)：以 origin 为原点的加权直方图（3x3 盒核）慢速混入 S_h（generic 变体直接替换）。"""
        half = int(self.p["struct_half"])
        m = 2 * half + 1
        dy = torch.round(ys.to(F64) - float(origin[0])).long()
        dx = torch.round(xs.to(F64) - float(origin[1])).long()
        hist = torch.zeros(m + 2, m + 2, dtype=F64)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                iy, ix = dy + oy + half + 1, dx + ox + half + 1
                ok = (iy >= 0) & (iy < m + 2) & (ix >= 0) & (ix < m + 2)
                if ok.any():
                    hist.index_put_((iy[ok], ix[ok]), a[ok] / 9.0, accumulate=True)
        hist = hist[1:-1, 1:-1]
        mass = float(hist.sum())
        if mass <= 0:
            return
        if self.p["update"] == "generic":
            beta = 1.0
        else:
            beta = float(self.p["struct_rate"]) * min(1.0, float(a.sum()) / float(self.p["struct_full"]))
        S = (1.0 - beta) * h.S + beta * (hist / mass)
        h.S = S / S.sum()

    def _update_velocity(self, h, k):
        """用第 k 窗及之前 lag 窗的测量拟合速度，存为外推用的先验速度（不足两个测量时保持原值）。"""
        js = sorted(jj for jj in h.meas if k - int(self.p["lag"]) <= jj <= k)
        if len(js) < 2:
            return
        zs = torch.stack([h.meas[jj][0] for jj in js])
        ws = torch.tensor([h.meas[jj][1] for jj in js], dtype=F64)
        fit = line_fit(torch.tensor(js, dtype=F64), zs, ws, float(k))
        if fit is not None:
            h.velocity = fit[2]

    def _new(self, k, anchor, velocity, origin):
        h = Hypothesis(self.next_id, k, anchor, velocity, self.template.clone(), origin)
        self.next_id += 1
        return h

    def step(self, k, ys, xs, probs, g, mu0, tube_C=None):
        """处理第 k 窗（式 2-12），并保存窗末快照。

        ys, xs   本窗事件的像素坐标 [n]（任意设备）
        probs    本窗事件的初判概率 sigma(m) [n]
        g, mu0   本窗 SNN 活动估计与前端背景 [H, W]（可带前置的单例维度）
        tube_C   本窗运动管道膜电位 [V, H, W]（可带批维 1）；None 表示不走管道入口
        返回本窗统计 dict（存活 / 已确认假设数、本窗生成数）。
        """
        p = self.p
        k = int(k)
        self.k = k
        ys = ys.detach().to("cpu").long().view(-1)
        xs = xs.detach().to("cpu").long().view(-1)
        probs = probs.detach().to("cpu", F64).view(-1)
        g = g.detach().to("cpu", F64).reshape(self.height, self.width)
        mu0 = mu0.detach().to("cpu", F64).reshape(self.height, self.width)
        n = int(ys.numel())
        if p["update"] == "confident":
            source = (probs >= float(p["confident_tau"])).to(F64)
        else:
            source = probs
        gate_rel = float(p["gate_rel"])
        cover = torch.zeros(self.height, self.width, dtype=torch.bool)

        # 式 (2)(3)：预测、门与关联（只用截止到上一窗的测量）
        preds = []
        total_phi = torch.zeros(n, dtype=F64)
        for h in self.alive:
            fld = self.field(h, k, k - 1)
            if fld is None or fld["qmax"] <= 0:
                preds.append((h, None, None, None, 0.0, 0.0, 0))
                continue
            bys, bxs, (y0, y1, x0, x1) = fld["box"]
            gate_px = fld["q"] >= gate_rel * fld["qmax"]
            cover[bys[gate_px], bxs[gate_px]] = True
            A_k = float(g[bys[gate_px], bxs[gate_px]].sum())
            B_k = float(mu0[bys[gate_px], bxs[gate_px]].sum())
            inside = (ys >= y0) & (ys <= y1) & (xs >= x0) & (xs <= x1)
            idx = inside.nonzero().view(-1)
            phi = torch.zeros(0, dtype=F64)
            if idx.numel():
                phi = distribution(h.S, fld["center"], fld["sigma"], ys[idx], xs[idx]) / fld["qmax"]
                keep = phi >= gate_rel
                idx, phi = idx[keep], phi[keep]
                total_phi.index_add_(0, idx, phi)
            preds.append((h, fld, idx, phi, A_k, B_k, int(idx.numel())))
        denom = torch.clamp(total_phi, min=1.0)

        # 式 (4)(7)(8)(9)：测量、结构、活动量与确认
        for h, fld, idx, phi, A_k, B_k, n_obs in preds:
            prior = [jj for jj in h.activity if jj < k]
            A_prev = h.activity[max(prior)] if prior else 0.0
            h.activity[k] = A_k
            if A_prev > 0 and fld is not None:
                e = n_obs * math.log1p(A_prev / max(B_k, 1e-9)) - A_prev
            else:
                e = 0.0
            h.C = max(0.0, h.C + e)
            valid = False
            if fld is not None and idx.numel():
                a = source[idx] * phi / denom[idx]
                N = float(a.sum())
                if N >= float(p["min_effective"]):
                    pos = torch.stack([ys[idx].to(F64), xs[idx].to(F64)], 1)
                    z = (a[:, None] * pos).sum(0) / N
                    h.meas[k] = (z, N)
                    h.support += 1
                    origin = fld["center"] if p["update"] == "generic" else z
                    self._update_structure(h, ys[idx], xs[idx], a, origin)
                    self._update_velocity(h, k)
                    valid = True
            h.miss = 0 if valid else h.miss + 1
            if not h.confirmed:
                if h.C >= float(p["confirm_theta"]):
                    h.status, h.confirmed_at = "confirmed", k
                    self.stats["confirmed"] += 1
                    self.stats["confirm_delays"].append(k - h.birth)
                    h.zero_run = 0
                else:
                    h.zero_run = h.zero_run + 1 if h.C <= 0 else 0

        # 式 (10)：结束
        patience = int(p["miss_patience"])
        keep = []
        for h in self.alive:
            if h.miss >= patience or (not h.confirmed and h.zero_run >= patience):
                self.stats["ended"] += 1
            else:
                keep.append(h)
        self.alive = keep

        # 式 (11)：合并（窗末分布重叠连续 2 窗）
        self._merge(k)

        # 式 (12)：生成
        births = self._birth_snn(k, ys, xs, probs, g, cover)
        births += self._birth_tube(k, tube_C, g, cover)

        # 窗末快照（冻结参照）与过期数据清理
        snap = {}
        for h in self.alive:
            if h.confirmed:
                center, sigma = self.estimate(h, k, k)
                snap[h.hid] = (center.clone(), self._sigma_for(sigma), h.S.clone())
        self.snapshots[k] = snap
        horizon = k - int(p["keep_windows"])
        for old in [kk for kk in self.snapshots if kk <= horizon]:
            del self.snapshots[old]
        for h in self.alive:
            for old in [jj for jj in h.meas if jj <= horizon]:
                del h.meas[old]
            for old in [jj for jj in h.activity if jj <= horizon]:
                del h.activity[old]
        return {"alive": len(self.alive), "confirmed": len(self.confirmed_hypotheses()), "births": births}

    def _rank(self, h):
        """合并时保留的优先级：已确认 > 支持窗数多 > 更早生成。"""
        return (1 if h.confirmed else 0, h.support, -h.hid)

    def _merge(self, k):
        if len(self.alive) < 2:
            self.overlap_run = {}
            return
        fields = {h.hid: self.field(h, k, k) for h in self.alive}
        runs, dropped = {}, set()
        for i, a in enumerate(self.alive):
            for b in self.alive[i + 1:]:
                fa, fb = fields[a.hid], fields[b.hid]
                if fa is None or fb is None:
                    continue
                ya0, ya1, xa0, xa1 = fa["box"][2]
                yb0, yb1, xb0, xb1 = fb["box"][2]
                if ya1 < yb0 or yb1 < ya0 or xa1 < xb0 or xb1 < xa0:
                    continue
                y0, y1, x0, x1 = min(ya0, yb0), max(ya1, yb1), min(xa0, xb0), max(xa1, xb1)
                ay, ax = torch.arange(y0, y1 + 1), torch.arange(x0, x1 + 1)
                ys = ay[:, None].expand(len(ay), len(ax)).reshape(-1)
                xs = ax[None, :].expand(len(ay), len(ax)).reshape(-1)
                qa = distribution(a.S, fa["center"], fa["sigma"], ys, xs)
                qb = distribution(b.S, fb["center"], fb["sigma"], ys, xs)
                key = (a.hid, b.hid)
                if float(torch.minimum(qa, qb).sum()) >= float(self.p["merge_overlap"]):
                    runs[key] = self.overlap_run.get(key, 0) + 1
                    if runs[key] >= 2:
                        dropped.add(min(a, b, key=self._rank).hid)
        self.overlap_run = runs
        if dropped:
            self.stats["merged"] += len(dropped)
            self.alive = [h for h in self.alive if h.hid not in dropped]

    def _cover_box(self, cover, h):
        fld = self.field(h, h.birth, h.birth)
        if fld is None:
            return 0.0
        bys, bxs, _ = fld["box"]
        gate_px = fld["q"] >= float(self.p["gate_rel"]) * fld["qmax"]
        cover[bys[gate_px], bxs[gate_px]] = True
        return fld, gate_px

    def _birth_snn(self, k, ys, xs, probs, g, cover):
        """式 (12) SNN 入口：高置信、不在任何门内的事件按切比雪夫距离成簇。"""
        p = self.p
        if ys.numel() == 0:
            return 0
        sel = ((probs >= float(p["birth_conf"])) & ~cover[ys, xs]).nonzero().view(-1)
        if sel.numel() < int(p["birth_min_events"]):
            return 0
        link = int(p["birth_link"])
        pix = {}
        for i in sel.tolist():
            pix.setdefault((int(ys[i]), int(xs[i])), []).append(i)
        seen, births = set(), 0
        for start in list(pix):
            if start in seen:
                continue
            stack, members = [start], []
            seen.add(start)
            while stack:
                cy, cx = stack.pop()
                members.extend(pix[(cy, cx)])
                for oy in range(-link, link + 1):
                    for ox in range(-link, link + 1):
                        nb = (cy + oy, cx + ox)
                        if nb in pix and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            if len(members) < int(p["birth_min_events"]):
                continue
            idx = torch.tensor(members, dtype=torch.long)
            w = probs[idx]
            N = float(w.sum())
            pos = torch.stack([ys[idx].to(F64), xs[idx].to(F64)], 1)
            z = (w[:, None] * pos).sum(0) / N
            h = self._new(k, z, torch.zeros(2, dtype=F64), "snn")
            if N >= float(p["min_effective"]):
                h.meas[k] = (z, N)
                h.support = 1
            self._update_structure(h, ys[idx], xs[idx], w, z)
            res = self._cover_box(cover, h)
            if res:
                fld, gate_px = res
                h.activity[k] = float(g[fld["box"][0][gate_px], fld["box"][1][gate_px]].sum())
            self.alive.append(h)
            self.stats["births_snn"] += 1
            births += 1
        return births

    def _birth_tube(self, k, tube_C, g, cover):
        """式 (12) 管道入口：max_v C_v >= candidate_theta 且不在任何门内的位置，按膜电位从高到低生成候选。"""
        p = self.p
        if tube_C is None or int(self.velocities.shape[0]) == 0:
            return 0
        C = tube_C.detach()
        if C.dim() == 4:
            C = C[0]
        cmax, vidx = C.max(dim=0)
        mask = cmax >= float(p["candidate_theta"])
        if not bool(mask.any()):
            return 0
        pos = mask.nonzero().to("cpu")
        vals = cmax[mask].to("cpu", F64)
        vs = vidx[mask].to("cpu")
        births = 0
        for i in torch.argsort(vals, descending=True).tolist():
            y, x = int(pos[i, 0]), int(pos[i, 1])
            if y >= self.height or x >= self.width or bool(cover[y, x]):
                continue
            h = self._new(k, torch.tensor([float(y), float(x)], dtype=F64), self.velocities[int(vs[i])].clone(), "tube")
            res = self._cover_box(cover, h)
            if res:
                fld, gate_px = res
                h.activity[k] = float(g[fld["box"][0][gate_px], fld["box"][1][gate_px]].sum())
            self.alive.append(h)
            self.stats["births_tube"] += 1
            births += 1
            if births >= int(p["max_tube_births"]):
                break
        return births

    def summary(self):
        """序列级统计：生成、确认、合并、结束的次数与确认延迟（窗）。"""
        out = dict(self.stats)
        delays = out.pop("confirm_delays")
        out["confirm_delay_median"] = float(sorted(delays)[len(delays) // 2]) if delays else None
        return out

    def clone_hypothesis(self, hid):
        """复制一个假设（新编号，快照一并复制；测试"重复假设不改变读出"用）。"""
        h = self.find(hid)
        if h is None:
            raise KeyError(hid)
        dup = copy.deepcopy(h)
        dup.hid = self.next_id
        self.next_id += 1
        self.alive.append(dup)
        for snap in self.snapshots.values():
            if h.hid in snap:
                center, sigma, S = snap[h.hid]
                snap[dup.hid] = (center.clone(), sigma, S.clone())
        return dup
