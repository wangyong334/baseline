"""模块三：目标级航迹头（Track Head；Stream-SNN-Innovations.md 13.60）。

问题：流式逐像素记忆没有"目标"这个单位。证据钉在像素上、起落同速，目标离开后旧位置继续报目标（残影 / 拖尾误报），
      目标弱的几窗断掉（断裂 / 断续与第一拍漏检）。
原理：贝叶斯多目标滤波 = 卡尔曼形状态 + 伯努利存在概率（对标 CNSF 2026、Residual Kalman 2026、MOTIP 2025）。
接口：只读骨干（E3a）的逐事件 logit z0 与 dec1 特征，不回写骨干（避免输出回写造成的自激；骨干照常逐层并行）。
每窗一次（B 条流并行，每条流 N 个航迹槽）：
    1. 预测   匀速模型：每个坐标轴的状态为 [位置, 速度]，两轴共用 2x2 协方差 P；过程噪声 q 学出来
    2. 量测   证据图 = 每格（cell x cell 像素）事件概率之和，再取 3x3 格之和；先在每条航迹的验证门（gate_sigma 倍标准差）
              内取证据最强的格，几条航迹抢同一格时存在概率高的优先；量测位置 = 概率加权质心，
              量测特征 = 质量、平均概率、事件数、极性、空间扩散、概率加权的平均 dec1 特征
    3. 更新   卡尔曼增益 K = P H^T / (H P H^T + R)；量测噪声 R 由量测特征预测（网络给出"这个量测多可靠"）
    4. 存在   GRU 累积航迹级证据（有无量测、量测特征、归一化新息、不确定度、速度、年龄、连续漏检）-> 存在概率
    5. 出生   所有验证门之外的局部峰（质量 >= birth_floor）；附带"到现有航迹当前位置、近期轨迹的距离"，
              由存在监督学会不让拖尾（落在近期轨迹上的峰）长成新目标
    6. 消亡   存在概率 < death_exist，或连续漏检 > max_coast 窗，或位置离开画布
    7. 修正   最终 logit = z0 + Δ(z0, 事件特征, 极性, 窗内时间, 事件与航迹的关系)；Δ 的末层零初始化，起步与 z0 逐位相同
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


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))


def _dist(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1) + 1e-6)


class TrackHead(nn.Module):
    """目标级航迹头。forward_chunk(events, z0, feat, state, steps, keep) -> (z, state, aux)。"""

    def __init__(self, feat_dim, height, width, cell=4, slots=10, hidden=32, history=21, gate_sigma=3.0,
                 gate_max_px=24.0, gate_floor=0.5, birth_floor=1.0, max_births=4, death_exist=0.02,
                 max_coast=20, match_min_px=8.0, corr_hidden=32, r_min=0.25):
        super(TrackHead, self).__init__()
        self.feat_dim = int(feat_dim)
        self.height, self.width, self.cell = int(height), int(width), int(cell)
        self.gh, self.gw = -(-self.height // self.cell), -(-self.width // self.cell)
        self.slots, self.hidden, self.history = int(slots), int(hidden), int(history)
        self.gate_sigma, self.gate_max_px = float(gate_sigma), float(gate_max_px)
        self.gate_floor, self.birth_floor = float(gate_floor), float(birth_floor)
        self.max_births, self.death_exist, self.max_coast = int(max_births), float(death_exist), float(max_coast)
        self.match_min_px, self.r_min = float(match_min_px), float(r_min)
        self.meas_dim = 5 + self.feat_dim
        self.u_dim = self.meas_dim + 11
        self.ctx_dim = 7
        self.log_q = nn.Parameter(torch.tensor(math.log(0.25)))       # 过程噪声（像素/窗^2 的加速度方差）
        self.log_v0 = nn.Parameter(torch.tensor(math.log(4.0)))       # 出生时的速度方差（(像素/窗)^2）
        self.r_net = _mlp(self.meas_dim, 16, 1)                        # 量测特征 -> 量测噪声 R（像素^2）
        nn.init.zeros_(self.r_net[2].weight)
        nn.init.constant_(self.r_net[2].bias, math.log(math.expm1(4.0)))   # 初始 R = 4 + r_min
        self.gru = nn.GRUCell(self.u_dim, self.hidden)
        self.exist = nn.Linear(self.hidden, 1)
        self.none_logit = nn.Parameter(torch.zeros(()))
        self.corr = _mlp(1 + self.feat_dim + 2 + self.ctx_dim, int(corr_hidden), 1)
        nn.init.zeros_(self.corr[2].weight)
        nn.init.zeros_(self.corr[2].bias)
        gy = torch.arange(self.gh, dtype=torch.float32).repeat_interleave(self.gw)
        gx = torch.arange(self.gw, dtype=torch.float32).repeat(self.gh)
        half = (self.cell - 1) / 2.0
        self.register_buffer("centers", torch.stack([gx * self.cell + half, gy * self.cell + half], 1),
                             persistent=False)

    # ------------------------------------------------------------------ 状态
    def init_state(self, batch, device):
        B, N, H, D = int(batch), self.slots, self.history, self.hidden
        zeros = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        return {"active": torch.zeros(B, N, dtype=torch.bool, device=device),
                "pos": zeros(B, N, 2), "vel": zeros(B, N, 2), "P": zeros(B, N, 2, 2), "h": zeros(B, N, D),
                "ex": zeros(B, N), "age": zeros(B, N), "miss": zeros(B, N),
                "hist": zeros(B, N, H, 2), "hist_ok": torch.zeros(B, N, H, dtype=torch.bool, device=device)}

    @staticmethod
    def detach_state(state):
        return None if state is None else {k: v.detach() for k, v in state.items()}

    @staticmethod
    def _reset(state, mask):
        """mask [B] bool：这些流开始新录像，航迹全部清空。"""
        out = {}
        for key, value in state.items():
            view = mask.view((-1,) + (1,) * (value.dim() - 1))
            out[key] = torch.where(view, torch.zeros_like(value), value)
        return out

    # ------------------------------------------------------------------ 量测
    def _maps(self, eb, ex, ey, prob, pol, feat, batch):
        """每格累加量 -> 3x3 格之和 [B, C, G]。"""
        C, G = MEAS_SUMS + self.feat_dim, self.gh * self.gw
        cell = (torch.div(ey, self.cell, rounding_mode="floor") * self.gw
                + torch.div(ex, self.cell, rounding_mode="floor"))
        xf, yf = ex.to(prob.dtype), ey.to(prob.dtype)
        cols = [prob, prob * prob, torch.ones_like(prob), prob * pol, prob * xf, prob * yf, prob * xf * xf,
                prob * yf * yf]
        vals = torch.cat([torch.stack(cols, 1), prob[:, None] * feat], 1)            # [Ne, C]
        chans = torch.arange(C, device=prob.device)
        flat = (eb[:, None] * C + chans[None, :]) * G + cell[:, None]
        buf = torch.zeros(batch * C * G, dtype=prob.dtype, device=prob.device)
        buf = buf.index_put((flat.reshape(-1),), vals.reshape(-1), accumulate=True)
        sums = buf.view(batch, C, self.gh, self.gw)
        sums = F.avg_pool2d(sums, 3, stride=1, padding=1, count_include_pad=True) * 9.0
        return sums.view(batch, C, G)

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
        """state：init_state 的字典；ev：本窗事件 b/x/y/p/t_local/z0/feat。返回 (z [Ne], 新状态, info)。"""
        if reset_mask is not None:
            state = self._reset(state, reset_mask)
        act = state["active"]
        B, N = act.shape
        dev, cell = act.device, float(self.cell)
        z0, feat = ev["z0"], ev["feat"]
        prob = torch.sigmoid(z0)
        idx_n = torch.arange(N, device=dev)

        # 1. 预测（匀速，离散白噪声加速度模型，dt = 1 窗）
        q = torch.exp(self.log_q)
        P = state["P"]
        P00 = P[..., 0, 0] + 2.0 * P[..., 0, 1] + P[..., 1, 1] + 0.25 * q
        P01 = P[..., 0, 1] + P[..., 1, 1] + 0.5 * q
        P11 = P[..., 1, 1] + q
        Ppred = torch.stack([torch.stack([P00, P01], -1), torch.stack([P01, P11], -1)], -2)
        actv, actm = act[..., None], act[..., None, None]
        pos = torch.where(actv, state["pos"] + state["vel"], state["pos"])
        vel = state["vel"]
        Ppred = torch.where(actm, Ppred, P)

        # 2. 量测：验证门内证据最强的格
        sums = self._maps(ev["b"], ev["x"], ev["y"], prob, ev["p"].to(prob.dtype), feat, B)   # [B, C, G]
        mass = sums[:, 0]
        d2 = ((self.centers[None, None] - pos[:, :, None]) ** 2).sum(-1)                  # [B, N, G]
        gate_r2 = (self.gate_sigma ** 2 * (Ppred[..., 0, 0] + cell * cell)).clamp(max=self.gate_max_px ** 2)
        in_gate = (d2 <= gate_r2[..., None]) & actv
        score = torch.where(in_gate, mass[:, None].expand_as(d2), torch.full_like(d2, -1.0))
        best, cidx = score.max(-1)
        has = act & (best >= self.gate_floor)
        ex_prev = state["ex"]
        same = (cidx[:, :, None] == cidx[:, None, :]) & has[:, :, None] & has[:, None, :]
        better = (ex_prev[:, None, :] > ex_prev[:, :, None]) | (
            (ex_prev[:, None, :] == ex_prev[:, :, None]) & (idx_n[None, None, :] < idx_n[None, :, None]))
        has = has & ~(same & better).any(-1)
        bidx = torch.arange(B, device=dev)[:, None].expand(B, N)
        s_gate = sums.permute(0, 2, 1)[bidx, cidx]                                       # [B, N, C]
        f, zpos = self._measure(s_gate)

        # 3. 卡尔曼更新
        R = self._noise(f)
        S = Ppred[..., 0, 0] + R
        K = Ppred[..., :, 0] / S[..., None]                                              # [B, N, 2]：位置、速度增益
        innov = zpos - pos
        hv = has[..., None]
        pos = torch.where(hv, pos + K[..., 0:1] * innov, pos)
        vel = torch.where(hv, vel + K[..., 1:2] * innov, vel)
        P_upd = Ppred - K[..., :, None] * Ppred[..., 0, :][..., None, :]
        P = torch.where(has[..., None, None], P_upd, Ppred)
        d2n = (innov ** 2).sum(-1) / S
        hf = has.to(prob.dtype)
        zero = torch.zeros_like(hf)
        miss = torch.where(act, torch.where(has, zero, state["miss"] + 1.0), zero)
        age = torch.where(act, state["age"] + 1.0, zero)

        # 4. 航迹级证据积累 -> 存在
        speed = torch.sqrt((vel ** 2).sum(-1) + 1e-6) / cell
        u = torch.cat([hf[..., None], f * hf[..., None], (torch.log1p(d2n) * hf)[..., None],
                       (torch.log(S) * hf)[..., None], torch.log(Ppred[..., 0, 0].clamp(min=1e-6))[..., None],
                       (speed.clamp(max=4.0) / 4.0)[..., None], (torch.log1p(age) / 3.0)[..., None],
                       (miss / self.max_coast)[..., None], torch.zeros(B, N, 4, device=dev, dtype=prob.dtype)], -1)
        h_new = self.gru(u.reshape(B * N, self.u_dim), state["h"].reshape(B * N, self.hidden)).view(B, N, self.hidden)
        h = torch.where(actv, h_new, state["h"])
        hist, hist_ok = state["hist"], state["hist_ok"]

        # 5. 出生：所有验证门之外的局部峰
        in_any = in_gate.any(1)
        peak = F.max_pool2d(mass.view(B, 1, self.gh, self.gw), 3, stride=1, padding=1).view(B, -1)
        cand = (mass >= self.birth_floor) & (mass >= peak) & ~in_any
        k = min(self.max_births, mass.shape[1])
        cv, ci = torch.where(cand, mass, torch.full_like(mass, -1.0)).topk(k, dim=1)
        cv_l, ci_l, free_l = cv.cpu().tolist(), ci.cpu().tolist(), (~act).cpu().tolist()
        bb, ss, cc = [], [], []
        for b in range(B):
            free = [n for n in range(N) if free_l[b][n]]
            for value, c in zip(cv_l[b], ci_l[b]):
                if value < self.birth_floor or not free:
                    break
                bb.append(b)
                ss.append(free.pop(0))
                cc.append(c)
        born = torch.zeros(B, N, dtype=torch.bool, device=dev)
        if bb:
            bt = torch.tensor(bb, dtype=torch.long, device=dev)
            st = torch.tensor(ss, dtype=torch.long, device=dev)
            ct = torch.tensor(cc, dtype=torch.long, device=dev)
            f_b, z_b = self._measure(sums.permute(0, 2, 1)[bt, ct])
            scale = 8.0 * cell
            act_s = act[bt]
            dcur = torch.where(act_s, _dist(z_b[:, None], pos[bt]), torch.full_like(act_s, BIG, dtype=prob.dtype))
            dcur = dcur.amin(1)
            hok = hist_ok[bt] & act_s[..., None]
            dh = torch.where(hok, _dist(z_b[:, None, None], hist[bt]), torch.full_like(hok, BIG, dtype=prob.dtype))
            dh = dh.amin(2).amin(1)
            R_b = self._noise(f_b)
            M = int(bt.shape[0])
            ones, zeros1 = torch.ones(M, 1, device=dev, dtype=prob.dtype), torch.zeros(M, 1, device=dev, dtype=prob.dtype)
            logR = torch.log(R_b)[:, None]
            u_b = torch.cat([ones, f_b, zeros1, logR, logR, zeros1, zeros1, zeros1, ones,
                             (dcur / scale).clamp(max=1.0)[:, None], (dh / scale).clamp(max=1.0)[:, None],
                             (dh < 2.0 * cell).to(prob.dtype)[:, None]], -1)
            h_b = self.gru(u_b, torch.zeros(M, self.hidden, device=dev, dtype=prob.dtype))
            v0 = torch.exp(self.log_v0).expand(M)
            P_b = torch.stack([torch.stack([R_b, torch.zeros_like(R_b)], -1),
                               torch.stack([torch.zeros_like(R_b), v0], -1)], -2)
            pos = pos.index_put((bt, st), z_b)
            vel = vel.index_put((bt, st), torch.zeros(M, 2, device=dev, dtype=prob.dtype))
            P = P.index_put((bt, st), P_b)
            h = h.index_put((bt, st), h_b)
            born = born.index_put((bt, st), torch.ones(M, dtype=torch.bool, device=dev))
            age = age.index_put((bt, st), torch.zeros(M, device=dev, dtype=prob.dtype))
            miss = miss.index_put((bt, st), torch.zeros(M, device=dev, dtype=prob.dtype))
            hist_ok = hist_ok & ~born[..., None]
        live = act | born

        # 6. 存在概率；7. 消亡
        ex_logit = self.exist(h).squeeze(-1)
        ex = torch.sigmoid(ex_logit)
        m = self.gate_max_px
        outside = (pos[..., 0] < -m) | (pos[..., 0] > self.width + m) | (pos[..., 1] < -m) | (pos[..., 1] > self.height + m)
        die = live & ((ex < self.death_exist) | (miss > self.max_coast) | outside)
        alive = live & ~die
        info = {"ex_logit": ex_logit, "live": live, "pos": pos, "born": born, "has": has}
        av, am = alive[..., None], alive[..., None, None]
        pos, vel = torch.where(av, pos, torch.zeros_like(pos)), torch.where(av, vel, torch.zeros_like(vel))
        P, h = torch.where(am, P, torch.zeros_like(P)), torch.where(av, h, torch.zeros_like(h))
        ex = torch.where(alive, ex, torch.zeros_like(ex))
        age, miss = torch.where(alive, age, zero), torch.where(alive, miss, zero)
        hist = torch.cat([pos[:, :, None], hist[:, :, :-1]], 2)
        hist_ok = torch.cat([alive[:, :, None], hist_ok[:, :, :-1]], 2) & alive[..., None]
        new_state = {"active": alive, "pos": pos, "vel": vel, "P": P, "h": h, "ex": ex, "age": age, "miss": miss,
                     "hist": hist, "hist_ok": hist_ok}

        # 8. 逐事件修正
        z = self._correct(ev, z0, feat, new_state)
        return z, new_state, info

    def _context(self, ev, st):
        """事件与航迹的关系 [Ne, ctx_dim]：无航迹权重、加权存在、加权归一化距离、加权速度、到最近航迹当前位置的距离、
        到航迹 2～20 窗之前位置（近期轨迹）的最近距离、本流活航迹数占比。距离按 8 格归一并截到 1。"""
        N, cell = self.slots, float(self.cell)
        eb = ev["b"]
        dtype = ev["z0"].dtype
        exy = torch.stack([ev["x"], ev["y"]], 1).to(dtype)
        act_e, ex_e = st["active"][eb], st["ex"][eb]
        d = _dist(exy[:, None], st["pos"][eb])                                           # [Ne, N]
        delta2 = d * d / (st["P"][eb][..., 0, 0] + cell * cell)
        speed = torch.sqrt((st["vel"][eb] ** 2).sum(-1) + 1e-6) / cell
        logit_w = torch.where(act_e, -0.5 * delta2 + torch.log(ex_e.clamp(min=1e-6)), torch.full_like(d, -1e4))
        w = torch.softmax(torch.cat([logit_w, self.none_logit.expand(int(d.shape[0]), 1)], 1), 1)
        wt, wn = w[:, :N], w[:, N]
        scale = 8.0 * cell
        dcur = torch.where(act_e, d, torch.full_like(d, BIG)).amin(1)
        hist = st["hist"][eb][:, :, 2:]                                                   # 2～20 窗之前的位置
        hok = st["hist_ok"][eb][:, :, 2:]
        dh = _dist(exy[:, None, None], hist)
        tail = torch.where(hok, dh, torch.full_like(dh, BIG)).amin(2).amin(1)
        return torch.stack([wn, (wt * ex_e).sum(1), (wt * delta2.clamp(max=25.0) / 25.0).sum(1),
                            (wt * speed.clamp(max=4.0) / 4.0).sum(1), (dcur / scale).clamp(max=1.0),
                            (tail / scale).clamp(max=1.0), act_e.to(dtype).sum(1) / float(N)], 1)

    def _correct(self, ev, z0, feat, st):
        if int(z0.shape[0]) == 0:
            return z0
        ctx = self._context(ev, st)
        inp = torch.cat([z0.clamp(-15.0, 15.0)[:, None] / 5.0, feat, ev["p"].to(z0.dtype)[:, None],
                         ev["t_local"].to(z0.dtype)[:, None], ctx], 1)
        return z0 + self.corr(inp).squeeze(-1)

    # ------------------------------------------------------------------ 片段
    def forward_chunk(self, events, z0, feat, state, steps, keep=None, labels=None, tid=None, log=None):
        """events：片段事件（t 为片段内窗号、b 为流号）；z0 [Ne]、feat [Ne, F] 来自骨干。

        keep [T, B]（或 None）：keep[t, b] = 0 表示第 b 条流在第 t 窗开始一段新录像，先清空该流航迹。
        labels / tid 给出时计算航迹监督（存在 BCE 之和、质心 L1 之和及其个数）。log 为列表时逐窗记录航迹（可视化）。
        返回 (z [Ne]，顺序与 events 相同, 片段结束时的状态, aux)。
        """
        if state is None:
            batch = int(keep.shape[1]) if keep is not None else int(events["b"].max()) + 1 if events["b"].numel() else 1
            state = self.init_state(batch, z0.device)
        aux = {"exist_sum": z0.new_zeros(()), "exist_n": 0, "pos_sum": z0.new_zeros(()), "pos_n": 0}
        idx_parts, z_parts = [], []
        for t in range(int(steps)):
            idx = torch.nonzero(events["t"] == t, as_tuple=False).flatten()
            ev = {key: events[key][idx] for key in ("b", "x", "y", "p", "t_local")}
            ev["z0"], ev["feat"] = z0[idx], feat[idx]
            reset = (keep[t] == 0) if keep is not None else None
            z_t, state, info = self.forward_window(state, ev, reset)
            idx_parts.append(idx)
            z_parts.append(z_t)
            if labels is not None:
                self._supervise(info, ev, labels[idx], tid[idx], aux)
            if log is not None:
                log.append({key: state[key].detach().cpu().numpy() for key in ("active", "pos", "vel", "ex")})
        if idx_parts and sum(int(i.shape[0]) for i in idx_parts):
            z = z0.index_put((torch.cat(idx_parts),), torch.cat(z_parts))
        else:
            z = z0
        return z, state, aux

    # ------------------------------------------------------------------ 监督
    def _supervise(self, info, ev, label, tid, aux):
        live, pos, ex_logit = info["live"], info["pos"], info["ex_logit"]
        B, N = live.shape
        target = torch.zeros_like(ex_logit)
        pos_t = torch.zeros_like(pos)
        pmask = torch.zeros_like(live)
        gb, gc, gr = window_targets(ev["b"], ev["x"], ev["y"], label, tid, self.match_min_px)
        if gb.numel() and bool(live.any()):
            dist = _dist(gc[:, None], pos[gb])                                            # [Gt, N]
            ok = live[gb] & (dist <= gr[:, None])
            dist_np = torch.where(ok, dist, torch.full_like(dist, np.inf)).detach().cpu().numpy()
            gb_np = gb.cpu().numpy()
            pairs = greedy_match(dist_np, gb_np)
            if pairs:
                rows = torch.tensor([r for r, _ in pairs], dtype=torch.long, device=pos.device)
                cols = torch.tensor([c for _, c in pairs], dtype=torch.long, device=pos.device)
                bsel = gb[rows]
                target = target.index_put((bsel, cols), torch.ones_like(cols, dtype=target.dtype))
                pos_t = pos_t.index_put((bsel, cols), gc[rows])
                pmask = pmask.index_put((bsel, cols), torch.ones_like(cols, dtype=torch.bool))
        if bool(live.any()):
            aux["exist_sum"] = aux["exist_sum"] + F.binary_cross_entropy_with_logits(
                ex_logit[live], target[live], reduction="sum")
            aux["exist_n"] += int(live.sum())
        if bool(pmask.any()):
            aux["pos_sum"] = aux["pos_sum"] + F.smooth_l1_loss((pos[pmask] - pos_t[pmask]) / 4.0,
                                                               torch.zeros_like(pos[pmask]), reduction="sum")
            aux["pos_n"] += int(pmask.sum())


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
