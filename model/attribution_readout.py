"""V2-2 回溯分布修正读出（V2 定稿设计 v1.0 第 8 节，式 (13)-(17)；对照读出见第 10 节）。
设计页：https://claude.ai/artifact/UqE8gG3SjbyCr5hBBfDRSU

第 k 窗的事件 i 在第 k+d 窗末读出一次（假设库已经处理完第 k+d 窗）：
    h*       = argmax_{已确认的 h, phi_h(x_i) >= gate_rel} q_h(x_i; k | k+d)                  式 (14)
    rho^d_i  = q_{h*}(x_i; k | k+d)                  后续观测平滑后，事件所在窗目标在事件处的分布值
    rho^0_i  = h* 在第 k 窗末已确认：当时快照 q(x_i; k | k)（冻结参照）                       式 (13)
               否则：固定参考 (2 r_ref + 1)^-2（以事件为中心的均匀分布，与未来观测无关）
    gamma_i  = 1[h* 存在，且其有效测量窗数 >= min_support]                                     式 (15)
    Delta_i  = gamma_i * clip( log((rho^d_i + eps) / (rho^0_i + eps)), -cap, cap )               式 (16)
调用方组合最终 logit：z_i = m_i + w * Delta_i（式 17；w 在验证集上校准，不宣称是精确后验）。
活动量在分子分母中取同一个值而约掉，所以不进入 Delta；未来活动的强弱只通过定位精度起作用。

对照读出（同一次运行、同一个假设库、同样的等待，设计页第 10 节）：
    backfill  gamma_i * 1[phi_{h*}(x_i; k | k+d) >= backfill_rel]           区域回填：区域内的事件得到同样的加分
    abs       gamma_i * clip(log((A * rho^d + eps_abs) / (mu0_i + eps_abs)), -cap, cap)   绝对值融合（不相对初判）
    direct    gamma_i * A * rho^d / (A * rho^d + mu0_i)                     只用结构，不用 SNN 初判（直接当概率）
    snnref    参照换成 SNN 活动估计在以事件为中心的 (2 r_ref + 1)^2 窗内的归一化值（盲区探测）
    A 为 h* 在第 k 窗的活动量 A_h(k)，mu0_i 为事件处第 k 窗的前端背景。
每个事件另给 case：0 无修正（gamma = 0）/ 1 快照参照（已有假设）/ 2 固定参考。

接口与 TubeReadout 一致：每窗末调用 step(k, key, ys, xs, mu0, g)，返回本步到期的 [(key, d, 结果 dict, 实际发布窗号)]；
序列结束调用 flush()，尚未到期的延迟在最后一窗读出（实际发布窗号 < k+d，延迟被截断）。本类不修改假设库。
"""
import torch
import torch.nn.functional as F

from model.target_hypotheses import F64, distribution

VARIANTS = ("backfill", "abs", "direct", "snnref")


class AttributionReadout(object):
    """回溯分布修正读出（无可学习参数）。

    参数:
        bank          HypothesisBank（只读）
        delays        固定等待的窗数列表（>= 1）
        cap, eps      修正上限 c 与比值下限 epsilon（式 16）
        ref_radius    固定参考半径 r_ref（rho_ref = (2 r_ref + 1)^-2），snnref 的归一化窗口也用它
        min_support   参与修正所需的有效测量窗数 n_min（式 15）
        variants      额外计算哪些对照读出（VARIANTS 的子集）
        backfill_rel  区域回填的相对阈值
        eps_abs       abs / direct 的强度下限
        eps_g         snnref 中 SNN 活动估计的下限
    """

    def __init__(self, bank, delays, cap=3.0, eps=1e-3, ref_radius=7, min_support=3, variants=VARIANTS,
                 backfill_rel=0.1, eps_abs=1e-3, eps_g=1e-3):
        self.bank = bank
        self.delays = sorted(set(int(d) for d in delays))
        if any(d < 1 for d in self.delays):
            raise ValueError("延迟必须 >= 1（延迟 0 就是网络输出本身）")
        unknown = set(variants) - set(VARIANTS)
        if unknown:
            raise ValueError("未知的对照读出: %s" % sorted(unknown))
        if float(cap) <= 0 or float(eps) <= 0 or int(ref_radius) < 0 or int(min_support) < 1:
            raise ValueError("cap、eps 必须 > 0，ref_radius >= 0，min_support >= 1")
        self.cap, self.eps = float(cap), float(eps)
        self.ref_radius = int(ref_radius)
        self.rho_ref = 1.0 / float((2 * self.ref_radius + 1) ** 2)
        self.min_support = int(min_support)
        self.variants = tuple(v for v in VARIANTS if v in variants)
        self.backfill_rel = float(backfill_rel)
        self.eps_abs, self.eps_g = float(eps_abs), float(eps_g)
        self.max_delay = max(self.delays) if self.delays else 0
        self.pending = []
        self.last_k = -1

    def reset(self):
        self.pending = []
        self.last_k = -1

    def _plane(self, tensor):
        return tensor.detach().reshape(self.bank.height, self.bank.width)

    def _register(self, k, key, ys, xs, mu0, g):
        """登记第 k 窗的事件：坐标，以及对照读出需要的第 k 窗量（事件处背景、snnref 参照）。"""
        entry = {"k": int(k), "key": key,
                 "ys": ys.detach().to("cpu").long().view(-1), "xs": xs.detach().to("cpu").long().view(-1)}
        if "abs" in self.variants or "direct" in self.variants:
            m = self._plane(mu0)
            entry["mu0"] = m[ys.to(m.device).long(), xs.to(m.device).long()].to("cpu", F64)
        if "snnref" in self.variants:
            gm = self._plane(g).to(torch.float64) + self.eps_g
            r = self.ref_radius
            size = 2 * r + 1
            # 画面外按 0 计入：窗内和 = 有效像素上 (g + eps_g) 之和
            total = F.avg_pool2d(gm[None, None], size, stride=1, padding=r, count_include_pad=True)[0, 0] * size * size
            yy, xx = ys.to(gm.device).long(), xs.to(gm.device).long()
            entry["snnref"] = (gm[yy, xx] / total[yy, xx]).to("cpu", F64)
        return entry

    def step(self, k, key, ys, xs, mu0=None, g=None):
        """第 k 窗末调用（假设库已 step(k)）：先对到期的旧窗事件读出，再登记本窗事件。

        ys, xs  本窗事件坐标 [n]；mu0, g 本窗前端背景与 SNN 活动估计 [H, W]（abs / direct / snnref 需要）。
        """
        out = []
        k = int(k)
        for entry in self.pending:
            d = k - entry["k"]
            if d in self.delays:
                out.append((entry["key"], d, self._readout(entry, k), k))
        self.pending = [e for e in self.pending if k - e["k"] < self.max_delay]
        if self.max_delay > 0:
            self.pending.append(self._register(k, key, ys, xs, mu0, g))
        self.last_k = k
        return out

    def flush(self):
        """序列结束：尚未到期的延迟在最后一窗读出（同一事件的多个截断延迟共用一次计算）。"""
        out = []
        for entry in self.pending:
            due = [d for d in self.delays if d > self.last_k - entry["k"]]
            if due:
                res = self._readout(entry, self.last_k)
                out.extend((entry["key"], d, res, self.last_k) for d in due)
        self.pending = []
        return out

    def _readout(self, entry, K):
        """式 (13)-(16) 与对照读出；返回 {名称: CPU 张量 [n]}。"""
        bank = self.bank
        k, ys, xs = entry["k"], entry["ys"], entry["xs"]
        n = int(ys.numel())
        gate_rel = float(bank.p["gate_rel"])
        best_q = torch.zeros(n, dtype=F64)
        best_phi = torch.zeros(n, dtype=F64)
        best = torch.full((n,), -1, dtype=torch.long)
        hyps = {}
        for h in bank.confirmed_hypotheses():                     # 式 (14)：选最能解释事件的已确认假设
            q, phi = bank.density(h, k, K, ys, xs)
            better = (phi >= gate_rel) & (q > best_q)
            if bool(better.any()):
                best_q[better], best_phi[better], best[better] = q[better], phi[better], h.hid
                hyps[h.hid] = h
        gamma = best >= 0
        for hid, h in hyps.items():                                # 式 (15)：支持不足不修正
            if h.support < self.min_support:
                gamma &= best != hid
        rho0 = torch.full((n,), self.rho_ref, dtype=F64)
        case = torch.zeros(n, dtype=torch.long)
        case[gamma] = 2
        for hid, h in hyps.items():                                # 式 (13)：第 k 窗末已确认则取快照
            sel = gamma & (best == hid)
            if not bool(sel.any()):
                continue
            # 分子分母在同一批事件上用同一条计算路径求值：状态相同时逐位相等（P1 在浮点下也严格成立）
            center, sigma, S = bank.state(h, k, K)
            best_q[sel] = distribution(S, center, sigma, ys[sel], xs[sel])
            snap = bank.snapshot_state(k, hid)
            if snap is not None:
                rho0[sel] = distribution(snap[2], snap[0], snap[1], ys[sel], xs[sel])
                case[sel] = 1
        gf = gamma.to(F64)
        delta = gf * torch.clamp(torch.log((best_q + self.eps) / (rho0 + self.eps)), -self.cap, self.cap)
        res = {"attr": delta, "case": case}
        if "backfill" in self.variants:
            res["backfill"] = gf * (best_phi >= self.backfill_rel).to(F64)
        if "abs" in self.variants or "direct" in self.variants:
            A = torch.zeros(n, dtype=F64)
            for hid, h in hyps.items():
                A[best == hid] = float(bank.activity_at(h, k))
            lam, mu = A * best_q, entry["mu0"]
            if "abs" in self.variants:
                res["abs"] = gf * torch.clamp(torch.log((lam + self.eps_abs) / (mu + self.eps_abs)), -self.cap, self.cap)
            if "direct" in self.variants:
                res["direct"] = gf * lam / torch.clamp(lam + mu, min=1e-12)
        if "snnref" in self.variants:
            res["snnref"] = gf * torch.clamp(torch.log((best_q + self.eps) / (entry["snnref"] + self.eps)),
                                             -self.cap, self.cap)
        return res


def rho_reference(ref_radius):
    """固定参考的每像素概率 (2 r_ref + 1)^-2。"""
    return 1.0 / float((2 * int(ref_radius) + 1) ** 2)


def correction_logit(logit, delta, weight):
    """式 (17)：z = m + w * Delta（张量逐元素）。"""
    return logit + float(weight) * delta


__all__ = ["AttributionReadout", "VARIANTS", "rho_reference", "correction_logit"]
