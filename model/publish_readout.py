"""V3（方案 A）：等待安全的逐事件发布（Wait-Safe Evidence Publishing），阶段 1 = 发布层，不训练、无可学习参数。

对象：每个已发生事件的目标/背景标签的最终发布时刻。处理节拍不变（每 50 ms 一窗，网络、前端与判决层照常更新，
发布不回写任何共享状态）；第 k 窗出生的事件在第 k .. k+D 窗之间择机发布，发布即最终，不再修改。
依据：证据是否充分。事件 i 在年龄 d（出生后第 d 窗末）的证据分数
    z_i(d) = m_i + w * F_i(d)                                                              (1)
    m_i     出生窗的网络 mark logit（归属：这个事件像不像目标，出生时定下，等待中不变）
    F_i(d)  出生后 d 窗沿各速度管道的存在证据 = logsumexp_v R_i,v(d) - log V，R 为判决层逐窗证据 ell 沿管道的累加，
            与 V2 的延迟读出（TubeReadout）逐位相同；F_i(0) = 0
    w       V2 的融合权重（fusion_weight）
事件证据缓存 = 锚定证据链（可选，anchor）：每个待发布事件、每条管道另记"链是否还连着"——出生时连着，某窗管道位置的
    足迹里没有事件就断开；断开后只计入负证据。这样事件分到的未来证据只来自以它为起点、每窗连续的活动，
    目标后来路过这里不能再给它加分（model/evidence_neuron.anchored_accumulate）。
归属门控（可选，gate = g）：m_i < g 的事件只能被未来证据减分，不能被"等成"目标（存在与归属分开）：
    z_i(d) = m_i + w * min(F_i(d), 0)   当 m_i < g                                          (2)
发布规则（截断的双边序贯检验；θ 为最终阈值，a >= 0、b > 0 为提前发布所需的裕量）：
    年龄 d < D 时   z >= θ + a·s(d) → 发布"目标"（upper）；z <= θ - b·s(d) → 发布"背景"（lower）；否则继续等   (3)
    年龄 d = D 时   发布 1[z >= θ]（deadline）                                              (4)
    s(d) = 1 - d/D（linear，边界随年龄线性收拢到 θ）或 1（step，期限前不收）
    序列提前结束（真实序列末尾）时，尚未发布的事件按 1[z >= θ] 发布（eos）。
由 a >= 0、b > 0 可知任何发布都满足 标签 = 1[z_pub >= θ]，所以发布分数与标签一致。
一个待发布事件就是一个脉冲发布单元：膜电位 = 累积的存在证据（兴奋受归属门控、抑制不受），两个阈值随年龄收拢，
越过任一阈值即发放（发布），期限到时强制发放；只有尚未发布的事件占计算。

    PublishRule   规则本身（式 1–4），numpy 与 torch 通用；replay 为离线逐年龄回放（tools/publish_replay.py 用）
    PublishUnits  在线发布层：每窗读判决层的证据增量 ell，推进待发布事件，输出本窗发布的记录
"""
import math

import numpy as np
import torch

from model.evidence_neuron import anchored_accumulate

UPPER, LOWER, DEADLINE, EOS = 0, 1, 2, 3
REASONS = ("upper", "lower", "deadline", "eos")
COLLAPSES = ("linear", "step")


class PublishRule(object):
    """式 (1)–(4) 的发布规则。

    输入: theta 最终阈值（logit）；upper a >= 0、lower b > 0（nat）；deadline D >= 0（窗）；collapse linear / step；
          gate 归属门槛（logit，None = 不门控）；weight 融合权重 w。
    """

    def __init__(self, theta, upper, lower, deadline, collapse="linear", gate=None, weight=1.0):
        if int(deadline) != float(deadline) or int(deadline) < 0:
            raise ValueError("deadline 必须是非负整数（窗）")
        if float(upper) < 0 or float(lower) <= 0:
            raise ValueError("需要 upper >= 0、lower > 0，否则提前发布的背景标签可能与最终阈值矛盾")
        if collapse not in COLLAPSES:
            raise ValueError("collapse 必须是 %s 之一" % (COLLAPSES,))
        self.theta = float(theta)
        self.upper = float(upper)
        self.lower = float(lower)
        self.deadline = int(deadline)
        self.collapse = collapse
        self.gate = None if gate is None else float(gate)
        self.weight = float(weight)

    def margins(self, age):
        """年龄 age（0 <= age < D）时上下界离 θ 的距离 (a·s, b·s)。"""
        s = 1.0 if self.collapse == "step" else 1.0 - float(age) / float(self.deadline)
        return self.upper * s, self.lower * s

    def score(self, logit, evidence):
        """式 (1)(2)：z = m + w·F，门控时 m < gate 的事件只保留 F 的负部。numpy 数组与 torch 张量通用。
        门控的写法 F - relu(F)·1[m < g] 在两种库里都逐位精确（F > 0 时减去自身得 0，F <= 0 时不变）。"""
        if self.gate is None:
            return logit + self.weight * evidence
        positive = (evidence + abs(evidence)) * 0.5
        return logit + self.weight * (evidence - positive * ((logit < self.gate) * 1.0))

    def decide(self, z, age):
        """年龄 age 时的决定：返回 (发布目标, 发布背景) 两个布尔掩码；age >= D 时强制 (z >= θ, z < θ)。"""
        if age >= self.deadline:
            up = z >= self.theta
            return up, ~up
        a, b = self.margins(age)
        up = z >= self.theta + a
        return up, (z <= self.theta - b) & ~up

    def replay(self, z_by_age, available=None):
        """离线逐年龄回放（与 PublishUnits 的在线决定逐事件相同）。

        输入: z_by_age [D+1, N]（年龄 0..D 的证据分数，numpy）；available [N] 每个事件最后可观察的年龄
              （序列末尾截断时 < D；None 表示都到 D）。
        输出: (label bool [N], age int64 [N], reason int8 [N], z_pub float64 [N])
        """
        z_by_age = np.asarray(z_by_age, dtype=np.float64)
        n = z_by_age.shape[1]
        last = np.full(n, self.deadline, np.int64) if available is None else np.minimum(available, self.deadline)
        label = np.zeros(n, bool)
        age = np.full(n, -1, np.int64)
        reason = np.full(n, -1, np.int8)
        z_pub = np.zeros(n, np.float64)
        open_ = np.ones(n, bool)
        for d in range(self.deadline + 1):
            z = z_by_age[d]
            if d < self.deadline:
                up, down = self.decide(z, d)
                for mask, lab, why in ((open_ & up, True, UPPER), (open_ & down, False, LOWER)):
                    label[mask], age[mask], reason[mask], z_pub[mask] = lab, d, why, z[mask]
                open_ &= ~(up | down)
                end = open_ & (last == d)                    # 序列在这一年龄结束：按最终阈值发布
                why = EOS
            else:
                end = open_
                why = DEADLINE
            label[end], age[end], reason[end], z_pub[end] = z[end] >= self.theta, d, why, z[end]
            open_ &= ~end
        return label, age, reason, z_pub


class PublishUnits(object):
    """在线发布层（一条序列一个实例）。每处理完第 k 窗调用 step，序列结束调用 flush。

    step(state, k, b, y, x, key, logit, support)：state 为判决层本窗之后的状态（用 state["ell"]）；b/y/x/logit 为第 k 窗
    新事件（与 TubeReadout 的调用约定相同）。先用第 k 窗的 ell 推进已登记的待发布事件，再登记新事件并做年龄 0 的决定。
    anchor=True（事件证据缓存 = 锚定证据链）时 support 为本窗的支持场 [B,1,H,W]，每个待发布事件、每条管道另记
    "链是否还连着"，累加用 model/evidence_neuron.anchored_accumulate（与 TubeReadout(anchor=True) 同一规则）。
    返回本步发布的记录列表，每条 (key, pos, label, z, publish_window, reason, age)：pos 为这些事件在其出生窗
    事件列表中的位置（long），label 布尔，z 为发布时的证据分数（与 logit 同设备同 dtype）。
    不修改判决层状态；只有尚未发布的事件占用计算。
    """

    def __init__(self, cusum, rule, anchor=False):
        self.cusum = cusum
        self.rule = rule
        self.anchor = bool(anchor)
        self.pending = []
        self.last_k = -1

    def _score(self, entry, fresh=False):
        """式 (1)(2)。F 与 TubeReadout 的读出分数同一表达式（逐位相同）；刚出生（年龄 0）时 F 恰为 0，
        直接取零（全零向量的 logsumexp 减 log V 在浮点下会留下约 1e-7 的残差）。"""
        logit = entry["logit"]
        if fresh:
            evidence = torch.zeros_like(logit)
        else:
            evidence = torch.logsumexp(entry["run"], dim=0) - math.log(self.cusum.n_hypotheses)
        return self.rule.score(logit, evidence)

    def _decide(self, entry, z, age, k):
        """对一个出生窗的待发布事件做年龄 age 的决定，返回 (记录列表, 剩余事件的条目或 None)。"""
        up, down = self.rule.decide(z, age)
        forced = age >= self.rule.deadline
        records = []
        for mask, why in ((up, DEADLINE if forced else UPPER), (down, DEADLINE if forced else LOWER)):
            if bool(mask.any()):
                records.append((entry["key"], entry["pos"][mask], up[mask], z[mask], int(k), why, int(age)))
        rest = ~(up | down)
        if forced or not bool(rest.any()):
            return records, None
        kept = {name: entry[name][rest] for name in ("b", "y", "x", "pos", "logit")}
        kept.update(k=entry["k"], key=entry["key"], run=entry["run"][:, rest])
        if self.anchor:
            kept["alive"] = entry["alive"][:, rest]
        return records, kept

    def step(self, state, k, b, y, x, key, logit, support=None):
        out = []
        k = int(k)
        if self.anchor and support is None:
            raise ValueError("anchor=True 时每步都要给出本窗的支持场 support")
        if self.pending:
            cat = lambda name: torch.cat([e[name] for e in self.pending]) if len(self.pending) > 1 else self.pending[0][name]
            k_from = torch.cat([torch.full_like(e["y"], e["k"]) for e in self.pending])
            where = (cat("b"), cat("y"), cat("x"))
            values = self.cusum.gather_along_many(state["ell"], k, *where, k_from)
            if self.anchor:
                V = self.cusum.n_hypotheses
                held = self.cusum.gather_along_many(support.expand(-1, V, -1, -1), k, *where, k_from) > 0
            start, kept = 0, []
            for entry in self.pending:
                n = int(entry["y"].shape[0])
                if self.anchor:
                    entry["run"], entry["alive"] = anchored_accumulate(entry["run"], entry["alive"],
                                                                       values[:, start:start + n], held[:, start:start + n])
                else:
                    entry["run"] = entry["run"] + values[:, start:start + n]
                start += n
                records, rest = self._decide(entry, self._score(entry), k - entry["k"], k)
                out += records
                if rest is not None:
                    kept.append(rest)
            self.pending = kept
        n = int(y.shape[0])
        if n:
            entry = {"k": k, "key": key, "b": b, "y": y, "x": x, "logit": logit,
                     "pos": torch.arange(n, device=y.device), "run": logit.new_zeros(self.cusum.n_hypotheses, n)}
            if self.anchor:
                entry["alive"] = torch.ones(self.cusum.n_hypotheses, n, dtype=torch.bool, device=y.device)
            records, rest = self._decide(entry, self._score(entry, fresh=True), 0, k)
            out += records
            if rest is not None:
                self.pending.append(rest)
        self.last_k = k
        return out

    def flush(self):
        """真实序列结束：尚未发布的事件按最终阈值发布（eos），发布窗号为最后一窗。"""
        out = []
        for entry in self.pending:
            z = self._score(entry, fresh=entry["k"] == self.last_k)
            label = z >= self.rule.theta
            out.append((entry["key"], entry["pos"], label, z, int(self.last_k), EOS, int(self.last_k - entry["k"])))
        self.pending = []
        return out
