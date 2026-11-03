"""模块三 RCL：回溯式变点标注层（Retrospective Change-point Labeling；方案 RCL_V1_PLAN_20261101.md）。

v2（11-01，第一阶段诊断后的修正，CLAUDE.md 1.0.0）：
  - 证据改为线性对数似然比 l = g(X) - mu（v1 的 log(1 + g) 把证据压到 <= 2.4 nats，目标格的在场平均只有 0.06～0.10）；
  - 事件输出改为"在场 x 归属"：P = P_{s|s+D}(c_e) x m。目标事件必在占据格内，所以 P(y=1) = P(占据) P(y=1 | 占据)，
    是恒等分解（不是近似）；v1 的贝叶斯因子以 V5 的 p0 为先验锚，V5 漏掉的第一拍离阈值 7.65 nats，
    即使未来在场完美也只救得回 22%（val 1102 个中的 245 个）；
  - 训练加转移损失 L_dyn：训练时状态（占据）已知，出生 / 存活 / 位移核按"已知上一窗占据预测下一窗占据"直接拟合
    （有监督 HMM 的标准做法）；v1 只靠后验 BCE 间接学，位移核学成了均匀的（不动 0.05，标签 0.38）。
  v1 的写法保留为选项（evidence_form = log1p、fusion = bayes，configs/changepoint_v1.yaml），用于复现第一阶段。

v3（11-02，v2 失效诊断后的重设计，Inn 13.80；configs/changepoint_v3.yaml）：
  - v2 失效的主因：在场通路只读当前窗原始计数，格级检测 AUC val / test 只有 0.90 / 0.86，而 V5 的输出汇到格上是
    0.985 / 0.989；乘积输出又让这个更弱的检测器否决 V5（误压的目标 97～98% 卡在在场）。两个估计器估同一件事，
    权力给了信息少的那个。
  - evidence_source = backbone：证据按混合 NN-HMM（Bourlard & Morgan）写：骨干的报告尺度 logit z0 是缩放对数似然比，
    格内事件按标记泊松叠加给出格的对数似然比 sum_e softplus(z0_e + beta) - mu（mu = 目标的期望事件数 = 沉默代价，
    beta = 目标 / 背景强度比的对数，都是学出的标量）；原始计数通路保留为残差（零初始化），与骨干项一起过 3x3 卷积
    （初始化 = 只取本格的骨干项，让网络自己学多格目标的证据汇聚）。
  - recursion = union：一个来源格的各个去向互斥（求和），不同来源独立（求并 1 - prod(1 - .)），在场有界、
    不动极限仍是精确两状态 HMM；RTS 超过 1 的部分用光滑饱和代替硬截断（保留梯度）。
  - factor_training：每个因子只按自己的标签训练：证据由 L_pres，出生 / 存活 / 位移核只由 L_dyn（标签上的有监督 HMM），
    归属头由 L_mem + L_evt（L_evt 里平滑在场 detach）。

v4（11-02，Inn 13.81；configs/changepoint_v4.yaml = 框占据 + fusion mixture，下面 (E8m)）：v3 在本地暴露乘积式的结构缺陷
（框移走 = 格里的目标"消失"，平滑压低尾部格的过去在场；转移参数按标签拟合后被误压的 V5 正确目标 85.8% 在随后被框让出的格里）。
  (E8m) fusion = mixture（定稿）：P_e = S_e m_e + (1 - S_e) b_e（全概率公式；S = 本格回溯平滑在场 = 门控，
        m = 在场条件下的归属 MLP，b = 后备专家 = sigmoid(V5 校准 logit + MLP(本格 / 足迹场的平滑与过滤在场、滞后))，
        后备的最后一层零初始化 = 未来不提供信息时退回 V5）。m 只由 L_evt 训练（L_mem 只看标签占据格，与使用分布不一致）。
  本地核对过、未采用的写法（作为消融保留，Inn 13.81）：state = centre（中心在场覆盖不足）、attribution / evidence /
  learned（加性写法只用 L_evt 训练都学成"不用未来"）。下面是这些写法的说明：
  - state = centre：状态 = 目标中心所在格，每个目标一个粒子（精确 HMM / RTS 验证覆盖的情形）；标签 = 外接框中心格。
  (E1c) 证据：中心在 c 的对数似然比 l(c) = sum_e softplus(z0_e + beta + h(c_e - c)) - mu + 原始计数残差
        （Gilholm & Salmond 的扩展目标泊松模型：目标事件的强度 = mu x 足迹 f，h = log f 学出来；V5 的 z0 = 缩放对数似然比）
  (E8a) fusion = attribution：logit P_e = u_e + log(exp(b_u) + sum_c' S(c') exp(g(c_e - c')))，u_e = a z0_e + c + r(特征)
        （扩展目标的事件归属 = 事件自身的似然比 x 附近目标的强度；b_u = 未检出目标的强度，PMBM 的泊松未检出项：
        附近没有可信目标时输出退回 V5 的判断，附近目标被未来否定时压低）。a = 1、c = -log w、r = 0、b_u = 0 初始化时
        输出 = V5（附近有在场中心时略抬高）。L_evt 训练 u、b_u、g；中心在场 S 只由 L_pres、L_dyn 训练。
  - 足迹 h / g、mu 按训练集标签初始化（init_from_labels：目标事件相对中心格的位移分布、每个目标每窗的期望事件数），
    转移参数按中心轨迹做最大似然（train_track_head.fit_dynamics），之后都照常学习。

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
          commit = adaptive：以判决阈值为中心的 SPRT 边界，logit P >= logit tau* + logit(1 - alpha) 提交 1；
          暂定为 1 且 logit P <= logit tau* - logit(1 - alpha) 提交 0；否则 s+D 提交。提交后不再改写。
          等价于"一种判法的贝叶斯风险 <= 另一种的 alpha / (1 - alpha)"，代价与 tau* 相同；tau* = 0.5 时即 P >= 1 - alpha / P <= alpha。
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
SOFTCAP = 1e-3
CHANGEPOINT_KEYS = ("defer_max", "defer_alpha", "commit", "rings", "cell", "evidence_channels", "member_hidden",
                    "evidence_form", "fusion", "evidence_source", "recursion", "factor_training", "state",
                    "footprint_radius", "presence_tolerance", "future_stats", "commit_lag", "evidence_pool",
                    "gate_grad", "raw_residual", "expert_input")
COMMITS = ("fixed", "adaptive")
EVIDENCE_FORMS = ("linear", "log1p")
FUSIONS = ("product", "bayes", "attribution", "evidence", "learned", "mixture")
STAT_SCALE = 5.0
EVIDENCE_SOURCES = ("raw", "backbone")
RECURSIONS = ("additive", "union")
STATES = ("box", "centre")
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


def softcap(x, d=SOFTCAP):
    """x 本身（x <= 1 - d）；更大时 C1 光滑地饱和到 1（RTS 近似越过 1 时代替硬截断，保留梯度）。"""
    over = (x - (1.0 - d)).clamp(min=0.0)
    return torch.where(x <= 1.0 - d, x, 1.0 - d * torch.exp(-over / d))


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


def occupancy_boxes(seq, cell, centre=False):
    """(E12) 目标轨迹占据：每窗一组 [cy0, cy1, cx0, cx1]（格坐标，含两端）。有事件的窗取该目标事件外接框；
    同一目标编号首末出现之间没有事件的窗，外接框中心与半宽线性插值，不往外推。没有参数。
    目标 = label 为 1 的事件，按 target_id 分组（EV-Flying 的目标编号可以是 0，不能用编号判断是不是目标）。
    centre = True（v4，state = centre）：每个目标只取外接框中心所在的格（一个目标一个粒子），插值规则相同。"""
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
        if centre:
            boxes = [((x0 + x1) / 2.0,) * 2 + ((y0 + y1) / 2.0,) * 2 for x0, x1, y0, y1 in boxes]
        arr = np.array([[int(y0 // cell), int(y1 // cell), int(x0 // cell), int(x1 // cell)]
                        for x0, x1, y0, y1 in boxes], dtype=np.int64).reshape(-1, 4)
        cells.append(arr)
    return cells


def footprint_statistics(seq, cell, radius):
    """v4 的标签统计（扩展目标泊松模型的最大似然，只用于初始化学出的量）：目标事件所在格相对其目标中心格的位移直方图
    [2R+1, 2R+1]（超出半径的不计）、目标事件总数、目标中心窗数（含插值的缺口窗）。"""
    size = 2 * int(radius) + 1
    hist = np.zeros((size, size), dtype=np.float64)
    n_events, n_centres = 0, 0
    sel = seq.label > 0.5
    if not np.any(sel):
        return hist, 0, 0
    win = np.zeros(seq.label.shape[0], dtype=np.int64)
    for k in range(int(seq.n_windows)):
        win[seq.order[int(seq.bounds[k]):int(seq.bounds[k + 1])]] = k
    centres = {}
    for k, arr in enumerate(_centre_points(seq)):
        for tid, cyp, cxp in arr:
            centres[(k, tid)] = (int(cyp // cell), int(cxp // cell))
    n_centres = len(centres)
    for e in np.nonzero(sel)[0]:
        c = centres.get((int(win[e]), int(seq.target_id[e])))
        if c is None:
            continue
        n_events += 1
        dy, dx = int(seq.y[e]) // cell - c[0], int(seq.x[e]) // cell - c[1]
        if abs(dy) <= radius and abs(dx) <= radius:
            hist[dy + radius, dx + radius] += 1.0
    return hist, n_events, n_centres


def _centre_points(seq):
    """每窗 [(target_id, 中心 y 像素, 中心 x 像素)]：与 occupancy_boxes(centre=True) 同一插值规则。"""
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
            present.setdefault(int(tid), {})[k] = ((ys[mk].min() + ys[mk].max()) / 2.0,
                                                   (xs[mk].min() + xs[mk].max()) / 2.0)
    out = [[] for _ in range(n)]
    for tid, wins in present.items():
        ks = sorted(wins)
        for a, b in zip(ks, ks[1:] + [None]):
            out[a].append((tid,) + wins[a])
            if b is None or b - a <= 1:
                continue
            for k in range(a + 1, b):
                w = (k - a) / float(b - a)
                out[k].append((tid, (1 - w) * wins[a][0] + w * wins[b][0], (1 - w) * wins[a][1] + w * wins[b][1]))
    return out


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
                 evidence_channels=8, member_hidden=32, evidence_form="linear", fusion="product",
                 evidence_source="raw", recursion="additive", factor_training=False, state="box",
                 footprint_radius=3, presence_tolerance=0, future_stats="smoothed", commit_lag=None,
                 evidence_pool="sum", gate_grad="presence", raw_residual=True, expert_input="feature"):
        super(ChangepointLayer, self).__init__()
        if commit not in COMMITS:
            raise ValueError("commit 必须是 %s 之一" % (COMMITS,))
        if evidence_form not in EVIDENCE_FORMS or fusion not in FUSIONS:
            raise ValueError("evidence_form 必须是 %s 之一，fusion 必须是 %s 之一" % (EVIDENCE_FORMS, FUSIONS))
        if evidence_source not in EVIDENCE_SOURCES or recursion not in RECURSIONS:
            raise ValueError("evidence_source 必须是 %s 之一，recursion 必须是 %s 之一" % (EVIDENCE_SOURCES, RECURSIONS))
        if evidence_source == "backbone" and evidence_form != "linear":
            raise ValueError("evidence_source = backbone 时证据是线性的对数似然比（evidence_form = linear）")
        if state not in STATES:
            raise ValueError("state 必须是 %s 之一" % (STATES,))
        if state == "centre" and evidence_source != "backbone":
            raise ValueError("state = centre 的证据是足迹加权的骨干标记泊松项（evidence_source = backbone）")
        self.state, self.foot_radius, self.pres_tol = str(state), int(footprint_radius), int(presence_tolerance)
        # evaluation-only ablations (do not change the trained model):
        #   future_stats = past: the same head reads only the statistics known at the event's arrival (no future);
        #   commit_lag = k: fixed commit at lag k <= defer_max (the lag input stays normalised by defer_max)
        if future_stats not in ("smoothed", "past"):
            raise ValueError("future_stats 必须是 smoothed 或 past")
        if future_stats == "past" and fusion not in ("learned", "mixture"):
            raise ValueError("future_stats = past 只用于 fusion = learned / mixture 的同头对照")
        self.future_stats = str(future_stats)
        # v4.1: evidence_pool = robust adds a dependence-robust channel log(1 + sum_e exp(z0_e + beta)) (~ the strongest
        # event of the cell) next to the independent-marks sum; the conv starts on the robust channel only
        if evidence_pool not in ("sum", "robust"):
            raise ValueError("evidence_pool 必须是 sum 或 robust")
        if evidence_pool == "robust" and (evidence_source != "backbone" or state != "box"):
            raise ValueError("evidence_pool = robust 只用于框占据状态的骨干证据")
        self.evidence_pool = str(evidence_pool)
        # gate_grad = task (v4.1): the event loss also trains the presence evidence through the mixture gate
        # (standard mixture-of-experts gate training; L_pres stays as the auxiliary occupancy supervision);
        # the transition parameters stay label-only under factor_training
        if gate_grad not in ("presence", "task"):
            raise ValueError("gate_grad 必须是 presence 或 task")
        if gate_grad == "task" and fusion not in ("learned", "mixture"):
            raise ValueError("gate_grad = task 只用于 fusion = learned / mixture")
        self.gate_task = gate_grad == "task"
        # raw_residual = False (v4.1): the presence evidence is the backbone term only (hybrid NN-HMM). The raw-count
        # residual measures absolute activity, which shifts with the background density: in dense noise it exceeded the
        # silence cost and opened the gate over the background (local diagnosis, Inn 13.84)
        self.raw_residual = bool(raw_residual)
        if not self.raw_residual and (evidence_source != "backbone" or state != "box"):
            raise ValueError("raw_residual = False 只用于框占据状态的骨干证据")
        # expert_input = score (v4.1): the mixture experts read only the backbone's per-event score (+ polarity, time in
        # window), not its internal dec1 features: the backbone judges appearance, this layer judges temporal consistency,
        # and a head trained on cross-fitted backbone outputs transfers to the full backbone (same score semantics)
        if expert_input not in ("feature", "score"):
            raise ValueError("expert_input 必须是 feature 或 score")
        if expert_input == "score" and fusion != "mixture":
            raise ValueError("expert_input = score 只用于 fusion = mixture")
        self.expert_input = str(expert_input)
        n_in = int(feat_dim) + 3 if self.expert_input == "feature" else 3
        self.commit_lag = int(defer_max) if commit_lag is None else int(commit_lag)
        if not 1 <= self.commit_lag <= int(defer_max):
            raise ValueError("需要 1 <= commit_lag <= defer_max")
        self.evidence_form, self.fusion = str(evidence_form), str(fusion)
        self.evidence_source, self.recursion = str(evidence_source), str(recursion)
        self.factor_training = bool(factor_training)
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
        n_raw = int(evidence_channels) if self.raw_residual else 0
        self.ev1 = (nn.Conv2d(self.channels, n_raw, self.cell, stride=self.cell, bias=False)
                    if self.raw_residual else None)
        backbone = self.evidence_source == "backbone"
        centre = self.state == "centre"
        n_back = 0 if (not backbone or centre) else (2 if self.evidence_pool == "robust" else 1)
        self.ev2 = nn.Conv2d(n_raw + n_back, 1, 3, padding=1, bias=False)
        if backbone:
            with torch.no_grad():
                self.ev2.weight.zero_()
                if not centre:                       # box: start at the backbone term of the own cell (robust: max-like)
                    self.ev2.weight[0, -1, 1, 1] = 1.0
            self.obs_bias = nn.Parameter(torch.zeros(()))
        size_f = 2 * self.foot_radius + 1
        if centre:                                   # log footprint h(d), d = event cell - centre cell (E1c)
            self.foot_logit = nn.Parameter(torch.zeros(size_f, size_f))
            fy, fx = np.meshgrid(np.arange(-self.foot_radius, self.foot_radius + 1),
                                 np.arange(-self.foot_radius, self.foot_radius + 1), indexing="ij")
            self.register_buffer("foot_dy", torch.from_numpy(fy.reshape(-1).astype(np.int64)), persistent=False)
            self.register_buffer("foot_dx", torch.from_numpy(fx.reshape(-1).astype(np.int64)), persistent=False)
        self.mu_raw = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.rho_raw = nn.Parameter(torch.tensor(_scalar_logit(1e-3)))
        self.p_raw = nn.Parameter(torch.tensor(_scalar_logit(0.9)))
        self.pi_logit = nn.Parameter(torch.zeros(len(self.offsets)))
        if self.fusion == "attribution":            # (E8a) event term + log target intensity field
            self.att_slope = nn.Parameter(torch.ones(()))
            self.att_bias = nn.Parameter(torch.tensor(-self.log_w))
            self.att_undetected = nn.Parameter(torch.zeros(()))
            self.att_foot = nn.Parameter(torch.zeros(size_f, size_f))
            self.att_res = nn.Sequential(nn.Linear(int(feat_dim) + 3, int(member_hidden)), nn.ReLU(),
                                         nn.Linear(int(member_hidden), 1))
            with torch.no_grad():                    # residual starts at zero: the output starts at V5
                self.att_res[2].weight.zero_()
                self.att_res[2].bias.zero_()
        if self.fusion == "mixture":                # (E8m) presence-gated experts: S m + (1 - S) b
            self.member = nn.Sequential(nn.Linear(n_in, int(member_hidden)), nn.ReLU(),
                                        nn.Linear(int(member_hidden), 1))
        if self.fusion in ("learned", "mixture"):   # (E8l) z0 + MLP(sufficient statistics of past / future presence)
            self.att_foot = nn.Parameter(torch.zeros(size_f, size_f))
            hid = int(member_hidden)
            self.fuse_static = nn.Linear(n_in, hid)
            self.fuse_dyn = nn.Linear(5, hid, bias=False)
            self.fuse_out = nn.Linear(hid, 1)
            with torch.no_grad():                    # zero last layer: the output starts at V5
                self.fuse_out.weight.zero_()
                self.fuse_out.bias.zero_()
        elif self.fusion == "evidence":             # (E8e) z0 + kappa_e x future log-LR of the event's place
            self.att_foot = nn.Parameter(torch.zeros(size_f, size_f))
            self.kappa = nn.Sequential(nn.Linear(int(feat_dim) + 3, int(member_hidden)), nn.ReLU(),
                                       nn.Linear(int(member_hidden), 1))
            with torch.no_grad():                    # kappa starts at softplus(0) = 0.69 for every event
                self.kappa[2].weight.zero_()
                self.kappa[2].bias.zero_()
        elif self.fusion not in ("mixture", "attribution"):
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

    def commit_bounds(self):
        """自适应提交的 (lo, hi)，校准 logit 尺度：以判决阈值 tau* 为中心、各差 logit(1 - alpha)。"""
        half = _scalar_logit(1.0 - self.alpha)
        return self.tau_logit - half, self.tau_logit + half

    def kernel(self):
        """转移核 [K, K]：33 个位移上的 softmax 概率放在对应位置（前向卷积、后向相关共用）。"""
        w = torch.softmax(self.pi_logit, 0)
        return (w[:, None, None] * self.basis).sum(0)

    def dynamics(self, detach=False):
        """递推用的 (rho, p, 位移概率 [K], 转移核)。detach = 只由 L_dyn 训练（factor_training）。"""
        w = torch.softmax(self.pi_logit, 0)
        out = (self.rho(), self.p(), w, (w[:, None, None] * self.basis).sum(0))
        return tuple(x.detach() for x in out) if detach else out

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
            out = {"survival": p, "mean_life_windows": 1.0 / max(1.0 - p, 1e-9), "birth": float(self.rho()),
                   "silence_cost": float(self.mu()), "move_prob_by_ring": by_ring}
            if self.evidence_source == "backbone" and self.state == "box":   # marked-Poisson: beta, own-cell gain
                out.update(obs_bias=float(self.obs_bias), obs_gain=float(self.ev2.weight[0, -1, 1, 1]))
                if self.evidence_pool == "robust":                         # gain of the independent-sum channel
                    out.update(obs_gain_sum=float(self.ev2.weight[0, -2, 1, 1]))
            if self.state == "centre":
                out.update(obs_bias=float(self.obs_bias), obs_gain=float(torch.exp(self.foot_logit).sum()),
                           foot_centre=float(self.foot_logit[self.foot_radius, self.foot_radius]))
            if self.fusion == "evidence":
                out.update(att_foot_centre=float(self.att_foot[self.foot_radius, self.foot_radius]),
                           att_foot_max=float(self.att_foot.max()), kappa_bias=float(self.kappa[2].bias))
            if self.fusion in ("learned", "mixture"):
                out.update(att_foot_centre=float(self.att_foot[self.foot_radius, self.foot_radius]),
                           att_foot_max=float(self.att_foot.max()),
                           fuse_out_norm=float(self.fuse_out.weight.norm()), fuse_out_bias=float(self.fuse_out.bias))
            if self.fusion == "attribution":
                out.update(att_slope=float(self.att_slope), att_bias=float(self.att_bias),
                           att_undetected=float(self.att_undetected),
                           att_foot_centre=float(self.att_foot[self.foot_radius, self.foot_radius]),
                           att_foot_max=float(self.att_foot.max()))
            return out

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

    def evidence(self, x, v=None):
        """(E1) l = g(X) - mu（linear）或 log(1 + g(X)) - mu（log1p，v1），[B, hc, wc]；g >= 0 且 g(0) = 0。
        evidence_source = backbone（v3）：l = h([relu(ev1(X)), v]) - mu，v = backbone_map（格内骨干的标记泊松项），
        h 是无偏置的 3x3 卷积、不再取 ReLU（证据可正可负）；零输入、没有事件时仍是 -mu。"""
        if self.ev1 is None:                            # backbone term only (raw_residual = False)
            v = v.to(self.mu_raw.dtype) if v.dim() == 4 else v[:, None].to(self.mu_raw.dtype)
            return self.ev2(v).squeeze(1) - self.mu()
        h = F.relu(self.ev1(x))
        if self.evidence_source == "backbone":
            if v is None:
                raise ValueError("evidence_source = backbone 需要格内骨干项 v（backbone_map）")
            if self.state == "centre":                  # (E1c) footprint-weighted backbone term + raw residual
                return v.to(h.dtype) + self.ev2(h).squeeze(1) - self.mu()
            v = v.to(h.dtype) if v.dim() == 4 else v[:, None].to(h.dtype)     # robust pool: [B, 2, hc, wc]
            return self.ev2(torch.cat([h, v], 1)).squeeze(1) - self.mu()
        g = F.relu(self.ev2(h)).squeeze(1)
        return (g if self.evidence_form == "linear" else torch.log1p(g)) - self.mu()

    def backbone_map(self, z0, b, cy, cx, batch):
        """(E1') 每格本窗事件的 sum_e softplus(z0_e + beta)，[B, hc, wc]。z0 = 骨干报告尺度 logit = 缩放对数似然比
        （正样本权重 = 训练集负 / 正之比）；标记泊松叠加下，事件 e 让"格里有目标"的对数似然比增加 log(1 + r LR_e)，
        r = 目标 / 背景强度比 = exp(beta)。没有事件的格为 0。"""
        z = z0.to(self.obs_bias.dtype) + self.obs_bias
        b, cy, cx = b.long(), cy.long(), cx.long()
        if self.state == "centre":
            # (E1c) a target centred at c explains event e with log footprint h(c_e - c):
            # V(c) = sum_e softplus(z0_e + beta + h(c_e - c)), over the (2R+1)^2 centres around each event
            ty, tx = cy[:, None] - self.foot_dy[None, :], cx[:, None] - self.foot_dx[None, :]
            ok = (ty >= 0) & (ty < self.hc) & (tx >= 0) & (tx < self.wc)
            val = F.softplus(z[:, None] + self.foot_logit.reshape(1, -1))[ok]
            flat = ((b[:, None] * self.hc + ty) * self.wc + tx)[ok]
        else:
            val, flat = F.softplus(z), (b * self.hc + cy) * self.wc + cx
        out = torch.zeros(batch * self.hc * self.wc, device=val.device, dtype=val.dtype)
        if int(flat.shape[0]):
            # accumulate=True: deterministic on CUDA in torch 1.9 (same path as encode)
            out = out.index_put((flat,), val, accumulate=True)
        out = out.view(batch, self.hc, self.wc)
        if self.evidence_pool != "robust":
            return out
        # (E1r) dependence-robust pool: softplus(log sum_e exp(z_e)) per cell (shift by the window max for stability)
        lse = torch.zeros(batch * self.hc * self.wc, device=z.device, dtype=z.dtype)
        if int(flat.shape[0]):
            k = z.detach().max()
            lse = lse.index_put((flat,), torch.exp(z - k), accumulate=True)
        has = lse > 0
        lse = torch.where(has, F.softplus(torch.log(lse.clamp(min=1e-30)) + (k if int(flat.shape[0]) else 0.0)),
                          torch.zeros_like(lse))
        return torch.stack([out, lse.view(batch, self.hc, self.wc)], 1)

    def event_term(self, feat, p, t_local, z0):
        """(E8a) 事件自身的一项 u_e = a z0_e + c + r(特征)；r 是零初始化的残差 MLP（输入同骨干读出层 + z0）。"""
        x = torch.cat([feat.to(z0.dtype), p.to(z0.dtype)[:, None], t_local.to(z0.dtype)[:, None], z0[:, None]], 1)
        return self.att_slope * z0 + self.att_bias + self.att_res(x).squeeze(-1)

    def inheritance(self, feat, p, t_local, z0):
        """(E8e) 事件继承所在位置未来证据的系数 kappa_e = softplus(MLP) >= 0（理论值：目标事件 1、背景事件 0）。"""
        x = torch.cat([feat.to(z0.dtype), p.to(z0.dtype)[:, None], t_local.to(z0.dtype)[:, None], z0[:, None]], 1)
        return F.softplus(self.kappa(x).squeeze(-1))

    def _expert_x(self, feat, p, t_local, lp0):
        """专家的事件输入：[dec1 特征, p, t_local, V5 校准 logit]；expert_input = score 时去掉 dec1 特征。"""
        parts = [p.to(lp0.dtype)[:, None], t_local.to(lp0.dtype)[:, None], lp0[:, None]]
        return torch.cat(([feat.to(lp0.dtype)] if self.expert_input == "feature" else []) + parts, 1)

    def fuse_static_term(self, feat, p, t_local, lp0):
        """(E8l) 融合 MLP 第一层里只依赖事件本身的部分（到达时算一次）：W_s [dec1 特征, p, t_local, V5 校准 logit] + b。"""
        return self.fuse_static(self._expert_x(feat, p, t_local, lp0))

    def fuse_learned(self, h_static, lp0, s_now, f_now, past, lag):
        """(E8l) 校准 logit = V5 + MLP(充分统计量)：本格平滑 / 过滤在场的 logit、足迹场的平滑 / 过滤值、滞后比例。
        判决只依赖 V5 的因果后验与回溯平滑的在场（变点后验）这几项统计量；怎样组合交给网络学（最后一层零初始化）。"""
        x = torch.stack([logit(s_now) / STAT_SCALE, past[:, 0] / STAT_SCALE, f_now / STAT_SCALE,
                         past[:, 1] / STAT_SCALE, lag.to(lp0.dtype) / float(self.defer_max)], 1)
        return lp0 + self.fuse_out(F.relu(h_static + self.fuse_dyn(x))).squeeze(-1)

    def target_field(self, S):
        """(E8a) 目标强度场 log sum_c' S(c') exp(g(c - c'))，S: [..., hc, wc] -> 同形状（g 为事件格 - 中心格的位移）。"""
        r = self.foot_radius
        shape = S.shape
        w = torch.flip(torch.exp(self.att_foot), (0, 1))[None, None].to(S.dtype)
        f = F.conv2d(S.reshape(-1, 1, shape[-2], shape[-1]), w, padding=r)
        return torch.log(f.clamp(min=1e-30)).reshape(shape)

    def init_from_labels(self, hist, n_events, n_centres):
        """v4：用训练集标签初始化足迹的形状（之后照常学习）：h = log f（目标事件相对中心格的位移分布，归一化），
        归属的足迹 g = log f - max log f。沉默代价 mu 不用标签均值：目标的事件数离散得厉害（暗弱目标常常只有几个事件），
        固定均值的泊松会把它们判成强反证据（本地核对：mu = 12.75 时第一拍附近的强度场 log F 中位 -13），mu 从 1 起学。
        返回的 target_events_per_window 只作记录。"""
        hist = np.asarray(hist, dtype=np.float64)
        f = hist / max(float(hist.sum()), 1.0)
        logf = np.log(np.maximum(f, 1e-6))
        rate = float(n_events) / max(float(n_centres), 1.0)
        with torch.no_grad():
            if self.state == "centre":
                self.foot_logit.copy_(torch.from_numpy(logf).to(self.foot_logit.dtype))
            if self.fusion in ("attribution", "evidence", "learned", "mixture"):
                self.att_foot.copy_(torch.from_numpy(logf - logf.max()).to(self.att_foot.dtype))
        return {"footprint_peak": float(f.max()), "footprint_inside": float(hist.sum()) / max(float(n_events), 1.0),
                "target_events_per_window": rate}

    def _shifted(self, f):
        """[B, K - 1, hc, wc]：每个非零位移 o 的 f(x - o)（来源在画布外 = 0）。"""
        r, (h, w) = self.radius, f.shape[-2:]
        fp = F.pad(f, (r, r, r, r))
        return torch.stack([fp[:, r - dy:r - dy + h, r - dx:r - dx + w] for dy, dx in self.offsets[1:]], 1)

    def _pooled_logit(self, P):
        """位置容忍的在场（presence_tolerance = r）：logit P(本格 +-r 格内有目标中心) = logit(1 - prod(1 - P))，[B, hc, wc]。
        中心点检测的标准做法（CenterNet / CornerNet 不惩罚中心附近的预测）：一个目标的证据天然落在中心周围几格，
        逐格监督中心会惩罚位置差一格的在场，只能整体压低在场。"""
        r = self.pres_tol
        lk = torch.log1p(-P.clamp(max=1.0 - EPS))
        ones = torch.ones(1, 1, 2 * r + 1, 2 * r + 1, dtype=P.dtype, device=P.device)
        s = F.conv2d(F.pad(lk[:, None], (r, r, r, r)), ones).squeeze(1)          # log prod(1 - P) <= 0
        return torch.log((-torch.expm1(s)).clamp(min=1e-12)) - s

    def _pooled_label(self, occ):
        r = self.pres_tol
        return F.max_pool2d(occ[:, None], 2 * r + 1, stride=1, padding=r).squeeze(1)

    def _union_prior(self, P, rho, p, w):
        """(E2u) 预测在场：本格（出生或留在原格，互斥）与各邻格移入（独立）求并。"""
        own = (1.0 - P) * rho + P * p * w[0]
        mv = p * w[1:, None, None] * self._shifted(P)
        keep = torch.log1p(-own.clamp(max=1.0 - EPS)) + torch.log1p(-mv.clamp(max=1.0 - EPS)).sum(1)
        return -torch.expm1(keep)

    def transition_loss(self, prev, cur, k):
        """L_dyn：已知上一窗的占据 prev（标签），按 (E2) 预测这一窗的占据，与标签 cur 的逐格 BCE（未求和）。
        recursion = union 时用 (E2u)，与前向同一个预测式。"""
        if self.recursion == "union":
            pred = self._union_prior(prev, self.rho(), self.p(), torch.softmax(self.pi_logit, 0))
        else:
            pred = (1.0 - prev) * self.rho() + self.p() * self._scatter(prev, k)
        return F.binary_cross_entropy(pred.clamp(EPS, 1.0 - EPS), cur, reduction="none")

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

    def forward_step(self, P, ell, k, dyn=None):
        """(E2)~(E4)：上一窗过滤在场 P、本窗证据 ell -> (本窗过滤在场, 本窗预测在场)。dyn = dynamics() 的结果
        （不给时用当前参数，k 与之一致）。recursion = union（v3）：(E3) 的来源归一化不变；落到同一格的质量里，
        本格的出生与留在原格互斥（相加），各邻格来源独立（1 - prod(1 - .)），所以在场 < 1，不需要截断。"""
        rho, p, w = (self.rho(), self.p(), torch.softmax(self.pi_logit, 0)) if dyn is None else dyn[:3]
        lam = torch.exp(ell.clamp(-CLAMP, CLAMP))
        Z = (1.0 - P) * ((1.0 - rho) + rho * lam) + P * ((1.0 - p) + p * self._gather(lam, k, 1.0))
        if self.recursion == "union":
            own = ((1.0 - P) * rho + P * p * w[0]) * lam / Z
            mv = p * w[1:, None, None] * lam[:, None] * self._shifted(P / Z)
            keep = torch.log1p(-own.clamp(max=1.0 - EPS)) + torch.log1p(-mv.clamp(max=1.0 - EPS)).sum(1)
            return (-torch.expm1(keep)).clamp(EPS, 1.0 - EPS), self._union_prior(P, rho, p, w).clamp(EPS, 1.0 - EPS)
        post = (1.0 - P) * rho * lam / Z + lam * p * self._scatter(P / Z, k)
        pred = (1.0 - P) * rho + p * self._scatter(P, k)
        return post.clamp(EPS, 1.0 - EPS), pred.clamp(EPS, 1.0 - EPS)

    def rts_step(self, filt_u, pred_next, smooth_next, k, dyn=None):
        """(E5)：由 u+1 的平滑在场倒推 u 的平滑在场。recursion = union：越过 1 的部分光滑饱和（softcap）而不是硬截断。"""
        p = self.p() if dyn is None else dyn[1]
        ratio = smooth_next / pred_next
        term = (1.0 - p) * (1.0 - smooth_next) / (1.0 - pred_next) + p * self._gather(ratio, k, 1.0)
        s = filt_u * term
        if self.recursion == "union":
            s = softcap(s)
        return s.clamp(EPS, 1.0 - EPS)

    def membership(self, feat, p, t_local, lp0):
        if self.fusion == "mixture":
            x = self._expert_x(feat, p, t_local, lp0)
        else:
            x = torch.cat([feat.to(lp0.dtype), p.to(lp0.dtype)[:, None], t_local.to(lp0.dtype)[:, None],
                           lp0[:, None]], 1)
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
        k = self.kernel()                                   # L_dyn always trains the transition parameters
        dyn = self.dynamics(detach=train and self.factor_training)
        k_rec = dyn[3]
        lp0_all = z0 - self.log_w
        lo_commit, hi_commit = self.commit_bounds()
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
            eb = ev["b"].long()
            cy = torch.div(ev["y"].long(), self.cell, rounding_mode="floor").clamp(max=self.hc - 1)
            cx = torch.div(ev["x"].long(), self.cell, rounding_mode="floor").clamp(max=self.wc - 1)
            # (E1)~(E4) evidence and forward
            v = self.backbone_map(z0[idx], eb, cy, cx, batch) if self.evidence_source == "backbone" else None
            ell = self.evidence(self.encode(ev, batch) if self.ev1 is not None else None, v)
            P, pr = self.forward_step(P, ell, k_rec, dyn)
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
                smooth[j] = self.rts_step(filt[j], pred[j + 1], smooth[j + 1], k_rec, dyn)
            if train and self.pres_tol > 0:                 # L_pres on presence within +-tol cells (centre state)
                tot = F.binary_cross_entropy_with_logits(self._pooled_logit(P), self._pooled_label(occ[-1]),
                                                         reduction="sum")
                cnt = P.numel()
                for lag in range(1, n):
                    valid = (lag < age).to(P.dtype)[:, None, None]
                    s = smooth[n - 1 - lag]
                    tot = tot + (F.binary_cross_entropy_with_logits(
                        self._pooled_logit(s), self._pooled_label(occ[n - 1 - lag]), reduction="none") * valid).sum()
                    cnt += int(valid.sum()) * self.hc * self.wc
            elif train:                                     # L_pres: filtered + every smoothing lag actually used
                tot = F.binary_cross_entropy_with_logits(logit(P), occ[-1], reduction="sum")
                cnt = P.numel()
                for lag in range(1, n):
                    valid = (lag < age).to(P.dtype)[:, None, None]
                    s = smooth[n - 1 - lag]
                    tot = tot + (F.binary_cross_entropy_with_logits(logit(s), occ[n - 1 - lag], reduction="none")
                                 * valid).sum()
                    cnt += int(valid.sum()) * self.hc * self.wc
            if train:
                aux["exist_sum"] = aux["exist_sum"] + tot
                aux["exist_n"] += cnt
            # waiting events: one more window of future evidence
            if q is not None and int(q["b"].shape[0]):
                q = dict(q, lag=q["lag"] + 1)
                stack = torch.stack(smooth)                                       # [n, B, hc, wc]
                j = (n - 1 - q["lag"]).clamp(min=0)
                s_val = stack[j, q["b"], q["cy"], q["cx"]]
                if self.fusion in ("learned", "mixture"):     # (E8l) V5 + MLP(statistics of past / future presence)
                    S_all = stack.detach() if (train and self.factor_training and not self.gate_task) else stack
                    fld = self.target_field(S_all)[j, q["b"], q["cy"], q["cx"]]
                    s_now = S_all[j, q["b"], q["cy"], q["cx"]]
                    if self.future_stats == "past":          # same head, statistics at arrival only (ablation)
                        s_now, fld = torch.sigmoid(q["q"][:, 0]), q["q"][:, 1]
                    z_cal = self.fuse_learned(q["h"] if self.fusion == "mixture" else q["m"], q["lp0"], s_now, fld,
                                              q["q"], q["lag"])
                    if self.fusion == "mixture":              # (E8m) P = S m + (1 - S) b, b = the fallback above
                        z_cal = logit((s_now * q["m"] + (1.0 - s_now) * torch.sigmoid(z_cal)).clamp(EPS, 1.0 - EPS))
                elif self.fusion == "evidence":               # (E8e) z0 + kappa_e (log F_smooth - log F_filt)
                    S_all = stack.detach() if (train and self.factor_training) else stack
                    fld = self.target_field(S_all)[j, q["b"], q["cy"], q["cx"]]
                    L = (fld - q["q"]).clamp(-CLAMP, CLAMP)        # q = log F of the filtered presence at arrival
                    z_cal = q["lp0"] + q["m"] * L
                elif self.fusion == "attribution":            # (E8a) u_e + log(undetected + target field)
                    S_all = stack.detach() if (train and self.factor_training) else stack
                    fld = self.target_field(S_all)[j, q["b"], q["cy"], q["cx"]]
                    z_cal = q["m"] + torch.logaddexp(self.att_undetected.expand_as(fld), fld)
                elif self.fusion == "product":                # P = presence x membership
                    if train and self.factor_training:        # presence has its own label (L_pres): L_evt trains m only
                        s_val = s_val.detach()
                    z_cal = logit((s_val * q["m"]).clamp(EPS, 1.0 - EPS))
                else:                                         # v1: Bayes factor on the V5 prior, presence detached
                    L = (logit(s_val) - logit(q["q"])).detach()
                    z_cal = fuse(q["lp0"], q["m"], q["q"], L)
                if train:
                    aux["evt_sum"] = aux["evt_sum"] + F.binary_cross_entropy_with_logits(z_cal, q["y"], reduction="sum")
                    aux["evt_n"] += int(z_cal.shape[0])
                z_cal = z_cal.detach()
                deadline = q["lag"] >= self.commit_lag
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
                lp0 = lp0_all[idx]
                if self.fusion == "attribution":              # queue slot "m" carries the event term u_e
                    m = self.event_term(feat[idx], events["p"][idx], events["t_local"][idx], z0[idx])
                elif self.fusion == "evidence":               # queue slot "m" carries kappa_e
                    m = self.inheritance(feat[idx], events["p"][idx], events["t_local"][idx], z0[idx])
                elif self.fusion == "learned":                # queue slot "m" carries the static first-layer term
                    m = self.fuse_static_term(feat[idx], events["p"][idx], events["t_local"][idx], lp0)
                elif self.fusion == "mixture":                # membership m (L_mem) + fallback's static term h
                    m = self.membership(feat[idx], events["p"][idx], events["t_local"][idx], lp0)
                else:
                    m = self.membership(feat[idx], events["p"][idx], events["t_local"][idx], lp0)
                if train and self.fusion not in ("attribution", "evidence", "learned"):     # L_mem
                    inside = occ[-1][eb, cy, cx] > 0.5
                    if bool(inside.any()):
                        aux["mem_sum"] = aux["mem_sum"] + F.binary_cross_entropy(
                            m[inside], labels[idx][inside].to(m.dtype), reduction="sum")
                        aux["mem_n"] += int(inside.sum())
                key = keys[idx] if keys is not None else idx          # file index (int64) or training label
                if self.fusion == "evidence":                 # past-only field of the event's place (E8e)
                    q_val = self.target_field(P.detach() if (train and self.factor_training) else P)[eb, cy, cx]
                elif self.fusion in ("learned", "mixture"):   # past-only statistics: logit S_filt, log F_filt
                    Pd = P.detach() if (train and self.factor_training and not self.gate_task) else P
                    q_val = torch.stack([logit(Pd[eb, cy, cx]), self.target_field(Pd)[eb, cy, cx]], 1)
                else:
                    q_val = P[eb, cy, cx].detach()
                entry = {"b": eb, "cy": cy, "cx": cx, "lag": torch.zeros_like(eb), "key": key, "z0": z0[idx],
                         "lp0": lp0, "m": m, "q": q_val,
                         "y": labels[idx].to(z0.dtype) if train else torch.zeros_like(lp0),
                         "prov": (lp0 >= self.tau_logit).to(z0.dtype), "zlast": lp0.detach()}
                if self.fusion == "mixture":
                    entry["h"] = self.fuse_static_term(feat[idx], events["p"][idx], events["t_local"][idx], lp0)
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
