"""位置级告警（CUSUM 的 M >= theta，带复位）的评估。与原 utils/eval.py 的逐事件 Pd/Fa 是不同的随机量，单独报告。

对每个告警阈值累计：
    目标级    每个目标第一次"真告警"所在窗 k -> 首次告警延迟 = (k+1)*窗长 - 该目标第一个事件的时间；
              检出率 = 至少有一次真告警的目标占比。
              "真告警"：告警像素与该目标在本窗或上一窗的某个事件的切比雪夫距离 <= radius。
    虚警      告警像素中离所有目标事件（本窗与上一窗）都超过 radius 的像素，按 8 连通域计数；
              虚警率 = 连通域数 / (窗数 × 画面像素数)。
    上界      bound_per_location_window 为早先的并集界 V/(e^theta - 1)（保留旧键，便于和旧结果对照）；
              tight_bound_per_location_window 为紧界 (1 + 1/L)/(e^theta - 1)（L 为每条序列的平均窗数），与 V 无关。
              两者都约束"告警位置数"（每个像素 M >= theta 记一次），连通域数不多于告警位置数。
虚警诊断（09-24）：
    虚警到目标的距离  每个虚警连通域到"最近 history_windows 窗内任一目标事件"的切比雪夫距离，分档计数；
                      近处的虚警是目标在场时放错位置的告警（阶段 0 实测：大多数虚警都在目标 8 像素内），
                      远处或近 1 s 没有目标时是背景（热像素、成簇 OFF 事件、未标注的物体）
    重复位置          虚警连通域 2 像素以内在之前 history_windows 窗里出现过虚警（持续的静态干扰或未标注物体）
    去重虚警事件      与前 2 窗的虚警像素相距 dedup_radius 以上的虚警连通域才算一次新的虚警事件；另给每小时数
    纯背景窗          本窗与之前 pure_windows 窗内都没有目标事件。分两类报告（外部评审 09-24）：
                      start = 序列里第一个目标出现之前（干净的 H0：状态从未见过目标）；
                      after = 目标离开 1 s 以上之后（可能还带着目标留下的膜电位，不是干净的 H0）。
                      上界约束的是期望：不要求每次有限样本都低于它，所以另给单侧检验的 p 值——
                      P(Poisson(紧界下的期望数) >= 实测虚警连通域数)，连通域数不多于告警位置数，p 值小才算越界的证据
只依赖 numpy；膨胀、连通域与距离变换优先用 scipy，没有时用 cv2。
"""
import numpy as np

DISTANCE_BINS = (8, 16, 32, 64)


def _label(mask):
    """8 连通域标记，返回 (标记图, 个数)。"""
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        return labels, int(n)
    except ImportError:
        import cv2
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels, int(n) - 1


def _label_count(mask):
    """8 连通域个数。"""
    return _label(mask)[1]


def _dilate(mask, radius):
    """方形结构元（边长 2*radius+1）的二值膨胀，即切比雪夫距离 <= radius 的区域。"""
    if radius <= 0:
        return mask
    try:
        from scipy import ndimage
        return ndimage.binary_dilation(mask, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool))
    except ImportError:
        import cv2
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def poisson_upper_p(n, mu):
    """P(X >= n)，X ~ Poisson(mu)。没有 scipy 时用带连续性修正的正态近似。"""
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import poisson
        return float(poisson.sf(int(n) - 1, float(mu)))
    except ImportError:
        import math
        z = (float(n) - 0.5 - float(mu)) / math.sqrt(max(float(mu), 1e-12))
        return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chebyshev_distance(sources, shape):
    """每个像素到 sources（布尔图中为真的像素）的切比雪夫距离；sources 为空时返回 None。"""
    if not sources.any():
        return None
    try:
        from scipy import ndimage
        return ndimage.distance_transform_cdt(~sources, metric="chessboard").astype(np.float64)
    except ImportError:
        # 没有 scipy 时逐档膨胀（只需要分档，档位之外记为无穷远）
        dist = np.full(shape, np.inf)
        dist[sources] = 0.0
        for radius in DISTANCE_BINS:
            grown = _dilate(sources, radius)
            dist[grown & np.isinf(dist)] = float(radius)
        return dist


class AlarmEvaluator(object):
    """用法：每条序列 begin_sequence(seq) -> 每窗每个阈值 update(theta, k, alarm) -> end_sequence()；最后 summary()。

    参数: thetas 告警阈值列表；height/width 画面大小（告警图需裁到这个大小）；window_ms 窗长；radius 真告警的容差（像素）；
          hypotheses 速度假设数 V（只用于写出旧的并集界）；history_windows 距离与重复位置诊断回看的窗数；
          pure_windows 纯背景窗要求之前多少窗内没有目标事件；dedup_radius 去重虚警事件的距离（像素）。
    """

    def __init__(self, thetas, height, width, window_ms, radius, hypotheses, history_windows=20, pure_windows=20,
                 dedup_radius=8):
        self.thetas = [float(t) for t in thetas]
        self.height, self.width = int(height), int(width)
        self.window_ms, self.radius = float(window_ms), int(radius)
        self.hypotheses = int(hypotheses)
        self.history_windows, self.pure_windows = int(history_windows), int(pure_windows)
        self.dedup_radius = int(dedup_radius)
        self.records = {t: [] for t in self.thetas}
        self.false_components = {t: 0 for t in self.thetas}
        self.alarm_pixels = {t: 0 for t in self.thetas}
        self.distance_counts = {t: np.zeros(len(DISTANCE_BINS) + 2, dtype=np.int64) for t in self.thetas}
        self.repeat_components = {t: 0 for t in self.thetas}
        self.false_events = {t: 0 for t in self.thetas}
        self.pure_alarm_pixels = {t: 0 for t in self.thetas}
        self.pure_false_components = {t: 0 for t in self.thetas}
        self.pure_split = {part: {"windows": 0, "pixels": {t: 0 for t in self.thetas},
                                  "components": {t: 0 for t in self.thetas}} for part in ("start", "after")}
        self.windows = 0
        self.sequences = 0
        self.pure_window_count = 0

    def begin_sequence(self, seq, n_windows):
        """准备一条序列：每窗的目标事件坐标与编号、每个目标的首个事件时间、哪些窗是纯背景窗。"""
        pos = (seq.label == 1) & (seq.target_id != 0)
        window = seq.t // int(self.window_ms)
        self.targets = {}
        for k in np.unique(window[pos]):
            sel = pos & (window == k)
            self.targets[int(k)] = (seq.y[sel].astype(np.int64), seq.x[sel].astype(np.int64), seq.target_id[sel])
        self.t_first = {float(tid): float(seq.t[pos & (seq.target_id == tid)].min())
                        for tid in np.unique(seq.target_id[pos])}
        self.first_alarm = {t: {} for t in self.thetas}
        # 纯背景窗：[k - pure_windows, k] 内没有任何目标事件
        has_target = np.zeros(int(n_windows), dtype=bool)
        for k in self.targets:
            if 0 <= k < int(n_windows):
                has_target[k] = True
        recent = np.zeros(int(n_windows), dtype=bool)
        for k in range(int(n_windows)):
            recent[k] = has_target[max(0, k - self.pure_windows):k + 1].any()
        self.pure = ~recent
        self.pure_window_count += int(self.pure.sum())
        first = min(self.targets) if self.targets else int(n_windows)
        self.pure_start = self.pure & (np.arange(int(n_windows)) < first)
        self.pure_split["start"]["windows"] += int(self.pure_start.sum())
        self.pure_split["after"]["windows"] += int((self.pure & ~self.pure_start).sum())
        # 每个像素最近一次出现虚警的窗号（重复位置与去重用）
        self.last_false = {t: np.full((self.height, self.width), -10 ** 9, dtype=np.int64) for t in self.thetas}
        self.windows += int(n_windows)
        self.sequences += 1

    def _near(self, ys, xs):
        """与给定事件坐标切比雪夫距离 <= radius 的像素区域。"""
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[np.clip(ys, 0, self.height - 1), np.clip(xs, 0, self.width - 1)] = True
        return _dilate(mask, self.radius)

    def _recent_targets(self, k):
        """第 k 窗及之前 history_windows 窗内全部目标事件所在像素（布尔图）。"""
        mask = np.zeros((self.height, self.width), dtype=bool)
        for j in range(k - self.history_windows, k + 1):
            if j in self.targets:
                ys, xs, _ = self.targets[j]
                mask[np.clip(ys, 0, self.height - 1), np.clip(xs, 0, self.width - 1)] = True
        return mask

    def _diagnose(self, theta, k, false_mask):
        """虚警连通域的分档：到近期目标的距离、是否重复位置、是否新的虚警事件。"""
        labels, n = _label(false_mask)
        if n == 0:
            return
        dist = _chebyshev_distance(self._recent_targets(k), (self.height, self.width))
        last = self.last_false[theta]
        seen_recently = _dilate(last >= k - self.history_windows, 2)
        seen_just_now = _dilate(last >= k - 2, self.dedup_radius)
        counts = self.distance_counts[theta]
        for c in range(1, n + 1):
            comp = labels == c
            if dist is None:
                counts[-1] += 1                                   # 近期没有目标
            else:
                d = float(dist[comp].min())
                slot = next((i for i, edge in enumerate(DISTANCE_BINS) if d <= edge), len(DISTANCE_BINS))
                counts[slot] += 1
            if (comp & seen_recently).any():
                self.repeat_components[theta] += 1
            if not (comp & seen_just_now).any():
                self.false_events[theta] += 1
        last[false_mask] = int(k)

    def update(self, theta, k, alarm):
        """alarm: numpy 布尔 [height, width]，第 k 窗、阈值 theta 的告警图。"""
        if not alarm.any():
            return
        self.alarm_pixels[theta] += int(alarm.sum())
        if 0 <= k < self.pure.shape[0] and self.pure[k]:
            n_comp = _label_count(alarm)
            self.pure_alarm_pixels[theta] += int(alarm.sum())
            self.pure_false_components[theta] += n_comp
            part = self.pure_split["start" if self.pure_start[k] else "after"]
            part["pixels"][theta] += int(alarm.sum())
            part["components"][theta] += n_comp
        parts = [self.targets[j] for j in (k - 1, k) if j in self.targets]
        if not parts:
            self.false_components[theta] += _label_count(alarm)
            self._diagnose(theta, k, alarm)
            return
        ys = np.concatenate([p[0] for p in parts])
        xs = np.concatenate([p[1] for p in parts])
        ids = np.concatenate([p[2] for p in parts])
        false_mask = alarm & ~self._near(ys, xs)
        self.false_components[theta] += _label_count(false_mask)
        self._diagnose(theta, k, false_mask)
        first = self.first_alarm[theta]
        for tid in np.unique(ids):
            tid = float(tid)
            if tid not in first and (alarm & self._near(ys[ids == tid], xs[ids == tid])).any():
                first[tid] = int(k)

    def end_sequence(self):
        """把本序列每个目标的首次告警延迟记下来（未告警记 None）。"""
        for theta in self.thetas:
            for tid, t0 in self.t_first.items():
                k = self.first_alarm[theta].get(tid)
                latency = None if k is None else (k + 1) * self.window_ms - t0
                self.records[theta].append(latency)

    def summary(self):
        """{阈值: 统计字典}。"""
        out = {}
        pixels = float(self.height * self.width)
        hours = self.windows * self.window_ms / 3.6e6
        mean_length = self.windows / float(max(self.sequences, 1))
        names = ["<=%d" % edge for edge in DISTANCE_BINS] + [">%d" % DISTANCE_BINS[-1], "no_target_recently"]
        for theta in self.thetas:
            lat = np.array([v for v in self.records[theta] if v is not None], dtype=np.float64)
            n = len(self.records[theta])
            tight = (1.0 + 1.0 / max(mean_length, 1.0)) / (np.exp(theta) - 1.0)
            pure_cells = self.pure_window_count * pixels
            row = {"n_targets": n, "n_detected": int(lat.size),
                   "detection_rate": float(lat.size) / n if n else float("nan"),
                   "false_components": int(self.false_components[theta]), "alarm_pixels": int(self.alarm_pixels[theta]),
                   "false_alarm_rate": self.false_components[theta] / max(self.windows * pixels, 1.0),
                   "bound_per_location_window": self.hypotheses / (np.exp(theta) - 1.0),
                   "tight_bound_per_location_window": float(tight),
                   "false_components_per_hour": self.false_components[theta] / hours if hours > 0 else float("nan"),
                   "false_events": int(self.false_events[theta]),
                   "false_events_per_hour": self.false_events[theta] / hours if hours > 0 else float("nan"),
                   "false_repeat_components": int(self.repeat_components[theta]),
                   "false_distance_to_recent_target": dict(zip(names, [int(c) for c in self.distance_counts[theta]])),
                   "pure_windows": int(self.pure_window_count),
                   "pure_alarm_pixels": int(self.pure_alarm_pixels[theta]),
                   "pure_false_components": int(self.pure_false_components[theta]),
                   "pure_alarm_location_rate": (self.pure_alarm_pixels[theta] / pure_cells) if pure_cells else None,
                   "pure_expected_alarms_at_bound": float(tight * pure_cells),
                   "pure_violation_p": poisson_upper_p(self.pure_false_components[theta], tight * pure_cells)}
            for name, part in self.pure_split.items():
                cells = part["windows"] * pixels
                row.update({"pure_%s_windows" % name: int(part["windows"]),
                            "pure_%s_alarm_pixels" % name: int(part["pixels"][theta]),
                            "pure_%s_false_components" % name: int(part["components"][theta]),
                            "pure_%s_alarm_location_rate" % name: (part["pixels"][theta] / cells) if cells else None,
                            "pure_%s_expected_alarms_at_bound" % name: float(tight * cells),
                            "pure_%s_violation_p" % name: poisson_upper_p(part["components"][theta], tight * cells)})
            if lat.size:
                row.update({"latency_mean_ms": float(lat.mean()), "latency_median_ms": float(np.median(lat)),
                            "latency_p90_ms": float(np.percentile(lat, 90))})
            out[str(theta)] = row
        return out
