"""位置级告警（CUSUM 的 M >= theta，带复位）的评估。与原 utils/eval.py 的逐事件 Pd/Fa 是不同的随机量，单独报告。

对每个告警阈值累计：
    目标级    每个目标第一次"真告警"所在窗 k -> 首次告警延迟 = (k+1)*窗长 - 该目标第一个事件的时间；
              检出率 = 至少有一次真告警的目标占比。
              "真告警"：告警像素与该目标在本窗或上一窗的某个事件的切比雪夫距离 <= radius。
    虚警      告警像素中离所有目标事件（本窗与上一窗）都超过 radius 的像素，按 8 连通域计数；
              虚警率 = 连通域数 / (窗数 × 画面像素数)，可与理论上界 V/(e^theta - 1)（每位置每窗）对照（后者是上界，
              连通域数又不多于告警位置数，所以实测值应远低于它）。
只依赖 numpy；膨胀与连通域优先用 scipy，没有时用 cv2。
"""
import numpy as np


def _label_count(mask):
    """8 连通域个数。"""
    try:
        from scipy import ndimage
        return int(ndimage.label(mask, structure=np.ones((3, 3), dtype=int))[1])
    except ImportError:
        import cv2
        return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]) - 1


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


class AlarmEvaluator(object):
    """用法：每条序列 begin_sequence(seq) -> 每窗每个阈值 update(theta, k, alarm) -> end_sequence()；最后 summary()。

    参数: thetas 告警阈值列表；height/width 画面大小（告警图需裁到这个大小）；window_ms 窗长；radius 真告警的容差（像素）；
          hypotheses 速度假设数 V（只用于写出理论上界）。
    """

    def __init__(self, thetas, height, width, window_ms, radius, hypotheses):
        self.thetas = [float(t) for t in thetas]
        self.height, self.width = int(height), int(width)
        self.window_ms, self.radius = float(window_ms), int(radius)
        self.hypotheses = int(hypotheses)
        self.records = {t: [] for t in self.thetas}
        self.false_components = {t: 0 for t in self.thetas}
        self.alarm_pixels = {t: 0 for t in self.thetas}
        self.windows = 0

    def begin_sequence(self, seq, n_windows):
        """准备一条序列：每窗的目标事件坐标与编号、每个目标的首个事件时间。"""
        pos = (seq.label == 1) & (seq.target_id != 0)
        window = seq.t // int(self.window_ms)
        self.targets = {}
        for k in np.unique(window[pos]):
            sel = pos & (window == k)
            self.targets[int(k)] = (seq.y[sel].astype(np.int64), seq.x[sel].astype(np.int64), seq.target_id[sel])
        self.t_first = {float(tid): float(seq.t[pos & (seq.target_id == tid)].min())
                        for tid in np.unique(seq.target_id[pos])}
        self.first_alarm = {t: {} for t in self.thetas}
        self.windows += int(n_windows)

    def _near(self, ys, xs):
        """与给定事件坐标切比雪夫距离 <= radius 的像素区域。"""
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[np.clip(ys, 0, self.height - 1), np.clip(xs, 0, self.width - 1)] = True
        return _dilate(mask, self.radius)

    def update(self, theta, k, alarm):
        """alarm: numpy 布尔 [height, width]，第 k 窗、阈值 theta 的告警图。"""
        if not alarm.any():
            return
        self.alarm_pixels[theta] += int(alarm.sum())
        parts = [self.targets[j] for j in (k - 1, k) if j in self.targets]
        if not parts:
            self.false_components[theta] += _label_count(alarm)
            return
        ys = np.concatenate([p[0] for p in parts])
        xs = np.concatenate([p[1] for p in parts])
        ids = np.concatenate([p[2] for p in parts])
        self.false_components[theta] += _label_count(alarm & ~self._near(ys, xs))
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
        for theta in self.thetas:
            lat = np.array([v for v in self.records[theta] if v is not None], dtype=np.float64)
            n = len(self.records[theta])
            row = {"n_targets": n, "n_detected": int(lat.size),
                   "detection_rate": float(lat.size) / n if n else float("nan"),
                   "false_components": int(self.false_components[theta]), "alarm_pixels": int(self.alarm_pixels[theta]),
                   "false_alarm_rate": self.false_components[theta] / max(self.windows * pixels, 1.0),
                   "bound_per_location_window": self.hypotheses / (np.exp(theta) - 1.0)}
            if lat.size:
                row.update({"latency_mean_ms": float(lat.mean()), "latency_median_ms": float(np.median(lat)),
                            "latency_p90_ms": float(np.percentile(lat, 90))})
            out[str(theta)] = row
        return out
