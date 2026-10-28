"""E1 对照：训练时注入合成背景噪声（Inn 13.46 之后的"最简单替代解释"检验）。

只作用于训练录像（在读入之后、切窗之前），验证 / 测试保持原样。
噪声模型：每像素每窗的齐次泊松背景，ON / OFF 各一半，标签 0、target_id 0。
每条录像以概率 prob 加噪；加噪时总速率 r（每像素每窗，两种极性合计）在 [rate_min, rate_max] 上按对数均匀抽样。
默认常数的来源：
    rate_min = 0.002  训练集逐序列背景率的中位数（Inn 13.45：ON 0.0021）
    rate_max = 0.04   = 1/25，5x5 邻域每窗随机巧合负荷等于 1 的阈值（Inn 13.37 的物理分界）
prob = 0 时完全不调用随机数，训练与原来逐位相同。
"""
import numpy as np

from dataset.stream_windows import StreamSequence, split_windows, time_bin_and_local

NOISE_AUG_DEFAULTS = {"noise_aug_prob": 0.0, "noise_aug_rate_min": 0.002, "noise_aug_rate_max": 0.04}


def noise_aug_settings(cfg):
    """从展平配置读出 (prob, rate_min, rate_max)，并检查取值。"""
    prob = float(cfg.get("noise_aug_prob", NOISE_AUG_DEFAULTS["noise_aug_prob"]))
    lo = float(cfg.get("noise_aug_rate_min", NOISE_AUG_DEFAULTS["noise_aug_rate_min"]))
    hi = float(cfg.get("noise_aug_rate_max", NOISE_AUG_DEFAULTS["noise_aug_rate_max"]))
    if not 0.0 <= prob <= 1.0 or not 0.0 < lo <= hi:
        raise ValueError("noise_aug 取值不合法: prob=%s rate=[%s, %s]" % (prob, lo, hi))
    return prob, lo, hi


def inject_background_noise(seq, rate, height, width, window_ms, n_windows, time_bins, rng):
    """返回加了齐次泊松背景噪声的新 StreamSequence（原序列不变）。

    rate 是每像素每窗的总事件率（ON、OFF 各 rate/2）；坐标、毫秒时间在有效画面和录像时长内均匀分布。
    新事件接在原事件之后，原事件的下标不变（预测回填顺序不受影响）。
    """
    total_ms = int(window_ms) * int(n_windows)
    count = int(rng.poisson(float(rate) * height * width * n_windows))
    if count == 0:
        return seq
    x = rng.randint(0, width, size=count).astype(np.int64)
    y = rng.randint(0, height, size=count).astype(np.int64)
    t = rng.randint(0, total_ms, size=count).astype(np.int64)
    p = rng.randint(0, 2, size=count).astype(np.int8)
    x_all, y_all, t_all = (np.concatenate([a, b]) for a, b in ((seq.x, x), (seq.y, y), (seq.t, t)))
    order, bounds = split_windows(t_all, window_ms, n_windows)
    inner_bin, t_local = time_bin_and_local(t_all, window_ms, time_bins)
    return StreamSequence(
        name=seq.name, x=x_all, y=y_all, t=t_all,
        p=np.concatenate([seq.p, p]),
        label=np.concatenate([seq.label, np.zeros(count, dtype=np.float32)]),
        target_id=np.concatenate([seq.target_id, np.zeros(count, dtype=np.float64)]),
        inner_bin=inner_bin, t_local=t_local, order=order, bounds=bounds)


class NoiseAugment(object):
    """按配置给训练录像加噪；每次读入录像调用一次。stats 记录本轮加噪的录像数与速率。"""

    def __init__(self, cfg, rng):
        self.prob, self.lo, self.hi = noise_aug_settings(cfg)
        self.cfg, self.rng = cfg, rng
        self.stats = {"loaded": 0, "noised": 0, "rates": []}

    @property
    def enabled(self):
        return self.prob > 0.0

    def __call__(self, seq):
        self.stats["loaded"] += 1
        if not self.enabled or self.rng.rand() >= self.prob:
            return seq
        rate = float(np.exp(self.rng.uniform(np.log(self.lo), np.log(self.hi))))
        self.stats["noised"] += 1
        self.stats["rates"].append(rate)
        c = self.cfg
        n_windows = int(seq.bounds.shape[0]) - 1
        return inject_background_noise(seq, rate, int(c["height"]), int(c["width_px"]), int(c["window_ms"]),
                                       n_windows, int(c["time_bins"]), self.rng)

    def summary(self):
        rates = self.stats["rates"]
        return {"loaded": self.stats["loaded"], "noised": self.stats["noised"],
                "rate_median": float(np.median(rates)) if rates else None}
