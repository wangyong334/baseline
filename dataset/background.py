"""V6b 事件表示层：每个事件带上所在像素的背景率（Stream-SNN-Innovations.md 13.15）。

问题（登记表 #3、#4①）：k、θ 与卷积权重都是全局的。同一组门限既要用于每像素每窗约 0.002 个背景事件的普通序列，
也要用于约 0.02 个的极密序列（EV-UAV 的传感器背景噪声），还要用于热像素、闪烁像素；网络没有逐像素调门限的输入。
做法：原 12 个计数通道不变，另加 4 个背景率通道（逐像素、区域 × 正、负极性），式 (B1)~(B5)：
    (B1) 时间保护区  第 k 窗的估计只用第 0..k-1-G 窗（排除最近 G 窗，与 CFAR 的保护单元同理）：正在经过的目标不计入自己的背景
    (B2) 逐像素  mu_pix = (S + n0 * mu_ref) / (W + n0)，S <- beta * S + C，W <- beta * W + 1（C 为一窗的计数，按 B1 延迟 G 窗进入）
    (B3) 区域    p_reg  = (R + n0 * mu_ref) / (W + n0)，R = O 在 (2r+1) x (2r+1) 个 cell x cell 粗格内对真实像素的平均，
                 O <- beta * O + 1[C > 0]：区域通道是"附近一个像素在一窗内发放的概率"（占有率），即空间巧合在零假设下的概率；
                 用占有率而不是计数，一个热像素（每窗几十个事件）只按 1 计，不会把整个区域抬成"密集背景"
    (B4) 编码    B = log1p(mu / mu_ref) / log1p(1 / mu_ref)：mu = mu_ref 时 B = log2 / log1p(1/mu_ref)；
                 逐像素通道每窗 1 个事件时 B = 1，区域通道每个像素每窗都发放时 B = 1
    (B5) 门控    B 只在本窗有事件（任一极性）的像素上非零：背景率是事件的属性。输入的稀疏度与原 12 通道相同，enc1 仍按事件驱动计算，
                 没有事件的像素输入严格为 0（BIR 的精确静息态不变）
只用过去：第 k 窗的 B 只依赖第 0..k-1-G 窗的计数（因果）；本窗计数只用于门控。没有可学习参数，常数都按规则导出：
    mu_ref   训练集背景事件（标签 0）每像素、每窗、每极性的平均数 = train_neg / (2 * 训练窗数 * height * width_px)
    n0       先验折合的窗数，取 1：第 0 窗也有定义的最弱先验
    G        10 窗（0.5 s）：盖住相对速度 >= 0.1 身长/窗的目标在一个像素上的停留（EV-UAV 中位 0.4 身长/窗、停留约 2.5 窗，
             EV-Flying 约 1 身长/窗）。不设保护区时，中位目标两三窗就把自己像素的估计抬到极密噪声的水平（13.16）
    tau      5 s（100 窗）：远长于 G 与目标停留；又短于一段 8 s 录像
    cell, r  8 像素（网络最粗层 enc4 的步长）、r = 2：区域边长 40 像素，覆盖 dec1 读出的感受野（冲激测试 38 像素），远大于目标
为什么不按背景归一化（V2 的 count / mu0）：泊松检验的充分统计量是 (计数, 背景率) 这一对，比值不是；比值在稀疏处会把孤立噪声放大。
这里两者都给，对数域里 w * log(C) - v * log(mu) 既能表示"除以背景"（v = w），也能表示"不看背景"（v = 0），由网络学。
状态与清零：背景状态属于一段录像的一次连续处理，与网络状态同步清零（多流训练中某条流开始新录像、batchify 切开的后半截）。
BackgroundTrack 按"下一个要处理的窗号"缓存状态，所以同一段录像切开的几截可以在不同的流里并行处理。
注意：
    - 评估的 reset_each_window 与 --reset-every 只清网络状态，不清背景状态（背景属于事件表示，不是网络的记忆）；
      所以带背景的模型的"清零后 IoU"不再是"完全没有历史"，与 V5 的同名数字不可直接比。
    - 多流训练里，若同一录像的两截同时在跑、且其中一截先结束而释放了来源（只会出现在 batchify 且录像长于一条流时，
      如 EBSSA），另一截下一次取窗会报"背景状态不连续"。届时让 StreamSources 按截而不是按录像管理来源。
"""
import math

import torch
import torch.nn.functional as F

BACKGROUND_DEFAULTS = {"tau_ms": 5000.0, "prior_windows": 1.0, "guard_windows": 10, "cell_px": 8, "region_cells": 5}
BACKGROUND_CHANNELS = 4
BACKGROUND_NAMES = ("bg_pixel_pos", "bg_pixel_neg", "bg_region_pos", "bg_region_neg")


def background_settings(value):
    """校验 DATA.background：None / false 为关闭（返回 None）；字典为开启（补全默认值，键见 BACKGROUND_DEFAULTS）。"""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("background 必须是 null / false 或配置字典，收到 %r" % (value,))
    unknown = set(value) - set(BACKGROUND_DEFAULTS)
    if unknown:
        raise ValueError("未知 background 配置: %s" % sorted(unknown))
    out = dict(BACKGROUND_DEFAULTS)
    out.update(value)
    tau, n0 = float(out["tau_ms"]), float(out["prior_windows"])
    if not (math.isfinite(tau) and tau > 0 and math.isfinite(n0) and n0 > 0):
        raise ValueError("background.tau_ms 与 background.prior_windows 必须是有限正数")
    for key, low in (("cell_px", 1), ("region_cells", 1), ("guard_windows", 0)):
        if isinstance(out[key], bool) or int(out[key]) != out[key] or int(out[key]) < low:
            raise ValueError("background.%s 必须是 >= %d 的整数" % (key, low))
    if int(out["region_cells"]) % 2 != 1:
        raise ValueError("background.region_cells 必须是奇数（区域以事件所在的粗格为中心）")
    return {"tau_ms": tau, "prior_windows": n0, "guard_windows": int(out["guard_windows"]),
            "cell_px": int(out["cell_px"]), "region_cells": int(out["region_cells"])}


def background_enabled(cfg):
    """配置是否开启背景率通道。"""
    return background_settings(cfg.get("background")) is not None


def input_channel_count(cfg):
    """网络输入通道数：12 个计数通道（2 + 2 * time_bins），开启背景时再加 4 个。"""
    return 2 + 2 * int(cfg["time_bins"]) + (BACKGROUND_CHANNELS if background_enabled(cfg) else 0)


def reference_rate(stats, cfg):
    """mu_ref = 训练集背景事件数 / (2 极性 * 训练窗数 * height * width_px)；只用训练集统计（tools/stream_train_stats.py）。"""
    try:
        windows = float(stats["audit"]["train"]["windows"])
    except (KeyError, TypeError):
        raise ValueError("统计文件缺少 audit.train.windows（训练集窗数），无法导出背景先验 mu_ref")
    negatives = float(stats["train_neg"])
    if windows <= 0 or negatives <= 0:
        raise ValueError("训练集窗数与背景事件数必须为正")
    return negatives / (2.0 * windows * float(cfg["height"]) * float(cfg["width_px"]))


def resolve_background(cfg, stats):
    """开启背景时把 mu_ref 写进 cfg["background_mu_ref"]（随 checkpoint 的配置一起保存）；已有的值必须与统计一致。

    返回 mu_ref；关闭时返回 None，且配置里不得残留 background_mu_ref。
    """
    if not background_enabled(cfg):
        if cfg.get("background_mu_ref") is not None:
            raise ValueError("background 关闭，但配置里有 background_mu_ref")
        return None
    mu = reference_rate(stats, cfg)
    old = cfg.get("background_mu_ref")
    if old is not None and abs(float(old) - mu) > 1e-9 * mu:
        raise ValueError("配置中的 background_mu_ref %r 与统计文件导出的 %r 不一致" % (old, mu))
    cfg["background_mu_ref"] = mu
    return mu


class BackgroundEncoder(object):
    """式 (B1)~(B5) 的逐窗计算（无可学习参数，不参与反传）。

    状态 = (pending, S, O, W)：pending 为还在保护区内的最近 G 窗计数（元组，最旧的在前），S / O 为已移出保护区的
    计数 / 占有率的指数和 [2,Hp,Wp] float32，W 为对应的权重和（浮点数）。fresh() 为一段录像开始时的状态（只有先验）。
    step(counts, state) -> (B [4,Hp,Wp], 新状态)：counts 为本窗正 / 负极性整窗计数 [2,Hp,Wp]（float32，与输入通道 0 / 1 同序）；
    B 的通道顺序见 BACKGROUND_NAMES，只由 S、O、W（第 k-1-G 窗及以前）算出，本窗计数只用于门控和进入保护区。
    """

    def __init__(self, settings, mu_ref, window_ms, height, width, pad_height, pad_width, device):
        self.n0 = float(settings["prior_windows"])
        self.guard = int(settings["guard_windows"])
        self.beta = math.exp(-float(window_ms) / float(settings["tau_ms"]))
        self.mu_ref = float(mu_ref)
        if not (math.isfinite(self.mu_ref) and self.mu_ref > 0):
            raise ValueError("mu_ref 必须是有限正数")
        self.unit = math.log1p(1.0 / self.mu_ref)
        self.cell, self.cells = int(settings["cell_px"]), int(settings["region_cells"])
        if int(pad_height) % self.cell or int(pad_width) % self.cell:
            raise ValueError("画布 %dx%d 不能被 cell_px=%d 整除" % (int(pad_height), int(pad_width), self.cell))
        if not (0 < int(height) <= int(pad_height) and 0 < int(width) <= int(pad_width)):
            raise ValueError("传感器尺寸必须在画布之内")
        self.device = device
        self.shape = (2, int(pad_height), int(pad_width))
        valid = torch.zeros((1, 1) + self.shape[1:], dtype=torch.float32, device=device)
        valid[..., :int(height), :int(width)] = 1.0
        self.region_count = self._region_mean(valid)[0]                      # [1,Hp,Wp]：区域内真实像素的占比
        if not bool((self.region_count > 0).all()):
            raise ValueError("有区域完全落在补零区内")

    def _region_mean(self, value):
        """[N,C,Hp,Wp] -> 每个像素所在区域（(2r+1)^2 个粗格）的平均，按粗格展开回原分辨率。"""
        coarse = F.avg_pool2d(value, self.cell, stride=self.cell)
        coarse = F.avg_pool2d(coarse, self.cells, stride=1, padding=self.cells // 2, count_include_pad=False)
        return coarse.repeat_interleave(self.cell, dim=2).repeat_interleave(self.cell, dim=3)

    def fresh(self):
        zeros = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        return (), zeros, zeros.clone(), 0.0

    def rates(self, state):
        """状态对应的 (mu_pix, p_reg)，各 [2,Hp,Wp]：逐像素每窗的期望背景事件数（式 B2）、区域内一个像素每窗发放的概率（式 B3）。"""
        _, s, o, w = state
        prior, denom = self.n0 * self.mu_ref, w + self.n0
        region = self._region_mean(o.unsqueeze(0))[0] / self.region_count
        return (s + prior) / denom, (region + prior) / denom

    def step(self, counts, state):
        if tuple(counts.shape) != self.shape:
            raise ValueError("counts 应为 %s，收到 %s" % (self.shape, tuple(counts.shape)))
        pixel, region = self.rates(state)
        code = torch.log1p(torch.cat([pixel, region], 0) / self.mu_ref) / self.unit
        gate = (counts.sum(0, keepdim=True) > 0).to(code.dtype)
        pending, s, o, w = state
        pending = pending + (counts.clone(),)                 # 拷贝：不让缓存拖住整段片段的计数张量
        if len(pending) > self.guard:                         # 最旧的一窗移出保护区，进入估计
            old, pending = pending[0], pending[1:]
            s = self.beta * s + old
            o = self.beta * o + (old > 0).to(old.dtype)
            w = self.beta * w + 1.0
        return code * gate, (pending, s, o, w)


class BackgroundTrack(object):
    """一段录像的背景状态：按"下一个要处理的窗号"缓存。

    take(start, fresh)：fresh=True 或 start=0 时从先验开始（录像开头、或与网络状态一起清零的新起点）；
    否则必须已经处理过第 start-1 窗，否则报错（背景通道只能按时间顺序构造，不能跳窗）。
    put(end, state)：处理完第 end-1 窗后的状态。
    """

    def __init__(self, encoder):
        self.encoder, self.cache = encoder, {}

    def take(self, start, fresh):
        start = int(start)
        if fresh or start == 0:
            self.cache.pop(start, None)
            return self.encoder.fresh()
        if start not in self.cache:
            raise RuntimeError("背景状态不连续：第 %d 窗之前的窗还没有处理（只能按时间顺序构造，或用 fresh=True 从先验开始）"
                               % start)
        return self.cache.pop(start)

    def put(self, end, state):
        self.cache[int(end)] = state

    def encode(self, counts, start, fresh=False):
        """counts [T,2,Hp,Wp]（第 start..start+T-1 窗）-> B [T,4,Hp,Wp]，并缓存处理完后的状态。"""
        state = self.take(start, fresh)
        out = []
        for t in range(int(counts.shape[0])):
            code, state = self.encoder.step(counts[t], state)
            out.append(code)
        self.put(int(start) + int(counts.shape[0]), state)
        return torch.stack(out)


def make_background_track(cfg, device):
    """配置开启背景时为一段录像构造 BackgroundTrack，否则返回 None。需要先 resolve_background(cfg, stats)。"""
    settings = background_settings(cfg.get("background"))
    if settings is None:
        return None
    if cfg.get("background_mu_ref") is None:
        raise ValueError("background 已开启但配置里没有 background_mu_ref（应先调用 resolve_background）")
    encoder = BackgroundEncoder(settings, cfg["background_mu_ref"], cfg["window_ms"], cfg["height"], cfg["width_px"],
                                cfg["pad_height"], cfg["pad_width"], device)
    return BackgroundTrack(encoder)
