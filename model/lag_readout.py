"""V6 第一版：滞后读出头（自定时判决神经元的证据部分，Stream-SNN-Innovations.md 13.10~13.12）。

对象：第 k 窗出生的事件，在第 k+d 窗（d = 1..D）再读一次网络状态，给出"它是不是目标"的新分数。
    logit_d = z0 + head_d([此刻的状态特征, 出生窗的特征, z0, p, t_local])
    出生窗的特征 = dec1 在事件原像素的 M；骨干带背景率通道时（V6b，birth_extra = 4）再接上事件像素在出生窗的背景率：
    背景越密，未来窗里"碰巧"出现的支持越多，后续证据要按背景打折，只有读到背景率的头才能做到
    z0      V5 原读出头在出生窗给出的 logit（d = 0 就是它，不重算、不改变）
    head_d  每个滞后一个两层 MLP；最后一层零初始化，所以训练开始时 logit_d = z0（不比零等待差）
此刻的状态特征（variant）：
    pixel       只读 dec1 在事件原像素的 M（"静止假设"对照，13.7 的 E1）
    multiscale  再读 dec2 / dec3 / enc4 在事件所在粗格的 M，以及 enc4 的 3x3 邻域均值（结构对齐，13.7 的 E2）
骨干不改、不回传梯度：本模块只读各层的 M（BIR 的状态，forward / forward_chunk 的 u_pre）。

LagBuffer 保存尚未读满 D 个滞后的事件，训练（多条并行流、按片段）与评估（单序列、逐窗）共用同一套记账：
    g     事件出生窗在流时间轴上的序号
    end   事件所在录像在流时间轴上的结束窗号（开区间）；g + d >= end 的滞后不存在（录像已结束）
    第 d 个滞后在流时间轴上的第 g + d 窗读取，每个 (事件, d) 恰好处理一次。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evspsegnet_stream import LAYER_NAMES

LAG_VARIANTS = ("pixel", "multiscale")
# 读哪些层的 M：名字 -> LAYER_NAMES 中的序号与相对 dec1 的下采样倍数
MAP_LAYERS = {"dec1": (6, 1), "dec2": (5, 2), "dec3": (4, 4), "enc4": (3, 8)}
if any(LAYER_NAMES[index] != name for name, (index, _) in MAP_LAYERS.items()):
    raise RuntimeError("MAP_LAYERS 与网络的 LAYER_NAMES 顺序不一致: %s" % (LAYER_NAMES,))
BUFFER_FIELDS = ("g", "end", "b", "y", "x", "p", "t_local", "z0", "birth", "label", "key")


def variant_layers(variant):
    """该 variant 需要读取的层名。"""
    if variant not in LAG_VARIANTS:
        raise ValueError("lag variant 必须是 %s 之一，收到 %r" % (LAG_VARIANTS, variant))
    return ("dec1",) if variant == "pixel" else ("dec1", "dec2", "dec3", "enc4")


def lag_maps(u_pre, variants):
    """把各层的 M 整理成按 (窗, 流) 展平的特征图字典。

    输入: u_pre  7 个张量，逐窗 forward 时为 [B,C,h,w]，forward_chunk 时为 [T,B,C,h,w]
          variants  需要的 variant 列表
    输出: {层名: [T*B, C, h, w]}；multiscale 另有 "enc4_mean"（3x3 邻域均值，边界只平均有效格）。
          展平序号 = 窗号 * B + 流号，与 forward_chunk 读出头的约定相同。
    """
    names = sorted({name for v in variants for name in variant_layers(v)})
    maps = {}
    for name in names:
        tensor = u_pre[MAP_LAYERS[name][0]]
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        maps[name] = tensor.reshape((-1,) + tuple(tensor.shape[2:]))
    if "multiscale" in variants:
        maps["enc4_mean"] = F.avg_pool2d(maps["enc4"], 3, stride=1, padding=1, count_include_pad=False)
    return maps


def gather_map(feature_map, flat, y, x, stride):
    """在 [N_maps, C, h, w] 的第 flat 张图上取 (y//stride, x//stride) 处的 C 维特征，返回 [n, C]。"""
    h, w = int(feature_map.shape[2]), int(feature_map.shape[3])
    yy = torch.clamp(torch.div(y, stride, rounding_mode="floor"), max=h - 1)
    xx = torch.clamp(torch.div(x, stride, rounding_mode="floor"), max=w - 1)
    return feature_map[flat, :, yy, xx]


class LagReadout(nn.Module):
    """D 个滞后读出头（一个 variant）。

    channels  骨干的 (c1, c2, c3, c4)；lags = D；hidden 为 MLP 隐层宽度（与 V5 读出头相同取 32）；
    birth_extra 出生窗特征里 dec1 之外的维数（骨干读出头读的背景率通道数，没有为 0）。
    forward(d, now, birth, z0, p, t_local) -> logit_d [n]，d = 1..D；birth 为 [n, c1 + birth_extra]。
    """

    def __init__(self, channels, lags=5, variant="pixel", hidden=32, birth_extra=0):
        super(LagReadout, self).__init__()
        variant_layers(variant)
        if int(lags) < 1:
            raise ValueError("lags 必须 >= 1")
        if isinstance(birth_extra, bool) or int(birth_extra) != birth_extra or int(birth_extra) < 0:
            raise ValueError("birth_extra 必须是非负整数")
        c1, c2, c3, c4 = (int(c) for c in channels)
        self.variant, self.lags, self.c1, self.birth_extra = variant, int(lags), c1, int(birth_extra)
        self.now_dim = c1 if variant == "pixel" else c1 + c2 + c3 + 2 * c4
        self.birth_dim = c1 + self.birth_extra
        in_dim = self.now_dim + self.birth_dim + 3
        self.heads = nn.ModuleList()
        for _ in range(self.lags):
            head = nn.Sequential(nn.Linear(in_dim, int(hidden)), nn.ReLU(), nn.Linear(int(hidden), 1))
            nn.init.zeros_(head[2].weight)
            nn.init.zeros_(head[2].bias)
            self.heads.append(head)

    def now_features(self, maps, flat, y, x):
        """此刻（判决窗）在事件位置的状态特征 [n, now_dim]。"""
        parts = [gather_map(maps["dec1"], flat, y, x, 1)]
        if self.variant == "multiscale":
            for name in ("dec2", "dec3", "enc4"):
                parts.append(gather_map(maps[name], flat, y, x, MAP_LAYERS[name][1]))
            parts.append(gather_map(maps["enc4_mean"], flat, y, x, MAP_LAYERS["enc4"][1]))
        return torch.cat(parts, 1)

    def forward(self, d, now, birth, z0, p, t_local):
        if not 1 <= int(d) <= self.lags:
            raise ValueError("滞后 d 必须在 1..%d 内" % self.lags)
        inp = torch.cat([now, birth, z0.unsqueeze(1), p.unsqueeze(1), t_local.unsqueeze(1)], 1)
        return z0 + self.heads[int(d) - 1](inp).squeeze(1)


def birth_features(maps, flat, y, x, extra=None):
    """出生窗 dec1 在事件原像素的 M [n, c1]（与骨干读出头读到的特征相同）；extra 为 [n, e] 时接在后面（背景率）。"""
    feat = gather_map(maps["dec1"], flat, y, x, 1)
    return feat if extra is None else torch.cat([feat, extra.to(feat.dtype)], 1)


class LagBuffer(object):
    """尚未读满 D 个滞后的事件（字段见 BUFFER_FIELDS；全部是同一设备上的张量）。"""

    def __init__(self, lags):
        self.lags = int(lags)
        self.data = None

    def __len__(self):
        return 0 if self.data is None else int(self.data["g"].shape[0])

    def add(self, **fields):
        if set(fields) != set(BUFFER_FIELDS):
            raise ValueError("LagBuffer.add 需要字段 %s" % (BUFFER_FIELDS,))
        if int(fields["g"].shape[0]) == 0:
            return
        if self.data is None:
            self.data = {k: v for k, v in fields.items()}
        else:
            self.data = {k: torch.cat([self.data[k], fields[k]]) for k in BUFFER_FIELDS}

    def due(self, d, start, end):
        """第 d 个滞后落在流时间 [start, end) 内、且录像尚未结束的条目（布尔掩码）。"""
        if self.data is None:
            return None
        decision = self.data["g"] + int(d)
        return (decision >= int(start)) & (decision < int(end)) & (decision < self.data["end"])

    def prune(self, chunk_end):
        """丢掉之后不再有任何滞后要读的条目：保留 min(g + D, end - 1) >= chunk_end。"""
        if self.data is None:
            return
        last = torch.minimum(self.data["g"] + self.lags, self.data["end"] - 1)
        keep = last >= int(chunk_end)
        self.data = {k: v[keep] for k, v in self.data.items()}

    def select(self, mask):
        return {k: v[mask] for k, v in self.data.items()}


def lag_logits(readouts, buffer, maps, start, steps, batch):
    """对缓存中第 d 个滞后落在 [start, start+steps) 的条目，计算每个 variant 每个 d 的 logit。

    输入: readouts {variant: LagReadout}；maps 为 lag_maps(u_pre, 全部 variant) 的结果（同一片段，各 variant 共用）；
          batch = 流数 B
    输出: 列表 [(variant, d, 条目字典, logit [n])]（只含非空的组）
    """
    out = []
    if buffer.data is None:
        return out
    lags = max(r.lags for r in readouts.values())
    for d in range(1, lags + 1):
        mask = buffer.due(d, start, start + steps)
        if mask is None or not bool(mask.any()):
            continue
        rows = buffer.select(mask)
        flat = (rows["g"] + d - int(start)) * int(batch) + rows["b"]
        for variant, readout in readouts.items():
            if d > readout.lags:
                continue
            now = readout.now_features(maps, flat, rows["y"], rows["x"])
            logit = readout(d, now, rows["birth"], rows["z0"], rows["p"], rows["t_local"])
            out.append((variant, d, rows, logit))
    return out
