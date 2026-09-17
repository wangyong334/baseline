"""流式 SNN V1 网络：二维脉冲 U-Net 骨干 + 逐事件连续读出头（纯 PyTorch，兼容 torch 1.9）。

结构（输入 [B,12,264,352]，3 次 stride=2 下采样）：
    enc1  Conv3x3 12->12 s1 -> 增益 -> LIF1      264x352
    enc2  Conv3x3 12->24 s2 -> 增益 -> LIF2      132x176
    enc3  Conv3x3 24->48 s2 -> 增益 -> LIF3       66x88
    enc4  Conv3x3 48->48 s2 -> 增益 -> LIF4       33x44
    up3   ConvT2x2 48->48 (S4)，与 S3 拼接 -> dec3 Conv3x3 96->48 -> 增益 -> LIF5   66x88
    up2   ConvT2x2 48->24 (S5)，与 S2 拼接 -> dec2 Conv3x3 48->24 -> 增益 -> LIF6  132x176
    up1   ConvT2x2 24->12 (S6)，与 S1 拼接 -> dec1 Conv3x3 24->12 -> 增益 -> LIF7  264x352
    读出  MLP([U_pre7(x,y), p, t_local]) -> 原始 logit（模型内不做 sigmoid）
所有卷积 bias=False；默认不使用 BN/GN（零输入 -> 零电流）。
原版的问题：ConvT 输出是实数，与跳连脉冲拼接后，解码卷积有一半输入是实数（只能按 MAC 计）。

merged_decoder=True（电流合并解码器）时，每个解码阶段写成两路突触电流之和：
    up3   ConvT4x4 s2 p1 48->48 (S4)  +  dec3 Conv3x3 48->48 (S3)  -> 增益 -> LIF5   66x88
    up2   ConvT4x4 s2 p1 48->24 (S5)  +  dec2 Conv3x3 24->24 (S2)  -> 增益 -> LIF6  132x176
    up1   ConvT4x4 s2 p1 24->12 (S6)  +  dec1 Conv3x3 12->12 (S1)  -> 增益 -> LIF7  264x352
所有突触运算的输入都是 0/1 脉冲，仍为 7 组状态、读出不变。原版 ConvT2x2 与解码卷积之间没有非线性，
Conv3x3(cat[ConvT2x2(S_deep), S_skip]) 可以精确合成上式，训练好的原版权重可无损换算，
见 merge_decoder_state_dict 与 tools/convert_to_merged_decoder.py。
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lif2d_stream import ChannelGain, ReLUNeuron, StreamingLIF2d
from utils.stream_common import gains_from_positive_samples

LAYER_NAMES = ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")
DECODER_STAGES = (("up3", "dec3"), ("up2", "dec2"), ("up1", "dec1"))


def make_norm(kind, channels):
    """归一化插槽：'none' 返回恒等映射；'groupnorm_noshift' 返回无仿射参数的 GroupNorm（逐窗、零输入仍为零）。"""
    if kind == "none":
        return nn.Identity()
    if kind == "groupnorm_noshift":
        return nn.GroupNorm(min(4, channels), channels, affine=False)
    raise ValueError("未知的 norm 类型: %s" % kind)


class SpikingConvBlock(nn.Module):
    """一个"卷积 -> 归一化插槽 -> 逐通道增益 -> 神经元"单元。

    forward(x, state, extra_current=None) -> (spikes, new_state, u_pre, pre_gain_current)
        extra_current 为另一路突触电流（电流合并解码器中来自 ConvT），在归一化之前与本层卷积相加；
        pre_gain_current 是乘增益之前的电流，只用于增益校准统计。
    """

    def __init__(self, in_ch, out_ch, stride, neuron, norm, lif_kwargs):
        super(SpikingConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = make_norm(norm, out_ch)
        self.gain = ChannelGain(out_ch)
        if neuron == "lif":
            self.neuron = StreamingLIF2d(out_ch, **lif_kwargs)
        elif neuron == "relu":
            self.neuron = ReLUNeuron()
        else:
            raise ValueError("未知的 neuron 类型: %s" % neuron)
        self.out_ch = out_ch

    def forward(self, x, state, extra_current=None):
        current = self.conv(x)
        if extra_current is not None:
            current = current + extra_current
        current = self.norm(current)
        spikes, new_state, u_pre = self.neuron(self.gain(current), state)
        return spikes, new_state, u_pre, current


class EvSpSegNetStream(nn.Module):
    """流式脉冲分割网络（V1）。

    forward(x, events, states, state_mode=None, collect=False) -> (logits, new_states, info)
        x          [B,12,H,W] 当前窗口输入
        events     字典：b/y/x 为 long [N]，p/t_local 为 float [N]（当前窗口的逐事件信息）
        states     上一窗口返回的 7 个膜电位列表，或 None
        state_mode 'carry'（使用 states）或 'reset_each_window'（忽略 states，等价 beta=0）；
                   None 时使用构造参数中的默认值
        collect    True 时 info 返回每层的脉冲、u_pre、增益前电流（用于监控与校准）
        logits     [N] 每个事件的原始 logit
        new_states 7 个软复位后的膜电位（ReLU 版本为 None 列表）
    """

    def __init__(self, in_channels=12, channels=(12, 24, 48, 48), neuron="lif", norm="none",
                 state_mode="carry", readout_hidden=32, dt_ms=50.0, tau_init_ms=200.0,
                 tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0, merged_decoder=False):
        super(EvSpSegNetStream, self).__init__()
        if state_mode not in ("carry", "reset_each_window"):
            raise ValueError("state_mode 必须是 carry 或 reset_each_window")
        if not isinstance(merged_decoder, bool):
            raise ValueError("merged_decoder 必须是 YAML 布尔值（true/false）")
        c1, c2, c3, c4 = channels
        lif = dict(dt_ms=dt_ms, tau_init_ms=tau_init_ms, tau_min_ms=tau_min_ms,
                   tau_max_ms=tau_max_ms, v_threshold=v_threshold)
        block = lambda i, o, s: SpikingConvBlock(i, o, s, neuron, norm, lif)  # noqa: E731
        self.enc1 = block(in_channels, c1, 1)
        self.enc2 = block(c1, c2, 2)
        self.enc3 = block(c2, c3, 2)
        self.enc4 = block(c3, c4, 2)
        if merged_decoder:
            # ConvT4x4/s2/p1 负责深层脉冲，解码卷积只接跳连脉冲；两路电流在神经元内相加
            upsample = lambda i, o: nn.ConvTranspose2d(i, o, 4, stride=2, padding=1, bias=False)  # noqa: E731
            self.up3, self.dec3 = upsample(c4, c3), block(c3, c3, 1)
            self.up2, self.dec2 = upsample(c3, c2), block(c2, c2, 1)
            self.up1, self.dec1 = upsample(c2, c1), block(c1, c1, 1)
        else:
            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2, bias=False)
            self.dec3 = block(c3 + c3, c3, 1)
            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2, bias=False)
            self.dec2 = block(c2 + c2, c2, 1)
            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2, bias=False)
            self.dec1 = block(c1 + c1, c1, 1)
        self.readout = nn.Sequential(nn.Linear(c1 + 2, readout_hidden), nn.ReLU(),
                                     nn.Linear(readout_hidden, 1))
        self.state_mode = state_mode
        self.neuron_kind = neuron
        self.merged_decoder = merged_decoder
        # 读出消融开关，默认关闭；只在评估阶段由命令行设置，不影响训练
        self.readout_ablation = "none"
        self.v_threshold = float(v_threshold)
        # 标记增益是否已校准；随 checkpoint 保存，评估时禁止重新校准
        self.register_buffer("gain_calibrated", torch.tensor(False))

    def blocks(self):
        """按前向拓扑顺序返回 7 个脉冲卷积单元（校准必须按这个顺序逐层进行）。"""
        return [getattr(self, name) for name in LAYER_NAMES]

    def forward(self, x, events, states, state_mode=None, collect=False):
        mode = self.state_mode if state_mode is None else state_mode
        if mode not in ("carry", "reset_each_window"):
            raise ValueError("state_mode 必须是 carry 或 reset_each_window")
        prev = states if (mode == "carry" and states is not None) else [None] * 7
        if len(prev) != 7:
            raise ValueError("states 应为 7 个膜电位，收到 %d 个" % len(prev))
        s1, n1, u1, i1 = self.enc1(x, prev[0])
        s2, n2, u2, i2 = self.enc2(s1, prev[1])
        s3, n3, u3, i3 = self.enc3(s2, prev[2])
        s4, n4, u4, i4 = self.enc4(s3, prev[3])
        if self.merged_decoder:
            # 两路突触电流在神经元内相加：跳连脉冲经解码卷积，深层脉冲经 ConvT（作为 extra_current）
            s5, n5, u5, i5 = self.dec3(s3, prev[4], self.up3(s4))
            s6, n6, u6, i6 = self.dec2(s2, prev[5], self.up2(s5))
            s7, n7, u7, i7 = self.dec1(s1, prev[6], self.up1(s6))
        else:
            s5, n5, u5, i5 = self.dec3(torch.cat([self.up3(s4), s3], 1), prev[4])
            s6, n6, u6, i6 = self.dec2(torch.cat([self.up2(s5), s2], 1), prev[5])
            s7, n7, u7, i7 = self.dec1(torch.cat([self.up1(s6), s1], 1), prev[6])
        # 高级索引 u7[b, :, y, x] 的结果形状为 [N, C]
        feat = u7[events["b"], :, events["y"], events["x"]]
        extra = torch.stack([events["p"].to(feat.dtype), events["t_local"].to(feat.dtype)], 1)
        # 读出消融（只用于评估，不重新训练）：用来判断预测到底依赖哪一路信息
        #   none          正常
        #   zero_extra    把 p 与 t_local 置零 -> 只剩网络特征；若性能不掉说明这两个标量没有泄漏
        #   zero_feature  把网络特征置零 -> 只剩 p 与 t_local；若性能仍高则说明标签可由它们直接推出
        if self.readout_ablation == "zero_extra":
            extra = torch.zeros_like(extra)
        elif self.readout_ablation == "zero_feature":
            feat = torch.zeros_like(feat)
        elif self.readout_ablation != "none":
            raise ValueError("未知的 readout_ablation: %s" % self.readout_ablation)
        logits = self.readout(torch.cat([feat, extra], 1)).squeeze(1)
        info = None
        if collect:
            info = {"spikes": [s1, s2, s3, s4, s5, s6, s7],
                    "u_pre": [u1, u2, u3, u4, u5, u6, u7],
                    "current": [i1, i2, i3, i4, i5, i6, i7]}
        return logits, [n1, n2, n3, n4, n5, n6, n7], info


def count_parameters(net):
    """返回参数量：总数与按顶层模块分组的明细（字典）。"""
    groups = {}
    for name, p in net.named_parameters():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
    return {"total": int(sum(groups.values())), "by_module": groups}


def merge_decoder_state_dict(state_dict):
    """把原版 V1 权重精确换算为电流合并解码器（merged_decoder=True）的权重，不改变网络函数。

    原版每个解码阶段：I = Conv3x3_p1( cat[ ConvT2x2_s2(S_deep), S_skip ] )
                        = Conv3x3_a( ConvT2x2_s2(S_deep) ) + Conv3x3_b(S_skip)
    第一项是两个线性、平移等变的运算串联，等于一个 ConvT4x4_s2_p1：
      低分辨率位置 i 经 ConvT2x2_s2 覆盖高分辨率 2i..2i+1，再经 Conv3x3_p1 覆盖 2i-1..2i+2；
      ConvT4x4_s2_p1 把位置 i 写到 2i-1+k（k=0..3），覆盖范围完全相同。
    因此合并核就是该组合对单位冲激的响应。换算在 float64 下完成，再转回原精度。

    只适用于 ConvT 与解码卷积之间没有非线性的结构：原版 V1（含 ReLU 对照；groupnorm_noshift 也可以，
    因为归一化作用在两路之和上）。已删除的 10 层版本在 ConvT 后插了 LIF，其权重含 up*_spike，
    形状与原版相同却不是线性串联，这里显式拒绝，避免被静默换算成错误结果。
    输入: 原版 EvSpSegNetStream 的 state_dict。输出: 新的 OrderedDict，除 up*/dec*.conv 外原样复制。
    """
    if any(key.split(".")[0].endswith("_spike") for key in state_dict):
        raise ValueError("权重含 up*_spike（已删除的 10 层脉冲解码器），ConvT 与解码卷积之间有 LIF，不能换算")
    merged = OrderedDict((key, value.detach().clone()) for key, value in state_dict.items())
    for up, dec in DECODER_STAGES:
        up_weight = state_dict[up + ".weight"]
        conv_weight = state_dict[dec + ".conv.weight"]
        if tuple(up_weight.shape[2:]) != (2, 2):
            raise ValueError("%s 的核为 %s，不是原版 2x2（可能已经是合并形式）" % (up, tuple(up_weight.shape[2:])))
        if tuple(conv_weight.shape[2:]) != (3, 3):
            raise ValueError("%s.conv 的核不是 3x3" % dec)
        deep_ch, up_ch = int(up_weight.shape[0]), int(up_weight.shape[1])
        if int(conv_weight.shape[1]) <= up_ch:
            raise ValueError("%s.conv 输入通道 %d 不大于 ConvT 输出通道 %d，没有跳连部分"
                             % (dec, int(conv_weight.shape[1]), up_ch))
        impulse = torch.zeros(deep_ch, deep_ch, 3, 3, dtype=torch.float64)
        channel = torch.arange(deep_ch)
        impulse[channel, channel, 1, 1] = 1.0                      # 每个输入通道在 (1,1) 放一个单位冲激
        response = F.conv2d(F.conv_transpose2d(impulse, up_weight.detach().cpu().double(), stride=2),
                            conv_weight[:, :up_ch].detach().cpu().double(), padding=1)
        merged[up + ".weight"] = response[:, :, 1:5, 1:5].to(
            dtype=up_weight.dtype, device=up_weight.device).contiguous()
        merged[dec + ".conv.weight"] = conv_weight[:, up_ch:].detach().clone().contiguous()
    return merged


def calibrate_gains(net, sequence_factory, quantile, samples_per_channel, min_positive,
                    gain_min, gain_max, seed):
    """一次性、逐层顺序的增益校准（只允许用训练集数据）。

    过程：对 LIF1..LIF7 依次执行——固定前面各层已校准的增益，
    在校准数据上按网络自身的 state_mode 连续前向（带状态），
    收集本层"乘增益之前"的正电流，求逐通道 q 分位数，令 增益 = v_th / q。
    必须逐层进行：一次性在所有增益为 1 时统计，前层沉默会导致深层拿不到有效统计量。

    输入:
        sequence_factory  无参函数，每次调用返回一个可迭代对象，产出若干"序列"；
                          每个序列是 (x, events) 的可迭代序列（按时间顺序）
        samples_per_channel 每个窗口每个通道最多保留的随机样本数（用固定种子采样，结果可复现）
    输出: 每层的校准报告列表；并把 net.gain_calibrated 置为 True。
    """
    if bool(net.gain_calibrated):
        raise RuntimeError("增益已经校准过，禁止重复校准（评估阶段不得重新校准）")
    generator = torch.Generator().manual_seed(int(seed))
    reports = []
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for layer_index, block in enumerate(net.blocks()):
            n_ch = block.out_ch
            samples = [[] for _ in range(n_ch)]
            positive_counts = np.zeros(n_ch, dtype=np.int64)
            for sequence in sequence_factory():
                states = None
                for x, events in sequence:
                    _, states, info = net(x, events, states, collect=True)
                    current = info["current"][layer_index]
                    for c in range(n_ch):
                        channel = current[:, c]
                        values = channel[channel > 0]
                        n = int(values.numel())
                        positive_counts[c] += n
                        if n == 0:
                            continue
                        if n > samples_per_channel:
                            pick = torch.randint(0, n, (samples_per_channel,), generator=generator)
                            values = values[pick.to(values.device)]
                        samples[c].append(values.float().cpu().numpy())
            merged = [np.concatenate(s) if s else np.zeros(0) for s in samples]
            gains, detail = gains_from_positive_samples(
                merged, positive_counts, net.v_threshold, quantile, min_positive, gain_min, gain_max)
            block.gain.set_gain(torch.from_numpy(gains))
            reports.append({"layer": LAYER_NAMES[layer_index],
                            "positive_counts": positive_counts.tolist(),
                            "gain": gains.tolist(), **detail})
    net.gain_calibrated.fill_(True)
    net.train(was_training)
    return reports


def estimate_operations(net, height, width, firing_rates, events_per_window):
    """估计单个窗口的理论运算量（面向神经形态硬件的理论值，不代表 GPU 实测能耗）。

    约定：
        稠密运算数 dense = C_out * C_in * k*k * H_out * W_out（ConvT 按输入位置计）
        输入是脉冲的部分记为 SOP ≈ 输入发放率 * dense（常用近似）
        输入是实数的部分记为 MAC：第一层（实数计数输入）、解码器中 ConvT 输出的通道、读出 MLP
        电流合并解码器中 ConvT 与解码卷积的输入都是脉冲，解码阶段全部记为 SOP
    输入: firing_rates 为 {层名: 该层平均发放率}；events_per_window 为平均每窗事件数（读出 MLP 按事件计）。
    输出: {"dense_ops", "mac", "sop", "per_layer": [...]}
    """
    h, w = int(height), int(width)
    layers = []

    def conv_ops(conv, hw_out):
        """卷积的稠密运算数 = C_out * C_in * k*k * 输出像素数。"""
        k = conv.kernel_size[0] * conv.kernel_size[1]
        return conv.out_channels * conv.in_channels * k * hw_out

    def add(name, mac, sop, dense):
        """记录一层的 MAC / SOP / 稠密运算数。"""
        layers.append({"layer": name, "mac": float(mac), "sop": float(sop), "dense": float(dense)})

    size = {"enc1": h * w, "enc2": (h // 2) * (w // 2), "enc3": (h // 4) * (w // 4),
            "enc4": (h // 8) * (w // 8)}
    d = conv_ops(net.enc1.conv, size["enc1"])
    add("enc1", d, 0.0, d)                                           # 实数输入 -> MAC
    for name, src in (("enc2", "enc1"), ("enc3", "enc2"), ("enc4", "enc3")):
        d = conv_ops(getattr(net, name).conv, size[name])
        add(name, 0.0, firing_rates[src] * d, d)
    for up, dec, deep, skip, hw in (("up3", "dec3", "enc4", "enc3", size["enc3"]),
                                    ("up2", "dec2", "dec3", "enc2", size["enc2"]),
                                    ("up1", "dec1", "dec2", "enc1", size["enc1"])):
        conv_t = getattr(net, up)
        k = conv_t.kernel_size[0] * conv_t.kernel_size[1]
        d_up = conv_t.in_channels * conv_t.out_channels * k * (hw // 4)
        add(up, 0.0, firing_rates[deep] * d_up, d_up)               # 输入是深层脉冲
        conv = getattr(net, dec).conv
        if net.merged_decoder:
            d_dec = conv_ops(conv, hw)                               # 只接跳连脉冲
            add(dec, 0.0, firing_rates[skip] * d_dec, d_dec)
        else:
            k = conv.kernel_size[0] * conv.kernel_size[1]
            real_part = conv.out_channels * conv_t.out_channels * k * hw  # ConvT 输出为实数 -> MAC
            spike_part = conv.out_channels * (conv.in_channels - conv_t.out_channels) * k * hw
            add(dec, real_part, firing_rates[skip] * spike_part, real_part + spike_part)
    lin1, lin2 = net.readout[0], net.readout[2]
    per_event = lin1.in_features * lin1.out_features + lin2.in_features * lin2.out_features
    add("readout", per_event * float(events_per_window), 0.0, per_event * float(events_per_window))
    if net.neuron_kind != "lif":                                     # ReLU 对照没有脉冲，全部按 MAC 计
        for layer in layers:
            layer["mac"], layer["sop"] = layer["dense"], 0.0
    state_size = dict(size)
    for level in (1, 2, 3):
        state_size["dec%d" % level] = size["enc%d" % level]
    state_elements = sum(block.out_ch * state_size[name]
                         for name, block in zip(LAYER_NAMES, net.blocks()))
    return {"dense_ops": sum(l["dense"] for l in layers), "mac": sum(l["mac"] for l in layers),
            "sop": sum(l["sop"] for l in layers), "per_layer": layers,
            "state_elements": state_elements if net.neuron_kind == "lif" else 0,
            "note": "SOP 为基于发放率的理论估计；当前 GPU 实现仍执行稠密卷积；"
                    "未计入膜电位更新、增益、访存及预后处理成本，state_elements 按单序列统计"}
