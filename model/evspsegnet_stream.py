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
所有突触运算的输入都是 0/1 脉冲（neuron=tif 时为多级电平 theta*q，传输的是整数变化量 Δq），
仍为 7 组状态、读出不变。原版 ConvT2x2 与解码卷积之间没有非线性，
Conv3x3(cat[ConvT2x2(S_deep), S_skip]) 可以精确合成上式，训练好的原版权重可无损换算，
见 merge_decoder_state_dict 与 tools/convert_to_merged_decoder.py。

两种执行方式（数学等价，见 tests/test_stream_layerwise.py 与 tools/check_stream_execution.py）：
    forward        逐窗：每个窗口依次经过 7 层（流式推理、增益校准、正式评估）
    forward_chunk  逐层：一个 TBPTT 片段的 T 个窗口先整体过完一层再进入下一层（训练加速）
"""
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lif2d_stream import (ChannelGain, ReLUNeuron, StreamingGraded2d,
                                StreamingLIF2d, StreamingDualMemoryLIF2d, StreamingTIF2d,
                                SurrogateSpike)
from utils.stream_common import gains_from_positive_samples

LAYER_NAMES = ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")
DECODER_STAGES = (("up3", "dec3"), ("up2", "dec2"), ("up1", "dec1"))
# 发出消息的层：输出（脉冲 / Δq）被其他层接收。dec1 的输出不被任何层接收（读出头读取的是 dec1 的 u_pre），不发消息。
# 改动 forward / forward_chunk 的连接时必须同步修改；tests/test_stream_comm.py 用扰动逐层核对这一声明。
MESSAGE_LAYERS = ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2")
# surrogate：10-19 两端代理梯度（实验失败，默认关）
# adapt：输入层工作点自校准（只在 enc1 打开，偏差校正的慢基线，Stream-SNN-Innovations.md 12.10），默认关；
#   关时与基础版逐位相同。属事件表示层的补充项（第 3 项移交事件表示层，12.7）
# adapt_debias / adapt_layers：只为读入第三版（evisseg_stream_tif_adapt3）的配置保留，取值必须是 true / [enc1]；
#   10-22 在全部 7 层上做的两版（基线从 0 开始、偏差校正）已否定并删除，不能再构造（12.5、12.10）
TIF_DEFAULTS = {"levels": 7, "quantize": True, "k_init": 0.05, "surrogate": False,
                "adapt": False, "adapt_tau_init_ms": 2000.0, "adapt_tau_min_ms": 1000.0, "adapt_tau_max_ms": 10000.0}
TIF_NEURON_KEYS = ("levels", "quantize", "k_init", "surrogate",
                   "adapt", "adapt_tau_init_ms", "adapt_tau_min_ms", "adapt_tau_max_ms")
TIF_ADAPT_LAYERS = ("enc1",)
TIF_ADAPT_TAU_KEYS = ("adapt_tau_min_ms", "adapt_tau_init_ms", "adapt_tau_max_ms")
# 已删除的机制留在旧 checkpoint 配置里的键（10-19 删除搬运，10-22 删除本地参照与滞回汇报）：只接受"未启用"的取值，读入时丢弃
TIF_REMOVED_KEYS = {"eps_init": None, "transport": False, "transport_ref_init": 0.0, "transport_layers": None,
                    "reference": False, "hysteresis": False}


def tif_settings(tif):
    """校验 TIF 配置字典并补全默认值；None 表示全部取默认（见 TIF_DEFAULTS）。

    旧配置里已删除机制的键：搬运与本地参照必须是关闭的（transport / reference: false、transport_ref_init: 0），
    否则报错；eps_init、transport_layers 只在搬运时起作用，直接丢弃。
    """
    if tif is None:
        tif = {}
    if not isinstance(tif, dict):
        raise ValueError("tif 必须为配置字典")
    if tif.get("transport", False) is not False or float(tif.get("transport_ref_init", 0.0)) != 0.0:
        raise ValueError("TIF 的膜电位搬运已于 10-19 删除，不能构造或加载带搬运的配置")
    if tif.get("reference", False) is not False:
        raise ValueError("TIF 的本地参照（TIF-R）已于 10-22 删除，不能构造或加载带参照的配置")
    if tif.get("hysteresis", False) is not False:
        raise ValueError("TIF 的滞回汇报已于 10-22 删除（12.12~12.13），不能构造或加载带滞回的配置")
    tif = {key: value for key, value in tif.items() if key not in TIF_REMOVED_KEYS}
    debias, layers = tif.pop("adapt_debias", None), tif.pop("adapt_layers", None)
    if tif.get("adapt", False) is True:
        if debias is not True or layers is None or list(layers) != list(TIF_ADAPT_LAYERS):
            raise ValueError("tif.adapt 现在只表示 enc1 上的偏差校正自校准：配置须写 adapt_debias: true、"
                             "adapt_layers: [enc1]（全部 7 层的两版已于 10-22 否定并删除）")
    elif debias not in (None, False) or layers is not None:
        raise ValueError("tif.adapt_debias / tif.adapt_layers 只能与 tif.adapt: true 同用")
    unknown = set(tif) - set(TIF_DEFAULTS)
    if unknown:
        raise ValueError("未知 TIF 配置: %s" % sorted(unknown))
    out = dict(TIF_DEFAULTS)
    out.update(tif)
    if isinstance(out["levels"], bool) or int(out["levels"]) != out["levels"] or int(out["levels"]) < 1:
        raise ValueError("tif.levels 必须是 >= 1 的整数")
    if not all(isinstance(out[key], bool) for key in ("quantize", "surrogate", "adapt")):
        raise ValueError("tif.quantize / tif.surrogate / tif.adapt 必须为 YAML 布尔值")
    taus = [float(out[key]) for key in TIF_ADAPT_TAU_KEYS]
    if not (all(np.isfinite(taus)) and 0 < taus[0] < taus[1] < taus[2]):
        raise ValueError("需要 0 < tif.adapt_tau_min_ms < tif.adapt_tau_init_ms < tif.adapt_tau_max_ms")
    out.update(zip(TIF_ADAPT_TAU_KEYS, taus))
    value = float(out["k_init"])
    if not (np.isfinite(value) and value > 0):
        raise ValueError("tif.k_init 必须是有限正数")
    out["k_init"] = value
    out["levels"] = int(out["levels"])
    return out


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

    def __init__(self, in_ch, out_ch, stride, neuron, norm, lif_kwargs, tif_kwargs=None):
        super(SpikingConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = make_norm(norm, out_ch)
        self.gain = ChannelGain(out_ch)
        if neuron == "lif":
            self.neuron = StreamingLIF2d(out_ch, **lif_kwargs)
        elif neuron == "graded":
            self.neuron = StreamingGraded2d(out_ch, **lif_kwargs)
        elif neuron == "relu":
            self.neuron = ReLUNeuron()
        elif neuron == "tif":
            self.neuron = StreamingTIF2d(out_ch, **dict(lif_kwargs, **(tif_kwargs or {})))
        else:
            raise ValueError("未知的 neuron 类型: %s" % neuron)
        self.out_ch = out_ch

    def _currents(self, x, extra_current=None):
        """返回（增益后电流，增益前电流）；供普通前向和浅层观测残差共用，避免重复卷积。"""
        current = self.conv(x)
        if extra_current is not None:
            current = current + extra_current
        pre_gain = self.norm(current)
        return self.gain(pre_gain), pre_gain

    def forward(self, x, state, extra_current=None, return_driving_current=False):
        driving, pre_gain = self._currents(x, extra_current)
        spikes, new_state, u_pre = self.neuron(driving, state)
        result = (spikes, new_state, u_pre, pre_gain)
        return result + (driving,) if return_driving_current else result

    def forward_steps(self, x, state, carry, keep_u_pre, extra_current=None, collect_memory=False,
                      return_driving_current=False, keep=None):
        """一个片段的多窗口版本：卷积在 T*B 维上批量计算，神经元再按时间递推。

        输入: x [T,B,C_in,H,W]；extra_current 为已展平的 [T*B,C_out,H',W'] 或 None；
              keep [T,B] 或 None（多流训练中某条流在片段中途开始新录像时清零该流状态，见 mask_state）。
        输出: (spikes [T,B,...], 最后一步膜电位 [B,...], u_pre [T,B,...] 或 None)。不返回增益前电流（校准仍逐窗做）。
        """
        steps, batch = int(x.shape[0]), int(x.shape[1])
        current, _ = self._currents(x.reshape((steps * batch,) + tuple(x.shape[2:])), extra_current)
        current = current.view((steps, batch) + tuple(current.shape[1:]))
        if collect_memory and isinstance(self.neuron, StreamingDualMemoryLIF2d):
            result = self.neuron.forward_steps(current, state, carry, keep_u_pre, collect_memory=True, keep=keep)
        else:
            result = self.neuron.forward_steps(current, state, carry, keep_u_pre, keep=keep)
        return result + (current,) if return_driving_current else result


class EvSpSegNetStream(nn.Module):
    """流式脉冲分割网络（V1）。

    forward(x, events, states, state_mode=None, collect=False) -> (logits, new_states, info)
        x          [B,12,H,W] 当前窗口输入
        events     字典：b/y/x 为 long [N]，p/t_local 为 float [N]（当前窗口的逐事件信息）
        states     上一窗口返回的 7 组状态列表，或 None；双仓层的状态为 (U, M)
        state_mode 'carry'（使用 states）或 'reset_each_window'（忽略 U/M 全部历史）；
                   None 时使用构造参数中的默认值
        collect    True 时 info 返回每层的脉冲、u_pre、增益前电流（用于监控与校准）
        logits     [N] 每个事件的原始 logit
        new_states 7 组状态：普通 LIF 为软复位后的 U，双仓为 (U, M)，TIF 为信念 M（adapt 时为 (M, b)），ReLU 为 None
    neuron="tif" 时 tif 为 TIF 配置字典（键见 TIF_DEFAULTS；None 取默认）；TIF 参数不消耗随机数，
    同种子下卷积与读出的初始化与 LIF 版本逐位相同。
    """

    def __init__(self, in_channels=12, channels=(12, 24, 48, 48), neuron="lif", norm="none",
                 state_mode="carry", readout_hidden=32, dt_ms=50.0, tau_init_ms=200.0,
                 tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0, merged_decoder=False,
                 dual_memory=None, subthreshold_residual=None, tif=None):
        super(EvSpSegNetStream, self).__init__()
        if state_mode not in ("carry", "reset_each_window"):
            raise ValueError("state_mode 必须是 carry 或 reset_each_window")
        if not isinstance(merged_decoder, bool):
            raise ValueError("merged_decoder 必须是 YAML 布尔值（true/false）")
        if tif is not None and neuron != "tif":
            raise ValueError("tif 配置只用于 neuron: tif")
        self.tif_config = tif_settings(tif) if neuron == "tif" else None
        c1, c2, c3, c4 = channels
        lif = dict(dt_ms=dt_ms, tau_init_ms=tau_init_ms, tau_min_ms=tau_min_ms,
                   tau_max_ms=tau_max_ms, v_threshold=v_threshold)
        neuron_tif = (None if self.tif_config is None
                      else {key: self.tif_config[key] for key in TIF_NEURON_KEYS})
        self.tif_adapt_layers = ()
        if self.tif_config is not None and self.tif_config["adapt"]:
            self.tif_adapt_layers = TIF_ADAPT_LAYERS

        def layer_tif(name):
            # 不在 adapt_layers 里的层就是基础版 TIF（TIF 参数不消耗随机数，初始化与层的选择无关）
            if neuron_tif is None or name in self.tif_adapt_layers:
                return neuron_tif
            return dict(neuron_tif, adapt=False)

        block = lambda name, i, o, s: SpikingConvBlock(i, o, s, neuron, norm, lif, layer_tif(name))  # noqa: E731
        self.enc1 = block("enc1", in_channels, c1, 1)
        self.enc2 = block("enc2", c1, c2, 2)
        self.enc3 = block("enc3", c2, c3, 2)
        self.enc4 = block("enc4", c3, c4, 2)
        if merged_decoder:
            # ConvT4x4/s2/p1 负责深层脉冲，解码卷积只接跳连脉冲；两路电流在神经元内相加
            upsample = lambda i, o: nn.ConvTranspose2d(i, o, 4, stride=2, padding=1, bias=False)  # noqa: E731
            self.up3, self.dec3 = upsample(c4, c3), block("dec3", c3, c3, 1)
            self.up2, self.dec2 = upsample(c3, c2), block("dec2", c2, c2, 1)
            self.up1, self.dec1 = upsample(c2, c1), block("dec1", c1, c1, 1)
        else:
            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2, bias=False)
            self.dec3 = block("dec3", c3 + c3, c3, 1)
            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2, bias=False)
            self.dec2 = block("dec2", c2 + c2, c2, 1)
            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2, bias=False)
            self.dec1 = block("dec1", c1 + c1, c1, 1)
        self.readout = nn.Sequential(nn.Linear(c1 + 2, readout_hidden), nn.ReLU(),
                                     nn.Linear(readout_hidden, 1))
        self.dual_memory_layers = ()
        if dual_memory is not None:
            if not isinstance(dual_memory, dict):
                raise ValueError("dual_memory 必须为配置字典")
            if neuron != "lif":
                raise ValueError("dual_memory 仅支持 neuron: lif")
            allowed = {"layers", "tau_init_ms", "tau_min_ms", "tau_max_ms", "kappa_max"}
            if set(dual_memory) - allowed:
                raise ValueError("未知双仓配置: %s" % sorted(set(dual_memory) - allowed))
            selected = dual_memory.get("layers")
            if (not isinstance(selected, (list, tuple)) or not selected
                    or any(n not in LAYER_NAMES for n in selected)
                    or len(set(selected)) != len(selected)):
                raise ValueError("dual_memory.layers 必须是 enc1/enc2/enc3/enc4/dec3/dec2/dec1 的非空、无重复列表")
            self.dual_memory_layers = tuple(n for n in LAYER_NAMES if n in selected)
            for name in self.dual_memory_layers:
                selected_block = getattr(self, name)
                selected_block.neuron = StreamingDualMemoryLIF2d(
                    selected_block.out_ch, **lif,
                    memory_tau_init_ms=float(dual_memory.get("tau_init_ms", 300.0)),
                    memory_tau_min_ms=float(dual_memory.get("tau_min_ms", 50.0)),
                    memory_tau_max_ms=float(dual_memory.get("tau_max_ms", 1000.0)),
                    kappa_max=float(dual_memory.get("kappa_max", 0.5)))
        # 浅层分级残差只表达当前窗口 enc1 电流，绝不保存为状态。新模块放在基础网络构造之后，
        # 因而启用该机制不会消耗随机数并改变原 U-Net 参数的初始化。
        if subthreshold_residual is None:
            subthreshold_residual = {}
        if not isinstance(subthreshold_residual, dict):
            raise ValueError("subthreshold_residual 必须为配置字典")
        residual_allowed = {"vertical", "event_skip", "local_kernel", "vertical_mode",
                            "vertical_peak_kernel", "vertical_threshold_init",
                            "vertical_threshold_min", "vertical_threshold_max",
                            "vertical_scale_max", "event_skip_mode",
                            "event_threshold_init", "event_threshold_min",
                            "event_threshold_max", "event_correction_max"}
        if set(subthreshold_residual) - residual_allowed:
            raise ValueError("未知亚阈值残差配置: %s" % sorted(set(subthreshold_residual) - residual_allowed))
        self.subthreshold_vertical = subthreshold_residual.get("vertical", False)
        self.subthreshold_event_skip = subthreshold_residual.get("event_skip", False)
        if not isinstance(self.subthreshold_vertical, bool) or not isinstance(self.subthreshold_event_skip, bool):
            raise ValueError("subthreshold_residual.vertical/event_skip 必须为 YAML 布尔值")
        self.residual_local_kernel = int(subthreshold_residual.get("local_kernel", 3))
        if self.residual_local_kernel < 1 or self.residual_local_kernel % 2 != 1:
            raise ValueError("subthreshold_residual.local_kernel 必须为正奇数")
        if (self.subthreshold_vertical or self.subthreshold_event_skip) and neuron != "lif":
            raise ValueError("亚阈值残差机制只支持 neuron: lif")
        if (self.subthreshold_vertical or self.subthreshold_event_skip) and "enc1" in self.dual_memory_layers:
            raise ValueError("亚阈值残差要求 enc1 为普通 LIF；双仓应部署在 enc2 及更深层")
        self.subthreshold_vertical_mode = str(subthreshold_residual.get("vertical_mode", "graded"))
        if self.subthreshold_vertical_mode not in ("graded", "competitive"):
            raise ValueError("subthreshold_residual.vertical_mode 必须为 graded 或 competitive")
        self.residual_vertical_peak_kernel = int(subthreshold_residual.get("vertical_peak_kernel", 3))
        if self.residual_vertical_peak_kernel < 1 or self.residual_vertical_peak_kernel % 2 != 1:
            raise ValueError("subthreshold_residual.vertical_peak_kernel 必须为正奇数")
        threshold_min = float(subthreshold_residual.get("vertical_threshold_min", 0.2))
        threshold_init = float(subthreshold_residual.get("vertical_threshold_init", 0.5))
        threshold_max = float(subthreshold_residual.get("vertical_threshold_max", 0.9))
        self.residual_vertical_scale_max = float(subthreshold_residual.get("vertical_scale_max", 0.5))
        if not 0.0 < threshold_min < threshold_init < threshold_max <= 1.0:
            raise ValueError("需要 0 < vertical_threshold_min < vertical_threshold_init < "
                             "vertical_threshold_max <= 1")
        if not 0.0 < self.residual_vertical_scale_max < 1.0:
            raise ValueError("vertical_scale_max 必须在 (0,1) 内")
        self.residual_vertical_threshold_min = threshold_min
        self.residual_vertical_threshold_max = threshold_max
        if self.subthreshold_vertical and self.subthreshold_vertical_mode == "competitive":
            fraction = (threshold_init - threshold_min) / (threshold_max - threshold_min)
            threshold_raw = float(np.log(fraction / (1.0 - fraction)))
            self.residual_vertical_threshold_raw = nn.Parameter(
                torch.full((c1,), threshold_raw, dtype=torch.float32))
            # 零初始化让新路径在训练开始时严格等同 enc234 基线；若残差有用，梯度先打开逐通道幅度，
            # 随后的梯度再联合调整竞争阈值。tanh 有界且允许通道选择促进或抑制。
            self.residual_vertical_scale_raw = nn.Parameter(torch.zeros(c1, dtype=torch.float32))
        self.subthreshold_event_skip_mode = str(subthreshold_residual.get("event_skip_mode", "legacy"))
        if self.subthreshold_event_skip_mode not in ("legacy", "selective"):
            raise ValueError("subthreshold_residual.event_skip_mode 必须为 legacy 或 selective")
        if (self.subthreshold_event_skip_mode == "selective"
                and not (self.subthreshold_vertical and self.subthreshold_vertical_mode == "competitive")):
            raise ValueError("selective 横向路径要求同时启用 competitive 纵向编码")
        self.residual_event_correction_max = float(
            subthreshold_residual.get("event_correction_max", 0.25))
        if not 0.0 < self.residual_event_correction_max <= 1.0:
            raise ValueError("event_correction_max 必须在 (0,1] 内")
        if self.subthreshold_event_skip:
            self.residual_gate = nn.Linear(c1, 1, bias=True)
            self.residual_project = nn.Linear(c1, c1, bias=False)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.zeros_(self.residual_gate.bias)
            nn.init.zeros_(self.residual_project.weight)
            if self.subthreshold_event_skip_mode == "selective":
                event_threshold_min = float(subthreshold_residual.get("event_threshold_min", 0.5))
                event_threshold_init = float(subthreshold_residual.get("event_threshold_init", 0.8))
                event_threshold_max = float(subthreshold_residual.get("event_threshold_max", 0.98))
                if not 0.0 < event_threshold_min < event_threshold_init < event_threshold_max <= 1.0:
                    raise ValueError("需要 0 < event_threshold_min < event_threshold_init < "
                                     "event_threshold_max <= 1")
                fraction = ((event_threshold_init - event_threshold_min)
                            / (event_threshold_max - event_threshold_min))
                self.residual_event_threshold_min = event_threshold_min
                self.residual_event_threshold_max = event_threshold_max
                self.residual_event_threshold_raw = nn.Parameter(torch.tensor(
                    float(np.log(fraction / (1.0 - fraction))), dtype=torch.float32))
        self.subthreshold_vertical_enabled = self.subthreshold_vertical
        self.subthreshold_event_skip_enabled = self.subthreshold_event_skip
        self.state_mode = state_mode
        self.neuron_kind = neuron
        self.merged_decoder = merged_decoder
        # 读出消融开关，默认关闭；只在评估阶段由命令行设置，不影响训练
        self.readout_ablation = "none"
        self.v_threshold = float(v_threshold)
        # 标记增益是否已校准；随 checkpoint 保存，评估时禁止重新校准
        self.register_buffer("gain_calibrated", torch.tensor(False))

    def message_layers(self):
        """发出消息的层名（见 MESSAGE_LAYERS）；通信预算与消息统计只通过这个接口得知网络连接。"""
        return MESSAGE_LAYERS

    def blocks(self):
        """按前向拓扑顺序返回 7 个脉冲卷积单元（校准必须按这个顺序逐层进行）。"""
        return [getattr(self, name) for name in LAYER_NAMES]

    @property
    def spiking(self):
        """输出是否为 0/1 脉冲（只有 lif 是；graded 与 relu 输出实数）。"""
        return self.neuron_kind == "lif"

    @property
    def stateful(self):
        """是否保留跨窗状态（lif、graded、tif 保留，relu 没有状态）。"""
        return self.neuron_kind in ("lif", "graded", "tif")

    @property
    def is_tif(self):
        return self.neuron_kind == "tif"

    def start_tif_stats(self, per_window=False):
        """开始累计各 TIF 层的诊断统计（消息率、电平占用；per_window 时另记逐窗 |Δq| 率）；非 TIF 网络不做任何事。"""
        if self.is_tif:
            for block in self.blocks():
                block.neuron.start_stats(per_window)

    def stop_tif_stats(self):
        """停止累计并返回 {层名: 统计}；非 TIF 网络返回空字典。"""
        if not self.is_tif:
            return {}
        return {name: block.neuron.stop_stats() for name, block in zip(LAYER_NAMES, self.blocks())}

    @property
    def has_dual_memory(self):
        return bool(self.dual_memory_layers)

    @property
    def has_subthreshold_residual(self):
        return self.subthreshold_vertical or self.subthreshold_event_skip

    def set_subthreshold_ablation(self, mode):
        """同权重推理消融；不删除模块，也不改变 enc1 的 LIF 状态动力学。"""
        if mode not in ("none", "vertical_off", "skip_off", "all_off"):
            raise ValueError("未知亚阈值残差消融: %s" % mode)
        self.subthreshold_vertical_enabled = self.subthreshold_vertical and mode not in ("vertical_off", "all_off")
        self.subthreshold_event_skip_enabled = self.subthreshold_event_skip and mode not in ("skip_off", "all_off")

    def _observation_residual(self, driving_current, spikes):
        """当前观测的有界有符号亚阈值残差；I=0 时严格为零，不会重播旧膜电位。"""
        return (1.0 - spikes.detach()) * torch.clamp(
            driving_current / self.v_threshold, min=-1.0, max=1.0)

    def _competitive_residual_code(self, residual):
        """把当前观测残差编码为 {-1,0,+1} 稀疏码，供纵向与选择性横向路径共享。"""
        original_shape = residual.shape
        if residual.dim() == 5:
            residual = residual.reshape((-1,) + tuple(residual.shape[2:]))
        magnitude = residual.abs()
        fraction = torch.sigmoid(self.residual_vertical_threshold_raw).to(residual.dtype)
        threshold = (self.residual_vertical_threshold_min
                     + (self.residual_vertical_threshold_max
                        - self.residual_vertical_threshold_min) * fraction).view(1, -1, 1, 1)
        selected = SurrogateSpike.apply(magnitude - threshold)
        if self.residual_vertical_peak_kernel > 1:
            pad = self.residual_vertical_peak_kernel // 2
            local_max = F.max_pool2d(magnitude.detach(), self.residual_vertical_peak_kernel,
                                     stride=1, padding=pad)
            selected = selected * magnitude.detach().ge(local_max).to(residual.dtype)
        return (residual.sign() * selected).reshape(original_shape)

    def _vertical_residual(self, residual):
        """纵向编码：原连续残差，或局部竞争后的稀疏双极脉冲。

        competitive 模式只保留超过逐通道可学习阈值、且为同通道局部幅值峰值的位置。
        输出为 {-g_c, 0, +g_c}，g_c 有界且零初始化；g_c 可折叠进 enc2 卷积核，
        因而在事件驱动实现中仍可按稀疏加/减累加，而不要求整幅连续 MAC。
        """
        if self.subthreshold_vertical_mode == "graded":
            return residual
        code = self._competitive_residual_code(residual)
        scale = (self.residual_vertical_scale_max
                 * torch.tanh(self.residual_vertical_scale_raw).to(residual.dtype))
        scale = scale.view((1, 1, -1, 1, 1) if residual.dim() == 5 else (1, -1, 1, 1))
        return code * scale

    def subthreshold_parameter_values(self):
        """返回竞争式纵向及选择性横向阈值；只供诊断，不参与前向。"""
        result = {}
        if self.subthreshold_vertical and self.subthreshold_vertical_mode == "competitive":
            fraction = torch.sigmoid(self.residual_vertical_threshold_raw)
            result["threshold"] = (self.residual_vertical_threshold_min
                                   + (self.residual_vertical_threshold_max
                                      - self.residual_vertical_threshold_min) * fraction)
            result["scale"] = (self.residual_vertical_scale_max
                               * torch.tanh(self.residual_vertical_scale_raw))
        if self.subthreshold_event_skip and self.subthreshold_event_skip_mode == "selective":
            fraction = torch.sigmoid(self.residual_event_threshold_raw)
            result["event_threshold"] = (self.residual_event_threshold_min
                                         + (self.residual_event_threshold_max
                                            - self.residual_event_threshold_min) * fraction)
        return result

    def _event_features(self, u_pre, residual, deep_current, events):
        """读取逐事件 dec1 特征，并在配置开启时加入受深层局部活动约束的瞬时浅层修正。"""
        if u_pre.dim() == 5:
            steps, batch = int(u_pre.shape[0]), int(u_pre.shape[1])
            flat_index = events["t"] * batch + events["b"]
            shape = (steps * batch,) + tuple(u_pre.shape[2:])
            u_map = u_pre.reshape(shape)
            d_map = deep_current.reshape(shape)
        else:
            flat_index = events["b"]
            u_map, d_map = u_pre, deep_current
        feat = u_map[flat_index, :, events["y"], events["x"]]
        if not self.subthreshold_event_skip_enabled:
            return feat, None, None
        r_map = residual.reshape(u_map.shape) if residual.dim() == 5 else residual
        pad = self.residual_local_kernel // 2
        # 门控只读取深层支持，不允许这条浅层捷径反向塑造深层特征；先 detach 可避免池化图占用训练显存。
        gate_map = d_map.detach()
        local_deep = F.avg_pool2d(gate_map, self.residual_local_kernel, stride=1, padding=pad)
        activity = gate_map.abs().sum(1, keepdim=True)
        if self.subthreshold_event_skip_mode == "selective":
            # 深层活动必须接近本地峰值；比值归一化避免固定阈值受卷积幅值尺度支配。
            local_max = F.max_pool2d(activity, self.residual_local_kernel, stride=1, padding=pad)
            ratio = activity / (local_max + torch.finfo(activity.dtype).eps)
            fraction = torch.sigmoid(self.residual_event_threshold_raw).to(activity.dtype)
            threshold = (self.residual_event_threshold_min
                         + (self.residual_event_threshold_max
                            - self.residual_event_threshold_min) * fraction)
            deep_support = (SurrogateSpike.apply(ratio - threshold)
                            * activity.gt(0).to(activity.dtype))
            # 横向只读取与纵向相同的竞争式浅层峰值，不再读取所有连续残差。
            r_map = self._competitive_residual_code(r_map)
            shallow_support = r_map.detach().abs().sum(1, keepdim=True).gt(0).to(activity.dtype)
            support = deep_support * shallow_support
        else:
            support = F.max_pool2d(activity, self.residual_local_kernel,
                                   stride=1, padding=pad).gt(0).detach()
        deep_event = local_deep[flat_index, :, events["y"], events["x"]]
        support_event = support[flat_index, 0, events["y"], events["x"]].to(feat.dtype)
        gate = support_event * torch.sigmoid(self.residual_gate(deep_event)).squeeze(1)
        residual_event = r_map[flat_index, :, events["y"], events["x"]]
        correction = self.residual_project(residual_event * gate.unsqueeze(1))
        if self.subthreshold_event_skip_mode == "selective":
            correction = self.residual_event_correction_max * torch.tanh(correction)
        return feat + correction, gate, correction

    def set_memory_read_enabled(self, enabled, layers=None):
        """只关闭新增记忆的读取，保留所有层的普通 LIF 历史；用于同权重消融。"""
        if not self.has_dual_memory:
            raise ValueError("当前模型没有双仓记忆")
        selected = self.dual_memory_layers if layers is None else layers
        if (not isinstance(selected, (list, tuple)) or not selected
                or any(n not in self.dual_memory_layers for n in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("Memory read ablation requires unique, configured dual-memory layers")
        for name in selected:
            getattr(self, name).neuron.memory_read_enabled = bool(enabled)

    def forward(self, x, events, states, state_mode=None, collect=False):
        mode = self.state_mode if state_mode is None else state_mode
        if mode not in ("carry", "reset_each_window"):
            raise ValueError("state_mode 必须是 carry 或 reset_each_window")
        prev = states if (mode == "carry" and states is not None) else [None] * 7
        if len(prev) != 7:
            raise ValueError("states 应为 7 个膜电位，收到 %d 个" % len(prev))
        if self.has_subthreshold_residual:
            s1, n1, u1, i1, drive1 = self.enc1(x, prev[0], return_driving_current=True)
            residual1 = self._observation_residual(drive1, s1)
        else:
            s1, n1, u1, i1 = self.enc1(x, prev[0])
            residual1 = None
        vertical1 = self._vertical_residual(residual1) if self.subthreshold_vertical_enabled else None
        enc2_input = s1 + vertical1 if vertical1 is not None else s1
        s2, n2, u2, i2 = self.enc2(enc2_input, prev[1])
        s3, n3, u3, i3 = self.enc3(s2, prev[2])
        s4, n4, u4, i4 = self.enc4(s3, prev[3])
        if self.merged_decoder:
            # 两路突触电流在神经元内相加：跳连脉冲经解码卷积，深层脉冲经 ConvT（作为 extra_current）
            s5, n5, u5, i5 = self.dec3(s3, prev[4], self.up3(s4))
            s6, n6, u6, i6 = self.dec2(s2, prev[5], self.up2(s5))
            deep1 = self.up1(s6)
            s7, n7, u7, i7 = self.dec1(s1, prev[6], deep1)
        else:
            s5, n5, u5, i5 = self.dec3(torch.cat([self.up3(s4), s3], 1), prev[4])
            s6, n6, u6, i6 = self.dec2(torch.cat([self.up2(s5), s2], 1), prev[5])
            deep1 = self.up1(s6)
            s7, n7, u7, i7 = self.dec1(torch.cat([deep1, s1], 1), prev[6])
        feat, residual_gate, residual_correction = self._event_features(
            u7, residual1, deep1, events)
        logits = self._readout(feat, events)
        info = None
        if collect:
            info = {"spikes": [s1, s2, s3, s4, s5, s6, s7],
                    "u_pre": [u1, u2, u3, u4, u5, u6, u7],
                    "current": [i1, i2, i3, i4, i5, i6, i7]}
            if self.has_subthreshold_residual:
                info.update(subthreshold_residual=residual1,
                            subthreshold_vertical_signal=vertical1,
                            event_residual_gate=residual_gate,
                            event_residual_correction=residual_correction)
            if self.has_dual_memory:
                info["memory"] = {name: state[1].detach()
                                  for name, state in zip(LAYER_NAMES, [n1, n2, n3, n4, n5, n6, n7])
                                  if name in self.dual_memory_layers}
        return logits, [n1, n2, n3, n4, n5, n6, n7], info

    def forward_chunk(self, x, events, states, state_mode=None, collect=False, keep=None):
        """逐层时间并行：一次处理一个片段的 T 个连续窗口，与按时间顺序调用 forward T 次数学等价。

        本网络层与层之间没有时间反馈，每层的输出序列只取决于它的输入序列，
        所以可以逐层推进：卷积把 T 个窗口当作一个 batch 算完，只有神经元内部的膜电位按时间递推。
        输入:
            x       [T,B,12,H,W]
            events  在 forward 的字段之外多一个 t（事件所在窗口在片段内的序号，long [N]）；b 为序列在 batch 内的序号
            states  片段开始前的 7 个膜电位 [B,C,H,W]，或 None
            keep    [T,B] 或 None：多流训练时，keep[t,b]=0 表示第 b 条流在第 t 窗开始一段新录像，
                    所有层在该窗递推前把这条流的状态清零（等价于该流从 None 重新开始，梯度在此截断）
        输出: (logits [N]，顺序与 events 相同, 片段结束时的 7 个膜电位, info)
            collect=True 时 info 含每层 [T,B,C,H,W] 的 spikes 与 u_pre（不含增益前电流）。
        流式推理与增益校准仍使用 forward。
        """
        mode = self.state_mode if state_mode is None else state_mode
        if mode not in ("carry", "reset_each_window"):
            raise ValueError("state_mode 必须是 carry 或 reset_each_window")
        prev = states if (mode == "carry" and states is not None) else [None] * 7
        if len(prev) != 7:
            raise ValueError("states 应为 7 个膜电位，收到 %d 个" % len(prev))
        if x.dim() != 5:
            raise ValueError("forward_chunk 的输入应为 [T,B,C,H,W]，收到 %d 维" % x.dim())
        carry = mode == "carry"
        steps, batch = int(x.shape[0]), int(x.shape[1])
        if keep is not None and tuple(keep.shape) != (steps, batch):
            raise ValueError("keep 应为 [T,B]=(%d,%d)，收到 %s" % (steps, batch, tuple(keep.shape)))

        def flat(seq):
            return seq.reshape((steps * batch,) + tuple(seq.shape[2:]))

        def unflat(tensor):
            return tensor.view((steps, batch) + tuple(tensor.shape[1:]))

        memories = {}

        def run_layer(name, inputs, extra_current=None, keep_u_pre=None, return_driving_current=False):
            keep_flag = collect if keep_u_pre is None else keep_u_pre
            result = getattr(self, name).forward_steps(
                inputs, prev[LAYER_NAMES.index(name)], carry, keep_flag,
                extra_current=extra_current, collect_memory=collect,
                return_driving_current=return_driving_current, keep=keep)
            if collect and name in self.dual_memory_layers:
                memories[name] = result[3]
            return result[:3] + ((result[-1],) if return_driving_current else ())

        if self.has_subthreshold_residual:
            s1, n1, u1, drive1 = run_layer("enc1", x, return_driving_current=True)
            residual1 = self._observation_residual(drive1, s1)
        else:
            s1, n1, u1 = run_layer("enc1", x)
            residual1 = None
        vertical1 = self._vertical_residual(residual1) if self.subthreshold_vertical_enabled else None
        enc2_input = s1 + vertical1 if vertical1 is not None else s1
        s2, n2, u2 = run_layer("enc2", enc2_input)
        s3, n3, u3 = run_layer("enc3", s2)
        s4, n4, u4 = run_layer("enc4", s3)
        if self.merged_decoder:
            s5, n5, u5 = run_layer("dec3", s3, self.up3(flat(s4)))
            s6, n6, u6 = run_layer("dec2", s2, self.up2(flat(s5)))
            deep1 = unflat(self.up1(flat(s6)))
            s7, n7, u7 = run_layer("dec1", s1, flat(deep1), keep_u_pre=True)
        else:
            s5, n5, u5 = run_layer("dec3", torch.cat([unflat(self.up3(flat(s4))), s3], 2))
            s6, n6, u6 = run_layer("dec2", torch.cat([unflat(self.up2(flat(s5))), s2], 2))
            deep1 = unflat(self.up1(flat(s6)))
            s7, n7, u7 = run_layer("dec1", torch.cat([deep1, s1], 2),
                                   keep_u_pre=True)
        feat, residual_gate, residual_correction = self._event_features(
            u7, residual1, deep1, events)
        logits = self._readout(feat, events)
        info = None
        if collect:
            info = {"spikes": [s1, s2, s3, s4, s5, s6, s7],
                    "u_pre": [u1, u2, u3, u4, u5, u6, u7],
                    }
            if self.has_subthreshold_residual:
                info.update(subthreshold_residual=residual1,
                            subthreshold_vertical_signal=vertical1,
                            event_residual_gate=residual_gate,
                            event_residual_correction=residual_correction)
            if self.has_dual_memory:
                info["memory"] = memories
        return logits, [n1, n2, n3, n4, n5, n6, n7], info

    def _readout(self, feat, events):
        """逐事件读出：MLP([膜电位特征, p, t_local]) -> 原始 logit [N]。"""
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
        return self.readout(torch.cat([feat, extra], 1)).squeeze(1)


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


def estimate_operations(net, height, width, firing_rates, events_per_window, input_density=None):
    """估计单个窗口的理论运算量（面向神经形态硬件的理论值，不代表 GPU 实测能耗）。

    约定：
        稠密运算数 dense = C_out * C_in * k*k * H_out * W_out（ConvT 按输入位置计）
        输入是脉冲的部分记为 SOP ≈ 输入发放率 * dense（常用近似）
        输入是实数的部分记为 MAC：第一层（实数计数输入）、解码器中 ConvT 输出的通道、读出 MLP
        电流合并解码器中 ConvT 与解码卷积的输入都是脉冲，解码阶段全部记为 SOP
    输入: firing_rates 为 {层名: 该层平均发放率}；events_per_window 为平均每窗事件数（读出 MLP 按事件计）。
        input_density 为输入中非零元素的占比（实测值，可选）。给出时额外返回事件驱动口径的 enc1 运算量：
        卷积按输入散射时一个非零输入驱动 C_out*k*k 次乘加，所以 enc1 运算数 = 稠密运算数 * 非零占比，
        与基线稀疏卷积"只在有事件的位置计算"是同一口径（见 tools/stream_energy.py）。
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
        if name == "enc2" and net.subthreshold_vertical_enabled:
            if net.subthreshold_vertical_mode == "competitive":
                # S1 与残差信号按构造互斥。逐通道幅度可折叠进卷积核，稀疏双极码按累加估计。
                density = min(1.0, firing_rates[src]
                              + firing_rates.get("subthreshold_vertical", 0.0))
                add(name, 0.0, density * d, d)
            else:
                add(name, d, 0.0, d)                                # S1+R1 含连续观测残差
        else:
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
    if net.subthreshold_event_skip_enabled:
        c1 = net.enc1.out_ch
        residual_event_ops = c1 + c1 * c1                  # 12->1 门 + 12->12 投影；不计池化/逐元素门控
        add("event_residual_skip", residual_event_ops * float(events_per_window), 0.0,
            residual_event_ops * float(events_per_window))
    tif_quantized = net.is_tif and net.tif_config["quantize"]
    if not net.spiking and not tif_quantized:  # graded / relu / 不量化的 TIF 输出实数，全部按 MAC 计
        for layer in layers:
            layer["mac"], layer["sop"] = layer["dense"], 0.0
    state_size = dict(size)
    for level in (1, 2, 3):
        state_size["dec%d" % level] = size["enc%d" % level]
    state_elements = sum(block.out_ch * state_size[name]
                         for name, block in zip(LAYER_NAMES, net.blocks()))
    result = {"dense_ops": sum(l["dense"] for l in layers), "mac": sum(l["mac"] for l in layers),
              "sop": sum(l["sop"] for l in layers), "per_layer": layers,
              "state_elements": state_elements if net.stateful else 0}
    if net.is_tif:
        cfg = net.tif_config
        update_ops = 7 + (4 if cfg["quantize"] else 0)  # 驱动、EMA、截断约 7 次；量化与 Δq 约 4 次
        adapt_elements = sum(block.out_ch * state_size[name]
                             for name, block in zip(LAYER_NAMES, net.blocks()) if name in net.tif_adapt_layers)
        adapt_ops = 5                                       # 基线 EMA 两乘一加、relu 与减法、偏差校正的一次除法
        result["state_elements"] += adapt_elements          # 自校准层每个神经元多一个基线 b
        result["tif"] = {
            "adapt_layers": list(net.tif_adapt_layers), "adapt_state_elements": adapt_elements,
            "adapt": cfg["adapt"],
            "levels": cfg["levels"], "quantize": cfg["quantize"],
            "message_payload_bits": int(math.ceil(math.log2(2 * cfg["levels"] + 1))),
            "sop_basis": "firing_rates 为每神经元每窗平均 |Δq|：SOP = 单位脉冲口径（每个单位电平变化一次突触累加）",
            "elementwise_ops_per_window": update_ops * state_elements + adapt_ops * adapt_elements,
            "surrogate": cfg["surrogate"],
            "synaptic_accumulator_elements": state_elements,
            "included_in_mac_sop": False,
            "note": ("逐元素运算单列，未折算进 MAC/SOP；事件驱动实现按 Δ 网络做法为每个神经元保存一份突触累积，"
                     "另列 synaptic_accumulator_elements；quantize=false 时层间是实数，全部按 MAC 计")}
    if net.has_dual_memory:
        extra_by_layer = {name: getattr(net, name).out_ch * state_size[name] for name in net.dual_memory_layers}
        extra = sum(extra_by_layer.values())
        result["state_elements"] += extra
        result["dual_memory"] = {
            "layers": list(net.dual_memory_layers), "additional_state_elements": extra,
            "state_elements_per_layer": extra_by_layer,
            "additional_state_bytes_fp32": extra * 4,
            "parameters": 2 * sum(getattr(net, name).out_ch for name in net.dual_memory_layers),
            # 每位置：Q 除阈值；M 两乘一加；旧 M 读取一乘一加。
            # 不含逐通道系数、autograd、监控、内存访问；无状态首窗略少。
            "elementwise_mul_per_window": 3 * extra,
            "elementwise_add_per_window": 2 * extra,
            "elementwise_div_per_window": extra,
            "tanh_per_window": extra,
            "included_in_mac_sop": False,
            "note": "额外逐元素算术与 tanh 单列，未折算为卷积 MAC/SOP 或总能耗；需实测耗时。"}
    if net.has_subthreshold_residual:
        competitive_parameters = (2 * net.enc1.out_ch
                                  if (net.subthreshold_vertical
                                      and net.subthreshold_vertical_mode == "competitive") else 0)
        event_parameters = (((net.enc1.out_ch + 1) + net.enc1.out_ch ** 2)
                            if net.subthreshold_event_skip else 0)
        if (net.subthreshold_event_skip
                and net.subthreshold_event_skip_mode == "selective"):
            event_parameters += 1
        result["subthreshold_residual"] = {
            "vertical": bool(net.subthreshold_vertical),
            "event_skip": bool(net.subthreshold_event_skip),
            "vertical_enabled": bool(net.subthreshold_vertical_enabled),
            "event_skip_enabled": bool(net.subthreshold_event_skip_enabled),
            "vertical_mode": net.subthreshold_vertical_mode,
            "vertical_peak_kernel": int(net.residual_vertical_peak_kernel),
            "event_skip_mode": net.subthreshold_event_skip_mode,
            "event_correction_max": float(net.residual_event_correction_max),
            "local_kernel": int(net.residual_local_kernel),
            "parameters": competitive_parameters + event_parameters,
            "note": ("graded vertical 使 enc2 按连续输入 MAC 计；competitive vertical 的逐通道"
                     "幅度可折叠进卷积核，按实测稀疏双极码密度计 SOP；event_skip 只在事件坐标"
                      "计线性层。selective 横向的池化、比较、clamp、tanh、门控和访存未折算。")}
    if input_density is not None:
        enc1 = layers[0]
        result["input_density"] = float(input_density)
        result["mac_event_driven"] = result["mac"] - enc1["mac"] + enc1["dense"] * float(input_density)
    result["note"] = ("SOP 为基于发放率的理论估计；当前 GPU 实现仍执行稠密卷积；"
                      "未计入膜电位更新、增益、访存及预后处理成本，state_elements 按单序列统计；"
                      "mac_event_driven 为事件驱动口径（enc1 只计非零输入），与基线稀疏卷积同口径")
    return result



def window_sop_summary(net, tif_stats, height, width):
    """评估（能耗口径，只用于报告）：由各层逐窗的 |Δq| 率（stop_stats 的 per_window_rate）算单窗 SOP 的分布。

    SOP 对各层发放率是线性的：SOP = base + sum_l c_l * r_l，c_l 由 estimate_operations 对各层逐一把发放率置 1 求得，
    与按网络自身结构的 estimate_operations 同口径。没有逐窗统计时返回 None。
    """
    if not tif_stats or any("per_window_rate" not in tif_stats.get(n, {}) for n in LAYER_NAMES):
        return None
    zero = {name: 0.0 for name in LAYER_NAMES}
    base = estimate_operations(net, height, width, zero, 0.0)["sop"]
    coef = []
    for name in LAYER_NAMES:
        rates = dict(zero)
        rates[name] = 1.0
        coef.append(estimate_operations(net, height, width, rates, 0.0)["sop"] - base)
    rates = np.array([tif_stats[n]["per_window_rate"] for n in LAYER_NAMES], dtype=np.float64)   # [7, 窗数]
    if rates.shape[1] == 0:
        return None
    values = base + np.asarray(coef, dtype=np.float64).dot(rates)
    return {"windows": int(values.size), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}
