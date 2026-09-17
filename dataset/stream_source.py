"""流式 SNN 的窗口数据来源：把一个 8 秒序列按窗口或按 TBPTT 片段转换成网络输入。

两种实现输出相同（输入逐位相同，事件顺序相同），由配置 input_device 选择：
    cpu  NumpyWindowSource：原实现，每个窗口在 CPU 上用 numpy 计数、归一化，再拷到设备
    gpu  TorchWindowSource：整条序列的事件一次性放到网络所在设备，按片段用 torch 计数，
         归一化查 numpy 预先算好的表（dataset/stream_windows.normalization_table）
         （"gpu" 指网络所在设备；单元测试里设备是 CPU，同一份代码照样运行）

接口:
    window(k)          -> (x [1,C,H,W], events{b,y,x,p,t_local}, labels [N], idx numpy [N])   供逐窗 forward
    chunk(start, end)  -> (x [T,1,C,H,W], events{t,b,y,x,p,t_local}, labels [N], idx numpy [N]) 供 forward_chunk
事件按窗口先后、窗内按时间稳定排序排列；idx 是这些事件在 NPZ 文件中的原始下标，用于回填预测。
"""
import numpy as np
import torch

from dataset.ev_uav_stream import window_to_device
from dataset.stream_windows import normalization_table, num_input_channels

INPUT_DEVICES = ("cpu", "gpu")


def make_window_source(seq, cfg, q99, device, dtype=torch.float32):
    """按 cfg["input_device"]（缺省 cpu，即原实现）构造窗口数据来源。

    dtype 为网络的浮点精度：float32 时不做任何转换；float64 只用于等价性核对。
    """
    kind = cfg.get("input_device", "cpu")
    if kind == "cpu":
        return NumpyWindowSource(seq, cfg, q99, device, dtype)
    if kind == "gpu":
        return TorchWindowSource(seq, cfg, q99, device, dtype)
    raise ValueError("input_device 必须是 %s 之一，收到 %r" % (INPUT_DEVICES, kind))


def _cast(x, events, labels, dtype):
    """把浮点输入转换到网络精度（float32 时原样返回，不产生额外运算）。"""
    if dtype == torch.float32:
        return x, events, labels
    events = {key: (value.to(dtype) if value.is_floating_point() else value) for key, value in events.items()}
    return x.to(dtype), events, labels.to(dtype)


class NumpyWindowSource(object):
    """原实现：逐窗调用 window_to_device。chunk 只是把逐窗结果拼起来，用于单独检验逐层执行。"""

    def __init__(self, seq, cfg, q99, device, dtype=torch.float32):
        self.seq, self.cfg, self.q99 = seq, cfg, q99
        self.device, self.dtype = device, dtype

    def window(self, k):
        x, events, labels, idx = window_to_device(self.seq, k, self.q99, self.cfg, self.device)
        x, events, labels = _cast(x, events, labels, self.dtype)
        return x, events, labels, idx

    def chunk(self, start, end):
        parts = [self.window(k) for k in range(start, end)]
        x = torch.stack([part[0] for part in parts])
        events = {key: torch.cat([part[1][key] for part in parts]) for key in ("b", "y", "x", "p", "t_local")}
        events["t"] = torch.cat([torch.full((int(part[2].shape[0]),), i, dtype=torch.long, device=self.device)
                                 for i, part in enumerate(parts)])
        labels = torch.cat([part[2] for part in parts])
        idx = np.concatenate([part[3] for part in parts])
        return x, events, labels, idx


class TorchWindowSource(object):
    """事件常驻设备的窗口数据来源。

    构造时（每条序列一次）：按时间排序后的事件及其"整窗通道 / 分箱通道"的平面下标上传到设备。
    每个片段：一次 index_put_(accumulate=True) 累加 T 个窗口的计数，再按通道查表归一化。
    不用 scatter_add_ / bincount：torch 1.9 在确定性模式下它们会报错；index_put_(accumulate=True)
    在读出层高级索引的反向中本来就在使用，确定性模式下可用。计数是整数，累加顺序不影响结果。
    """

    def __init__(self, seq, cfg, q99, device, dtype=torch.float32):
        self.seq, self.device, self.dtype = seq, device, dtype
        self.height, self.width = int(cfg["pad_height"]), int(cfg["pad_width"])
        self.channels = num_input_channels(cfg["time_bins"])
        self.plane = self.height * self.width
        self.order, self.bounds = seq.order, seq.bounds
        order = seq.order
        window = np.repeat(np.arange(seq.n_windows, dtype=np.int64), np.diff(seq.bounds))
        x, y = seq.x[order].astype(np.int64), seq.y[order].astype(np.int64)
        negative = (seq.p[order] == 0).astype(np.int64)             # 正极性 -> 0，负极性 -> 1
        pixel = y * self.width + x
        whole_key = negative * self.plane + pixel                    # 通道 0 / 1（与 count_channels 相同约定）
        bin_key = (2 + 2 * seq.inner_bin[order].astype(np.int64) + negative) * self.plane + pixel

        def upload(array, tensor_dtype):
            return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=tensor_dtype)

        self.window_of = upload(window, torch.long)
        self.whole_key = upload(whole_key, torch.long)
        self.bin_key = upload(bin_key, torch.long)
        self.x, self.y = upload(x, torch.long), upload(y, torch.long)
        self.p = upload(seq.p[order].astype(np.float32), torch.float32)
        self.t_local = upload(seq.t_local[order], torch.float32)
        self.label = upload(seq.label[order], torch.float32)
        table, self.n_sat = normalization_table(q99, cfg["input_clip"])
        self.table = upload(table.reshape(-1), torch.float32)
        self.table_base = upload(np.arange(self.channels, dtype=np.int64) * (self.n_sat + 1),
                                 torch.long).view(1, 1, self.channels, 1, 1)

    def chunk(self, start, end):
        a, b = int(self.bounds[start]), int(self.bounds[end])
        steps = int(end) - int(start)
        if steps <= 0:
            raise ValueError("片段至少需要 1 个窗口")
        volume = self.channels * self.plane
        counts = torch.zeros(steps * volume, dtype=torch.float32, device=self.device)
        if b > a:
            base = (self.window_of[a:b] - int(start)) * volume
            index = torch.cat([base + self.whole_key[a:b], base + self.bin_key[a:b]])
            counts.index_put_((index,), torch.ones(int(index.shape[0]), dtype=torch.float32, device=self.device),
                              accumulate=True)
        lookup = counts.to(torch.long).clamp_(max=self.n_sat).view(steps, 1, self.channels, self.height, self.width)
        x = self.table[lookup + self.table_base]
        events = {"t": self.window_of[a:b] - int(start),
                  "b": torch.zeros(b - a, dtype=torch.long, device=self.device),
                  "y": self.y[a:b], "x": self.x[a:b], "p": self.p[a:b], "t_local": self.t_local[a:b]}
        x, events, labels = _cast(x, events, self.label[a:b], self.dtype)
        return x, events, labels, self.order[a:b]

    def window(self, k):
        x, events, labels, idx = self.chunk(k, k + 1)
        del events["t"]
        return x[0], events, labels, idx
