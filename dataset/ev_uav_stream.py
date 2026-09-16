"""流式 SNN V1 的 PyTorch 数据接口：按序列加载 NPZ，并把窗口转换成网络输入张量。

一个样本 = 一个完整 8 秒序列（StreamSequence，numpy）；窗口在训练/评估循环中按时间顺序逐个构造。
DataLoader 的 worker 只做 numpy 读取与校验，不接触 CUDA。
"""
import os

import numpy as np
import torch

from dataset.stream_windows import build_window_input, load_npz_events


class EvUAVStream(torch.utils.data.Dataset):
    """按序列读取某个划分（train/val/test）下的 NPZ 文件。

    参数:
        root   数据集根目录（其下有 train/val/test 子目录）
        split  划分名
        cfg    展平后的配置字典（需要 height/width_px/window_ms/n_windows/time_bins）
        names  可选的文件名子集（例如训练子集、校准序列、单序列过拟合）
    __getitem__(i) 返回 StreamSequence。
    """

    def __init__(self, root, split, cfg, names=None):
        self.directory = os.path.join(root, split)
        available = sorted(n for n in os.listdir(self.directory) if n.endswith(".npz"))
        if not available:
            raise RuntimeError("目录中没有 NPZ 文件: %s" % self.directory)
        if names is not None:
            missing = sorted(set(names) - set(available))
            if missing:
                raise RuntimeError("找不到文件: %s" % missing[:5])
            available = sorted(names)
        self.names = available
        self.cfg = cfg

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        c = self.cfg
        return load_npz_events(os.path.join(self.directory, self.names[index]),
                               c["height"], c["width_px"], c["window_ms"],
                               c["n_windows"], c["time_bins"])


def first_item_collate(batch):
    """DataLoader 的 collate：batch_size=1 时直接取出唯一的序列对象。"""
    return batch[0]


def make_sequence_loader(dataset, shuffle, seed, num_workers):
    """构造按序列迭代的 DataLoader（batch_size=1）。

    shuffle=True 时用固定种子的生成器打乱序列顺序（每个 epoch 顺序不同但可复现）；
    序列内部的窗口顺序永远不打乱。
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle,
                                       num_workers=int(num_workers),
                                       collate_fn=first_item_collate, generator=generator)


def window_to_device(seq, k, q99, cfg, device):
    """把序列的第 k 个窗口转换成网络输入。

    输出: (x, events, labels, idx)
        x       float32 [1, 12, pad_height, pad_width]
        events  字典 b/y/x (long [N])、p/t_local (float [N])，N 为本窗事件数（可为 0）
        labels  float32 [N]
        idx     numpy 原始事件下标，用于把预测回填到文件顺序
    """
    inp, idx = build_window_input(seq, k, q99, cfg["time_bins"], cfg["pad_height"],
                                  cfg["pad_width"], cfg["input_clip"])
    x = torch.from_numpy(inp).unsqueeze(0).to(device)
    n = int(idx.shape[0])
    events = {
        "b": torch.zeros(n, dtype=torch.long, device=device),
        "y": torch.from_numpy(seq.y[idx]).to(device=device, dtype=torch.long),
        "x": torch.from_numpy(seq.x[idx]).to(device=device, dtype=torch.long),
        "p": torch.from_numpy(seq.p[idx].astype(np.float32)).to(device),
        "t_local": torch.from_numpy(seq.t_local[idx]).to(device),
    }
    labels = torch.from_numpy(seq.label[idx]).to(device)
    return x, events, labels, idx
