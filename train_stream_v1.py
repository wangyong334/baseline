"""V5 runtime helpers retained for the V15 feature-revision experiment.

The historical training CLI and abandoned V2--V13 experiment branches have been
removed. Use tools/v15/train.py for training/evaluation. Native input conversion,
calibration, stream planning and layer-parallel TBPTT assembly retain V5 behavior.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import json
import random
from pathlib import Path
import numpy as np
import torch
from dataset.ev_uav_stream import EvUAVStream, window_to_device
from dataset.stream_source import make_window_source
from dataset.stream_windows import num_input_channels
from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream, calibrate_gains
from model.lif2d_stream import StreamingLIF2d
from utils.stream_common import select_subset

STREAM_LAYOUTS = ("greedy", "batchify")


def seed_everything(seed, deterministic):
    """固定 Python/numpy/torch 随机种子；deterministic=True 时要求确定性算法。

    若冒烟测试报 "does not have a deterministic implementation"，把 YAML 中 deterministic 改为 false。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def load_stats(cfg):
    """读取训练集统计 JSON（由 tools/stream_train_stats.py 生成），返回 (stats, q99, pos_weight)。

    pos_weight 按规则 min(train_neg/train_pos, pos_weight_cap) 重新计算并与 JSON 核对。
    """
    with open(cfg["stats_path"], "r", encoding="utf-8") as stream:
        stats = json.load(stream)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if q99.shape[0] != 2 + 2 * int(cfg["time_bins"]):
        raise ValueError("统计文件的 q99 通道数与配置不一致")
    pos_weight = min(float(stats["train_neg"]) / float(stats["train_pos"]), float(cfg["pos_weight_cap"]))
    if abs(pos_weight - float(stats["pos_weight"])) > 1e-9:
        raise ValueError("统计文件中的 pos_weight 与当前 cap 规则不一致，请重新生成统计文件")
    return stats, q99, pos_weight


def build_net(cfg, device):
    """Construct the unchanged V5 backbone (LIF controls remain available).

    Historical decoder/memory/normalizer experiments cannot silently enter a
    V15 run. The parameter creation order matches the original V5 constructor.
    """
    for key in ("spiking_decoder", "merged_decoder", "dual_memory",
                "subthreshold_residual", "invariant_input"):
        if cfg.get(key):
            raise ValueError("V15 does not support the legacy option: " + key)
    if cfg.get("border", "zero") != "zero":
        raise ValueError("V15 preserves V5 zero-padding in its backbone")
    if cfg.get("tif", {}).get("adapt", False):
        raise ValueError("V15 uses base BIR, not the retired adaptive-input variant")
    net = EvSpSegNetStream(
        in_channels=2 + 2 * int(cfg["time_bins"]), channels=tuple(cfg["channels"]),
        neuron=cfg["neuron"], norm=cfg["norm"], state_mode=cfg["state_mode"],
        readout_hidden=int(cfg["readout_hidden"]), dt_ms=float(cfg["window_ms"]),
        tau_init_ms=float(cfg["tau_init_ms"]), tau_min_ms=float(cfg["tau_min_ms"]),
        tau_max_ms=float(cfg["tau_max_ms"]), v_threshold=float(cfg["v_threshold"]),
        tif=cfg.get("tif") if cfg["neuron"] == "tif" else None)
    return net.to(device)


def train_file_names(cfg):
    """Sorted training-only records; V15 does not use old cross-fitting stages."""
    return sorted(p.name for p in (Path(cfg["root"]) / "train").glob("*.npz"))


def write_json(path, payload):
    """以 UTF-8、缩进格式写 JSON。"""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def calibration_factory(cfg, q99, device):
    """构造增益校准的数据来源：固定种子选出的若干训练序列，每个序列取前 calib_windows 个连续窗口。

    返回 (factory, names)。factory() 每次调用都重新产出相同的数据，供逐层校准反复使用。
    """
    names = select_subset(train_file_names(cfg), int(cfg["calib_sequences"]), int(cfg["seed"]) + 1000)
    dataset = EvUAVStream(cfg["root"], "train", cfg, names)
    n_windows = int(cfg["calib_windows"])

    def windows_of(seq):
        """按时间顺序产出一个序列前 n_windows 个窗口的 (x, events)。"""
        for k in range(min(n_windows, seq.n_windows)):
            x, events, _, _ = window_to_device(seq, k, q99, cfg, device)
            yield x, events

    def factory():
        """每次调用都从头产出全部校准序列（逐层校准会调用 7 次）。"""
        for i in range(len(dataset)):
            yield windows_of(dataset[i])

    return factory, names


def run_calibration(net, cfg, q99, device, out_dir):
    """执行一次性增益校准，并把报告写到 out_dir/calibration.json。"""
    factory, names = calibration_factory(cfg, q99, device)
    reports = calibrate_gains(net, factory, float(cfg["calib_quantile"]),
                              int(cfg["calib_samples_per_channel"]), int(cfg["calib_min_positive"]),
                              float(cfg["calib_gain_min"]), float(cfg["calib_gain_max"]),
                              int(cfg["seed"]) + 4000)
    write_json(Path(out_dir) / "calibration.json", {"sequences": names, "layers": reports})
    for r in reports:
        g = np.asarray(r["gain"])
        print("[calib] %-5s gain min/mean/max = %.3f / %.3f / %.3f  fallback=%d/%d" % (
            r["layer"], g.min(), g.mean(), g.max(), sum(r["fallback"]), len(r["fallback"])), flush=True)
    return reports


def tau_statistics(net):
    """返回每个 LIF 层的 tau/beta 最小/平均/最大值，以及贴近上下界的通道占比（ReLU 版本返回空字典）。"""
    out = {}
    for name, block in zip(LAYER_NAMES, net.blocks()):
        neuron = block.neuron
        if not isinstance(neuron, StreamingLIF2d):
            continue
        with torch.no_grad():
            tau = neuron.tau().double().cpu()
            beta = neuron.beta().double().cpu()
        margin = 0.02 * (neuron.tau_max_ms - neuron.tau_min_ms)
        out[name] = {"tau_min": float(tau.min()), "tau_mean": float(tau.mean()), "tau_max": float(tau.max()),
                     "beta_min": float(beta.min()), "beta_mean": float(beta.mean()),
                     "beta_max": float(beta.max()),
                     "near_min_frac": float((tau < neuron.tau_min_ms + margin).double().mean()),
                     "near_max_frac": float((tau > neuron.tau_max_ms - margin).double().mean())}
    return out


def plan_streams(lengths, n_streams, k, rng, layout="greedy"):
    """把训练录像分配给 B 条并行流。lengths[i] 为数据集第 i 条录像的窗数；先按 rng 打乱录像顺序。

    layout="greedy"（默认，EV-UAV / EV-Flying 的结果都用它）：整段录像依次放进当前总窗数最少的流。
        最长的录像不超过"总窗数 / B"时各流长度接近；超过时（如 EBSSA 一段占训练集 29%）分不平，
        一轮末尾只剩一条流在跑，连续很多次更新来自同一段录像。
    layout="batchify"（语言模型 TBPTT 的标准排法）：打乱后首尾接成一条长带，等分成 B 段，每条流一样长、同时结束；
        跨等分点的录像被切开，后半截作为新起点（该流状态清零），每轮最多切 B-1 刀。
    每条流开头随机延迟 0..k-1 个空窗（随机片段相位）。录像内部的窗顺序不变。
    返回: [(delay, items), ...]，共 n_streams 条；items 元素为 (index, n_windows)（整段）或 (index, first, n_windows)
    （从录像第 first 窗起的 n_windows 窗）。
    """
    if layout not in STREAM_LAYOUTS:
        raise ValueError("stream_layout 必须是 %s 之一" % (STREAM_LAYOUTS,))
    order = rng.permutation(len(lengths))
    n_streams = int(n_streams)
    streams = [[] for _ in range(n_streams)]
    if layout == "greedy":
        totals = [0] * n_streams
        for i in order:
            s = int(np.argmin(totals))
            streams[s].append((int(i), int(lengths[i])))
            totals[s] += int(lengths[i])
    else:
        total = int(sum(int(n) for n in lengths))
        bounds = [s * total // n_streams for s in range(n_streams + 1)]
        pos, s = 0, 0
        for i in order:
            n, first = int(lengths[i]), 0
            while first < n:
                while pos >= bounds[s + 1]:
                    s += 1
                take = min(n - first, bounds[s + 1] - pos)
                streams[s].append((int(i), first, take))
                first += take
                pos += take
    return [(int(rng.randint(0, int(k))), items) for items in streams]


def plan_item(item):
    """计划里的一项 -> (index, first, n_windows)。"""
    return (item[0], 0, item[1]) if len(item) == 2 else tuple(item)


def stream_total_windows(plan):
    """各流时间轴长度（含开头延迟）的最大值，即一轮需要的窗数。"""
    return max(delay + sum(item[-1] for item in items) for delay, items in plan)


def stream_pieces(plan, chunk_start, chunk_end):
    """流时间轴上窗 [chunk_start, chunk_end) 这一片段内，每条流由哪些录像的哪些窗组成。

    返回: 每条流一个列表 [(index, a, b, offset, new_start)]：录像 index 的第 a..b-1 窗放在片段内第 offset 窗起；
          new_start=True 表示计划中的这一项从这一窗开始（整段录像的开头，或 batchify 切开后的后半截；该流状态在此清零）。
          没有录像的位置是空窗（零输入、无事件）。
    """
    out = []
    for delay, items in plan:
        pieces, start = [], delay
        for item in items:
            index, first, n = plan_item(item)
            if start >= chunk_end:
                break
            end = start + n
            if end > chunk_start:
                lo, hi = max(chunk_start, start), min(chunk_end, end)
                pieces.append((index, first + lo - start, first + hi - start, lo - chunk_start, lo == start))
            start = end
        out.append(pieces)
    return out


class StreamSources(object):
    """多流训练按需读取录像并构造窗口数据来源；录像的最后一窗用完即释放。"""

    def __init__(self, dataset, cfg, q99, device, dtype):
        self.dataset, self.cfg, self.q99, self.device, self.dtype = dataset, cfg, q99, device, dtype
        self.sources, self.bounds = {}, {}

    def get(self, index):
        if index not in self.sources:
            seq = self.dataset[index]
            self.sources[index] = make_window_source(seq, self.cfg, self.q99, self.device, self.dtype)
            self.bounds[index] = seq.bounds
        return self.sources[index]

    def events(self, index, lo, hi):
        """录像 index 第 lo..hi-1 窗的事件数（不经过网络，只读窗口边界）。"""
        self.get(index)
        return int(self.bounds[index][hi] - self.bounds[index][lo])

    def release(self, index):
        self.sources.pop(index, None)
        self.bounds.pop(index, None)


def assemble_streams(sources, pieces, steps, cfg, device, dtype):
    """把各流的片段组成拼成一个批：x [T,B,C,H,W]、events（b = 流号，t = 片段内窗号）、labels [N]、keep [T,B]。"""
    batch = len(pieces)
    x = torch.zeros((steps, batch, num_input_channels(cfg["time_bins"]), int(cfg["pad_height"]),
                     int(cfg["pad_width"])), dtype=dtype, device=device)
    keep = torch.ones((steps, batch), dtype=dtype, device=device)
    parts, labels = {key: [] for key in ("t", "b", "y", "x", "p", "t_local")}, []
    for b, stream in enumerate(pieces):
        for index, lo, hi, offset, new_start in stream:
            xs, events, lab, _ = sources.get(index).chunk(lo, hi)
            x[offset:offset + (hi - lo), b] = xs[:, 0]
            if new_start:
                keep[offset, b] = 0
            parts["t"].append(events["t"] + int(offset))
            parts["b"].append(torch.full_like(events["t"], b))
            for key in ("y", "x", "p", "t_local"):
                parts[key].append(events[key])
            labels.append(lab)
    if labels:
        events = {key: torch.cat(value) for key, value in parts.items()}
        labels = torch.cat(labels)
    else:
        long_empty = torch.zeros(0, dtype=torch.long, device=device)
        float_empty = torch.zeros(0, dtype=dtype, device=device)
        events = {"t": long_empty, "b": long_empty, "y": long_empty, "x": long_empty,
                  "p": float_empty, "t_local": float_empty}
        labels = float_empty
    return x, events, labels, keep


def stream_valid(pieces, steps, device, dtype):
    """[T,B]：该流在片段内该窗有录像为 1，空窗（流错开起点、流尾补齐）为 0。"""
    valid = torch.zeros((steps, len(pieces)), dtype=dtype, device=device)
    for b, stream in enumerate(pieces):
        for _, lo, hi, offset, _ in stream:
            valid[offset:offset + (hi - lo), b] = 1
    return valid
