"""TIF 诊断（只评估，不训练、不修改 checkpoint）：同一划分上并排运行几个 checkpoint，看误差出在哪里、机制是否按预期工作。

用法（服务器，EV-UAV-v1 目录）：
    python tools/diagnose_tif.py --config configs/evisseg_stream_tif_notransport.yaml --split test --device cuda:0 \
        --run tif=log/stream_tif_notransport_seed37/best_val_iou_seed37.pt \
        --run v1=log/stream_v1_streams_b4n1_seed37/best_val_iou_seed37.pt \
        --out log/diagnose_tif_test.json
前两个 --run 做逐序列配对比较（A - B）。网络结构、q99 取自 checkpoint；数据根目录与阈值等评估口径取自 --config。
全部用 carry 模式逐窗执行（与 --mode eval 的 carry 相同），打印的 IoU 应与 eval_*.json 的 carry IoU 相同，用于核对。
（10-19 删除搬运之前的版本叫 tools/diagnose_tif_transport.py，其中的搬运流量统计随搬运一起删除。）

一、事件级（判正规则 prob >= threshold，与 window_confusion 相同）
  1. 逐序列 TP/FP/FN、IoU 与噪声密度（每窗负事件数）。
  2. FP 分三类（像素坐标，切比雪夫距离）：
       near   距本窗某个正事件 <= near_px                                  目标周边（边界、光晕）
       trail  不是 near，但距前 trail_windows 窗某个正事件 <= near_px        目标尾迹（目标离开后残留的响应）
       far    其余                                                        远离目标（背景噪声、闪烁）
  3. 目标速度分层：每窗每个目标（target_id > 0）取正事件质心，速度 = 与上一窗质心的距离（像素/窗），
     上一窗没有该目标记为 new。每层给出召回 TP/(TP+FN) 与局部 IoU TP/(TP+FN+FP_near)，
     FP_near = 距该目标本窗正事件 <= near_px 的 FP。
二、层内状态相对目标的位置（只对 TIF 层；前向钩子只读，不改变任何输出）
  按该层分辨率划分区域（本窗正事件映射到格子后外扩 1 格；一格 = enc1/dec1 1 px、enc2/dec2 2、enc3/dec3 4、enc4 8）：
       now    本窗目标所在格
       trail  前 trail_windows 窗目标所在格中、本窗不是目标的格
       bg     其余
  记忆 M（本窗更新后）在三个区域的占比，以及密度比 = 占比 / 面积占比（now 越高、trail 越低越好）。
三、逐序列消息率：每个序列各层平均 |Δq|，以及它与每窗负事件数的秩相关（杂波越多消息越多 = 相关为正）。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from dataset.stream_windows import refill_by_index  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES  # noqa: E402
from model.lif2d_stream import StreamingTIF2d  # noqa: E402
from train_stream_v1 import build_net, seed_everything  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402
from utils.stream_metrics import iou_from_counts, window_confusion  # noqa: E402

STRIDE = dict(enc1=1, enc2=2, enc3=4, enc4=8, dec3=4, dec2=2, dec1=1)
SPEED_EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, math.inf)
REGIONS = ("now", "trail", "bg")
FP_KINDS = ("near", "trail", "far")
EVAL_KEYS = ("threshold", "pd_detT", "correct_thresh", "rolling_window", "segments", "timing_warmup_windows")
DATA_KEYS = ("height", "width_px", "pad_height", "pad_width", "window_ms", "n_windows", "time_bins", "input_clip")


def parse_args(argv=None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="TIF 搬运诊断（只评估）")
    parser.add_argument("--config", default="configs/evisseg_stream_tif.yaml",
                        help="只取数据根目录与评估口径（threshold 等）")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--run", action="append", required=True,
                        help="名称=checkpoint，可重复；前两个做逐序列配对比较（A - B）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--near-px", type=int, default=3, help="near / trail 的像素半径（切比雪夫距离）")
    parser.add_argument("--trail-windows", type=int, default=10, help="尾迹回看的窗数（默认 10 窗 = 0.5 s）")
    parser.add_argument("--max-sequences", type=int, default=0, help="只跑前几个序列（冒烟用），0 为全部")
    parser.add_argument("--out", required=True, help="结果 JSON 路径")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.near_px < 0 or args.trail_windows < 1 or args.max_sequences < 0:
        raise ValueError("--near-px >= 0，--trail-windows >= 1，--max-sequences >= 0")
    return args


def parse_runs(items):
    """把 名称=路径 列表解析为 [(名称, 路径)]，名称不可重复。"""
    runs = []
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError("--run 的格式是 名称=checkpoint，收到 %r" % item)
        if name in [n for n, _ in runs]:
            raise ValueError("--run 名称重复: %s" % name)
        runs.append((name, path))
    return runs


def load_run(path, cfg, device):
    """按 --mode eval 的规则加载 checkpoint，返回 (net, train_cfg, q99, epoch)。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到 checkpoint: %s" % path)
    ckpt = torch.load(path, map_location=device)
    train_cfg = dict(ckpt["config"])
    train_cfg["root"] = cfg["root"]
    for key in EVAL_KEYS:
        train_cfg[key] = cfg[key]
    train_cfg["execution"], train_cfg["input_device"] = "step", "cpu"
    seed_everything(int(train_cfg["seed"]), bool(train_cfg["deterministic"]))
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    net = build_net(train_cfg, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("checkpoint 中的增益未校准: %s" % path)
    net.eval()
    return net, train_cfg, q99, ckpt.get("epoch")


# ---------------------------------------------------------------------------
# 层内：目标区域与前向钩子
# ---------------------------------------------------------------------------


class TargetRegions(object):
    """每窗按层分辨率给出目标区域 (now, trail)，[1,1,h,w] 的 0/1 浮点图，两者不重叠。

    调用顺序：reset()（每条序列）→ 每窗 set_window(正事件 y, x) → 前向（钩子调用 masks）→ advance()。
    """

    def __init__(self, trail_windows):
        self.trail_windows = int(trail_windows)
        self.history, self.current = {}, {}
        self.ys = self.xs = None

    def reset(self):
        self.history, self.current = {}, {}

    def set_window(self, ys, xs):
        self.ys, self.xs = ys, xs
        self.current = {}

    def masks(self, stride, height, width, device):
        key = (int(stride), int(height), int(width))
        if key not in self.current:
            cell = torch.zeros(height * width, dtype=torch.float32, device=device)
            if self.ys is not None and self.ys.numel():
                cy = torch.clamp(torch.div(self.ys, stride, rounding_mode="floor"), max=height - 1)
                cx = torch.clamp(torch.div(self.xs, stride, rounding_mode="floor"), max=width - 1)
                flat = cy * width + cx
                # 用 accumulate=True：torch 1.9 确定性模式下 CUDA 上可用的写法（同 TorchWindowSource）
                cell.index_put_((flat,), torch.ones(flat.shape[0], dtype=torch.float32, device=device),
                                accumulate=True)
            cell = torch.clamp(cell, max=1.0).view(1, 1, height, width)
            now = F.max_pool2d(cell, 3, stride=1, padding=1)
            trail = torch.zeros_like(now)
            for past in self.history.get(key, ()):
                trail = torch.maximum(trail, past)
            self.current[key] = (now, trail * (1.0 - now))
        return self.current[key]

    def advance(self):
        for key, (now, _) in self.current.items():
            self.history.setdefault(key, deque(maxlen=self.trail_windows)).append(now)


class LayerProbe(object):
    """在每层 TIF 神经元上挂前向钩子，只读输出，按区域累计记忆。"""

    def __init__(self, net, regions):
        self.regions = regions
        self.sums, self.handles = {}, []
        for name in LAYER_NAMES:
            neuron = getattr(net, name).neuron
            if isinstance(neuron, StreamingTIF2d):
                self.handles.append(neuron.register_forward_hook(self._hook(name)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @staticmethod
    def _add(store, key, value, masks, scale=1.0):
        for region, mask in zip(REGIONS, masks):
            amount = (mask.sum() * scale if value is None else (value * mask).sum()).double()
            name = key + "_" + region
            store[name] = store[name] + amount if name in store else amount

    def _hook(self, name):
        stride = STRIDE[name]

        def hook(module, inputs, output):
            with torch.no_grad():
                memory = output[2]
                now, trail = self.regions.masks(stride, memory.shape[-2], memory.shape[-1], memory.device)
                masks = (now, trail, 1.0 - now - trail)
                store = self.sums.setdefault(name, {})
                self._add(store, "memory", memory, masks)
                self._add(store, "area", None, masks, float(memory.shape[0] * memory.shape[1]))

        return hook

    def summary(self):
        """各层汇总：记忆与面积在三个区域的占比、密度比。"""
        out = {}
        for name, store in self.sums.items():
            s = {key: float(value) for key, value in store.items()}

            def total(key):
                return sum(s[key + "_" + r] for r in REGIONS)

            def share(key, region):
                t = total(key)
                return s[key + "_" + region] / t if t > 0 else None

            layer = {"memory_share": {r: share("memory", r) for r in REGIONS},
                     "area_share": {r: share("area", r) for r in REGIONS}}
            layer["memory_density"] = {r: (layer["memory_share"][r] / layer["area_share"][r]
                                           if layer["memory_share"][r] is not None and layer["area_share"][r] else None)
                                       for r in REGIONS}
            out[name] = layer
        return out


def run_sequence_probe(net, seq, cfg, q99, device, regions):
    """carry 模式逐窗推理一个序列（与 run_sequence 的 step 分支相同），每窗先告诉钩子本窗目标在哪里。"""
    source = make_window_source(seq, cfg, q99, device, next(net.parameters()).dtype)
    regions.reset()
    states = None
    index_parts, prob_parts = [], []
    for k in range(seq.n_windows):
        x, events, labels, idx = source.window(k)
        positive = labels == 1
        regions.set_window(events["y"][positive], events["x"][positive])
        logits, states, _ = net(x, events, states, state_mode="carry")
        regions.advance()
        index_parts.append(idx)
        prob_parts.append(torch.sigmoid(logits).float().cpu().numpy())
    return refill_by_index(seq.n_events, index_parts, prob_parts)


# ---------------------------------------------------------------------------
# 事件级：FP 分类与目标速度
# ---------------------------------------------------------------------------


def dilate(mask, radius):
    """二值图的方形膨胀（切比雪夫半径 radius），可分离实现。"""
    out = mask.copy()
    for d in range(1, radius + 1):
        out[d:, :] |= mask[:-d, :]
        out[:-d, :] |= mask[d:, :]
    rows = out.copy()
    for d in range(1, radius + 1):
        out[:, d:] |= rows[:, :-d]
        out[:, :-d] |= rows[:, d:]
    return out


def count_near(fx, fy, tx, ty, radius):
    """(fx, fy) 中距 (tx, ty) 任一点的切比雪夫距离 <= radius 的个数（在目标外接框内局部膨胀）。"""
    if fx.size == 0 or tx.size == 0:
        return 0
    x0, x1 = int(tx.min()) - radius, int(tx.max()) + radius
    y0, y1 = int(ty.min()) - radius, int(ty.max()) + radius
    box = (fx >= x0) & (fx <= x1) & (fy >= y0) & (fy <= y1)
    if not box.any():
        return 0
    local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    local[ty - y0, tx - x0] = True
    local = dilate(local, radius)
    return int(local[fy[box] - y0, fx[box] - x0].sum())


def analyze_sequence(seq, probs, threshold, height, width, near_px, trail_windows):
    """一个序列的事件级统计：TP/FP/FN、FP 三类、每窗每个目标的速度与 TP/FN/FP_near。"""
    pred = probs >= float(threshold)
    pos = seq.label == 1
    counts = np.zeros(3, dtype=np.int64)
    kinds = dict.fromkeys(FP_KINDS, 0)
    history = deque(maxlen=int(trail_windows))
    last, targets = {}, []
    for k in range(seq.n_windows):
        idx = seq.window_index(k)
        pk, dk = pos[idx], pred[idx]
        xs, ys = seq.x[idx], seq.y[idx]
        counts += np.asarray(window_confusion(probs[idx], seq.label[idx], threshold)[:3], dtype=np.int64)
        pos_map = np.zeros((height, width), dtype=bool)
        pos_map[ys[pk], xs[pk]] = True
        near = dilate(pos_map, near_px)
        trail = np.zeros_like(near)
        for past in history:
            trail |= past
        trail &= ~near
        history.append(near)
        false_pos = dk & ~pk
        fx, fy = xs[false_pos], ys[false_pos]
        is_near, is_trail = near[fy, fx], trail[fy, fx]
        kinds["near"] += int(is_near.sum())
        kinds["trail"] += int(is_trail.sum())
        kinds["far"] += int((~is_near & ~is_trail).sum())
        tids = seq.target_id[idx]
        for tid in np.unique(tids[pk & (tids > 0)]):
            sel = pk & (tids == tid)
            cx, cy = float(xs[sel].mean()), float(ys[sel].mean())
            prev = last.get(tid)
            speed = math.hypot(cx - prev[1], cy - prev[2]) if prev is not None and prev[0] == k - 1 else None
            last[tid] = (k, cx, cy)
            targets.append({"window": k, "target": float(tid), "speed": speed, "pos": int(sel.sum()),
                             "tp": int((sel & dk).sum()), "fn": int((sel & ~dk).sum()),
                             "fp_near": count_near(fx, fy, xs[sel], ys[sel], near_px)})
    tp, fp, fn = (int(v) for v in counts)
    n_windows = max(seq.n_windows, 1)
    return {"name": seq.name, "windows": seq.n_windows,
            "events_per_window": seq.n_events / float(n_windows),
            "neg_per_window": float((~pos).sum()) / n_windows, "pos_per_window": float(pos.sum()) / n_windows,
            "pos_without_target": int((pos & (seq.target_id <= 0)).sum()),
            "tp": tp, "fp": fp, "fn": fn, "iou": iou_from_counts(tp, fp, fn),
            "fp_near": kinds["near"], "fp_trail": kinds["trail"], "fp_far": kinds["far"],
            "targets": targets}


def speed_table(records):
    """按目标速度（像素/窗）分层汇总召回与局部 IoU；new = 上一窗没有该目标。"""
    bins = [("new", None, None)] + [("%g-%g" % (a, b), a, b) for a, b in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:])]
    rows = []
    for label, a, b in bins:
        if a is None:
            sel = [r for r in records if r["speed"] is None]
        else:
            sel = [r for r in records if r["speed"] is not None and a <= r["speed"] < b]
        tp, fn, fp = (sum(r[key] for r in sel) for key in ("tp", "fn", "fp_near"))
        rows.append({"bin": label, "target_windows": len(sel), "pos": tp + fn, "tp": tp, "fn": fn, "fp_near": fp,
                     "recall": tp / float(tp + fn) if tp + fn else None,
                     "local_iou": tp / float(tp + fn + fp) if tp + fn + fp else None})
    return rows


def average_ranks(values):
    """秩（从 0 开始），并列取平均秩。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def spearman(a, b):
    """Spearman 秩相关（并列取平均秩）。有效样本少于 3 个或某一方全部并列时返回 None。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return None
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def paired(name_a, a, name_b, b):
    """逐序列配对比较 A - B：ΔFP（分三类）、ΔFN、ΔIoU，以及它们与噪声密度的秩相关。"""
    rows = []
    for sa, sb in zip(a, b):
        if sa["name"] != sb["name"]:
            raise AssertionError("配对的序列顺序不一致")
        row = {"name": sa["name"], "neg_per_window": sa["neg_per_window"], "pos_per_window": sa["pos_per_window"],
               "iou_a": sa["iou"], "iou_b": sb["iou"]}
        for key in ("tp", "fp", "fn", "fp_near", "fp_trail", "fp_far"):
            row["d_" + key] = sa[key] - sb[key]
        row["d_iou"] = sa["iou"] - sb["iou"]
        rows.append(row)
    keys = ("d_fp", "d_fp_near", "d_fp_trail", "d_fp_far", "d_fn", "d_iou")
    total = {key: sum(r[key] for r in rows if math.isfinite(r[key])) for key in keys[:-1]}
    ranked = sorted(rows, key=lambda r: -r["d_fp"])
    positive = sum(max(r["d_fp"], 0) for r in rows)
    top3 = sum(max(r["d_fp"], 0) for r in ranked[:3])
    return {"a": name_a, "b": name_b, "total": total,
            "top3_share_of_positive_d_fp": top3 / float(positive) if positive > 0 else None,
            "spearman_vs_neg_per_window": {key: spearman([r["neg_per_window"] for r in rows], [r[key] for r in rows])
                                           for key in keys},
            "sequences": ranked}


def message_vs_noise(seqs):
    """各层（及 7 层平均）逐序列 |Δq| 与每窗负事件数的秩相关、全部序列的平均 |Δq|；非 TIF 网络返回空字典。"""
    layers = [layer for layer in LAYER_NAMES if seqs and all(layer in s.get("message_abs_rate", {}) for s in seqs)]
    if not layers:
        return {}
    noise = [s["neg_per_window"] for s in seqs]
    out = {}
    for layer in layers + ["mean"]:
        if layer == "mean":
            rates = [float(np.mean([s["message_abs_rate"][n] for n in layers])) for s in seqs]
        else:
            rates = [s["message_abs_rate"][layer] for s in seqs]
        out[layer] = {"spearman_vs_neg_per_window": spearman(noise, rates), "mean_rate": float(np.mean(rates))}
    return out


def clean(obj):
    """把 NaN / inf 换成 None，保证 JSON 合法。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(value) for value in obj]
    return obj


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------


def fmt(value, spec="%.3f"):
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else spec % value


def print_run(name, run):
    t = run["totals"]
    print("== %s（epoch %s）IoU %s  TP %d FP %d FN %d  （应与 eval json 的 carry IoU 相同）"
          % (name, run["epoch"], fmt(t["iou"], "%.4f"), t["tp"], t["fp"], t["fn"]), flush=True)
    fp = max(t["fp"], 1)
    print("   FP 分类: " + "  ".join("%s %d (%.1f%%)" % (k, run["fp_kinds"][k], 100.0 * run["fp_kinds"][k] / fp)
                                  for k in FP_KINDS))
    print("   速度分层（像素/窗）  目标窗数  正事件  召回    局部IoU  FP_near")
    for row in run["speed"]:
        print("     %-8s %8d %8d  %s  %s  %7d" % (row["bin"], row["target_windows"], row["pos"],
                                                 fmt(row["recall"]), fmt(row["local_iou"]), row["fp_near"]))
    if run["layers"]:
        print("   层内记忆（now / trail / bg 占比；密度比 now、trail = 占比 / 面积占比）")
        for layer in LAYER_NAMES:
            if layer not in run["layers"]:
                continue
            s = run["layers"][layer]
            line = "     %-5s M %s / %s / %s  密度 %s %s" % (
                layer, fmt(s["memory_share"]["now"]), fmt(s["memory_share"]["trail"]), fmt(s["memory_share"]["bg"]),
                fmt(s["memory_density"]["now"], "%.1f"), fmt(s["memory_density"]["trail"], "%.1f"))
            print(line)
    if run.get("message_vs_noise"):
        m = run["message_vs_noise"]
        print("   逐序列 |Δq| 与每窗负事件数的秩相关（括号内为平均 |Δq|）: " + "  ".join(
            "%s %s (%s)" % (k, fmt(v["spearman_vs_neg_per_window"], "%+.2f"), fmt(v["mean_rate"], "%.4f"))
            for k, v in m.items()))


def print_paired(p):
    print("== 配对 %s - %s：ΔFP %+d（near %+d, trail %+d, far %+d），ΔFN %+d，ΔFP>0 的部分前 3 个序列占 %s" % (
        p["a"], p["b"], p["total"]["d_fp"], p["total"]["d_fp_near"], p["total"]["d_fp_trail"],
        p["total"]["d_fp_far"], p["total"]["d_fn"], fmt(p["top3_share_of_positive_d_fp"], "%.2f")))
    print("   与每窗负事件数的秩相关: " + "  ".join("%s %s" % (k, fmt(v, "%+.2f"))
                                         for k, v in p["spearman_vs_neg_per_window"].items()))
    print("   ΔFP 最大的序列: 名称  负事件/窗  ΔFP (near/trail/far)  ΔFN  IoU A / B")
    for r in p["sequences"][:8]:
        print("     %-28s %8.0f  %+6d (%+d/%+d/%+d)  %+6d  %s / %s" % (
            r["name"], r["neg_per_window"], r["d_fp"], r["d_fp_near"], r["d_fp_trail"], r["d_fp_far"], r["d_fn"],
            fmt(r["iou_a"], "%.4f"), fmt(r["iou_b"], "%.4f")))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv=None):
    """加载全部 checkpoint → 逐序列、逐 checkpoint 推理并统计 → 写 JSON、打印摘要。返回报告字典。"""
    args = parse_args(argv)
    runs = parse_runs(args.run)
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise RuntimeError("结果已存在: %s（使用 --overwrite 覆盖）" % out_path)
    cfg = load_flat_config(args.config)
    device = torch.device(args.device)
    loaded = {name: load_run(path, cfg, device) for name, path in runs}       # 先全部加载，坏路径尽早报错
    data_cfg = loaded[runs[0][0]][1]
    for name, (_, train_cfg, _, _) in loaded.items():
        diff = [key for key in DATA_KEYS if train_cfg.get(key) != data_cfg.get(key)]
        if diff:
            raise ValueError("%s 与 %s 的数据配置不同: %s" % (name, runs[0][0], diff))
    threshold = float(cfg["threshold"])
    height, width = int(data_cfg["height"]), int(data_cfg["width_px"])
    dataset = EvUAVStream(cfg["root"], args.split, data_cfg)
    n_seq = len(dataset) if not args.max_sequences else min(len(dataset), args.max_sequences)
    probes, collected = {}, {}
    for name, (net, _, _, _) in loaded.items():
        regions = TargetRegions(args.trail_windows)
        probes[name] = (regions, LayerProbe(net, regions))
        collected[name] = {"sequences": [], "targets": []}
    with torch.no_grad():
        for i in range(n_seq):
            seq = dataset[i]
            for name, (net, train_cfg, q99, _) in loaded.items():
                net.start_tif_stats()                       # 非 TIF 网络不做任何事
                probs = run_sequence_probe(net, seq, train_cfg, q99, device, probes[name][0])
                tif_stats = net.stop_tif_stats()
                record = analyze_sequence(seq, probs, threshold, height, width, args.near_px, args.trail_windows)
                record["message_abs_rate"] = {layer: v["message_abs_rate"] for layer, v in tif_stats.items() if v}
                record["speed"] = speed_table(record["targets"])      # 逐序列的速度分层（区分慢目标与噪声密集序列）
                collected[name]["targets"] += record.pop("targets")
                collected[name]["sequences"].append(record)
            print("[%d/%d] %s" % (i + 1, n_seq, seq.name), flush=True)
    report = {"split": args.split, "config": args.config, "threshold": threshold, "near_px": args.near_px,
              "trail_windows": args.trail_windows, "n_sequences": n_seq, "runs": {}}
    for name, path in runs:
        net, train_cfg, _, epoch = loaded[name]
        seqs = collected[name]["sequences"]
        tp, fp, fn = (sum(s[key] for s in seqs) for key in ("tp", "fp", "fn"))
        report["runs"][name] = {
            "checkpoint": path, "epoch": epoch, "neuron": train_cfg["neuron"], "tif": net.tif_config,
            "totals": {"tp": tp, "fp": fp, "fn": fn, "iou": iou_from_counts(tp, fp, fn)},
            "fp_kinds": {k: sum(s["fp_" + k] for s in seqs) for k in FP_KINDS},
            "speed": speed_table(collected[name]["targets"]),
            "layers": probes[name][1].summary(),
            "message_vs_noise": message_vs_noise(seqs),
            "sequences": seqs}
        probes[name][1].remove()
    if len(runs) >= 2:
        (na, _), (nb, _) = runs[0], runs[1]
        report["paired"] = paired(na, collected[na]["sequences"], nb, collected[nb]["sequences"])
    report = clean(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, ensure_ascii=False, allow_nan=False)
    for name, _ in runs:
        print_run(name, report["runs"][name])
    if "paired" in report:
        print_paired(report["paired"])
    print("DIAGNOSE FINISHED:", out_path, flush=True)
    return report


if __name__ == "__main__":
    main()
