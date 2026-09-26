"""诊断脉冲骨干的膜电位分布与前端特征尺度（不训练、不改模型，只读一个 checkpoint）。

动机：评估日志里的 `big_membrane_frac` 统计的是 |U_pre| > 10*v_th 的占比，**不分正负**。
正向饱和（一直超阈值发放）与负向深抑制（被压到很深、永远发不出来）需要完全不同的修法：
    正向饱和 -> 膜电位裁剪 / ALIF 自适应阈值
    负向抑制 -> 输入尺度问题（前端把 log1p(count)、age、dipole 三种量纲混在一起喂进 enc1）
本脚本把两者分开，并给出前端每个特征通道的实际动态范围，用来决定"改神经元"还是"改归一化"。

顺带回答第三件事的前置问题：每条序列里有没有"目标尚未出现 / 已经离开"的窗（纯背景告警率实验的前提）。

输出（JSON + 终端表格）：
    layers[层名]  分位数（以 v_th 为单位）、正负两侧的越界占比、超阈值占比、深抑制占比、逐窗发放率曲线
    features[通道名]  分位数、min/max、非零占比
    background   mu0 的分位数
    windows      每条序列的总窗数 / 有目标事件的窗数 / 纯背景窗数

用法:
    python tools/diagnose_membrane.py --config configs/evisseg_stream_v2.yaml \
        --checkpoint log/v21_seed37/best_val_iou_seed37.pt --split val --sequences 4 \
        --out log/diagnose/membrane_s37_val.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES  # noqa: E402
from train_stream_v2 import build_all, run_sequence  # noqa: E402
from utils.stream_run import seed_everything  # noqa: E402

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
SAMPLE_PER_CHUNK = 20000        # 每个片段、每层/每通道随机抽这么多个元素估分位数
SAMPLE_CAP = 400000             # 每层/每通道最多保留这么多样本


class Reservoir(object):
    """按固定上限保存随机样本，用来估计分位数（全量存不下，直方图又要预先定范围）。"""

    def __init__(self, seed=0):
        self.parts = []
        self.kept = 0
        self.seed = int(seed)
        self.generators = {}

    def generator(self, device):
        """取与 device 同设备的随机数生成器（torch 要求生成器与被采样张量同设备，CPU 的生成器不能用在 CUDA 上）。

        每个设备一个、只播种一次，所以同一次运行里抽样是可复现的。
        """
        key = str(device)
        if key not in self.generators:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self.generators[key] = gen
        return self.generators[key]

    def add(self, flat):
        """flat: 一维 torch 张量（可以在 GPU 上）。超过上限后按均匀概率丢弃，保证样本仍然无偏。"""
        n = int(flat.numel())
        if n == 0:
            return
        gen = self.generator(flat.device)
        take = min(n, SAMPLE_PER_CHUNK)
        if take < n:
            idx = torch.randint(0, n, (take,), device=flat.device, generator=gen)
            flat = flat[idx]
        if self.kept >= SAMPLE_CAP:
            if float(torch.rand((), device=flat.device, generator=gen).item()) > 0.25:
                return                                    # 已经够多，随机丢弃大部分新片段
            self.parts = self.parts[len(self.parts) // 2:]
            self.kept = sum(int(p.numel()) for p in self.parts)
        self.parts.append(flat.detach().float().cpu())
        self.kept += int(flat.numel())

    def summary(self, scale=1.0):
        """返回分位数与极值（都除以 scale；膜电位用 v_th 做单位，特征用 1）。"""
        if not self.parts:
            return None
        values = torch.cat(self.parts).numpy() / float(scale)
        qs = np.quantile(values, QUANTILES)
        return {"n_samples": int(values.size),
                "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, qs)},
                "min": float(values.min()), "max": float(values.max()),
                "mean": float(values.mean()), "std": float(values.std()),
                "nonzero_frac": float(np.count_nonzero(values) / values.size)}


class MembraneProbe(object):
    """替代 LayerMonitor 的诊断探针：把膜电位按正负分开统计，并记录逐窗发放率。

    接口与 LayerMonitor 相同的部分只有 update(info)，因此可以直接传给 run_sequence。
    额外提供 features(feats, mu0) 作为 run_sequence 的 feature_probe 钩子。
    """

    def __init__(self, v_threshold, n_windows, feature_names, seed=0):
        self.v_th = float(v_threshold)
        self.n_windows = int(n_windows)
        self.feature_names = list(feature_names)
        self.seed = int(seed)
        self.layers = {}
        self.feature_res = [Reservoir(self.seed) for _ in self.feature_names]
        self.background_res = Reservoir(self.seed)
        self.window_cursor = 0
        self.windows_seen = np.zeros(self.n_windows, dtype=np.int64)

    def begin_sequence(self):
        """每条序列开头调用：逐窗曲线的窗号从 0 重新开始。"""
        self.window_cursor = 0

    def _slot(self, name):
        if name not in self.layers:
            self.layers[name] = {
                "elements": 0, "spikes": 0.0,
                "above_th": 0.0,         # U_pre >= v_th：本窗会发放
                "deep_neg": 0.0,         # U_pre < -v_th：被压到阈值下方一个阈值以上
                "big_pos": 0.0,          # U_pre > 10*v_th：正向失控
                "big_neg": 0.0,          # U_pre < -10*v_th：负向失控
                "u_min": float("inf"), "u_max": float("-inf"),
                "res": Reservoir(self.seed),
                "window_spikes": np.zeros(self.n_windows, dtype=np.float64),
                "window_elements": np.zeros(self.n_windows, dtype=np.int64)}
        return self.layers[name]

    def update(self, info):
        """run_sequence 每个片段调用一次。info["spikes"] / info["u_pre"] 为每层 [T,B,C,H,W] 或 [B,C,H,W]。"""
        steps = None
        with torch.no_grad():
            for name, spikes, u_pre in zip(LAYER_NAMES, info["spikes"], info["u_pre"]):
                if spikes.dim() == 4:
                    spikes, u_pre = spikes.unsqueeze(0), u_pre.unsqueeze(0)
                steps = int(spikes.shape[0])
                d = self._slot(name)
                active = (spikes > 0)
                d["elements"] += int(active.numel())
                d["spikes"] += float(active.sum().item())
                d["above_th"] += float((u_pre >= self.v_th).sum().item())
                d["deep_neg"] += float((u_pre < -self.v_th).sum().item())
                d["big_pos"] += float((u_pre > 10.0 * self.v_th).sum().item())
                d["big_neg"] += float((u_pre < -10.0 * self.v_th).sum().item())
                d["u_min"] = min(d["u_min"], float(u_pre.min().item()))
                d["u_max"] = max(d["u_max"], float(u_pre.max().item()))
                d["res"].add(u_pre.reshape(-1))
                per_window = active.float().mean(dim=(1, 2, 3, 4)).cpu().numpy()
                for t in range(steps):
                    k = self.window_cursor + t
                    if k < self.n_windows:
                        d["window_spikes"][k] += float(per_window[t])
                        d["window_elements"][k] += 1
        if steps:
            for t in range(steps):
                k = self.window_cursor + t
                if k < self.n_windows:
                    self.windows_seen[k] += 1
            self.window_cursor += steps

    def features(self, feats, mu0):
        """run_sequence 的 feature_probe：feats [T,B,C,H,W]，mu0 [T,B,1,H,W]。"""
        with torch.no_grad():
            channels = feats.shape[2]
            for c in range(min(channels, len(self.feature_res))):
                self.feature_res[c].add(feats[:, :, c].reshape(-1))
            self.background_res.add(mu0.reshape(-1))

    def summary(self):
        """汇总成可写进 JSON 的字典。"""
        layers = {}
        for name, d in self.layers.items():
            n = max(d["elements"], 1)
            seen = np.maximum(d["window_elements"], 1)
            layers[name] = {
                "firing_rate": d["spikes"] / n,
                "frac_above_threshold": d["above_th"] / n,
                "frac_deep_negative": d["deep_neg"] / n,
                "frac_big_positive": d["big_pos"] / n,
                "frac_big_negative": d["big_neg"] / n,
                "u_min_in_vth": d["u_min"] / self.v_th,
                "u_max_in_vth": d["u_max"] / self.v_th,
                "u_pre_in_vth": d["res"].summary(self.v_th),
                "firing_rate_by_window": (d["window_spikes"] / seen).tolist()}
        features = {}
        for name, res in zip(self.feature_names, self.feature_res):
            s = res.summary()
            if s is not None:
                features[name] = s
        return {"v_threshold": self.v_th, "layers": layers, "features": features,
                "background_mu0": self.background_res.summary()}


def backbone_threshold(model):
    """取骨干各层实际使用的发放阈值（分位数以它为单位）。

    不直接用 model.v_threshold：那是构造时记下的配置值，单元测试或手工改过阈值时会和神经元不一致。
    ReLU 对照的神经元没有阈值，退回配置值（此时"超阈值占比"没有意义，报告里会和发放率对不上）。
    """
    values = {float(block.neuron.v_threshold) for block in model.blocks()
              if hasattr(block.neuron, "v_threshold")}
    if not values:
        return float(model.v_threshold)
    if len(values) > 1:
        raise SystemExit("骨干各层的发放阈值不一致: %s" % sorted(values))
    return values.pop()


def window_occupancy(seq, window_ms):
    """第三件事的前置检查：这条序列里哪些窗有目标事件、哪些窗是纯背景。"""
    pos = (seq.label == 1) & (seq.target_id != 0)
    if not np.any(pos):
        return {"n_windows": int(seq.n_windows), "target_windows": 0,
                "background_windows": int(seq.n_windows), "first_target_window": None,
                "last_target_window": None}
    window = (seq.t // int(window_ms)).astype(np.int64)
    with_target = np.unique(window[pos])
    with_target = with_target[(with_target >= 0) & (with_target < seq.n_windows)]
    return {"n_windows": int(seq.n_windows), "target_windows": int(with_target.size),
            "background_windows": int(seq.n_windows - with_target.size),
            "first_target_window": int(with_target.min()), "last_target_window": int(with_target.max())}


def polarity_stats(seq, window_ms, height, width):
    """逐事件的极性统计与空间集中度。

    为什么要单列：特征通道的 `nonzero_frac` 是**像素占用率**，不是事件占比。OFF 事件如果集中在少数像素上
    （热像素、强边缘），占用率会比事件占比低一个数量级——09-23 就因为混淆这两者做出过错误推断。
    这里直接给出事件数、点亮像素数、以及"每个点亮像素上平均多少个事件"。
    """
    plane = int(height) * int(width)
    window = (seq.t // int(window_ms)).astype(np.int64)
    key = window * plane + seq.y.astype(np.int64) * int(width) + seq.x.astype(np.int64)
    out = {"n_events": int(seq.t.shape[0]), "n_windows": int(seq.n_windows)}
    for name, mask in (("on", seq.p != 0), ("off", seq.p == 0)):
        n = int(np.count_nonzero(mask))
        lit = int(np.unique(key[mask]).size) if n else 0
        out[name] = {"events": n, "event_frac": n / max(out["n_events"], 1),
                     "lit_pixel_windows": lit,
                     "pixel_occupancy": lit / float(max(out["n_windows"], 1) * plane),
                     "events_per_lit_pixel": (n / lit) if lit else 0.0}
    return out


def merge_polarity(stats):
    """把多条序列的极性统计合并成一份。"""
    total = {"n_events": 0, "n_windows": 0}
    for name in ("on", "off"):
        total[name] = {"events": 0, "lit_pixel_windows": 0}
    for s in stats:
        total["n_events"] += s["n_events"]
        total["n_windows"] += s["n_windows"]
        for name in ("on", "off"):
            total[name]["events"] += s[name]["events"]
            total[name]["lit_pixel_windows"] += s[name]["lit_pixel_windows"]
    for name in ("on", "off"):
        d = total[name]
        d["event_frac"] = d["events"] / max(total["n_events"], 1)
        d["events_per_lit_pixel"] = (d["events"] / d["lit_pixel_windows"]) if d["lit_pixel_windows"] else 0.0
    return total


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="膜电位分布与前端特征尺度诊断")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--sequences", type=int, default=4, help="只跑前几条序列（0 = 全部）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-mode", default=None, choices=("carry", "reset_each_window"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default="log/diagnose/membrane.json")
    return parser.parse_args()


def build_from_checkpoint(args):
    """按 checkpoint 里的训练配置重建前端/网络/判决层（诊断不跑判决层，但 run_sequence 需要它）。"""
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt["config"]) if "config" in ckpt else None
    if cfg is None:
        raise SystemExit("checkpoint 里没有 config，无法重建模型")
    if args.data_root:
        cfg["root"] = args.data_root
    if args.state_mode:
        cfg["state_mode"] = args.state_mode
    seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    frontend, model, cusum = build_all(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, device, frontend, model, cusum, ckpt


def print_report(report):
    """把关键结论打成终端表格（正负分开是这个脚本的全部意义）。"""
    floor, ceil = report.get("neuron_u_floor"), report.get("neuron_u_ceil")
    bounds = "跨窗膜电位下界 %s、上界 %s" % ("无" if floor is None else "%g" % floor, "无" if ceil is None else "%g" % ceil)
    print("\n=== 膜电位分布（单位：v_th = %.3g；%s）===" % (report["v_threshold"], bounds))
    print("%-6s %8s %9s %9s %9s %9s %9s %9s %9s" % (
        "层", "发放率", "超阈值", "深抑制", "正向失控", "负向失控", "中位数", "1%分位", "99%分位"))
    for name in LAYER_NAMES:
        d = report["layers"].get(name)
        if d is None:
            continue
        q = d["u_pre_in_vth"]["quantiles"]
        print("%-6s %8.4f %8.2f%% %8.2f%% %8.2f%% %8.2f%% %9.2f %9.2f %9.2f" % (
            name, d["firing_rate"], 100 * d["frac_above_threshold"], 100 * d["frac_deep_negative"],
            100 * d["frac_big_positive"], 100 * d["frac_big_negative"],
            q["0.5"], q["0.01"], q["0.99"]))
    print("\n（正向失控 = U_pre > 10*v_th，负向失控 = U_pre < -10*v_th；"
          "两者之和即评估日志里的 big_membrane_frac）")
    if floor is not None:
        print("（设了下界：携带的膜电位 >= %g*v_th，所以 U_pre >= beta*%g + 本窗电流。跨窗累积已被截断，"
              "此时仍出现的负向失控来自单窗内的强负电流——看中位数是否回到单窗电流的量级，而不是期望它归零）"
              % (floor, floor))
    if report["neuron"] == "lif":
        gaps = [abs(d["frac_above_threshold"] - d["firing_rate"]) for d in report["layers"].values()]
        if max(gaps) > 1e-6:
            print("警告：超阈值占比与发放率不一致（最大差 %.3g），说明用来统计的阈值和神经元的不是同一个" % max(gaps))
    else:
        print("（%s 对照没有发放阈值，超阈值一列无意义）" % report["neuron"])

    print("\n=== 前端特征通道的动态范围 ===")
    print("%-16s %10s %10s %10s %10s %10s" % ("通道", "1%分位", "中位数", "99%分位", "最大值", "非零占比"))
    for name, d in report["features"].items():
        q = d["quantiles"]
        print("%-16s %10.4f %10.4f %10.4f %10.4f %9.2f%%" % (
            name, q["0.01"], q["0.5"], q["0.99"], d["max"], 100 * d["nonzero_frac"]))
    pol = report.get("polarity")
    if pol:
        print("\n=== 极性统计（逐事件，不是像素占用率）===")
        print("%-6s %12s %10s %16s %18s" % ("极性", "事件数", "事件占比", "点亮像素·窗", "每点亮像素事件数"))
        for name, label in (("on", "ON"), ("off", "OFF")):
            d = pol[name]
            print("%-6s %12d %9.2f%% %16d %18.1f" % (
                label, d["events"], 100 * d["event_frac"], d["lit_pixel_windows"], d["events_per_lit_pixel"]))
        ratio = pol["off"]["events_per_lit_pixel"] / max(pol["on"]["events_per_lit_pixel"], 1e-9)
        print("OFF 的空间集中度是 ON 的 %.1f 倍（>3 提示热像素或强边缘聚集）" % ratio)
        print("注意：上面特征表里的\"非零占比\"是像素占用率，与这里的事件占比不是一回事。")

    bg = report.get("background_mu0")
    if bg:
        print("背景强度 mu0：中位数 %.5g，1%%/99%% 分位 %.5g / %.5g，最大 %.5g" % (
            bg["quantiles"]["0.5"], bg["quantiles"]["0.01"], bg["quantiles"]["0.99"], bg["max"]))

    print("\n=== 纯背景窗的前置检查 ===")
    total = sum(w["n_windows"] for w in report["windows"])
    bgw = sum(w["background_windows"] for w in report["windows"])
    print("共 %d 条序列、%d 窗，其中没有目标事件的窗 %d 个（%.1f%%）" % (
        len(report["windows"]), total, bgw, 100.0 * bgw / max(total, 1)))
    for w in report["windows"]:
        print("  %-28s 总 %3d 窗，有目标 %3d 窗，纯背景 %3d 窗，目标出现在第 %s~%s 窗" % (
            w["name"], w["n_windows"], w["target_windows"], w["background_windows"],
            w["first_target_window"], w["last_target_window"]))

    print("\n=== 逐窗发放率（前 16 窗，看状态预热）===")
    for name in LAYER_NAMES:
        d = report["layers"].get(name)
        if d is None:
            continue
        curve = d["firing_rate_by_window"][:16]
        print("  %-6s %s" % (name, " ".join("%.3f" % v for v in curve)))


def main():
    """入口。"""
    args = parse_args()
    cfg, device, frontend, model, cusum, ckpt = build_from_checkpoint(args)
    dataset = EvUAVStream(cfg["root"], args.split, cfg, None)
    n = len(dataset) if args.sequences <= 0 else min(len(dataset), int(args.sequences))
    probe = MembraneProbe(backbone_threshold(model), int(cfg["n_windows"]), frontend.feature_names(),
                          seed=int(cfg["seed"]))
    windows = []
    with torch.no_grad():
        for i in range(n):
            seq = dataset[i]
            probe.begin_sequence()
            run_sequence(model, frontend, cusum, seq, cfg, device, cfg["state_mode"], monitor=probe,
                         with_cusum=False, feature_probe=probe.features)
            occ = window_occupancy(seq, cfg["window_ms"])
            occ["name"] = seq.name
            occ["polarity"] = polarity_stats(seq, cfg["window_ms"], cfg["pad_height"], cfg["pad_width"])
            windows.append(occ)
            print("已处理 %s（%d/%d）" % (seq.name, i + 1, n), flush=True)
    report = probe.summary()
    report["windows"] = windows
    report["polarity"] = merge_polarity([w["polarity"] for w in windows])
    report["checkpoint"] = args.checkpoint
    report["split"] = args.split
    report["epoch"] = ckpt.get("epoch")
    report["neuron"] = cfg["neuron"]
    report["neuron_u_floor"] = cfg.get("neuron_u_floor")
    report["neuron_u_ceil"] = cfg.get("neuron_u_ceil")
    print_report(report)
    out = args.out
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, ensure_ascii=False)
    print("\n报告:", out)


if __name__ == "__main__":
    main()
