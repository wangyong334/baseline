"""流式 SNN V1 的纯 Python/numpy 工具函数（不依赖 torch，可本地单元测试）。

包含：配置读取、学习率计划、TBPTT 分块、子集选择、训练历史汇总、
增益校准的数学部分、健康门槛检查。
"""
import math

import numpy as np


def load_flat_config(path):
    """读取 YAML 配置并把各个小节（DATA/NET/TRAIN...）展平成一个字典。

    与原仓库 configs/configs.py 的展平方式一致，但不在 import 时解析命令行。
    若不同小节出现同名键则抛出 ValueError，避免静默覆盖。
    """
    import yaml
    with open(path, "r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    flat = {}
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError("配置小节 %s 必须是字典" % section)
        for key, value in values.items():
            if key in flat:
                raise ValueError("配置键重复: %s" % key)
            flat[key] = value
    return flat


def linear_epoch_lr(epoch, epochs, start_lr, end_lr):
    """按 epoch 线性衰减学习率：第 0 轮 = start_lr，第 epochs-1 轮 = end_lr。

    与 train_mp.py 中的同名函数语义相同（在这里重写一份，是为了不导入会解析命令行的模块）。
    学习率在每个 epoch 开始时设定一次，epoch 内保持不变。
    """
    if epochs < 2:
        raise ValueError("线性学习率至少需要 2 个 epoch")
    if not (math.isfinite(start_lr) and math.isfinite(end_lr) and 0 < end_lr <= start_lr):
        raise ValueError("需要满足 0 < lr_end <= lr")
    if epoch <= 0:
        return float(start_lr)
    if epoch >= epochs - 1:
        return float(end_lr)
    return float(start_lr + (end_lr - start_lr) * epoch / (epochs - 1))


def make_chunks(n_windows, k, first_len):
    """把 [0, n_windows) 切成 TBPTT 片段 [(start, end), ...]。

    第一段长度为 first_len（1..k，随机化以免某些窗口永远位于片段开头），
    之后每段 k 个窗口，最后一段为剩余窗口。所有窗口恰好出现一次，不因偏移丢弃窗口。
    """
    if not 1 <= first_len <= k:
        raise ValueError("first_len 必须在 [1, k] 内")
    chunks = []
    start = 0
    end = min(first_len, n_windows)
    while start < n_windows:
        chunks.append((start, end))
        start = end
        end = min(start + k, n_windows)
    return chunks


def select_subset(names, size, seed):
    """用固定种子从文件名列表中无放回抽取 size 个，返回按名称排序的列表。

    用于：增益校准的序列、训练子集评估的序列。相同输入与种子必然得到相同结果。
    size >= 列表长度时返回全部（排序后）。
    """
    names = sorted(names)
    if size >= len(names):
        return names
    rng = np.random.RandomState(int(seed))
    picked = rng.choice(len(names), int(size), replace=False)
    return sorted(names[i] for i in picked)


def subset_due(epoch, epochs, every):
    """训练子集评估是否在本轮进行：每 every 轮一次（第 every-1、2*every-1... 轮），最后一轮总是评估。

    训练子集评估只用于诊断，不参与选模型；每轮的打乱与 TBPTT 首段长度按 (seed, epoch) 重新设种，
    因此评估频率不影响训练本身。every=1 即每轮评估（原行为）。
    """
    if int(every) < 1:
        raise ValueError("train_subset_every 必须 >= 1")
    return (int(epoch) + 1) % int(every) == 0 or int(epoch) == int(epochs) - 1


def summarize_history(history, seed, tail=5):
    """汇总训练历史（每轮一条记录），字段与 utils/aggregate.py 兼容。

    输入 history: 列表，每项至少包含 epoch 与 val_iou（可为 None）、val_acc。
    输出: best_val_iou（乐观有偏）、final_mean_iou（末 tail 轮均值，跨种子比较时引用它）等。
    """
    scored = [r for r in history if r.get("val_iou") is not None]
    summary = {"seed": seed, "epochs_completed": len(history), "validated_epochs": len(scored)}
    if not scored:
        return summary
    ious = [float(r["val_iou"]) for r in scored]
    accs = [float(r["val_acc"]) for r in scored if r.get("val_acc") is not None]
    window = min(int(tail), len(ious))
    tail_ious = ious[-window:]
    summary.update({
        "best_val_iou": max(ious),
        "best_val_iou_epoch": scored[int(np.argmax(ious))]["epoch"],
        "final_epoch_iou": ious[-1],
        "final_window": window,
        "final_mean_iou": float(np.mean(tail_ious)),
        "final_std_iou": float(np.std(tail_ious, ddof=1)) if window > 1 else 0.0,
        "final_mean_acc": float(np.mean(accs[-window:])) if accs else None,
        "selection_bias_gap": max(ious) - float(np.mean(tail_ious)),
    })
    return summary


def gains_from_positive_samples(samples, positive_counts, v_threshold, quantile,
                                min_positive, gain_min, gain_max, eps=1e-6):
    """增益校准的数学部分：由每个通道的正电流样本求逐通道增益。

    输入:
        samples         长度 C 的列表，第 c 项是通道 c 的正电流采样值（numpy 一维，可为空）
        positive_counts 长度 C，通道 c 在校准数据中真实出现的正电流元素总数（采样前）
        v_threshold     LIF 阈值，校准目标是 q 分位正电流 * 增益 = v_threshold
        quantile        分位数（V1 为 0.99）
        min_positive    正电流元素少于该数的通道回退使用整层分位数
        gain_min/max    初始增益的截断范围（防止极小分位数产生几百倍增益）
    输出: (gains, info)
        gains  float64 [C]
        info   每个通道使用的分位数、是否回退、原始增益，便于写日志核查
    规则: 通道 q < eps 时增益取 1.0（无法估计尺度时不做缩放）。
    """
    n_ch = len(samples)
    nonempty = [s for s in samples if s is not None and len(s)]
    pooled = np.concatenate(nonempty) if nonempty else np.zeros(0)
    q_layer = float(np.quantile(pooled, quantile)) if pooled.size else 0.0
    gains = np.ones(n_ch, dtype=np.float64)
    info = {"q_layer": q_layer, "q_channel": [], "fallback": [], "raw_gain": []}
    for c in range(n_ch):
        s = samples[c]
        use_layer = int(positive_counts[c]) < int(min_positive) or s is None or len(s) == 0
        q = q_layer if use_layer else float(np.quantile(s, quantile))
        raw = 1.0 if q < eps else float(v_threshold) / q
        gains[c] = min(max(raw, float(gain_min)), float(gain_max))
        info["q_channel"].append(q)
        info["fallback"].append(bool(use_layer))
        info["raw_gain"].append(raw)
    return gains, info


# 健康门槛的默认阈值：只用于打印警告，不会中断训练。
HEALTH_DEFAULTS = {
    "min_firing_rate": 1e-5,          # 整层平均发放率低于此值：疑似整层沉默
    "max_silent_channel_frac": 0.5,   # 超过一半通道在评估期间从未发放：疑似死神经元
    "max_always_on_frac": 0.1,        # 超过 10% 神经元几乎每窗都发放：疑似饱和
    "max_big_membrane_frac": 0.01,    # |U_pre| > 10*阈值 的元素占比过高：疑似膜电位失控
    "tau_bound_frac": 0.9,            # 超过 90% 通道的 tau 贴近上下界
}


def health_warnings(layer_stats, tau_stats, grad_norms, limits=None):
    """根据发放率、膜电位、tau、梯度范数给出健康警告列表（空列表 = 通过门槛）。

    输入:
        layer_stats  {层名: {"firing_rate", "silent_channel_frac", "always_on_frac", "big_membrane_frac"}}
        tau_stats    {层名: {"near_min_frac", "near_max_frac"}}（ReLU 版本可传空字典）
        grad_norms   {参数组名: 梯度范数}（评估阶段可传空字典）
    输出: 字符串列表，每条描述一个异常现象。
    注意: "梯度非空、没有 NaN" 不足以说明训练正常，所以这里检查的是分布而不是有无。
    """
    lim = dict(HEALTH_DEFAULTS)
    if limits:
        lim.update(limits)
    warnings = []
    for name, s in layer_stats.items():
        if s.get("firing_rate", 1.0) < lim["min_firing_rate"]:
            warnings.append("%s 发放率 %.2e 过低（整层沉默）" % (name, s["firing_rate"]))
        if s.get("silent_channel_frac", 0.0) > lim["max_silent_channel_frac"]:
            warnings.append("%s 有 %.0f%% 通道从未发放" % (name, 100 * s["silent_channel_frac"]))
        if s.get("always_on_frac", 0.0) > lim["max_always_on_frac"]:
            warnings.append("%s 有 %.1f%% 神经元几乎每窗发放（饱和）" % (name, 100 * s["always_on_frac"]))
        if s.get("big_membrane_frac", 0.0) > lim["max_big_membrane_frac"]:
            warnings.append("%s 膜电位过大元素占比 %.2f%%" % (name, 100 * s["big_membrane_frac"]))
    for name, s in tau_stats.items():
        if s.get("near_min_frac", 0.0) > lim["tau_bound_frac"]:
            warnings.append("%s 的 tau 几乎全部收缩到下界（模型可能放弃了长历史）" % name)
        if s.get("near_max_frac", 0.0) > lim["tau_bound_frac"]:
            warnings.append("%s 的 tau 几乎全部贴近上界" % name)
    for name, value in grad_norms.items():
        if not math.isfinite(value):
            warnings.append("%s 梯度范数非有限值" % name)
        elif value == 0.0:
            warnings.append("%s 梯度范数为 0" % name)
    return warnings
