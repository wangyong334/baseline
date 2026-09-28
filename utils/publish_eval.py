"""V3 阶段 1（等待安全的逐事件发布，默认关闭）在训练 / 评估入口里的接线。

--publish on 时 run_sequence 多一个读出 pub：每个事件在出生后 0..D 窗之间按证据是否充分择机发布
（规则见 model/publish_readout.py）；关闭时本模块的任何代码都不运行，V2 的读出逐位不变。
    add_arguments / apply_overrides / validate_config / EVAL_KEYS   命令行与配置
    build_rule            按配置构造发布规则（θ 缺省 = logit(threshold)，与其他读出同一工作点）
    readout_names         打开时新增的读出名 ["pub"]
    PublishRun            一条序列上的运行：每窗推进、序列末 flush、回填成逐事件的概率与发布信息
    PublishStats / report_lines   跨序列统计（目标事件的发布延迟分布、发布原因、提前发布的对错）与打印
pub 的概率 = sigmoid(z - θ + logit(threshold)) 再夹到发布标签所在的一侧，所以按 threshold 取阈值恰好得到发布的标签，
原 utils/eval.py 的 IoU / ACC / Pd / Fa 就是发布标签的指标；首次检出延迟按逐事件的实际发布窗计
（utils/stream_metrics.first_detection_latencies_by_event）。
"""
import math

import numpy as np
import torch

from dataset.stream_windows import refill_by_index
from model.publish_readout import COLLAPSES, REASONS, PublishRule, PublishUnits

NAME = "pub"
EVAL_KEYS = ("publish", "publish_deadline", "publish_theta", "publish_upper", "publish_lower", "publish_collapse",
             "publish_gate")
NONE = "none"


def gate_value(text):
    """--publish-gate 的取值：数值（logit），或 none（不门控）。"""
    return NONE if str(text).strip().lower() in ("none", "null") else float(text)


def add_arguments(parser):
    """V3 发布层的命令行参数（都是评估时开关，不需要重新训练）。"""
    parser.add_argument("--publish", choices=("on", "off"), default=None,
                        help="V3 逐事件自适应发布（读出 pub）；off 时与 V2 的读出逐位相同")
    parser.add_argument("--publish-deadline", type=int, default=None, help="最长等待 D（窗）")
    parser.add_argument("--publish-theta", type=float, default=None,
                        help="最终阈值 θ（logit）；不给 = logit(threshold)。按虚警率选定时在 val 上用 tools/publish_replay.py 求")
    parser.add_argument("--publish-upper", type=float, default=None, help="提前发布目标所需的裕量 a（nat，>= 0）")
    parser.add_argument("--publish-lower", type=float, default=None, help="提前发布背景所需的裕量 b（nat，> 0）")
    parser.add_argument("--publish-collapse", choices=COLLAPSES, default=None,
                        help="边界随年龄的收拢方式：linear（线性收到 θ）/ step（期限前不收）")
    parser.add_argument("--publish-gate", type=gate_value, default=None,
                        help="归属门槛（logit）：mark 低于它的事件只能被未来证据减分；none = 不门控")


def apply_overrides(cfg, args):
    """用命令行覆盖 cfg 里的发布层配置（原地修改）。"""
    for key, value in (("publish_deadline", args.publish_deadline), ("publish_theta", args.publish_theta),
                       ("publish_upper", args.publish_upper), ("publish_lower", args.publish_lower),
                       ("publish_collapse", args.publish_collapse), ("publish_gate", args.publish_gate)):
        if value is not None:
            cfg[key] = None if value == NONE else value
    if args.publish is not None:
        cfg["publish"] = args.publish == "on"


def enabled(cfg):
    return bool(cfg.get("publish"))


def is_publish_readout(name):
    return name == NAME


def theta_of(cfg):
    """最终阈值 θ（logit）：显式给出时用它，否则 = logit(threshold)。"""
    theta = cfg.get("publish_theta")
    if theta is not None:
        return float(theta)
    p = float(cfg["threshold"])
    return math.log(p / (1.0 - p))


def build_rule(cfg):
    """按配置构造发布规则。"""
    gate = cfg.get("publish_gate")
    return PublishRule(theta_of(cfg), float(cfg.get("publish_upper", 1.0)), float(cfg.get("publish_lower", 2.0)),
                       int(cfg.get("publish_deadline", 5)), cfg.get("publish_collapse", "linear"),
                       None if gate is None else float(gate), float(cfg.get("fusion_weight", 1.0)))


def describe(cfg):
    """实际生效的发布规则参数（写进评估结果，供 tools/publish_replay.py 做一致性比对）；关闭时为 None。
    记录生效值而不是原始配置：配置里没写的项取的是 build_rule 的缺省值。"""
    if not enabled(cfg):
        return None
    rule = build_rule(cfg)
    return {"deadline": rule.deadline, "theta": rule.theta, "upper": rule.upper, "lower": rule.lower,
            "collapse": rule.collapse, "gate": rule.gate, "weight": rule.weight}


def validate_config(cfg):
    """发布层打开时的配置检查（关闭时什么都不查）：规则参数不合法时构造就会报错。"""
    if not enabled(cfg):
        return
    if not 0.0 < float(cfg["threshold"]) < 1.0:
        raise ValueError("threshold 必须在 (0, 1) 内")
    build_rule(cfg)


def readout_names(cfg):
    """打开时新增的读出名。"""
    return [NAME] if enabled(cfg) else []


def published_probability(z, label, theta, threshold):
    """发布分数换成与发布标签一致的概率：sigmoid(z - θ + logit(threshold))，再夹到标签所在一侧
    （float32 下 >= threshold 恰好等于发布标签，避免边界上的取整翻转）。"""
    thr = np.float32(threshold)
    shift = math.log(float(threshold) / (1.0 - float(threshold))) - float(theta)
    prob = (1.0 / (1.0 + np.exp(-(np.asarray(z, np.float64) + shift)))).astype(np.float32)
    below = np.nextafter(thr, np.float32(0.0))
    return np.where(label, np.maximum(prob, thr), np.minimum(prob, below)).astype(np.float32)


class PublishRun(object):
    """V3 发布层在一条序列上的运行（train_stream_v2.run_sequence 在 cfg["publish"] 打开时构造）。

    window_info 是 run_sequence 维护的 {窗号: (原始事件下标, 网络 logit)}；发布记录按出生窗号取回原始事件下标。
    """

    def __init__(self, cfg, cusum, window_info):
        self.rule = build_rule(cfg)
        self.units = PublishUnits(cusum, self.rule)
        self.window_info = window_info
        self.parts = {name: ([], []) for name in ("z", "label", "window", "reason", "age")}

    def _collect(self, records):
        for key, pos, label, z, published, reason, age in records:
            idx = self.window_info[key][0][pos.cpu().numpy()]
            n = idx.shape[0]
            values = {"z": z.detach().to("cpu", torch.float64).numpy(), "label": label.cpu().numpy(),
                      "window": np.full(n, published, np.int64), "reason": np.full(n, reason, np.int64),
                      "age": np.full(n, age, np.int64)}
            for name, value in values.items():
                self.parts[name][0].append(idx)
                self.parts[name][1].append(value)

    def step(self, state, k, b, y, x, logits):
        """第 k 窗（判决层本窗之后）：推进待发布事件并登记本窗新事件。"""
        self._collect(self.units.step(state, k, b, y, x, k, logits))

    def flush(self):
        """真实序列结束：尚未发布的事件按最终阈值发布。"""
        self._collect(self.units.flush())

    def outputs(self, n_events, threshold):
        """回填到原始事件顺序。返回 (probs {"pub": 概率}, extra {z_pub, label_pub, age_pub, reason_pub, publish_pub})。
        publish_pub 为逐事件的实际发布窗号（run_sequence 的调用方按它算首次检出延迟；以 publish 开头的字段不写入导出）。"""
        z = refill_by_index(n_events, *self.parts["z"], dtype=np.float64)
        label = refill_by_index(n_events, *self.parts["label"], dtype=np.uint8).astype(bool)
        probs = {NAME: published_probability(z, label, self.rule.theta, threshold)}
        extra = {"z_pub": z.astype(np.float32), "label_pub": label.astype(np.float32),
                 "age_pub": refill_by_index(n_events, *self.parts["age"], dtype=np.float32),
                 "reason_pub": refill_by_index(n_events, *self.parts["reason"], dtype=np.float32),
                 "publish_pub": refill_by_index(n_events, *self.parts["window"], dtype=np.int64)}
        return probs, extra


class PublishStats(object):
    """跨序列统计：真实目标事件（与全部事件）按发布年龄与原因的计数，提前发布的对错。"""

    def __init__(self, deadline):
        self.deadline = int(deadline)
        self.age_target = np.zeros(self.deadline + 1, np.int64)
        self.age_all = np.zeros(self.deadline + 1, np.int64)
        self.reason_target = np.zeros(len(REASONS), np.int64)
        self.reason_all = np.zeros(len(REASONS), np.int64)
        self.early = {"upper_true": 0, "upper_false": 0, "lower_true_target": 0, "lower_background": 0}

    def update(self, labels, extra):
        lab = np.asarray(labels) > 0.5
        age = extra["age_pub"].astype(np.int64)
        reason = extra["reason_pub"].astype(np.int64)
        self.age_target += np.bincount(age[lab], minlength=self.deadline + 1)[:self.deadline + 1]
        self.age_all += np.bincount(age, minlength=self.deadline + 1)[:self.deadline + 1]
        self.reason_target += np.bincount(reason[lab], minlength=len(REASONS))
        self.reason_all += np.bincount(reason, minlength=len(REASONS))
        up, low = reason == 0, reason == 1
        self.early["upper_true"] += int(np.count_nonzero(up & lab))
        self.early["upper_false"] += int(np.count_nonzero(up & ~lab))
        self.early["lower_true_target"] += int(np.count_nonzero(low & lab))
        self.early["lower_background"] += int(np.count_nonzero(low & ~lab))

    def summary(self, window_ms):
        nt, na = max(int(self.age_target.sum()), 1), max(int(self.age_all.sum()), 1)
        ages = np.arange(self.deadline + 1)
        return {"deadline": self.deadline,
                "age_hist_target": (self.age_target / float(nt)).tolist(),
                "age_hist_all": (self.age_all / float(na)).tolist(),
                "mean_delay_target_ms": float(window_ms) * float((ages * self.age_target).sum()) / nt,
                "mean_delay_all_ms": float(window_ms) * float((ages * self.age_all).sum()) / na,
                "reason_target": dict(zip(REASONS, (self.reason_target / float(nt)).tolist())),
                "reason_all": dict(zip(REASONS, (self.reason_all / float(na)).tolist())),
                "early": dict(self.early)}


def report_lines(summary, mode):
    """eval 日志里的摘要行。"""
    s = summary
    hist = "  ".join("%d窗 %.1f%%" % (d, 100 * v) for d, v in enumerate(s["age_hist_target"]))
    reason = "  ".join("%s %.1f%%" % (k, 100 * v) for k, v in s["reason_target"].items())
    e = s["early"]
    return ["[%s] pub 目标事件发布年龄：%s | 平均 %.1f ms（全部事件 %.1f ms）" % (
                mode, hist, s["mean_delay_target_ms"], s["mean_delay_all_ms"]),
            "[%s] pub 目标事件发布原因：%s | 提前发布目标 %d（其中真目标 %d）、提前发布背景里的真目标 %d" % (
                mode, reason, e["upper_true"] + e["upper_false"], e["upper_true"], e["lower_true_target"])]
