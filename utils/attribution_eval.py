"""V2-2（回溯分布修正，默认关闭）在训练 / 评估入口里的全部接线，从 train_stream_v2.py 移出。

V2 冻结后 V2-2 只是一个评估时开关（--attr on）：关掉时本模块的任何代码都不运行，V2-1 的读出逐位不变。
    add_arguments / apply_overrides   命令行参数与配置覆盖
    validate_config                   打开时的配置检查
    EVAL_KEYS                         eval 模式一律取当前 YAML / 命令行的 V2-2 配置项
    build_attribution                 构造假设库与回溯读出（无可学习参数）
    readout_names                     打开时新增的读出名
    AttributionRun                    一条序列上的运行：每窗推进假设库、收集到期读出、序列末回填并组合 attr_fused
    AttributionStats / report_lines   跨序列的诊断统计与打印
读出与公式见 model/attribution_readout.py 与 model/target_hypotheses.py。
"""
import time

import numpy as np
import torch

from dataset.stream_windows import refill_by_index
from model.attribution_readout import VARIANTS, AttributionReadout
from model.target_hypotheses import DEFAULT_PARAMS as HYP_DEFAULTS, UPDATES as HYP_UPDATES, HypothesisBank

EVAL_KEYS = ("attr", "attr_delays", "attr_weight", "attr_variants", "attr_cap", "attr_eps", "attr_ref_radius",
             "attr_min_support", "attr_tube_birth", "hyp_update", "hyp_params")


def add_arguments(parser):
    """V2-2 的命令行参数（都是评估时开关，不需要重新训练）。"""
    parser.add_argument("--attr", choices=("on", "off"), default=None,
                        help="V2-2 回溯分布修正读出；off 时与 V2-1 baseline 的读出逐位相同")
    parser.add_argument("--attr-delays", type=int, nargs="+", default=None, help="V2-2 固定等待的窗数，例如 1 2 5 10")
    parser.add_argument("--attr-weight", type=float, default=None, help="式 (17) 的修正权重 w（在验证集上校准）")
    parser.add_argument("--attr-variants", nargs="*", choices=VARIANTS, default=None,
                        help="额外导出的对照读出（只算 IoU/ACC，不算 Pd/Fa）")
    parser.add_argument("--attr-update", choices=HYP_UPDATES, default=None,
                        help="假设库的更新方式：separate（默认）/ generic / confident")
    parser.add_argument("--attr-cap", type=float, default=None, help="修正上限 c")
    parser.add_argument("--attr-eps", type=float, default=None, help="比值下限 epsilon")
    parser.add_argument("--attr-ref-radius", type=int, default=None, help="固定参考半径 r_ref")
    parser.add_argument("--attr-min-support", type=int, default=None, help="参与修正所需的有效测量窗数 n_min")
    parser.add_argument("--attr-tube-birth", choices=("on", "off"), default=None,
                        help="是否用运动管道生成候选（off = 只用 SNN 高置信簇，检验判决层是否还有必要）")
    parser.add_argument("--hyp", nargs="+", default=None, metavar="KEY=VALUE",
                        help="覆盖假设库参数（model/target_hypotheses.py 的 DEFAULT_PARAMS），例如 --hyp lag=3 confirm_theta=5")


def apply_overrides(cfg, args):
    """用命令行覆盖 cfg 里的 V2-2 配置项（原地修改）。"""
    overrides = {"attr_delays": args.attr_delays, "attr_weight": args.attr_weight, "attr_variants": args.attr_variants,
                 "hyp_update": args.attr_update, "attr_cap": args.attr_cap, "attr_eps": args.attr_eps,
                 "attr_ref_radius": args.attr_ref_radius, "attr_min_support": args.attr_min_support}
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if args.attr is not None:
        cfg["attr"] = args.attr == "on"
    if args.attr_tube_birth is not None:
        cfg["attr_tube_birth"] = args.attr_tube_birth == "on"
    if args.hyp:
        cfg["hyp_params"] = dict(cfg.get("hyp_params") or {}, **parse_hyp_overrides(args.hyp))


def parse_hyp_overrides(items):
    """把 ["lag=3", "update=generic"] 解析成假设库参数字典，数值按 DEFAULT_PARAMS 中的类型转换。"""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--hyp 的格式是 KEY=VALUE，收到 %r" % item)
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in HYP_DEFAULTS:
            raise ValueError("未知的假设库参数 %r（可选：%s）" % (key, ", ".join(sorted(HYP_DEFAULTS))))
        default = HYP_DEFAULTS[key]
        if isinstance(default, str):
            out[key] = value
        elif isinstance(default, int):
            out[key] = int(float(value))
        else:
            out[key] = float(value)
    return out


def validate_config(cfg):
    """V2-2 打开时的配置检查（关闭时什么都不查）。"""
    if not cfg.get("attr"):
        return
    delays = [int(d) for d in cfg.get("attr_delays") or []]
    if not delays or min(delays) < 1:
        raise ValueError("attr_delays 至少一个且都 >= 1")
    unknown = set(cfg.get("attr_variants") or []) - set(VARIANTS)
    if unknown:
        raise ValueError("未知的对照读出: %s" % sorted(unknown))
    bad = set(cfg.get("hyp_params") or {}) - set(HYP_DEFAULTS)
    if bad:
        raise ValueError("未知的假设库参数: %s" % sorted(bad))
    if cfg.get("hyp_update", "separate") not in HYP_UPDATES:
        raise ValueError("hyp_update 必须是 %s 之一" % (HYP_UPDATES,))


def build_attribution(cfg, cusum):
    """按配置构造 V2-2 的假设库与回溯读出（无可学习参数）。保留窗数自动覆盖 lag + 最大等待。"""
    params = dict(cfg.get("hyp_params") or {})
    params.setdefault("update", cfg.get("hyp_update", "separate"))
    delays = [int(d) for d in cfg["attr_delays"]]
    lag = int(params.get("lag", HYP_DEFAULTS["lag"]))
    params["keep_windows"] = max(int(params.get("keep_windows", HYP_DEFAULTS["keep_windows"])), lag + max(delays) + 2)
    velocities = cusum.velocities if cfg.get("attr_tube_birth", True) else ()
    bank = HypothesisBank(int(cfg["pad_height"]), int(cfg["pad_width"]), velocities, **params)
    readout = AttributionReadout(bank, delays, cap=float(cfg.get("attr_cap", 3.0)),
                                 eps=float(cfg.get("attr_eps", 1e-3)), ref_radius=int(cfg.get("attr_ref_radius", 7)),
                                 min_support=int(cfg.get("attr_min_support", 3)),
                                 variants=tuple(cfg.get("attr_variants") or ()))
    return bank, readout


def readout_names(cfg):
    """V2-2 打开时新增的读出名：attr_d{d}（诊断）、attr_fused_d{d}（V2-1 融合读出 + 修正，主比较，
    两者都要有这个等待窗数）与 attr_<对照>_d{d}（对照读出，只算 IoU/ACC）。关闭时为空列表。"""
    if not cfg.get("attr"):
        return []
    delays = [int(d) for d in cfg["attr_delays"]]
    names = ["attr_d%d" % d for d in delays]
    names += ["attr_fused_d%d" % d for d in delays if d in [int(x) for x in cfg["readout_delays"]]]
    names += ["attr_%s_d%d" % (v, d) for v in VARIANTS if v in (cfg.get("attr_variants") or []) for d in delays]
    return names


def is_variant_readout(name):
    """V2-2 的对照读出（只算 IoU/ACC）：attr_<对照>_d{d}；主读出 attr_d{d} 与 attr_fused_d{d} 不算。"""
    return name.startswith("attr_") and name.split("_")[1] in VARIANTS


class AttributionRun(object):
    """V2-2 在一条序列上的运行（train_stream_v2.run_sequence 在 cfg["attr"] 打开时构造）。

    window_info 是 run_sequence 维护的 {窗号: (原始事件下标, 网络 logit)}，读出到期时按窗号取回。
    每窗调用 step：假设库处理第 k 窗，再读出到期的旧窗事件；序列末调用 flush，再用 outputs 取结果：
        probs  attr_d{d} = sigmoid(mark + w*Delta)、对照读出 attr_<对照>_d{d}（direct 直接是概率）、
               attr_fused_d{d} = sigmoid(mark + w_F*F(d) + w_A*Delta(d))（该等待窗数也有 V2-1 融合读出时）
        extra  delta_attr_d{d}（修正量）、case_attr_d{d}（0 无修正 / 1 冻结参照 / 2 固定参考 / 3 冻结参照且
               原关联已不覆盖）、publish_attr_d{d}（每窗实际发布的窗号），以及 attr_bank（假设库统计，非数组）
    """

    def __init__(self, cfg, cusum, n_windows, window_info):
        self.bank, self.readout = build_attribution(cfg, cusum)
        self.delays, self.variants = list(self.readout.delays), list(self.readout.variants)
        self.weight = float(cfg.get("attr_weight", 1.0))
        self.window_info = window_info
        self.n_windows = int(n_windows)
        self.names = []
        for d in self.delays:
            self.names += ["attr_d%d" % d, "delta_attr_d%d" % d, "case_attr_d%d" % d]
            self.names += ["attr_%s_d%d" % (v, d) for v in self.variants]
        self.parts = {name: ([], []) for name in self.names}
        self.publish = {d: np.zeros(self.n_windows, dtype=np.int64) for d in self.delays}
        self.time = {"seconds": 0.0, "alive": 0, "confirmed": 0}

    def _collect(self, results):
        for key, delay, res, published in results:
            idx, logit = self.window_info[key]
            z = logit.detach().to("cpu", torch.float64)
            outputs = {"attr_d%d" % delay: torch.sigmoid(z + self.weight * res["attr"]),
                       "delta_attr_d%d" % delay: res["attr"], "case_attr_d%d" % delay: res["case"].to(torch.float64)}
            for v in self.variants:
                outputs["attr_%s_d%d" % (v, delay)] = res[v] if v == "direct" else torch.sigmoid(z + self.weight * res[v])
            for name, value in outputs.items():
                self.parts[name][0].append(idx)
                self.parts[name][1].append(value.float().numpy())
            self.publish[delay][key] = published

    def step(self, k, ys, xs, logits, log_g, mu0, C):
        """第 k 窗：logits 为本窗事件的网络 logit，log_g / mu0 为本窗的强度场与背景 [H,W]，C 为判决层膜电位。"""
        t0 = time.perf_counter()
        g_k, mu_k = torch.exp(log_g), mu0
        info = self.bank.step(k, ys, xs, torch.sigmoid(logits), g_k, mu_k, C)
        self._collect(self.readout.step(k, k, ys, xs, mu_k, g_k))
        self.time["seconds"] += time.perf_counter() - t0
        self.time["alive"] += info["alive"]
        self.time["confirmed"] += info["confirmed"]

    def flush(self):
        """序列结束：尚未到期的延迟在最后一窗读出。"""
        self._collect(self.readout.flush())

    def outputs(self, n_events, logit_net, evidence, fusion_weight):
        """回填到原始事件顺序并组合 attr_fused。evidence 为 {d: F(d) 回填后的数组}（V2-1 的各等待窗数）。
        返回 (probs, extra)，键的顺序与移出前的 run_sequence 相同。"""
        refilled = {name: refill_by_index(n_events, idx_parts, val_parts)
                    for name, (idx_parts, val_parts) in self.parts.items()}
        probs = {name: refilled[name] for name in self.names if name.startswith("attr_")}
        for d in self.delays:                        # attr_fused_d{d} = sigmoid(mark + w_F * F(d) + w_A * Delta(d))
            if d in evidence:
                z = (logit_net.astype(np.float64) + fusion_weight * evidence[d].astype(np.float64)
                     + self.weight * refilled["delta_attr_d%d" % d].astype(np.float64))
                probs["attr_fused_d%d" % d] = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        extra = {name: refilled[name] for name in self.names if not name.startswith("attr_")}
        return probs, extra

    def publish_fields(self):
        """{publish_attr_d{d}: 每窗实际发布的窗号}。"""
        return {"publish_attr_d%d" % d: self.publish[d] for d in self.delays}

    def bank_info(self):
        """假设库的耗时与计数（AttributionStats.update 的 bank_info）。"""
        return dict(self.time, windows=self.n_windows, stats=dict(self.bank.stats))


class AttributionStats(object):
    """V2-2 的诊断统计（按等待窗数，跨序列累加）：修正覆盖了多少目标 / 背景事件、平均修正量与方向，
    以及假设库的生成、确认、合并、结束次数、平均存活数与耗时。只读 run_sequence 的导出，不影响任何读出。"""

    def __init__(self, delays):
        self.delays = list(delays)
        self.per = {d: {"target": 0, "background": 0, "target_case1": 0, "target_case2": 0, "target_case3": 0,
                        "background_case1": 0, "background_case2": 0, "background_case3": 0,
                        "target_up": 0, "background_down": 0, "background_up": 0,
                        "target_delta_sum": 0.0, "background_delta_sum": 0.0} for d in self.delays}
        self.bank = {"seconds": 0.0, "windows": 0, "alive": 0, "confirmed": 0, "births_snn": 0, "births_tube": 0,
                     "confirmed_total": 0, "merged": 0, "ended": 0}
        self.confirm_delays = []

    def update(self, labels, extra, bank_info):
        target = np.asarray(labels) > 0.5
        for d in self.delays:
            delta, case = extra["delta_attr_d%d" % d], extra["case_attr_d%d" % d]
            p, cov = self.per[d], case > 0
            for tag, mask in (("target", target), ("background", ~target)):
                p[tag] += int(mask.sum())
                p[tag + "_case1"] += int((mask & (case == 1)).sum())
                p[tag + "_case2"] += int((mask & (case == 2)).sum())
                p[tag + "_case3"] += int((mask & (case == 3)).sum())
                p[tag + "_delta_sum"] += float(delta[mask & cov].sum())
            p["target_up"] += int((target & cov & (delta > 0)).sum())
            p["background_down"] += int((~target & cov & (delta < 0)).sum())
            p["background_up"] += int((~target & cov & (delta > 0)).sum())
        if bank_info:
            b = self.bank
            for key in ("seconds", "windows", "alive", "confirmed"):
                b[key] += bank_info[key]
            st = bank_info["stats"]
            for key in ("births_snn", "births_tube", "merged", "ended"):
                b[key] += st[key]
            b["confirmed_total"] += st["confirmed"]
            self.confirm_delays += list(st["confirm_delays"])

    def summary(self):
        out = {}
        for d, p in self.per.items():
            cov_t = p["target_case1"] + p["target_case2"] + p["target_case3"]
            cov_b = p["background_case1"] + p["background_case2"] + p["background_case3"]
            out["d%d" % d] = dict(p, target_covered_frac=cov_t / float(max(p["target"], 1)),
                                  background_covered_frac=cov_b / float(max(p["background"], 1)),
                                  target_delta_mean=p["target_delta_sum"] / float(max(cov_t, 1)),
                                  background_delta_mean=p["background_delta_sum"] / float(max(cov_b, 1)),
                                  target_up_frac=p["target_up"] / float(max(cov_t, 1)),
                                  background_down_frac=p["background_down"] / float(max(cov_b, 1)))
        b = dict(self.bank)
        w = float(max(b["windows"], 1))
        b.update(ms_per_window=1000.0 * b["seconds"] / w, alive_per_window=b["alive"] / w,
                 confirmed_per_window=b["confirmed"] / w,
                 confirm_delay_median=float(np.median(self.confirm_delays)) if self.confirm_delays else None)
        out["bank"] = b
        return out


def report_lines(summary, mode):
    """AttributionStats.summary() 的打印行（eval 模式输出到日志）。"""
    b = summary["bank"]
    lines = ["[%s] V2-2 假设库：每窗存活 %.1f / 已确认 %.1f，生成 SNN %d + 管道 %d，确认 %d（中位延迟 %s 窗），"
             "合并 %d，结束 %d，耗时 %.1f ms/窗" % (
                 mode, b["alive_per_window"], b["confirmed_per_window"], b["births_snn"], b["births_tube"],
                 b["confirmed_total"], b["confirm_delay_median"], b["merged"], b["ended"], b["ms_per_window"])]
    for key, p in summary.items():
        if key == "bank":
            continue
        lines.append("[%s] V2-2 %s：修正覆盖 目标 %.1f%% / 背景 %.3f%%；平均修正 目标 %+.3f / 背景 %+.3f；"
                     "目标上调 %.1f%%，背景下调 %.1f%%；情形 1/2/3 目标 %d/%d/%d，背景 %d/%d/%d" % (
                         mode, key, 100 * p["target_covered_frac"], 100 * p["background_covered_frac"],
                         p["target_delta_mean"], p["background_delta_mean"], 100 * p["target_up_frac"],
                         100 * p["background_down_frac"], p["target_case1"], p["target_case2"], p["target_case3"],
                         p["background_case1"], p["background_case2"], p["background_case3"]))
    return lines
