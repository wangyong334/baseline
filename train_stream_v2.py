"""流式 SNN V2（方案一，设计 v2-1）的训练与评估入口。

结构：证据前端（无参数）-> V1 电流合并解码器的脉冲 U-Net 骨干 -> mark 头 + 强度头 -> 漂移 CUSUM 判决（无参数）
    mark 头   零延迟的逐事件判断，逐事件 BCE 训练（无 pos_weight）
    强度头    本窗的目标强度场 g（每像素每窗目标事件数），对 3x3 平滑后的目标计数做泊松负对数似然训练；
              CUSUM 把上一窗的 g 按各速度假设平移，作为本窗"可预测"的目标强度
    CUSUM     只在评估时运行：把上一窗的强度场按速度假设平移作为本窗预测，得到各管道的逐窗证据；
              延迟 d 窗的逐事件读出 = 网络 log-odds（先验）+ 之后 d 窗沿各速度管道的似然比（TubeReadout）
    V2-2      只在评估时运行（--attr on，默认）：稀疏目标假设库（model/target_hypotheses.py）+ 回溯分布修正读出
              （model/attribution_readout.py）；attr_d{d} = sigmoid(mark + w * Delta)，设计页式 (13)-(17)
模式（--mode）：
    smoke    校准增益 + 1 个训练序列前 32 窗的一次更新，打印损失、梯度、发放率、显存、耗时
    overfit  单序列过拟合诊断
    train    正式训练：每个序列一次参数更新；每轮在 val 上评估 net 读出（mark 头）并据此保存最优
    eval     评估 checkpoint（默认只跑 carry，见 eval_state_modes）；读出 net（零延迟）、fused_d{d}（V2-1）
             与 attr_d{d}（V2-2），都在第 k+d 窗末发布
训练与评估都按片段执行（骨干逐层时间并行、事件常驻设备），没有逐窗计时；主指标调用原 utils/eval.py。
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")   # 必须在导入 torch 之前设置

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream, make_sequence_loader  # noqa: E402
from dataset.stream_features import FEATURE_GROUPS, EventChunkSource, EvidenceFrontEnd  # noqa: E402
from dataset.stream_windows import refill_by_index  # noqa: E402
from model.attribution_readout import VARIANTS as ATTR_VARIANTS, AttributionReadout  # noqa: E402
from model.evidence_neuron import DriftCUSUM, TubeReadout, velocity_grid  # noqa: E402
from model.evidence_snn import EvidenceSNN  # noqa: E402
from model.evspsegnet_stream import calibrate_gains, count_parameters, estimate_operations  # noqa: E402
from model.lif2d_stream import detach_states  # noqa: E402
from model.target_hypotheses import DEFAULT_PARAMS as HYP_DEFAULTS, UPDATES as HYP_UPDATES, HypothesisBank  # noqa: E402
from train_stream_v1 import (LayerMonitor, peak_memory_gib, seed_everything, tau_statistics,  # noqa: E402
                             train_file_names, write_json)
from utils.alarm_metrics import AlarmEvaluator  # noqa: E402
from utils.evidence_loss import intensity_loss, mark_loss  # noqa: E402
from utils.stream_common import (health_warnings, linear_epoch_lr, load_flat_config, make_chunks,  # noqa: E402
                                 select_subset, subset_due, summarize_history)
from utils.stream_metrics import (first_detection_latencies_published, rolling_iou, segment_iou,  # noqa: E402
                                  summarize_latencies, window_confusion)

DESIGN_VERSION = "v2-2"
SOURCE_FILES = ("train_stream_v2.py", "dataset/stream_features.py", "dataset/stream_windows.py",
                "dataset/ev_uav_stream.py", "model/evidence_neuron.py", "model/evidence_snn.py",
                "model/target_hypotheses.py", "model/attribution_readout.py",
                "model/evspsegnet_stream.py", "model/lif2d_stream.py", "utils/evidence_loss.py",
                "utils/alarm_metrics.py", "utils/stream_common.py", "utils/stream_metrics.py", "utils/eval.py")
PRIMARY = "net"
# 判决层活跃比例统计的门控档位：门控后的 G 恰好是未门控 G 按 ε 截断，所以一次 ε=0 的运行就能读出各档
ACTIVITY_EPS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
# 每个读出另报这些阈值下的 IoU / ACC（只统计计数，几乎不花时间）：val 上选阈值、冻结到 test 不必再跑 Pd/Fa 扫描
IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# 决定前端输入、网络结构与神经元动力学的配置项。膜电位上下界、bg_mode 不在 state_dict 里，
# 续训或评估时 load_state_dict 查不出它们不一致，只能按配置核对（见 config_drift）
MODEL_KEYS = ("window_ms", "fe_taus_ms", "fe_dipole_taus_ms", "fe_dipole_radius", "bg_fast_ms", "bg_slow_ms",
              "bg_prior", "bg_prior_windows", "bg_floor", "bg_smooth_radius", "fe_features", "bg_mode",
              "channels", "neuron", "norm", "state_mode", "v_threshold", "neuron_u_floor", "neuron_u_ceil",
              "tau_init_ms", "tau_min_ms", "tau_max_ms", "merged_decoder", "head_hidden", "mark_prior",
              "intensity_prior", "log_g_max")
# 续训还要求优化过程一致（epochs 除外：延长训练是合理操作，只提示不拦截）
TRAINING_KEYS = MODEL_KEYS + ("seed", "lr", "lr_end", "tbptt_k", "grad_clip", "loss_mark_weight",
                              "loss_intensity_weight", "loss_intensity_smooth")
# 旧 checkpoint 里没有、后来才加的键，缺省值 = 加入之前的行为
CONFIG_DEFAULTS = {"fe_features": list(FEATURE_GROUPS), "bg_mode": "adaptive", "loss_intensity_smooth": 3,
                   "neuron_u_floor": None, "neuron_u_ceil": None}
# 需要重新训练才生效的命令行参数（参数名, 配置键）：eval 模式一律取 checkpoint 的值，给了不同的值就报错
RETRAIN_FLAGS = (("neuron", "neuron"), ("state_mode", "state_mode"), ("fe_features", "fe_features"),
                 ("bg_mode", "bg_mode"), ("fe_taus_ms", "fe_taus_ms"), ("fe_dipole_taus_ms", "fe_dipole_taus_ms"),
                 ("neuron_u_floor", "neuron_u_floor"), ("neuron_u_ceil", "neuron_u_ceil"),
                 ("loss_mark_weight", "loss_mark_weight"), ("loss_intensity_weight", "loss_intensity_weight"))


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------


def parse_args():
    """解析命令行参数。模型与训练超参全部来自 YAML，命令行只放本次运行相关的覆盖项。"""
    parser = argparse.ArgumentParser(description="Streaming SNN V2 (evidence / CUSUM)")
    parser.add_argument("--config", default="configs/evisseg_stream_v2.yaml")
    parser.add_argument("--mode", choices=("smoke", "overfit", "train", "eval"), required=True)
    parser.add_argument("--save-root", default=None, help="输出目录，默认取 YAML 的 save_root")
    parser.add_argument("--data-root", default=None, help="覆盖 YAML 中的数据根目录（例如合成数据）")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--state-mode", choices=("carry", "reset_each_window"), default=None)
    parser.add_argument("--neuron", choices=("lif", "graded", "relu"), default=None)
    parser.add_argument("--checkpoint", default=None, help="eval 模式使用的权重文件")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sequence", default=None, help="overfit 模式的序列文件名")
    parser.add_argument("--steps", type=int, default=300, help="overfit 模式的参数更新次数")
    parser.add_argument("--resume", action="store_true", help="train 模式从 save_root/last.pt 继续")
    parser.add_argument("--overwrite", action="store_true", help="eval 模式允许覆盖已有结果文件")
    parser.add_argument("--dump-dir", default=None, help="eval 模式：保存逐事件预测 NPZ（字段兼容 tools/sweep_threshold.py）")
    parser.add_argument("--train-subset-every", type=int, default=None)
    parser.add_argument("--tbptt-k", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 YAML 中的 epochs（本地冒烟用）")
    parser.add_argument("--tag", default=None, help="eval 模式：结果文件名后缀，用于区分同一 checkpoint 的多组判决层设置")
    parser.add_argument("--cusum-velocities", type=float, nargs="+", default=None,
                        help="覆盖速度假设的单轴取值，例如 --cusum-velocities 0（只保留静止假设，做运动补偿的对照）")
    parser.add_argument("--cusum-aggregate", choices=("sum", "mean", "lme"), default=None,
                        help="足迹证据聚合：lme/mean 对空间相关稳健，sum 要求像素独立")
    parser.add_argument("--cusum-track-tau-ms", type=float, default=None, help="管道强度记忆的时间常数，0 关闭")
    parser.add_argument("--cusum-footprint", type=int, default=None, help="证据聚合足迹边长（奇数，1 = 逐像素不聚合）")
    parser.add_argument("--cusum-memory-gain", type=float, default=None,
                        help="非对称记忆 λ ∈ [0,1]：加分用 max(新鲜强度, λ·rho·旧 G)，扣分仍用带记忆的 G；1 = 原实现")
    parser.add_argument("--cusum-gate-eps", type=float, default=None,
                        help="预测门控 ε：低于 ε 的强度与记忆置 0（稀疏执行的前提）；0 = 原实现")
    parser.add_argument("--cusum-reset-radius", type=int, default=None,
                        help="告警时复位邻域半径 r 内的全部假设（清掉共享过目标证据的兄弟管道）；0 = 原实现")
    parser.add_argument("--alarm-thetas", type=float, nargs="+", default=None,
                        help="打开位置级告警（可选扩展）并给出阈值列表；YAML 默认不算告警")
    parser.add_argument("--no-alarms", action="store_true", help="不算位置级告警（覆盖 YAML）")
    parser.add_argument("--eval-state-modes", nargs="+", choices=("carry", "reset_each_window"), default=None,
                        help="eval 模式跑哪几种骨干状态（默认取 YAML 的 eval_state_modes，即只跑 carry）")
    parser.add_argument("--attr", choices=("on", "off"), default=None,
                        help="V2-2 回溯分布修正读出；off 时与 V2-1 baseline 的读出逐位相同")
    parser.add_argument("--attr-delays", type=int, nargs="+", default=None, help="V2-2 固定等待的窗数，例如 1 2 5 10")
    parser.add_argument("--attr-weight", type=float, default=None, help="式 (17) 的修正权重 w（在验证集上校准）")
    parser.add_argument("--attr-variants", nargs="*", choices=ATTR_VARIANTS, default=None,
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
    parser.add_argument("--readout-delays", type=int, nargs="+", default=None, help="逐事件延迟读出的窗数")
    parser.add_argument("--eval-sequences", type=int, default=0,
                        help="eval 模式只用该划分按文件名排序的前 N 条序列（冒烟用；0 = 全部）。结果不能当正式数字")
    parser.add_argument("--eval-names", nargs="+", default=None,
                        help="eval 模式只评估这些序列（文件名，可省略 .npz），例如 --eval-names test_022 test_023")
    parser.add_argument("--fe-features", nargs="+", choices=FEATURE_GROUPS, default=None,
                        help="前端输出哪几组特征（消融；改变输入通道数，需要重新训练）")
    parser.add_argument("--bg-mode", choices=("adaptive", "constant"), default=None,
                        help="adaptive 自适应背景（默认）| constant 关掉背景归一化（消融；需要重新训练）")
    parser.add_argument("--fe-taus-ms", type=float, nargs="+", default=None,
                        help="覆盖时间矩的时间尺度（消融，例如去掉 2 s 尺度；需要重新训练）")
    parser.add_argument("--fe-dipole-taus-ms", type=float, nargs="+", default=None,
                        help="覆盖偶极所用的时间尺度（必须是 fe_taus_ms 的子集；需要重新训练）")
    parser.add_argument("--loss-mark-weight", type=float, default=None, help="逐事件 BCE 的权重")
    parser.add_argument("--loss-intensity-weight", type=float, default=None,
                        help="强度场泊松似然的权重（设 0 = 只留 mark 头的单头消融；需要重新训练）")
    parser.add_argument("--neuron-u-floor", type=float, default=None,
                        help="跨窗携带的膜电位下界（以 v_th 为单位，例如 -1）。不给 = 不设界，与旧实现逐位相同；"
                             "需要重新训练。动机见 09-23 的膜电位诊断：深层被压到 -30~-60 个阈值后无法恢复")
    parser.add_argument("--neuron-u-ceil", type=float, default=None,
                        help="跨窗携带的膜电位上界（以 v_th 为单位，例如 4）。不给 = 不设界；需要重新训练")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_config(args):
    """读取 YAML 并用命令行覆盖。返回展平的配置字典。"""
    cfg = load_flat_config(args.config)
    overrides = {"seed": args.seed, "state_mode": args.state_mode, "neuron": args.neuron,
                 "save_root": args.save_root, "root": args.data_root, "train_subset_every": args.train_subset_every,
                 "tbptt_k": args.tbptt_k, "epochs": args.epochs, "cusum_axis_velocities": args.cusum_velocities,
                 "cusum_aggregate": args.cusum_aggregate, "cusum_track_tau_ms": args.cusum_track_tau_ms,
                 "cusum_footprint": args.cusum_footprint, "cusum_memory_gain": args.cusum_memory_gain,
                 "cusum_gate_eps": args.cusum_gate_eps, "cusum_reset_radius": args.cusum_reset_radius,
                 "alarm_thetas": args.alarm_thetas,
                 "readout_delays": args.readout_delays, "fe_features": args.fe_features, "bg_mode": args.bg_mode,
                 "fe_taus_ms": args.fe_taus_ms, "fe_dipole_taus_ms": args.fe_dipole_taus_ms,
                 "loss_mark_weight": args.loss_mark_weight, "loss_intensity_weight": args.loss_intensity_weight,
                 "neuron_u_floor": args.neuron_u_floor, "neuron_u_ceil": args.neuron_u_ceil,
                 "attr_delays": args.attr_delays, "attr_weight": args.attr_weight, "attr_variants": args.attr_variants,
                 "hyp_update": args.attr_update, "attr_cap": args.attr_cap, "attr_eps": args.attr_eps,
                 "attr_ref_radius": args.attr_ref_radius, "attr_min_support": args.attr_min_support}
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if args.no_alarms:
        cfg["alarm_thetas"] = []
    if args.attr is not None:
        cfg["attr"] = args.attr == "on"
    if args.attr_tube_birth is not None:
        cfg["attr_tube_birth"] = args.attr_tube_birth == "on"
    if args.hyp:
        cfg["hyp_params"] = dict(cfg.get("hyp_params") or {}, **parse_hyp_overrides(args.hyp))
    validate_config(cfg)
    return cfg


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
    """构造任何模块之前的一致性检查：没有意义的组合直接报错，而不是训练几小时后才发现白跑了。"""
    if int(cfg["tbptt_k"]) < 1 or int(cfg["train_subset_every"]) < 1 or int(cfg["eval_chunk"]) < 1:
        raise ValueError("tbptt_k、train_subset_every、eval_chunk 必须 >= 1")
    w_mark, w_int = float(cfg["loss_mark_weight"]), float(cfg["loss_intensity_weight"])
    if w_mark < 0 or w_int < 0 or (w_mark == 0 and w_int == 0):
        raise ValueError("损失权重必须非负且不能同时为 0（mark %g，intensity %g）" % (w_mark, w_int))
    if cfg.get("neuron_u_floor") is not None or cfg.get("neuron_u_ceil") is not None:
        if cfg["neuron"] == "relu":
            raise ValueError("ReLU 对照没有膜电位，neuron_u_floor / neuron_u_ceil 对它无效")
        if cfg["state_mode"] == "reset_each_window":
            raise ValueError("reset_each_window 不携带跨窗膜电位，neuron_u_floor / neuron_u_ceil 对它无效")
    if cfg.get("attr"):
        delays = [int(d) for d in cfg.get("attr_delays") or []]
        if not delays or min(delays) < 1:
            raise ValueError("attr_delays 至少一个且都 >= 1")
        unknown = set(cfg.get("attr_variants") or []) - set(ATTR_VARIANTS)
        if unknown:
            raise ValueError("未知的对照读出: %s" % sorted(unknown))
        bad = set(cfg.get("hyp_params") or {}) - set(HYP_DEFAULTS)
        if bad:
            raise ValueError("未知的假设库参数: %s" % sorted(bad))
        if cfg.get("hyp_update", "separate") not in HYP_UPDATES:
            raise ValueError("hyp_update 必须是 %s 之一" % (HYP_UPDATES,))


def normalized(value):
    """配置值的比较形式：元组转列表、数值转浮点（YAML 的 1 与命令行的 1.0 视为相同，布尔值保持原样）。"""
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def config_value(cfg, key):
    """取一项配置用于比较；旧配置里没有的键按加入之前的行为取缺省值（CONFIG_DEFAULTS）。"""
    return normalized(cfg.get(key, CONFIG_DEFAULTS.get(key)))


def config_drift(saved, current, keys):
    """两份配置在 keys 上的差异 {键: (saved 的值, current 的值)}；没有差异时为空字典。"""
    return {key: (config_value(saved, key), config_value(current, key)) for key in keys
            if config_value(saved, key) != config_value(current, key)}


def eval_conflicts(args, cfg, ckpt_cfg):
    """eval 模式下命令行给了、但与 checkpoint 取值不同的"需要重新训练"的参数（它们不会生效，只会让结果被贴错标签）。"""
    out = []
    for attr, key in RETRAIN_FLAGS:
        if getattr(args, attr, None) is not None and config_value(cfg, key) != config_value(ckpt_cfg, key):
            out.append("--%s（命令行 %r，checkpoint %r）" % (attr.replace("_", "-"), cfg.get(key), ckpt_cfg.get(key)))
    return out


def build_frontend(cfg):
    """按配置构造证据前端（无参数）。"""
    return EvidenceFrontEnd(
        cfg["fe_taus_ms"], cfg["window_ms"], cfg["fe_dipole_taus_ms"], cfg["fe_dipole_radius"],
        cfg["bg_fast_ms"], cfg["bg_slow_ms"], cfg["bg_prior"], cfg["bg_prior_windows"], cfg["bg_floor"],
        cfg["bg_smooth_radius"], cfg.get("fe_features", FEATURE_GROUPS), cfg.get("bg_mode", "adaptive"))


def build_model(cfg, frontend):
    """按配置构造网络（骨干 + 两个头）。"""
    return EvidenceSNN(
        frontend.n_features, tuple(cfg["channels"]), cfg["neuron"], cfg["norm"], cfg["state_mode"],
        float(cfg["window_ms"]), float(cfg["tau_init_ms"]), float(cfg["tau_min_ms"]), float(cfg["tau_max_ms"]),
        float(cfg["v_threshold"]), bool(cfg["merged_decoder"]), int(cfg["head_hidden"]),
        float(cfg["mark_prior"]), float(cfg["intensity_prior"]), float(cfg["log_g_max"]),
        cfg.get("neuron_u_floor"), cfg.get("neuron_u_ceil"))


def build_cusum(cfg):
    """按配置构造漂移 CUSUM 判决层（无参数）。管道强度记忆的衰减 rho = exp(-窗长 / cusum_track_tau_ms)，0 表示关闭。
    三个评估时开关（非对称记忆 λ、门控 ε、复位半径 r）缺省时取原实现的值。"""
    tau = float(cfg.get("cusum_track_tau_ms", 0.0))
    decay = math.exp(-float(cfg["window_ms"]) / tau) if tau > 0 else 0.0
    return DriftCUSUM(velocity_grid(cfg["cusum_axis_velocities"]), int(cfg["cusum_footprint"]),
                      cfg["cusum_compensator"], cfg.get("cusum_nb_kappa"), cfg.get("cusum_aggregate", "lme"), decay,
                      float(cfg.get("cusum_memory_gain", 1.0)), float(cfg.get("cusum_gate_eps", 0.0)),
                      int(cfg.get("cusum_reset_radius", 0)))


def build_all(cfg, device):
    """构造前端、网络、CUSUM 并移动到设备。"""
    frontend = build_frontend(cfg).to(device)
    model = build_model(cfg, frontend).to(device)
    return frontend, model, build_cusum(cfg)


def source_hashes():
    """记录本次运行所用源码的 sha256。"""
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES if (here / name).exists()}


def gradient_group_norms(model):
    """按模块分组的梯度 L2 范数（骨干按层分组，如 backbone.enc1；头记为 head）。"""
    squares = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] == "backbone" else parts[0]
        value = 0.0 if param.grad is None else float(param.grad.detach().double().pow(2).sum())
        squares[group] = squares.get(group, 0.0) + value
    return {k: math.sqrt(v) for k, v in squares.items()}


def readout_names(cfg):
    """全部读出的名称：net（mark 头，零延迟）、fused_d{d}（V2-1：网络先验 + 之后 d 窗的管道证据）、
    attr_d{d}（V2-2 回溯分布修正）与 attr_<对照>_d{d}（对照读出，只算 IoU/ACC）。"""
    names = [PRIMARY] + ["fused_d%d" % int(d) for d in cfg["readout_delays"]]
    if cfg.get("attr"):
        delays = [int(d) for d in cfg["attr_delays"]]
        names += ["attr_d%d" % d for d in delays]
        names += ["attr_%s_d%d" % (v, d) for v in ATTR_VARIANTS if v in (cfg.get("attr_variants") or [])
                  for d in delays]
    return names


def readout_delay(name):
    """读出名里的等待窗数（net 为 0）。"""
    return 0 if name == PRIMARY else int(name.rsplit("_d", 1)[1])


def is_variant_readout(name):
    """V2-2 的对照读出（只算 IoU/ACC）：attr_<对照>_d{d}，不含主读出 attr_d{d}。"""
    return name.startswith("attr_") and not name[len("attr_d"):].isdigit()


def publish_field(name):
    """读出对应的发布窗号字段名。"""
    return ("publish_attr_d%d" if name.startswith("attr_") else "publish_d%d") % readout_delay(name)


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


# ---------------------------------------------------------------------------
# 增益校准
# ---------------------------------------------------------------------------


def calibration_factory(cfg, frontend, device):
    """增益校准数据：固定种子选出的训练序列，每个取前 calib_windows 窗，经前端得到特征。"""
    names = select_subset(train_file_names(cfg), int(cfg["calib_sequences"]), int(cfg["seed"]) + 1000)
    dataset = EvUAVStream(cfg["root"], "train", cfg, names)
    n_windows = int(cfg["calib_windows"])
    H, W = int(cfg["pad_height"]), int(cfg["pad_width"])

    def windows_of(seq):
        """按时间顺序产出一个序列前 n_windows 窗的 (features, None)。"""
        source = EventChunkSource(seq, cfg, device)
        state = frontend.init_state(1, H, W, device)
        _, feats, _, _ = frontend.run_chunk(state, source.chunk(0, min(n_windows, seq.n_windows)))
        for t in range(int(feats.shape[0])):
            yield feats[t], None

    def factory():
        for i in range(len(dataset)):
            yield windows_of(dataset[i])

    return factory, names


def run_calibration(model, frontend, cfg, device, out_dir):
    """一次性逐层增益校准（与 V1 相同的规则），报告写到 out_dir/calibration.json。"""
    factory, names = calibration_factory(cfg, frontend, device)
    reports = calibrate_gains(model.backbone, factory, float(cfg["calib_quantile"]),
                              int(cfg["calib_samples_per_channel"]), int(cfg["calib_min_positive"]),
                              float(cfg["calib_gain_min"]), float(cfg["calib_gain_max"]), int(cfg["seed"]) + 4000)
    write_json(Path(out_dir) / "calibration.json", {"sequences": names, "layers": reports})
    for r in reports:
        g = np.asarray(r["gain"])
        print("[calib] %-5s gain min/mean/max = %.3f / %.3f / %.3f  fallback=%d/%d" % (
            r["layer"], g.min(), g.mean(), g.max(), sum(r["fallback"]), len(r["fallback"])), flush=True)
    return reports


# ---------------------------------------------------------------------------
# 训练与推理
# ---------------------------------------------------------------------------


def train_sequence(model, frontend, seq, optimizer, cfg, device, rng, max_windows=None):
    """用一个序列训练一次（stateful TBPTT，梯度跨片段累加，序列结束时裁剪并更新一次）。

    每个片段：前端（no_grad）-> 骨干 + 头（逐层时间并行）-> 两项损失 -> backward -> 截断骨干状态。
    损失除以整条序列的事件数（没有事件时除以 1）。
    返回: {"loss_sum", "mark_sum", "intensity_sum", "events", "grad_norms", "missing_grads"}
    """
    k = int(cfg["tbptt_k"])
    n_windows = seq.n_windows if max_windows is None else min(int(max_windows), seq.n_windows)
    chunks = make_chunks(n_windows, k, int(rng.randint(1, k + 1)))
    total_events = int(seq.bounds[n_windows] - seq.bounds[0])
    denominator = float(max(total_events, 1))
    w_mark, w_int = float(cfg["loss_mark_weight"]), float(cfg["loss_intensity_weight"])
    dtype = next(model.parameters()).dtype
    source = EventChunkSource(seq, cfg, device, dtype)
    fe_state = frontend.init_state(1, int(cfg["pad_height"]), int(cfg["pad_width"]), device, dtype)
    states = None
    sums = {"loss_sum": 0.0, "mark_sum": 0.0, "intensity_sum": 0.0}
    optimizer.zero_grad(set_to_none=True)
    for start, end in chunks:
        chunk = source.chunk(start, end)
        with torch.no_grad():
            fe_state, feats, _, _ = frontend.run_chunk(fe_state, chunk)
        mark, log_g, states, _ = model.forward_chunk(feats, states)
        ev = chunk["events"]
        terms = []
        lm = mark_loss(mark[ev["t"], ev["b"], 0, ev["y"], ev["x"]], chunk["labels"])
        if lm is not None and w_mark > 0:
            terms.append(w_mark * lm)
            sums["mark_sum"] += float(lm.detach())
        li = intensity_loss(log_g, chunk["target_counts"], int(cfg.get("loss_intensity_smooth", 3)))
        if w_int > 0:
            terms.append(w_int * li)
            sums["intensity_sum"] += float(li.detach())
        if terms:
            loss = terms[0] if len(terms) == 1 else terms[0] + terms[1]
            (loss / denominator).backward()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError("损失出现非有限值: 序列 %s 窗口 %d-%d" % (seq.name, start, end))
            sums["loss_sum"] += value
        states = detach_states(states)
    norms = gradient_group_norms(model)
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
    optimizer.step()
    sums.update({"events": total_events, "grad_norms": norms, "missing_grads": missing})
    return sums


class AttributionStats(object):
    """V2-2 的诊断统计（按等待窗数，跨序列累加）：修正覆盖了多少目标 / 背景事件、平均修正量与方向，
    以及假设库的生成、确认、合并、结束次数、平均存活数与耗时。只读 run_sequence 的导出，不影响任何读出。"""

    def __init__(self, delays):
        self.delays = list(delays)
        self.per = {d: {"target": 0, "background": 0, "target_case1": 0, "target_case2": 0, "background_case1": 0,
                        "background_case2": 0, "target_up": 0, "background_down": 0, "background_up": 0,
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
            cov_t, cov_b = p["target_case1"] + p["target_case2"], p["background_case1"] + p["background_case2"]
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


class DecisionActivity(object):
    """判决层的活跃比例：稀疏同步执行时每窗需要计算的 (假设, 像素) 占比。

    每窗对每个门控档位 ε 统计：
        active     带记忆的强度 G >= ε 的 (假设, 像素) 占比
        footprint  按足迹膨胀后的占比——足迹内有一个活跃像素，该神经元的聚合证据就不为 0，需要计算
        c_positive 读出膜电位 C > 0 的占比（需要存储状态的神经元；不门控时的值，只作参考）
    只统计 >= 当前 gate_eps 的档位（门控后更低的档位没有意义），并总是包含 gate_eps 本身（> 0 时）。
    统计在设备上累加，序列结束时不做同步；summary() 时才取回。
    """

    def __init__(self, eps_list, cusum):
        gate = float(cusum.gate_eps)
        eps = sorted(set([float(e) for e in eps_list if float(e) >= gate] + ([gate] if gate > 0 else [])))
        self.eps, self.footprint = eps, int(cusum.footprint)
        self.active = [0.0] * len(eps)
        self.dilated = [0.0] * len(eps)
        self.c_positive = 0.0
        self.windows = 0

    def update(self, state):
        G, C = state["G"], state["C"]
        f = self.footprint
        for i, eps in enumerate(self.eps):
            act = (G >= eps).to(G.dtype)
            self.active[i] = self.active[i] + act.mean()
            dil = act if f == 1 else F.max_pool2d(act, f, stride=1, padding=f // 2)
            self.dilated[i] = self.dilated[i] + dil.mean()
        self.c_positive = self.c_positive + (C > 0).to(C.dtype).mean()
        self.windows += 1

    def summary(self):
        n = float(max(self.windows, 1))
        return {"eps": self.eps, "windows": self.windows,
                "active_fraction": [float(a) / n for a in self.active],
                "footprint_active_fraction": [float(d) / n for d in self.dilated],
                "c_positive_fraction": float(self.c_positive) / n}


def run_sequence(model, frontend, cusum, seq, cfg, device, state_mode, monitor=None, with_cusum=True,
                 input_stats=None, alarm_eval=None, feature_probe=None, activity=None):
    """推理一个完整序列（全部窗口，按时间顺序，按 eval_chunk 分片段执行）。

    返回: (probs, extra, confusion)
        probs      {读出名: 按文件原始事件顺序回填的概率 numpy float32 [N]}
                   net       sigmoid(mark)：本窗结束即发布，不做额外的跨窗等待
                   fused_d   sigmoid(mark + w*F(d))：轨迹证据融合分数，F 为之后 d 窗沿各速度管道的证据
                             （TubeReadout），w = fusion_weight（在验证集上校准，见 tools/calibrate_fusion.py）
        extra      {"logit_net": mark 的 logit，"evidence_d{d}": F(d)，"publish_d{d}": 每窗实际发布的窗号 [n_windows]}
                   （序列末尾不足 d 窗时在最后一窗发布，延迟被截断）
                   V2-2 打开时（cfg["attr"]）另有：probs 里的 attr_d{d} = sigmoid(mark + w*Delta) 与对照读出
                   attr_<对照>_d{d}（direct 直接是概率）；extra 里的 delta_attr_d{d}（修正量）、case_attr_d{d}
                   （0 无修正 / 1 快照参照 / 2 固定参考）、publish_attr_d{d}，以及 attr_bank（假设库统计，非数组）
        confusion  [n_windows, 4] net 读出每窗 TP/FP/FN/正事件数
    alarm_eval 不为空时，另外按 alarm_thetas 各维护一份带复位的膜电位，把每窗的位置级告警图交给它（utils/alarm_metrics.py）。
    feature_probe 不为空时，每个片段调用一次 feature_probe(feats, mu0)（诊断用钩子，见 tools/diagnose_membrane.py）。
    activity 不为空时，每窗把判决层状态交给它统计活跃比例（DecisionActivity）。
    调用方负责 torch.no_grad() 与 model.eval()。
    """
    threshold = float(cfg["threshold"])
    H, W = int(cfg["pad_height"]), int(cfg["pad_width"])
    dtype = next(model.parameters()).dtype
    source = EventChunkSource(seq, cfg, device, dtype)
    fe_state = frontend.init_state(1, H, W, device, dtype)
    delays = [int(d) for d in cfg["readout_delays"]] if with_cusum else []
    weight = float(cfg.get("fusion_weight", 1.0))
    c_state = cusum.init_state(1, H, W, device, dtype) if with_cusum else None
    readout = TubeReadout(cusum, delays) if with_cusum else None
    alarm_C = {}
    if alarm_eval is not None and with_cusum:
        alarm_eval.begin_sequence(seq, seq.n_windows)
        alarm_C = {theta: c_state["C"].clone() for theta in alarm_eval.thetas}
    prev_log_g = None
    names = [PRIMARY, "logit_net"] + ["fused_d%d" % d for d in delays] + ["evidence_d%d" % d for d in delays]
    bank = attr_readout = None
    attr_delays, attr_variants, attr_weight = [], [], float(cfg.get("attr_weight", 1.0))
    if with_cusum and cfg.get("attr"):
        bank, attr_readout = build_attribution(cfg, cusum)
        attr_delays, attr_variants = list(attr_readout.delays), list(attr_readout.variants)
        for d in attr_delays:
            names += ["attr_d%d" % d, "delta_attr_d%d" % d, "case_attr_d%d" % d]
            names += ["attr_%s_d%d" % (v, d) for v in attr_variants]
    attr_prob_names = [n for n in names if n.startswith("attr_")]
    attr_publish = {d: np.zeros(seq.n_windows, dtype=np.int64) for d in attr_delays}
    attr_time = {"seconds": 0.0, "alive": 0, "confirmed": 0}
    parts = {name: ([], []) for name in names}
    publish = {d: np.zeros(seq.n_windows, dtype=np.int64) for d in delays}
    window_info = {}                                 # 窗号 -> (原始事件下标, 网络 logit)，延迟读出用
    confusion = np.zeros((seq.n_windows, 4), dtype=np.int64)
    states = None
    size = int(cfg["eval_chunk"])
    n = seq.n_windows

    def collect(results):
        for key, delay, scores, published in results:
            idx, logit = window_info[key]
            parts["fused_d%d" % delay][0].append(idx)
            parts["fused_d%d" % delay][1].append(torch.sigmoid(logit + weight * scores).float().cpu().numpy())
            parts["evidence_d%d" % delay][0].append(idx)
            parts["evidence_d%d" % delay][1].append(scores.float().cpu().numpy())
            publish[delay][key] = published

    def collect_attr(results):
        for key, delay, res, published in results:
            idx, logit = window_info[key]
            z = logit.detach().to("cpu", torch.float64)
            outputs = {"attr_d%d" % delay: torch.sigmoid(z + attr_weight * res["attr"]),
                       "delta_attr_d%d" % delay: res["attr"], "case_attr_d%d" % delay: res["case"].to(torch.float64)}
            for v in attr_variants:
                outputs["attr_%s_d%d" % (v, delay)] = res[v] if v == "direct" else torch.sigmoid(z + attr_weight * res[v])
            for name, value in outputs.items():
                parts[name][0].append(idx)
                parts[name][1].append(value.float().numpy())
            attr_publish[delay][key] = published

    for start in range(0, n, size):
        end = min(start + size, n)
        chunk = source.chunk(start, end)
        fe_state, feats, mu0, total = frontend.run_chunk(fe_state, chunk)
        if input_stats is not None:
            input_stats["nonzero"] = input_stats["nonzero"] + (feats != 0).sum()
            input_stats["elements"] += int(feats.numel())
        if feature_probe is not None:
            feature_probe(feats, mu0)
        mark, log_g, states, info = model.forward_chunk(feats, states, state_mode=state_mode,
                                                         collect=monitor is not None)
        if monitor is not None:
            monitor.update(info)
        ev = chunk["events"]
        logits = mark[ev["t"], ev["b"], 0, ev["y"], ev["x"]]
        prob = torch.sigmoid(logits).float().cpu().numpy()
        logit_np = logits.float().cpu().numpy()
        idx_all = chunk["idx"]
        offset = 0
        for t, k in enumerate(range(start, end)):
            count = int(seq.bounds[k + 1] - seq.bounds[k])
            part_idx, part_prob = idx_all[offset:offset + count], prob[offset:offset + count]
            parts[PRIMARY][0].append(part_idx)
            parts[PRIMARY][1].append(part_prob)
            parts["logit_net"][0].append(part_idx)
            parts["logit_net"][1].append(logit_np[offset:offset + count])
            confusion[k] = window_confusion(part_prob, seq.label[part_idx], threshold)
            if with_cusum:
                c_state, _, _ = cusum.step(c_state, total[t], mu0[t], prev_log_g)
                if activity is not None:
                    activity.update(c_state)
                sl = slice(offset, offset + count)
                window_info[k] = (part_idx, logits[sl])
                collect(readout.step(c_state, k, ev["b"][sl], ev["y"][sl], ev["x"][sl], k))
                if bank is not None:                  # V2-2：假设库处理第 k 窗，再读出到期的旧窗事件
                    t_attr = time.perf_counter()
                    g_k, mu_k = torch.exp(log_g[t, 0, 0]), mu0[t, 0, 0]
                    info_k = bank.step(k, ev["y"][sl], ev["x"][sl], torch.sigmoid(logits[sl]), g_k, mu_k, c_state["C"])
                    collect_attr(attr_readout.step(k, k, ev["y"][sl], ev["x"][sl], mu_k, g_k))
                    attr_time["seconds"] += time.perf_counter() - t_attr
                    attr_time["alive"] += info_k["alive"]
                    attr_time["confirmed"] += info_k["confirmed"]
                for theta in alarm_C:
                    alarm_C[theta], _, _, alarm = cusum.accumulate(alarm_C[theta], c_state["shifts"], c_state["ell"],
                                                                   theta, reset=True)
                    alarm_eval.update(theta, k, alarm[0, 0, :alarm_eval.height, :alarm_eval.width].cpu().numpy())
            prev_log_g = log_g[t]
            offset += count
    if with_cusum:
        collect(readout.flush())
    if bank is not None:
        collect_attr(attr_readout.flush())
    if alarm_C:
        alarm_eval.end_sequence()
    refilled = {name: refill_by_index(seq.n_events, idx_parts, val_parts)
                for name, (idx_parts, val_parts) in parts.items()}
    prob_names = [PRIMARY] + ["fused_d%d" % d for d in delays] + attr_prob_names
    probs = {name: refilled[name] for name in prob_names}
    extra = {name: refilled[name] for name in refilled if name not in probs}
    extra.update({"publish_d%d" % d: publish[d] for d in delays})
    if bank is not None:
        extra.update({"publish_attr_d%d" % d: attr_publish[d] for d in attr_delays})
        extra["attr_bank"] = dict(attr_time, windows=n, stats=dict(bank.stats))
    return probs, extra, confusion


def evaluate_split(model, frontend, cusum, cfg, device, split, state_mode, names=None, full=False,
                   collect_layers=False, dump_dir=None):
    """在一个划分（或子集）上评估。

    full=False：只算 net 读出的 IoU/ACC（训练中每轮验证用，不跑 CUSUM）。
    full=True ：两类输出分开报告——
        逐事件分割  net 与 fused_d 的 IoU/ACC/Pd/Fa（原 utils/eval.py）与首次检出延迟（按每窗实际发布时间）
        位置级告警  各 alarm_thetas 下的目标检出率、首次告警延迟、虚警连通域率与理论上界（utils/alarm_metrics.py）
    另有 net 读出的逐窗 IoU、分段 IoU、骨干理论运算量。
    """
    from utils.eval import evalute        # 延迟导入：该模块依赖 cv2/pandas

    dataset = EvUAVStream(cfg["root"], split, cfg, names)
    readouts = readout_names(cfg) if full else [PRIMARY]
    roc_readouts = [name for name in readouts if full and not is_variant_readout(name)]
    evaluators = {name: evalute(SimpleNamespace(roc=name in roc_readouts, pd_detT=cfg["pd_detT"],
                                                correct_thresh=cfg["correct_thresh"]))
                  for name in readouts}
    attr = AttributionStats([int(d) for d in cfg.get("attr_delays") or []]) if (full and cfg.get("attr")) else None
    monitor = LayerMonitor(model.v_threshold) if (collect_layers or full) else None
    input_stats = {"nonzero": 0, "elements": 0} if full else None
    alarm_eval = None
    if full and cfg.get("alarm_thetas"):
        alarm_eval = AlarmEvaluator(cfg["alarm_thetas"], cfg["height"], cfg["width_px"], cfg["window_ms"],
                                    int(cfg.get("alarm_radius", 4)), cusum.n_hypotheses)
    activity = DecisionActivity(cfg.get("activity_eps", ACTIVITY_EPS), cusum) if full else None
    confusion_sum = np.zeros((int(cfg["n_windows"]), 4), dtype=np.int64)
    kept, total_events = [], 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            probs, extra, confusion = run_sequence(model, frontend, cusum, seq, cfg, device, state_mode, monitor,
                                                   with_cusum=full, input_stats=input_stats, alarm_eval=alarm_eval,
                                                   activity=activity)
            bank_info = extra.pop("attr_bank", None)
            if attr is not None:
                attr.update(seq.label, extra, bank_info)
            confusion_sum += confusion
            total_events += seq.n_events
            for name in readouts:
                evaluators[name].matches[str(i)] = {"seg_pred": torch.from_numpy(probs[name]),
                                                    "seg_gt": torch.from_numpy(seq.label)}
            if dump_dir is not None:
                zeros_i64 = np.zeros(seq.n_events, dtype=np.int64)
                fields = {"prob_" + name: probs[name].astype(np.float32) for name in readouts if name != PRIMARY}
                fields.update({name: value for name, value in extra.items() if not name.startswith("publish")})
                np.savez(os.path.join(dump_dir, seq.name), locs=np.stack([zeros_i64, seq.x, seq.y, seq.t], 1),
                         labels=seq.label.astype(np.float32), probabilities=probs[PRIMARY].astype(np.float32),
                         target_id=seq.target_id.astype(np.float64), **fields)
            if full:
                zeros = np.zeros(seq.n_events, dtype=np.float32)
                ev_locs = torch.from_numpy(np.stack([zeros, seq.x, seq.y, seq.t], 1).astype(np.float32))
                for name in roc_readouts:
                    evaluators[name].roc_update(ev_locs[:, 3], torch.from_numpy(probs[name]), seq.target_id,
                                                torch.from_numpy(seq.label), ev_locs, thresh=float(cfg["threshold"]))
                publish = {PRIMARY: np.arange(seq.n_windows)}
                publish.update({name: extra[publish_field(name)] for name in roc_readouts if name != PRIMARY})
                kept.append((seq.t, seq.label, seq.target_id, probs, publish))
    model.train(was_training)
    result = {"split": split, "state_mode": state_mode, "n_sequences": len(dataset), "readouts": {}}
    for name in readouts:
        iou, acc = evaluators[name].evaluate_iou_and_accuracy(thresh=float(cfg["threshold"]))
        result["readouts"][name] = {"iou": float(iou), "acc": float(acc), "delay_windows": readout_delay(name)}
        if full:
            result["readouts"][name]["iou_acc_by_threshold"] = {
                "%g" % th: [float(v) for v in evaluators[name].evaluate_iou_and_accuracy(thresh=th)]
                for th in IOU_THRESHOLDS}
    result["iou"], result["acc"] = result["readouts"][PRIMARY]["iou"], result["readouts"][PRIMARY]["acc"]
    if monitor is not None:
        result["layers"] = monitor.summary()
    if full:
        for name in roc_readouts:
            pd, fa = evaluators[name].cal_roc()
            records = []
            for t, label, target_id, probs, publish in kept:
                records += first_detection_latencies_published(t, label, target_id, probs[name], cfg["window_ms"],
                                                               cfg["threshold"], cfg["correct_thresh"], publish[name])
            result["readouts"][name].update({"pd": float(pd), "fa": float(fa), "latency": summarize_latencies(records)})
        if alarm_eval is not None:
            result["alarms"] = alarm_eval.summary()
        if attr is not None:
            result["attribution"] = attr.summary()
        tp, fp, fn, npos = (confusion_sum[:, j] for j in range(4))
        rates = {name: s["firing_rate"] for name, s in result["layers"].items()}
        density = float(input_stats["nonzero"]) / float(max(input_stats["elements"], 1))
        events_per_window = total_events / float(len(dataset) * int(cfg["n_windows"]))
        result.update({
            "segment_iou": segment_iou(tp, fp, fn, cfg["segments"]),
            "rolling_iou": [None if math.isnan(v) else v
                            for v in rolling_iou(tp, fp, fn, int(cfg["rolling_window"])).tolist()],
            "input_nonzero_fraction": density,
            "events_per_window": events_per_window,
            "operations_backbone": estimate_operations(
                model.backbone, cfg["pad_height"], cfg["pad_width"], rates, events_per_window, density),
            # 前端与判决层没有可学习参数，但都是每窗全画面的实数运算，不计入能耗口径会低估整机代价
            "operations_frontend": frontend.estimate_operations(
                cfg["pad_height"], cfg["pad_width"], events_per_window),
            "operations_decision": cusum.estimate_operations(
                cfg["pad_height"], cfg["pad_width"], events_per_window,
                cfg["readout_delays"], len(cfg.get("alarm_thetas") or [])),
        })
        if activity is not None and activity.windows:
            # 稀疏同步执行的运算量估计：按各门控档位的足迹活跃比例缩放逐 (假设, 像素) 的项
            act = activity.summary()
            result["decision_activity"] = act
            result["operations_decision_sparse"] = {
                "%g" % eps: cusum.estimate_operations(
                    cfg["pad_height"], cfg["pad_width"], events_per_window, cfg["readout_delays"],
                    len(cfg.get("alarm_thetas") or []), active_fraction=min(1.0, frac))
                for eps, frac in zip(act["eps"], act["footprint_active_fraction"])}
    return result


# ---------------------------------------------------------------------------
# 各模式
# ---------------------------------------------------------------------------


def prepare_run(args, cfg, allow_existing=False):
    """通用准备：种子、构造模块、输出目录、run_config.json。"""
    device = torch.device(args.device)
    seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    frontend, model, cusum = build_all(cfg, device)
    root = Path(cfg["save_root"])
    if root.exists() and not allow_existing:
        raise RuntimeError("输出目录已存在: %s（换一个 --save-root；train 模式要接着训用 --resume）" % root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / ("run_config_%s.json" % args.mode), {
        "design_version": DESIGN_VERSION, "args": vars(args), "config": cfg,
        "features": frontend.feature_names(), "velocities": cusum.velocities,
        "parameters": count_parameters(model), "parameters_backbone": count_parameters(model.backbone),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_sha256": source_hashes()})
    print("参数量:", count_parameters(model), "| 特征通道:", frontend.n_features, "| 速度假设:", cusum.n_hypotheses,
          flush=True)
    return device, frontend, model, cusum, root


def mode_smoke(args, cfg):
    """冒烟测试：校准 + 1 个训练序列前 32 窗的一次更新。"""
    device, frontend, model, cusum, root = prepare_run(args, cfg)
    run_calibration(model, frontend, cfg, device, root)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    seq = EvUAVStream(cfg["root"], "train", cfg, [train_file_names(cfg)[0]])[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = train_sequence(model, frontend, seq, optimizer, cfg, device, np.random.RandomState(int(cfg["seed"])),
                         max_windows=32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    monitor = LayerMonitor(model.v_threshold)
    model.eval()
    with torch.no_grad():
        source = EventChunkSource(seq, cfg, device)
        state = frontend.init_state(1, int(cfg["pad_height"]), int(cfg["pad_width"]), device)
        _, feats, mu0, _ = frontend.run_chunk(state, source.chunk(0, 32))
        mark, log_g, _, info = model.forward_chunk(feats, None, collect=True)
        monitor.update(info)
    model.train()
    layers, taus = monitor.summary(), tau_statistics(model)
    report = {"sequence": seq.name, "events_in_32_windows": out["events"],
              "loss_per_event": out["loss_sum"] / max(out["events"], 1),
              "mark_per_event": out["mark_sum"] / max(out["events"], 1),
              "intensity_per_event": out["intensity_sum"] / max(out["events"], 1),
              "grad_norms": out["grad_norms"], "missing_grads": out["missing_grads"],
              "train_ms_per_window": 1e3 * elapsed / 32, "peak_memory_gib": peak_memory_gib(device),
              "feature_nonzero_fraction": float((feats != 0).float().mean()),
              "mu0_mean": float(mu0.mean()), "log_g_mean": float(log_g.mean()),
              "layers": layers, "tau": taus, "warnings": health_warnings(layers, taus, out["grad_norms"])}
    write_json(root / "smoke.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    missing = out["missing_grads"]
    if cfg["state_mode"] == "reset_each_window":
        # reset 模式下 beta 不参与计算，时间常数参数 a 本来就没有梯度，不算错误
        missing = [name for name in missing if not name.endswith(".neuron.a")]
    if missing:
        raise RuntimeError("以下参数没有梯度: %s" % missing)
    print("SMOKE TEST FINISHED:", root, flush=True)


def mode_overfit(args, cfg):
    """单序列过拟合：同一序列反复训练 --steps 次，定期在该序列上评估 net 读出。"""
    device, frontend, model, cusum, root = prepare_run(args, cfg)
    run_calibration(model, frontend, cfg, device, root)
    name = args.sequence or train_file_names(cfg)[0]
    seq = EvUAVStream(cfg["root"], "train", cfg, [name])[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    rng = np.random.RandomState(int(cfg["seed"]))
    log = root / "overfit.jsonl"
    for step in range(1, int(args.steps) + 1):
        out = train_sequence(model, frontend, seq, optimizer, cfg, device, rng)
        if step == 1 or step % 20 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                _, _, confusion = run_sequence(model, frontend, cusum, seq, cfg, device, cfg["state_mode"],
                                               with_cusum=False)
            model.train()
            tp, fp, fn = (int(confusion[:, j].sum()) for j in range(3))
            record = {"step": step, "loss_per_event": out["loss_sum"] / max(out["events"], 1),
                      "iou": tp / max(tp + fp + fn, 1), "acc": tp / max(tp + fn, 1), "sequence": name}
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    save_checkpoint(root / "overfit_last.pt", model, optimizer, int(args.steps), float("nan"), cfg)
    print("OVERFIT FINISHED:", root, flush=True)


def save_checkpoint(path, model, optimizer, epoch, best_val_iou, cfg):
    """保存权重（含已校准增益）、优化器状态与配置；不保存膜电位、前端与 CUSUM 状态。"""
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best_val_iou": best_val_iou, "config": cfg, "design_version": DESIGN_VERSION}, str(path))


def load_resume_checkpoint(cfg, device):
    """读取 save_root/last.pt 并核对配置。在 prepare_run 之前调用：核对失败时不会覆盖原来的 run_config。"""
    path = Path(cfg["save_root"]) / "last.pt"
    if not path.exists():
        raise RuntimeError("找不到续训用的 checkpoint: %s" % path)
    ckpt = torch.load(str(path), map_location=device)
    saved = ckpt.get("config", {})
    drift = config_drift(saved, cfg, TRAINING_KEYS)
    if drift:
        raise RuntimeError("续训的配置与 checkpoint 不一致（膜电位上下界等不在 state_dict 里，加载权重时查不出来），"
                           "请用原训练的同一条命令再加 --resume。不一致的项：" +
                           "；".join("%s: checkpoint=%r 当前=%r" % (k, a, b) for k, (a, b) in sorted(drift.items())))
    if saved.get("epochs") is not None and int(saved["epochs"]) != int(cfg["epochs"]):
        print("[注意] epochs 由 %s 改为 %s，学习率计划随之改变" % (saved["epochs"], cfg["epochs"]), flush=True)
    return ckpt


def mode_train(args, cfg):
    """正式训练：每轮设定学习率 -> 逐序列训练 -> val（net 读出）与训练子集评估 -> 保存。"""
    ckpt = load_resume_checkpoint(cfg, torch.device(args.device)) if args.resume else None
    device, frontend, model, cusum, root = prepare_run(args, cfg, allow_existing=args.resume)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    seed, epochs = int(cfg["seed"]), int(cfg["epochs"])
    start_epoch, best, history = 0, -float("inf"), []
    metrics_path = root / "metrics.jsonl"
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best = int(ckpt["epoch"]) + 1, float(ckpt["best_val_iou"])
        history = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        history = [h for h in history if h["epoch"] < start_epoch]
        print("从 epoch %d 继续" % start_epoch, flush=True)
    else:
        if metrics_path.exists():
            raise RuntimeError("输出目录已有 metrics.jsonl，请换目录或使用 --resume")
        run_calibration(model, frontend, cfg, device, root)
    subset = select_subset(train_file_names(cfg), int(cfg["train_subset_size"]), seed + 2000)
    train_set = EvUAVStream(cfg["root"], "train", cfg)
    for epoch in range(start_epoch, epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["lr"]), float(cfg["lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = make_sequence_loader(train_set, True, seed * 1000 + epoch, cfg["num_workers"])
        rng = np.random.RandomState(seed * 1000 + epoch + 7)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        model.train()
        sums, event_sum, grad_acc, n_seq = {"loss_sum": 0.0, "mark_sum": 0.0, "intensity_sum": 0.0}, 0, {}, 0
        for seq in loader:
            out = train_sequence(model, frontend, seq, optimizer, cfg, device, rng)
            for key in sums:
                sums[key] += out[key]
            event_sum += out["events"]
            for key, value in out["grad_norms"].items():
                grad_acc[key] = grad_acc.get(key, 0.0) + value
            n_seq += 1
        train_seconds = time.perf_counter() - t0
        grad_mean = {k: v / max(n_seq, 1) for k, v in grad_acc.items()}
        val = evaluate_split(model, frontend, cusum, cfg, device, "val", cfg["state_mode"], collect_layers=True)
        sub = None
        if subset_due(epoch, epochs, int(cfg["train_subset_every"])):
            sub = evaluate_split(model, frontend, cusum, cfg, device, "train", cfg["state_mode"], names=subset)
        taus = tau_statistics(model)
        record = {"epoch": epoch, "seed": seed, "lr": lr,
                  "loss_per_event": sums["loss_sum"] / max(event_sum, 1),
                  "mark_per_event": sums["mark_sum"] / max(event_sum, 1),
                  "intensity_per_event": sums["intensity_sum"] / max(event_sum, 1),
                  "val_iou": val["iou"], "val_acc": val["acc"],
                  "train_subset_iou": sub["iou"] if sub else None, "train_subset_acc": sub["acc"] if sub else None,
                  "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - t0,
                  "peak_memory_gib": peak_memory_gib(device), "grad_norms": grad_mean,
                  "layers": val["layers"], "tau": taus, "warnings": health_warnings(val["layers"], taus, grad_mean)}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        history.append(record)
        if val["iou"] > best:
            best = val["iou"]
            save_checkpoint(root / ("best_val_iou_seed%d.pt" % seed), model, optimizer, epoch, best, cfg)
        save_checkpoint(root / "last.pt", model, optimizer, epoch, best, cfg)
        print("epoch %d lr %.2e loss/event %.5f (mark %.5f, intensity %.5f) val IoU %.4f ACC %.4f | "
              "train-subset IoU %s | %.0fs（训练 %.0fs）| 警告 %d" % (
                  epoch, lr, record["loss_per_event"], record["mark_per_event"], record["intensity_per_event"],
                  val["iou"], val["acc"], "%.4f" % sub["iou"] if sub else "-", record["epoch_seconds"],
                  train_seconds, len(record["warnings"])), flush=True)
        for w in record["warnings"]:
            print("  [health]", w, flush=True)
    summary = summarize_history(history, seed)
    write_json(root / "summary.json", summary)
    print("SUMMARY_JSON", json.dumps(summary, sort_keys=True), flush=True)
    print("TRAINING FINISHED:", root, flush=True)


def select_eval_names(directory, first_n=0, names=None):
    """eval 只评估的序列：names 给出文件名（可省略 .npz）时按给出的顺序；否则 first_n > 0 时取按文件名排序的前 N 条；
    都没有时返回 None（整个划分）。给出的名字不存在时报错。"""
    if names:
        if first_n:
            raise ValueError("--eval-names 与 --eval-sequences 只能给一个")
        wanted = [n if n.endswith(".npz") else n + ".npz" for n in names]
        missing = [n for n in wanted if not os.path.isfile(os.path.join(directory, n))]
        if missing:
            raise ValueError("找不到序列: %s（目录 %s）" % (missing, directory))
        return wanted
    if first_n:
        return sorted(n for n in os.listdir(directory) if n.endswith(".npz"))[:int(first_n)]
    return None


def mode_eval(args, cfg):
    """评估 checkpoint：骨干 carry 与 reset_each_window 各一遍，全部读出；结果写到 checkpoint 同目录。

    网络、前端、CUSUM 的配置一律取自 checkpoint；数据根目录、评估口径与 CUSUM 读出参数（阈值、延迟）取自当前 YAML。
    """
    if not args.checkpoint:
        raise ValueError("eval 模式需要 --checkpoint")
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_cfg = dict(ckpt["config"])
    conflicts = eval_conflicts(args, cfg, train_cfg)
    if conflicts:
        raise ValueError("eval 模式的网络、前端与神经元配置一律取自 checkpoint；下列参数需要重新训练才生效，"
                         "在这里给出只会让结果被贴错标签：" + "；".join(conflicts))
    # 判决层（CUSUM）与评估口径没有可学习参数，一律取自当前 YAML / 命令行：
    # 同一个 checkpoint 可以直接跑"静止 vs 运动补偿""lme vs sum""记忆开关"等对照，不必重训
    for key in ("root", "threshold", "pd_detT", "correct_thresh", "rolling_window", "segments",
                "readout_delays", "eval_chunk", "fusion_weight", "alarm_thetas", "alarm_radius",
                "cusum_axis_velocities", "cusum_footprint", "cusum_compensator", "cusum_nb_kappa",
                "cusum_aggregate", "cusum_track_tau_ms", "cusum_memory_gain", "cusum_gate_eps", "cusum_reset_radius",
                "activity_eps", "attr", "attr_delays", "attr_weight", "attr_variants", "attr_cap", "attr_eps",
                "attr_ref_radius", "attr_min_support", "attr_tube_birth", "hyp_update", "hyp_params"):
        if key in cfg:
            train_cfg[key] = cfg[key]
    train_cfg.setdefault("readout_delays", [1, 2, 5])
    seed_everything(int(train_cfg["seed"]), bool(train_cfg["deterministic"]))
    frontend, model, cusum = build_all(train_cfg, device)
    model.load_state_dict(ckpt["model"])
    if not bool(model.backbone.gain_calibrated):
        raise RuntimeError("checkpoint 中的增益未校准")
    suffix = ("_" + args.tag) if args.tag else ""
    out_path = Path(args.checkpoint).parent / ("eval_%s_%s%s.json" % (args.split, Path(args.checkpoint).stem, suffix))
    if out_path.exists() and not args.overwrite:
        raise RuntimeError("结果已存在: %s（使用 --overwrite 覆盖）" % out_path)
    results = {"checkpoint": args.checkpoint, "epoch": ckpt["epoch"], "design_version": DESIGN_VERSION,
               "tag": args.tag, "trained_design_version": ckpt.get("design_version"),
               "cusum": {"velocities": train_cfg["cusum_axis_velocities"], "aggregate": train_cfg.get("cusum_aggregate"),
                         "track_tau_ms": train_cfg.get("cusum_track_tau_ms"), "footprint": train_cfg["cusum_footprint"],
                         "memory_gain": cusum.memory_gain, "gate_eps": cusum.gate_eps,
                         "reset_radius": cusum.reset_radius, "alarm_thetas": train_cfg.get("alarm_thetas")},
               "attr": {key: train_cfg.get(key) for key in ("attr", "attr_delays", "attr_weight", "attr_variants",
                                                            "attr_cap", "attr_eps", "attr_ref_radius",
                                                            "attr_min_support", "attr_tube_birth", "hyp_update",
                                                            "hyp_params")},
               "parameters": count_parameters(model), "trained_state_mode": train_cfg["state_mode"],
               "neuron": train_cfg["neuron"], "readout_delays": train_cfg["readout_delays"],
               "neuron_u_floor": train_cfg.get("neuron_u_floor"), "neuron_u_ceil": train_cfg.get("neuron_u_ceil"),
               "config": train_cfg}
    modes = tuple(args.eval_state_modes or cfg.get("eval_state_modes") or ("carry", "reset_each_window"))
    names = select_eval_names(os.path.join(train_cfg["root"], args.split), args.eval_sequences, args.eval_names)
    if names is not None:
        results["eval_sequences"] = names
        print("只评估 %d 条序列 %s（子集结果，不能当整个划分的正式数字）" % (len(names), names), flush=True)
    for mode in modes:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        dump = None
        if args.dump_dir and mode == train_cfg["state_mode"]:
            dump = args.dump_dir
            os.makedirs(dump, exist_ok=True)
        t0 = time.perf_counter()
        r = evaluate_split(model, frontend, cusum, train_cfg, device, args.split, mode, full=True, dump_dir=dump,
                           names=names)
        r["seconds"] = time.perf_counter() - t0
        r["peak_memory_gib"] = peak_memory_gib(device)
        r["tau"] = tau_statistics(model)
        r["warnings"] = health_warnings(r["layers"], r["tau"], {})
        results[mode] = r
        for name, m in r["readouts"].items():
            if "pd" in m:
                print("[%s] %-18s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 延迟中位数 %s ms" % (
                    mode, name, m["iou"], m["acc"], m["pd"], m["fa"], m["latency"].get("latency_median_ms")), flush=True)
            else:
                print("[%s] %-18s IoU %.4f ACC %.4f（对照读出，只算 IoU/ACC）" % (mode, name, m["iou"], m["acc"]), flush=True)
        best = {name: max(m["iou_acc_by_threshold"].items(), key=lambda kv: kv[1][0])
                for name, m in r["readouts"].items()}
        print("[%s] 各读出 IoU 最高的阈值：%s" % (mode, "，".join(
            "%s %s→%.4f" % (name, th, v[0]) for name, (th, v) in best.items())), flush=True)
        if "attribution" in r:
            a = r["attribution"]
            b = a["bank"]
            print("[%s] V2-2 假设库：每窗存活 %.1f / 已确认 %.1f，生成 SNN %d + 管道 %d，确认 %d（中位延迟 %s 窗），"
                  "合并 %d，结束 %d，耗时 %.1f ms/窗" % (
                      mode, b["alive_per_window"], b["confirmed_per_window"], b["births_snn"], b["births_tube"],
                      b["confirmed_total"], b["confirm_delay_median"], b["merged"], b["ended"], b["ms_per_window"]),
                  flush=True)
            for key, p in a.items():
                if key == "bank":
                    continue
                print("[%s] V2-2 %s：修正覆盖 目标 %.1f%% / 背景 %.3f%%；平均修正 目标 %+.3f / 背景 %+.3f；"
                      "目标上调 %.1f%%，背景下调 %.1f%%（固定参考的目标事件 %d）" % (
                          mode, key, 100 * p["target_covered_frac"], 100 * p["background_covered_frac"],
                          p["target_delta_mean"], p["background_delta_mean"], 100 * p["target_up_frac"],
                          100 * p["background_down_frac"], p["target_case2"]), flush=True)
        print("[%s] 分段 IoU (net) %s" % (mode, {k: round(v, 4) for k, v in r["segment_iou"].items()}), flush=True)
        for theta, a in r.get("alarms", {}).items():
            print("[%s] 告警 theta=%s  检出率 %.3f  首次告警延迟中位数 %s ms  虚警率 %.3e（紧界 %.3e）"
                  "  去重虚警 %.0f/小时  到近期目标距离 %s  纯背景窗告警位置率 %s（%d 窗，紧界下期望 %.1f 个）" % (
                      mode, theta, a["detection_rate"], a.get("latency_median_ms"), a["false_alarm_rate"],
                      a["tight_bound_per_location_window"], a["false_events_per_hour"],
                      a["false_distance_to_recent_target"],
                      "—" if a["pure_alarm_location_rate"] is None else "%.3e" % a["pure_alarm_location_rate"],
                      a["pure_windows"], a["pure_expected_alarms_at_bound"]), flush=True)
            print("[%s]   theta=%s 纯背景 start %d 窗 / after %d 窗，越界检验 p = %.3g / %.3g" % (
                mode, theta, a["pure_start_windows"], a["pure_after_windows"], a["pure_start_violation_p"],
                a["pure_after_violation_p"]), flush=True)
        if "decision_activity" in r:
            act = r["decision_activity"]
            print("[%s] 判决层活跃比例（足迹膨胀后）%s" % (mode, "，".join(
                "ε=%g: %.4f" % (e, f) for e, f in zip(act["eps"], act["footprint_active_fraction"]))), flush=True)
    write_json(out_path, results)
    print("EVAL FINISHED:", out_path, flush=True)


def main():
    """入口：解析参数、读取配置、分发到对应模式。"""
    args = parse_args()
    cfg = build_config(args)
    {"smoke": mode_smoke, "overfit": mode_overfit, "train": mode_train, "eval": mode_eval}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
