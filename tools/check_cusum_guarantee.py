"""核对漂移 CUSUM 的虚警保证：纯背景下，任意"只用过去"的目标强度预测都不能让告警率超过理论上界。

理论（model/evidence_neuron.py）：每个速度假设一条管道；按位置告警、复位该位置全部假设时，每位置每窗的虚警率
    <= (1 + 1/L) / (e^theta - 1)   紧界，与速度假设数 V 无关（Q_k = sum_x mean_v R_v 的 Shiryaev-Roberts 论证）
    <= V / (e^theta - 1)           早先的并集界（仍输出，便于和旧结果对照）
判决层的评估时开关（--memory-gain λ、--gate-eps ε、--reset-radius r，缺省 = 原实现）任意取值都不应超过紧界，
每个开关单独跑一遍本脚本即为合成核对。
条件：预测只依赖过去；每个像素的补偿项不小于其背景条件分布的累积量生成函数（泊松时 mu0 >= 真实均值）；
      足迹聚合为 lme / mean 时对像素间相关性无要求，sum 时要求像素条件独立。

场景（每个场景 × 每个预测器 × 每个阈值，报告实测告警率 / 上界）：
    poisson_oracle        泊松背景，mu0 = 真实均值                        -> 应满足
    poisson_conservative  泊松背景，mu0 = 1.2 × 真实均值                  -> 应满足（更保守）
    poisson_under         泊松背景，mu0 = 0.3 × 真实均值                  -> 应违反（条件不成立的反例）
    negbin_poisson_comp   过离散背景（负二项），仍用泊松补偿              -> 可能违反（模型失配）
    negbin_negbin_comp    过离散背景，用负二项补偿（kappa 已知）          -> 应满足
    poisson_frontend      泊松背景，mu0 由 EvidenceFrontEnd 在线估计      -> 实际管线的情况
    poisson_sync          每个 3x3 块内像素取同一个泊松计数（逐像素边缘仍是泊松，但完全相关）
                          -> sum 聚合应违反（空间相关的反例），lme / mean 应满足
预测器（都只用上一窗及更早的计数）：chase（追着上一窗的事件）、constant（常数强度）、smooth（上一窗计数的邻域平均）。

用法: python tools/check_cusum_guarantee.py --size 64 --steps 1000 --out log/guarantee/check.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dataset.stream_features import EvidenceFrontEnd  # noqa: E402
from model.evidence_neuron import DriftCUSUM, velocity_grid  # noqa: E402


def sample_counts(generator, mu, size, kind, kappa, device):
    """一窗的背景计数：泊松；负二项（Gamma 混合泊松，形状 kappa、均值 mu）；或 3x3 块内完全相同的泊松计数。"""
    if kind == "sync":
        blocks = -(-size // 3)
        rate = torch.full((1, 1, blocks, blocks), mu, dtype=torch.float64, device=device)
        counts = torch.poisson(rate, generator=generator)
        return counts.repeat_interleave(3, 2).repeat_interleave(3, 3)[:, :, :size, :size]
    rate = torch.full((1, 1, size, size), mu, dtype=torch.float64, device=device)
    if kind == "negbin":
        gamma = torch.distributions.Gamma(torch.full_like(rate, kappa), torch.full_like(rate, kappa / mu))
        rate = gamma.sample()
    return torch.poisson(rate, generator=generator)


def predictor(name, prev, mu):
    """只用上一窗计数的目标强度预测（log g）。"""
    if name == "chase":
        return torch.log(0.5 * prev + 1e-3)
    if name == "constant":
        return torch.full_like(prev, math.log(0.6 * mu))
    if name == "smooth":
        return torch.log(F.avg_pool2d(prev, 5, 1, 2, count_include_pad=False) + 1e-3)
    raise ValueError(name)


def run(scenario, pred, theta, aggregate, args, device):
    """跑一个组合，返回每位置每窗的实测告警率。"""
    background = "negbin" if scenario.startswith("negbin") else ("sync" if scenario == "poisson_sync" else "poisson")
    compensator = "negbin" if scenario == "negbin_negbin_comp" else "poisson"
    velocities = velocity_grid(args.axis_velocities)
    cusum = DriftCUSUM(velocities, footprint=args.footprint, compensator=compensator, nb_kappa=args.kappa,
                       aggregate=aggregate, track_decay=args.track_decay, memory_gain=args.memory_gain,
                       gate_eps=args.gate_eps, reset_radius=args.reset_radius)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    size, mu = args.size, args.mu
    state = cusum.init_state(1, size, size, device, torch.float64)
    frontend = EvidenceFrontEnd([50.0], 50.0, [], bg_smooth_radius=7, bg_prior=mu, bg_floor=1e-4).double().to(device)
    fe_state = frontend.init_state(1, size, size, device, torch.float64)
    empty = torch.zeros(1, 2, size, size, dtype=torch.float64, device=device)
    prev = torch.zeros(1, 1, size, size, dtype=torch.float64, device=device)
    alarms, counted = 0, 0
    for k in range(args.steps):
        counts = sample_counts(generator, mu, size, background, args.kappa, device)
        if scenario == "poisson_frontend":
            mu0 = frontend.background(fe_state)
            fe_state, _, _, _ = frontend.step(fe_state, torch.cat([counts, torch.zeros_like(counts)], 1), empty, empty)
        else:
            factor = {"poisson_oracle": 1.0, "poisson_conservative": 1.2, "poisson_under": 0.3}.get(scenario, 1.0)
            mu0 = torch.full_like(counts, factor * mu)
        state, _, alarm = cusum.step(state, counts, mu0, predictor(pred, prev, mu) if k > 0 else None,
                                     theta=theta, reset_on_alarm=True)
        if k >= args.burn_in:
            alarms += int(alarm.sum())
            counted += size * size
        prev = counts
    return alarms / float(max(counted, 1)), cusum.n_hypotheses, args.steps - args.burn_in


def main():
    parser = argparse.ArgumentParser(description="漂移 CUSUM 虚警保证的模拟核对")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=20, help="前若干窗不计（在线背景估计需要预热）")
    parser.add_argument("--mu", type=float, default=0.05, help="背景每像素每窗期望事件数")
    parser.add_argument("--kappa", type=float, default=0.5, help="负二项形状参数（越小越过离散）")
    parser.add_argument("--footprint", type=int, default=3)
    parser.add_argument("--aggregates", nargs="+", default=["lme", "sum"], help="足迹聚合方式（lme / mean / sum）")
    parser.add_argument("--track-decay", type=float, default=0.82, help="管道强度记忆的每窗衰减（0 关闭）")
    parser.add_argument("--axis-velocities", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    parser.add_argument("--thetas", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    parser.add_argument("--predictors", nargs="+", default=["chase", "constant", "smooth"])
    parser.add_argument("--scenarios", nargs="+", default=["poisson_oracle", "poisson_conservative", "poisson_under",
                                                           "negbin_poisson_comp", "negbin_negbin_comp",
                                                           "poisson_frontend", "poisson_sync"])
    parser.add_argument("--memory-gain", type=float, default=1.0, help="非对称记忆 λ（1 = 原实现）")
    parser.add_argument("--gate-eps", type=float, default=0.0, help="预测门控 ε（0 = 原实现）")
    parser.add_argument("--reset-radius", type=int, default=0, help="告警复位邻域半径 r（0 = 原实现）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)                     # 负二项的 Gamma 采样用全局随机数
    rows = []
    print("判决层开关: memory_gain %g, gate_eps %g, reset_radius %d" % (
        args.memory_gain, args.gate_eps, args.reset_radius))
    print("%-22s %-5s %-9s %6s %12s %12s %8s %8s" % ("scenario", "aggr", "predictor", "theta", "alarm_rate",
                                                      "tight", "ratio", "ratio_V"))
    for scenario in args.scenarios:
        for aggregate in args.aggregates:
            for pred in args.predictors:
                for theta in args.thetas:
                    rate, V, L = run(scenario, pred, theta, aggregate, args, device)
                    bound = V / (math.exp(theta) - 1.0)
                    tight = (1.0 + 1.0 / max(L, 1)) / (math.exp(theta) - 1.0)
                    rows.append({"scenario": scenario, "aggregate": aggregate, "predictor": pred, "theta": theta,
                                 "alarm_rate": rate, "bound": bound, "ratio": rate / bound, "hypotheses": V,
                                 "tight_bound": tight, "ratio_tight": rate / tight})
                    print("%-22s %-5s %-9s %6.1f %12.3e %12.3e %8.3f %8.3f%s" % (
                        scenario, aggregate, pred, theta, rate, tight, rate / tight, rate / bound,
                        "  <-- 超过紧界" if rate > tight else ""), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump({"args": vars(args), "rows": rows}, stream, indent=2)


if __name__ == "__main__":
    main()
