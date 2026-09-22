# 流式 SNN V2（方案一，设计版本 v2-1）：预测—检验式证据积累

一句话：**网络给出一个可以被下一窗观测检验的目标强度预测；事件"如期出现"和"预测落空"都沿运动轨迹积累成证据；
何时报警由证据决定，并在明确的背景条件下有虚警上界。**

```text
事件 ─► 证据前端（无参数）──► 脉冲 U-Net 骨干（V1 电流合并解码器）──► 逐像素两个头
          背景强度 mu0（只用过去）                                      mark  ：本窗事件是目标的 logit
          时间矩（精确到 ms，4 个时间尺度）                              log_g ：本窗目标强度场
          极性偶极（运动方向）
                        上一窗 g 按 49 个速度假设平移 + 管道自身的强度记忆 ─► 漂移 CUSUM（无参数）
                             │
                             ├─► 输出 A  逐事件分割：net（零额外等待）、fused_d（mark + w·之后 d 窗的管道证据）
                             └─► 输出 B  位置级告警：M >= theta 即告警（带复位），告警时刻由证据决定
```

## v2-0 → v2-1 改了什么（针对评审意见）

| 评审指出的问题 | 处理 |
|---|---|
| CUSUM 的累积状态没有进入任何输出（把 C 清零，结果不变） | 位置级告警接入评估，作为独立输出 B；报告目标检出率、首次告警延迟、虚警连通域率与理论上界（`utils/alarm_metrics.py`） |
| 离开目标的管道只受网络当前判断驱动，网络不再预测时证据恒为 0、分数冻结 | 每条管道增加自己的"强度记忆" G_v = max(平移后的网络强度, rho·平移后的旧 G_v)：离开目标后仍预测"该有事件"，事件不来就持续产生负证据（`cusum_track_tau_ms`） |
| 3×3 邻域证据求和要求像素条件独立，空间相关背景（同步闪烁、成团杂波）下保证失效 | 足迹聚合改为 lme = log mean exp（或 mean），**对像素间任意相关都成立**（Jensen / 混合）；sum 保留为消融。评审的反例（逐像素泊松、完全相关）已写成单元测试：sum 超标，lme 满足 |
| mark + F 不是严格后验 | 改称"轨迹证据融合分数" = mark + w·F，w 用 `tools/calibrate_fusion.py` 在 val 上拟合后冻结到 test |
| 延迟读出的发布时间与统计口径不一致 | 按每窗的实际发布窗号计算首次检出延迟（序列末尾截断的延迟如实计入）；零延迟改称"零额外等待"（仍需等本窗结束）；网络计算与排队时间尚未计入 |
| 告警界不能直接套到原评估的 Fa 上 | 两者分开报告：原 Fa 按逐事件预测的连通域统计；告警界是"每位置每窗的告警率"，只用于输出 B |
| 需要压力测试与关键对照 | 合成数据新增成团突发、背景突变、目标闪现预设；对照实验见下表，全部用配置项实现 |

## 数学（详见 `model/evidence_neuron.py` 模块说明）

- **管道强度（预测隔室）**：`G_v(x,k) = max( g(x - d_v, k-1), rho * G_v(x - d_v, k-1) )`，只依赖过去 → 可预测。
- **逐像素证据**：`e_v = N log(1 + G_v/mu0) - psi`，泊松 `psi = G_v`；预测该有目标而事件没来时 `e = -G_v`（预测落空）。
- **足迹聚合**：`l_v = log mean_{y in S} exp(e_v(y))`（默认）。
- **判决神经元**：`C_v = max(0, C_v(x - d_v, k-1) + l_v)`，`M = logsumexp_v C_v - log V`，`M >= theta` 告警并复位。
- **保证**（条件：网络预测只用过去；每个像素的补偿不小于其背景条件分布的累积量生成函数，泊松时即 `mu0` 不低于真实均值；
  lme / mean 聚合对空间相关无要求）：每条管道 L 步内的期望告警次数 `<= L/(e^theta - 1)`，每位置每窗告警率 `<= V/(e^theta - 1)`，
  对任意网络权重成立。**条件本身（背景估计足够保守）在真实数据上没有保证**，需要在纯背景片段上实测并在 val 上定阈值。
- **融合分数**：`score_i(d) = mark_i + w * F_i(d)`，`F_i(d) = logsumexp_v sum_{m=k+1..k+d} l_v(沿管道) - log V`。

## 服务器运行顺序

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV && conda activate evuav && mkdir -p run log
python -m unittest discover -s tests -p "test_stream*.py" 2>&1 | tail -3        # 应为 96 个测试 OK
CFG=configs/evisseg_stream_v2.yaml; RUN=stream_v2; S=37; GPU=0

# 1. 虚警保证核对（含空间相关背景 poisson_sync：sum 应超标、lme 应满足；poisson_under 为条件不成立的反例）
python tools/check_cusum_guarantee.py --size 64 --steps 1000 --device cuda:0 \
    --out log/guarantee/check.json > run/check_cusum_guarantee.log 2>&1
# 2. 冒烟 / 3. 单序列过拟合（net IoU 应升到 0.9 左右，权重存为 overfit_last.pt）
CUDA_VISIBLE_DEVICES=$GPU python train_stream_v2.py --config $CFG --mode smoke --save-root log/${RUN}_smoke > run/${RUN}_smoke.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python train_stream_v2.py --config $CFG --mode overfit --steps 300 --save-root log/${RUN}_overfit > run/${RUN}_overfit.log 2>&1
# 4. 正式训练
nohup env CUDA_VISIBLE_DEVICES=$GPU python train_stream_v2.py --config $CFG --mode train --seed $S \
    --save-root log/${RUN}_seed$S > run/${RUN}_seed$S.log 2>&1 &
# 5. 评估（两类输出：逐事件读出 net/fused_d 与位置级告警 alarms）
for split in val test; do
  CUDA_VISIBLE_DEVICES=$GPU python train_stream_v2.py --config $CFG --mode eval --split $split \
      --checkpoint log/${RUN}_seed$S/best_val_iou_seed$S.pt --dump-dir log/verify/${RUN}_s${S}_$split
done
# 6. 在 val 上校准融合权重，冻结后用于 test（写出 log/verify/${RUN}_s37_test_cal_d{1,2,5}）
python tools/calibrate_fusion.py --val-dump log/verify/${RUN}_s37_val --test-dump log/verify/${RUN}_s37_test \
    --delays 1 2 5 --out-root log/verify/${RUN}_s37_test_cal
# 7. 等虚警率比较（V2 net、V2 校准后 fused_d1、V1）
python tools/sweep_threshold.py --target-fa 1e-5 6.5e-6 --dump-dir log/verify/${RUN}_s37_test \
    log/verify/${RUN}_s37_test_cal_d1 log/verify/v1_s37_test --out log/energy/threshold_sweep_v2_test.json
# 合成压力测试数据（同步闪烁 / 成团突发 / 背景突变 / 目标闪现 / 过离散）
python tools/synth_events.py --out /media/stephen/nvme0n1/wy_data/datasets/synth_v2 --n-train 40 --n-val 10 --n-test 10
```

eval 结果 `eval_<split>_<ckpt>.json`：`<carry|reset_each_window>.readouts.<net|fused_d*>` 为逐事件指标（原 eval 口径）与
首次检出延迟；`.alarms.<theta>` 为告警输出的检出率、首次告警延迟、虚警连通域率与上界。

## 消融与对照（全部用配置项/命令行实现，不改代码）

**A. 评估时切换，不用重训**（判决层与读出都没有可学习参数，同一份权重直接跑）：

| 消融 | 命令行 | 证明什么 |
|---|---|---|
| 只留静止假设 | `--cusum-velocities 0 --tag static` | 运动补偿（记忆跟着目标走）是否必要 |
| 关掉管道强度记忆 | `--cusum-track-tau-ms 0 --tag nomem` | 负证据（预测落空）的作用 |
| 聚合方式 | `--cusum-aggregate sum --tag sum` | 相关性稳健的代价（告警延迟） |
| 延迟读出 | `--readout-delays 1 2 5` | 延迟—精度曲线；与静止假设同样的 d 做"等待时间公平对照" |
| 骨干跨窗状态 | `--state-mode reset_each_window` | 骨干记忆的作用（注意前端与判决层仍有记忆） |
| 完全不用判决层 | 看 `net` 读出 | 判决层的净增益 |
| 告警阈值 / 足迹 / 补偿器 / 融合权重 | 配置项 | 各自的敏感度 |

**B. 需要重训**（改变了学习到的部分或输入通道）：

| 消融 | 配置 / 命令行 | 证明什么 |
|---|---|---|
| 单头（去掉强度头） | `loss_intensity_weight: 0` | 强度头的联合训练有没有帮助（此时判决层无预测可用） |
| 前端递进链 ①只有计数+固定背景 | `--fe-features count --bg-mode constant` | 近似 V1 的输入，作为前端链条的起点 |
| ②加自适应背景 | `--fe-features count` | 背景归一化值多少 |
| ③加多尺度时间矩 | `--fe-features count ratio age` | 长时间尺度（含 2 s）值多少 |
| ④加极性偶极（完整版） | 默认 | 偶极值多少 |
| 时间尺度集合 | `fe_taus_ms: [20, 100]` 等 | 2 s 尺度单独的贡献 |
| 神经元类型 | `--neuron relu` / `graded` | 脉冲骨干的贡献（同样前端、双头、判决） |
| TBPTT 长度、合并解码器 | `--tbptt-k` / `merged_decoder` | 沿用 V1 的结论，一般不重做 |

注：`bg_mode: constant` 同时关掉判决层的自适应补偿，虚警上界的前提（mu0 不低于真实背景）在事件密集的序列上会不成立——这正是"为什么需要背景模型"的证据，报告时要说明。

## 本地已核对的内容（CPU，无真实数据）

- **96 个单元测试全部通过**（原 77 + V2 的 19 个）。V2 部分包括：证据的期望（泊松/负二项解析求和）、管道强度记忆递推、
  三种聚合公式、CUSUM 与暴力计算一致、延迟读出与暴力计算一致（含实际发布窗）、独立背景下告警率不超界、背景低估时超界、
  **完全相关背景下 sum 超界而 lme / mean 满足**、告警评估（真告警/虚警连通域/首次告警延迟）、按发布窗的检出延迟、
  前端递推与直接求和一致、网络逐窗与片段等价、训练使全部参数更新、各读出覆盖全部事件。
- **虚警模拟**（48×48，400 窗，9 个假设）：独立泊松、在线背景估计下，lme 的实测告警率 ≤ 上界的 0.1%；
  同步背景（逐像素泊松、3×3 完全相关）下 sum 超出上界 1.2 倍（告警率比独立时高约 120 倍），lme 仍只有上界的 0.1%。
- **合成序列上的机制检查**（单序列过拟合权重，仅说明机制正确，不代表泛化）：
  net IoU 0.957；融合 d1 0.972；告警输出 θ=8–16 检出 3/3 目标，首次告警延迟中位数 85–135 ms，虚警连通域 0–1 个；
  **只保留静止假设时**，d5 召回从 0.978 降到 0.830，θ=16 只检出 1/3 目标——运动补偿（记忆跟着目标走）是必要的；
  sum 聚合告警更快（53–85 ms），即相关性稳健的代价约为 30–50 ms。

## 已知局限

1. 背景估计的保守性在真实数据上没有保证：先在纯背景片段上实测告警率，阈值在 val 上确定后冻结到 test。
2. 没有逐窗计时，首次检出延迟未含网络计算与排队时间。
3. 运算量统计只覆盖骨干；前端、双头与判决层的逐元素运算未计入，不能据此主张整套 V2 的能耗。
4. 超参数（时间尺度、背景先验、速度网格、足迹、记忆时间常数、两项损失权重）都是先验值，未调。
