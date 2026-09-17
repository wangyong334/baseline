# CLAUDE.md — EV-UAV 流式 SNN 研究

> 记录这个项目做过的事、当前进展和可继续的方向，用于持续迭代创新和写论文。
> 每完成一次迭代，更新第 1、3、4、7 节。最后更新：2026-09-17，分支 `research/windowed-streaming-snn-v1` @ `6cc2d31`。

## 1. 概况与进展

**目标**：以 ICCV 2025 论文 EV-SpSegNet（事件相机反无人机微小目标分割）为基线，做因果、低延迟、脉冲驱动的流式 SNN，形成论文。

**已完成（按时间）**
1. 基线工程（09-05 ~ 09-14）：双卡训练、线性学习率、基线 v2 实验协议
2. 早期探索 SNN v0 / 静态量化：只把基线里两处激活换成发放率 LIF，不是完整 SNN，已归档在 `main` 分支
3. 流式 SNN V1（v1-1，09-16）：test IoU 0.8363（3 种子），参数 0.105M，50 ms 块因果
4. 结果核验（v1-2，09-16）：确认 V1 没有评估错误或信息泄漏；发现原基线代码的时间下采样 bug
5. 基线 K5 修复（09-16）：基线 test IoU 从 0.6188 提到 0.8011，V1 的领先缩小到 +0.035
6. 解码器全脉冲化（方案 C 电流合并，09-17）：对已训练 V1 严格等价换算，指标不变，每窗理论能耗 2.24 → 0.60 mJ

**进行中**
- 方案 C 从零训练 seed37（`log/stream_merged_decoder_seed37`）
- 训练加速 A 与能耗统计工具（09-17）：代码和本地测试已完成，待在服务器 1 号卡上用 seed37 做等价核对、计时、基线能耗统计，然后重训

**主要发现**
- 原基线代码三次时间下采样都漏掉 `t%4==2` 的时间片，这部分事件 IoU 只有 0.27–0.32；修复后基线大幅提升
- V1 原解码器可以精确改写为两路脉冲电流之和，所有突触输入变成脉冲，能耗降 73%，精度不变
- 跨窗记忆的贡献很弱：carry 只比 reset 高 0.016，而种子间标准差是 0.013
- 单窗耗时里预处理约 13 ms（64%），网络约 7 ms；剩余理论能耗约 93% 来自 enc1 的实数输入
- V1 训练一轮约 540 s：训练约 410 s（25.7 ms/窗）+ 评估约 125 s（23%），GPU 利用率约 10%。时间主要耗在 CPU 逐窗预处理和逐窗调用小算子上，不在 GPU 计算
- TBPTT 16 窗意味着梯度只回传 800 ms，而 τ 上限是 2000 ms，这可能是跨窗记忆弱的原因之一（tau1000 变体更好也与此一致）

## 2. 基线：EV-SpSegNet

### 2.1 原论文

Chen et al., *Event-based Tiny Object Detection: A Benchmark Dataset and Baseline*, ICCV 2025（arXiv 2506.23575，PDF 在 `References/`）。

- **数据集 EV-UAV**：DAVIS346（346×260），147 条序列（train 99 / val 24 / test 24），每条 8 秒，逐事件标注。目标平均 6.8×5.4 像素，全部小于 32²，覆盖多种光照和场景。标注方法：按时间窗累积成帧画框，沿时间扩成 3D 框，框内事件即目标事件。NPZ 包含 `ev`、`evs_norm`（含 label 与 target_id）、`ev_loc`（整数 x, y, t）。
- **核心观察**：运动小目标在 (x, y, t) 点云中形成连续细长曲线，背景和噪声是离散的点或面，所以靠时空连续性而不是外观来判别。
- **网络**：体素化（1 px × 1 px × 1 ms，特征是事件归一化坐标与极性的均值）→ 3D 稀疏 U-Net（spconv）。核心模块 GDSCA = 逐点稀疏卷积 → 分组膨胀稀疏卷积（膨胀率 1–4）→ 稀疏 SE → Patch Attention。编码器下采样 stride [2,2,4]；逐体素 Sigmoid 输出，再映射回事件。
- **STC Loss**：用预测在 k×k×τ 邻域内的和算出时空相关性权重，加权 BCE，保留有连续轨迹支撑的事件、压制孤立噪声。论文 k=3、τ=5、γ=2，没有类别平衡。
- **评估**（`utils/eval.py`，阈值 0.9）：IoU 为全部序列拼接后的正类 IoU；ACC 实际是正类召回；Pd 按 50 ms 帧逐目标统计；Fa 为虚警 8 连通域数除以（帧数 × 像素数）。
- **论文结果**：IoU 55.18，ACC 65.02，Pd 77.53，Fa 1.63e-4，参数 4.0M，处理 8 秒数据 35.9 ms。
- **作者自陈局限**：静止或慢速目标不产生事件会漏检；未来方向是多模态。

### 2.2 我们的修订

| 提交 | 内容 |
|---|---|
| `64290b8` | 双卡模型并行训练 `train_mp.py`、`model/evspsegnet_mp.py` |
| `8768fbf` | 线性学习率 1e-3 → 1e-4 |
| `9e7db5f` / `0f1a424` | **基线 v2 协议**：每轮验证、种子可配置、末 5 轮均值统计（`utils/aggregate.py`）、STC γ 参数化；两组学习率对照后采用仓库学习率（`configs/evisseg_evuav_baseline_v2_repolr.yaml`） |
| `7081aa6` | **K5 修复**：conv2/3/4 与 inv_conv2/3/4 的时间核 3 → `[3,3,5]`。当前 `model/evspsegnet.py` 已是 K5，旧 K3 权重不兼容 |

**K5 修复的依据**：下采样 `kernel=3, stride=4, padding=1` 时，`t%4==2` 的输入位置不会被任何输出引用。验证过程：按 `t%4` 分组，原基线在余数 2 上明显塌陷，而 V1（没有 mod-4 结构）四组持平，3 种子 × val/test 全部复现；作者自己在 `basemodel.py` 的 `Downsample_block` 里写的就是 `[3,3,5]`，但没有被调用；改成 K5 后重训，四组持平，整体 IoU 提升 18 个点。核 5 同时加大了感受野和参数量，所以提升不能全部归因于补洞（`[3,3,4]` 可以分离，未做）。

### 2.3 基线仍存在的问题（潜在切入点）

- **Patch Attention 实际没有跨 patch 交互**：`unsqueeze(0)` 后序列长度为 1，注意力退化为逐体素线性变换（池化、反池化和残差仍在）。未修，修了可能进一步抬高基线。
- **参数量与论文不符**：发布配置 `width: 12` 下，K3 实际约 94 万参数，K5 约 108 万，论文写的是 4.0M。
- **代码与论文不一致**：学习率（代码 1e-3，论文 1e-2）、STC γ（代码 1，论文 2）。
- **`utils/eval.py` 的口径问题**：Pd/Fa 分帧用严格不等号，排除约 2% 的边界事件；帧数少算 1。为了与论文可比，没有修改。

## 3. 流式 SNN V1

### 3.1 设计（v1-3）

- **数据**：8 秒切成 160 个 50 ms 窗，每窗构造 12 通道稠密计数图（整窗正/负极性 + 5 个 10 ms bin × 正/负），`clip(log1p(C)/q99, 0, 3)` 归一化，q99 来自训练集。
- **神经元**：LIF，`U_pre = β·U + I`，超过阈值 1.0 发放并软复位（减阈值）；代理梯度 `1/(1+|v|)²`；τ 用 sigmoid 约束在 [50, 2000] ms，逐通道可学（初值 200 ms）。每窗只更新一次，膜电位跨窗保持。训练前逐层校准增益，使校准数据上的正电流 q99 对齐阈值。
- **网络**：7 层 2D 脉冲 U-Net（通道 12/24/48/48，分辨率 264×352 → 33×44），每层 `Conv3×3 → 增益 → LIF`，无 bias、无 BN。跳连传脉冲，读出取最后一层复位前膜电位 U_pre，与事件的极性 p、窗内时间 t_local 一起经 MLP 逐事件输出。参数 105,345。
- **损失与训练**：`BCEWithLogits`，pos_weight = min(负/正, 30)（没有用 STC Loss）；TBPTT 每 16 窗一段，一条序列一次更新；50 轮，学习率 1e-3 → 1e-4。
- **评估**：主指标调用原 `utils/eval.py`；同一权重分别以 carry（保持状态）和 reset_each_window（每窗清零）评估；另有逐窗 IoU、首次检出延迟、各层发放率、理论 MAC/SOP 运算量。

### 3.2 迭代记录

| 版本 | 改动 | 原因 | 结果 |
|---|---|---|---|
| v1-1 `8ac25ce` | 初版流式 SNN | 方案一"因果流式脉冲分割"：原方法全是离线的，要等满 8 秒 | test IoU 0.8363；reset 变体 0.8201；tau1000 变体 0.8498 |
| v1-2 `9bb2a0e` | 核验与诊断工具（`verify_predictions`、`diagnose_gap`） | V1 比基线高 21 个点，结果反常 | 数据对齐、指标实现、读出消融均无问题；定位到基线 bug |
| 基线 K5 `7081aa6` | 修复基线时间下采样 | 同上 | 基线 0.8011，V1 在 test 上领先 +0.035，val 上 +0.013 |
| 10 层 LIF（已删除） | 三个 ConvT 后各加增益 + LIF | 原解码器 ConvT 输出实数，占理论能耗 99% | 状态增加 195 万，冒烟时新层几乎不发放，需要重训；放弃，无训练结果 |
| **方案 C** `6cc2d31` | 电流合并解码器 `I = ConvT4×4_s2_p1(S_deep) + Conv3×3(S_skip)`，以及权重换算工具 | 同上；另外比较过最近邻上采样（需重训，能耗不更省） | 与原 V1 数学等价，可直接换算：3 种子指标差 ≤ 2e-4，每窗能耗 2.24 → 0.60 mJ，状态数不变 |
| 方案 C 从零训练 | 用合并形式直接训练 | 看这种参数化直接训练是否更好（函数空间更大） | 进行中（seed37） |
| 训练加速 A | GPU 构造输入（事件常驻显存、查表归一化，与 numpy 输入逐位相同）；片段内逐层时间并行 `forward_chunk`（卷积按 T 个窗口批量算，只有膜电位按时间递推）；训练子集评估可隔轮 | 训练 7–9 h、GPU 利用率 10%；预期约快 3–4 倍，要再快需融合 LIF 时间循环或多序列 batch | 本地 float64 等价测试通过；待服务器核对与计时。`--mode eval` 仍走原实现 |

补充分析：
- **稀疏卷积**：输入像素占用仅 0.5%，但卷积扩散加下采样后深层接近稠密；GPU 上收益小，所以没有采用。
- **是否真 SNN**：V1 属于直接训练的 SNN（代理梯度 + BPTT + 有状态 LIF）。第一层实数输入、末层读膜电位都是常见做法；原解码器的实数输入已由方案 C 解决。

### 3.3 模块状态

| 模块 | 当前 | 可继续迭代 |
|---|---|---|
| 输入表征 | 12 通道计数图，50 ms，5 bin | enc1 实数输入是剩余能耗主项；窗长和 bin 数没做过消融；预处理可挪到 GPU |
| 神经元 / 时序 | LIF，可学 τ | 跨窗记忆几乎没起作用 |
| 编码器 | 4 层脉冲卷积 | — |
| 解码器 | 全脉冲（方案 C） | 从零训练效果待定 |
| 读出 | U_pre + p + t_local 的 MLP | 分析显示不是瓶颈 |
| 损失 | BCE + pos_weight | 可引入因果版 STC 时空先验 |
| 脉冲必要性 | 未验证 | 同结构 ReLU 对照 |

## 4. 创新方向（持续更新）

| 方向 | 状态 | 说明 |
|---|---|---|
| 因果流式分割（时间轴 → SNN 时间步） | ✅ V1 | 50 ms 块因果，逐事件输出 |
| 解码器全脉冲（电流合并） | ✅ 方案 C | 严格等价，能耗降 73% |
| 发现并修复基线实现缺陷 | ✅ | 可作为复现性发现 |
| 脉冲必要性（同结构 ReLU 对照） | 待做 | `--neuron relu`，决定论文以 SNN 还是"轻量因果流式"为主线 |
| 强化跨窗时序记忆 | 待设计 | 可考虑：分析学到的 τ、循环连接、读出引入历史、按事件时间衰减、更长 TBPTT |
| 输入层脉冲化 / enc1 能耗 | 待做 | 改记账口径（只计非零输入）或把输入编码成脉冲 |
| 因果版 STC 时空相关性先验 | 候选 | 把论文的时空连续性思想做成跨窗形式 |
| 低延迟工程 | 待做 | 预处理挪 GPU，规范计时 |
| 窗长 / 时间分箱设计 | 候选 | 50 ms / 5 bin 尚未消融，可考虑自适应窗 |
| 静止 / 慢速目标 | 远期 | 论文自陈局限，多模态或记忆机制 |
| 公平比较 | 需要 | 参数量不对齐（0.105M 对 0.94M）；基线 Patch Attention 未修 |
| 同口径能效对比 | 工具已写，待跑 | 160 步是 160 段新数据，应按每 8 秒比较；基线稀疏卷积只在事件上计算，SNN 的 enc1 在全图上计 MAC。`baseline_energy.py` 测基线实际连接数，`stream_energy.py` 给出 SNN 稠密 / 事件驱动两种口径；决定论文能否主张节能 |
| 可并行训练的脉冲神经元 | 候选 | 训练时整段并行、推理时递推（去复位或线性递推，参考 PSN / 脉冲 SSM）；训练变快后可做完整 160 窗 BPTT，同时针对跨窗记忆弱 |

## 5. 代码地图

**原作者代码**：`train.py` / `test.py`（原入口）、`configs/configs.py` + `evisseg_evuav.yaml`、`dataset/ev_uav.py` + `basedataset.py`（NPZ 读取、体素化）、`model/evspsegnet.py`（主网络，已含 K5）、`model/basemodel.py`（GDBlock 等）、`utils/stcloss.py`、`utils/eval.py`（评估，可比性基准）、`lib/hais_ops/`（CUDA 体素化扩展）。

**基线工程**：`train_mp.py`（双卡入口，环境变量 `EVUAV_MODE` / `EVUAV_MP_SPLIT` / `EVUAV_SEED` / `EVUAV_SAVE_ROOT`）、`model/evspsegnet_mp.py`、`configs/evisseg_evuav_mp*.yaml` 与 `baseline_v2*.yaml`、`utils/aggregate.py`（多种子汇总）、`README_MODEL_PARALLEL.md`、`README_LINEAR_EXPERIMENT.md`。

**流式 SNN V1**

| 文件 | 作用 |
|---|---|
| `train_stream_v1.py` | 入口 `--mode smoke/overfit/train/eval`；TBPTT 训练、逐窗推理、评估、发放率监控、计时 |
| `dataset/stream_windows.py` | 纯 numpy：读 NPZ、切窗、12 通道计数、归一化、按下标回填、训练集统计 |
| `dataset/ev_uav_stream.py` | 按序列加载、窗口转张量 |
| `dataset/stream_source.py` | 窗口数据来源：numpy 逐窗（原实现）或设备端按片段构造（`input_device`） |
| `model/lif2d_stream.py` | LIF、代理梯度、逐通道增益、ReLU 对照 |
| `model/evspsegnet_stream.py` | 网络（`merged_decoder` 开关；逐窗 `forward` / 片段逐层 `forward_chunk`）、权重合并换算、增益校准、运算量估计 |
| `utils/stream_common.py`、`utils/stream_metrics.py` | 配置、学习率、TBPTT 分段、校准数学、健康检查；逐窗与延迟指标 |
| `configs/evisseg_stream_v1.yaml`、`evisseg_stream_merged_decoder.yaml` | V1、方案 C 配置（后者只多 `merged_decoder: true` 与保存目录） |
| `tests/test_stream_*.py` | 单元测试 |
| `README_STREAM_V1.md`、`README_STREAM_MERGED_DECODER.md` | 运行说明、方案 C 原理 |

**工具**：`tools/stream_train_stats.py`（训练集统计）、`dump_baseline_predictions.py`（导出基线逐事件预测并打印原指标）、`verify_predictions.py`（多方法对齐与指标核验）、`diagnose_gap.py`（按 t%4 分组、读出上限、eval 口径量化）、`convert_to_merged_decoder.py`（V1 → 方案 C 换算，含等价核对）、`check_stream_execution.py`（加速选项与原实现的等价核对）、`bench_stream_speed.py`（训练/评估计时分解）、`baseline_energy.py`（基线稀疏卷积实际运算量与能耗）、`stream_energy.py`（SNN 每 8 秒能耗的两种口径与基线对比）。

**其他**：`outputs/`（V1 示意图、合并等价性独立验证）、`References/`（论文 PDF）、V1 架构讲解页 <https://claude.ai/artifact/TYCEZjySwtjWbeFnRR9xWS>（对应原版解码器）。

**分支**：`research/windowed-streaming-snn-v1`（当前）；`main` = `archive/activation-v0`（SNN v0、静态量化）。

## 6. 服务器与命令习惯

- 服务器 `/media/stephen/nvme0n1/wy_data/EV-UAV`，conda 环境 `evuav`，Python 3.8 + torch 1.9.1，4 × RTX 4090；数据在 `/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset/{train,val,test}`。
- **代码同步用 SFTP**（本地改完上传），服务器上不用 git；SFTP 不会删除服务器上的旧文件，删除需要单独 `rm`。
- 长任务用 `nohup ... > run/<名称>.log 2>&1 &`，再用 `tail -f` 查看；输出放 `log/<名称>/`，逐事件预测 dump 放 `log/verify/`。
- 代码需兼容 Python 3.8 / torch 1.9（例如 1.9 的 `torch.testing.assert_close` 默认比较 stride）。本地没有 GPU 和数据，本地测试用 `.venv-stream-check/Scripts/python.exe -m unittest discover -s tests -p "test_stream*.py"`。

### 流式 V1 / 方案 C：训练 → 验证集 → 测试集

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV && conda activate evuav
CFG=configs/evisseg_stream_merged_decoder.yaml   # 原 V1：configs/evisseg_stream_v1.yaml
RUN=stream_merged_decoder; S=37; GPU=0

python -m unittest discover -s tests -p "test_stream*.py" 2>&1 | tail -3

# 冒烟
CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --config $CFG --mode smoke \
    --save-root log/${RUN}_smoke > run/${RUN}_smoke.log 2>&1

# 训练（每轮在 val 上评估，保存 best_val_iou_seed$S.pt；中断后加 --resume 续训）
nohup env CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --config $CFG --mode train \
    --seed $S --save-root log/${RUN}_seed$S > run/${RUN}_seed$S.log 2>&1 &
tail -f run/${RUN}_seed$S.log

# 验证集、测试集（carry 与 reset 各评一遍；可加 --dump-dir log/verify/${RUN}_s${S}_<split>）
for split in val test; do
  CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --mode eval --split $split \
      --checkpoint log/${RUN}_seed$S/best_val_iou_seed$S.pt
done

# 已训练的 V1 换算成方案 C（输出 *_merged.pt，之后同样用 --mode eval 评估）
CUDA_VISIBLE_DEVICES=3 python tools/convert_to_merged_decoder.py --device cuda:0 \
    --checkpoint log/stream_v1_seed$S/best_val_iou_seed$S.pt

# 汇总所有评估结果
python - <<'EOF'
import json, glob
E = lambda op: (op["mac"] * 4.6e-12 + op["sop"] * 0.9e-12) * 1e3
for f in sorted(glob.glob("log/*/eval_*_best_val_iou_seed*.json")):
    d = json.load(open(f)); c, r = d["carry"], d["reset_each_window"]
    print("%-72s IoU %.4f ACC %.4f Pd %.4f Fa %.2e | reset %.4f | %.3f mJ" % (
        f, c["iou"], c["acc"], c["pd"], c["fa"], r["iou"], E(c["operations"])))
EOF
```

### 训练加速与能耗：核对 → 计时 → 能耗 → 重训

```bash
CK=log/stream_v1_seed37/best_val_iou_seed37.pt; GPU=1
# 等价核对（输入逐位、float64 逻辑、float32 数值、整个验证集 IoU），最后一行应为 EXECUTION CHECK PASSED
CUDA_VISIBLE_DEVICES=$GPU python tools/check_stream_execution.py --checkpoint $CK --split val --full-split     > run/check_stream_execution_s37.log 2>&1
# 计时分解（四种组合 × 训练/评估，前向/反向，小输入对照只看规模变化；加速以重训日志 epoch_seconds 为准）
CUDA_VISIBLE_DEVICES=$GPU python tools/bench_stream_speed.py --checkpoint $CK     --reference-metrics log/stream_v1_seed37/metrics.jsonl > run/bench_stream_speed_s37.log 2>&1
# 能耗：基线实际运算量（GPU）→ SNN 两种口径并对比（CPU）
CUDA_VISIBLE_DEVICES=$GPU python tools/baseline_energy.py --config configs/evisseg_evuav_baseline_v2_repolr.yaml     --checkpoint log/baseline_k5_repolr_seed37/best_iou_seed37.pt --split test > run/baseline_energy_s37.log 2>&1
python tools/stream_energy.py --config configs/evisseg_stream_v1.yaml --split test     --eval-json log/stream_v1_seed37/eval_test_best_val_iou_seed37.json log/stream_v1_seed37/eval_test_best_val_iou_seed37_merged.json     --baseline-json log/baseline_k5_repolr_seed37/energy_test_best_iou_seed37.json
# 用加速选项重训（与 log/stream_v1_seed37 的曲线和耗时对照）
nohup env CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --config configs/evisseg_stream_v1.yaml --mode train     --seed 37 --execution layer --input-device gpu --train-subset-every 5     --save-root log/stream_v1_fast_seed37 > run/stream_v1_fast_seed37.log 2>&1 &
```

### 基线 K5（双卡）：训练 → 验证集 → 测试集

```bash
S=37
nohup env CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  EVUAV_MODE=train EVUAV_MP_SPLIT=2 EVUAV_SEED=$S EVUAV_SAVE_ROOT=log/baseline_k5_repolr_seed$S \
  python -u train_mp.py --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
  > run/baseline_k5_repolr_seed$S.log 2>&1 < /dev/null &
tail -n 30 -f run/baseline_k5_repolr_seed$S.log

# 验证集、测试集完整指标（只用原仓库代码，同时导出逐事件预测）
for split in val test; do
  CUDA_VISIBLE_DEVICES=0 python tools/dump_baseline_predictions.py \
      --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
      --checkpoint log/baseline_k5_repolr_seed$S/best_iou_seed$S.pt \
      --split $split --out-dir log/verify/baseline_k5_s${S}_$split
done
```

## 7. 指标记录

阈值 0.9；ACC 为正类召回；流式模型为 carry 模式；"a / b / c"对应种子 37 / 38 / 39。

### 测试集

| 版本 | 参数 | IoU | ACC | Pd | Fa |
|---|---|---|---|---|---|
| 论文 Tab.2 | 4.0M（论文值） | 0.5518 | 0.6502 | 0.7753 | 1.63e-4 |
| 官方权重 | — | 0.5844 | 0.6785 | 0.7848 | 8.49e-6 |
| 基线 v2（K3，有 bug） | ~0.94M | 0.5894 / 0.6243 / 0.6428 → **0.6188** | 0.694 | 0.806 | 1.79e-5 |
| 基线 K5 | ~1.08M | 0.8143 / 0.7946 / 0.7944 → **0.8011** | 0.8307 / 0.8085 / 0.8060 | — | — |
| **流式 V1** | 105,345 | 0.8226 / 0.8372 / 0.8492 → **0.8363** | 0.9183 / 0.9085 / 0.9190 | 0.9459 / 0.9478 / 0.9505 | 1.15e-5 / 9.75e-6 / 1.01e-5 |
| V1 reset 训练（s37） | 105,345 | 0.8201 | 0.8751 | 0.8705 | 5.62e-6 |
| V1 tau1000（s37） | 105,345 | 0.8498 | 0.9240 | 0.9503 | 1.05e-5 |
| **V1 换算为方案 C** | 训练 105,345 / 存储 123,057 | 0.8226 / 0.8372 / 0.8493 | 同 V1 | 0.9459 / 0.9476 / 0.9505 | 同 V1 |
| 方案 C 从零训练（s37） | 123,057 | 进行中 | | | |

### 验证集

| 版本 | IoU | ACC | Pd | Fa |
|---|---|---|---|---|
| 基线 K5 | 0.8592 / 0.8479 / 0.8386 | 0.8780 / 0.8700 / 0.8549 | — | — |
| **流式 V1** | 0.8756 / 0.8544 / 0.8551 → 0.8617 | 0.9221 / 0.8968 / 0.9128 | 0.9246 / 0.9045 / 0.9238 | 7.29e-6 / 6.92e-6 / 7.80e-6 |
| V1 reset 训练（s37） | 0.8152 | 0.8897 | 0.8362 | 9.38e-6 |
| V1 tau1000（s37） | 0.8667 | 0.9181 | 0.9217 | 7.86e-6 |
| V1 换算为方案 C | 同 V1 | 同 V1 | 同 V1 | 同 V1 |

### 效率

| 版本 | 理论能耗（每 50 ms 窗；MAC 4.6 pJ / AC 0.9 pJ，实测发放率） | 膜电位状态 | 单窗耗时 |
|---|---|---|---|
| 流式 V1 | 2.24 mJ/窗，8 s 约 358 mJ（MAC 占 99%） | 7 层，397 万 | 约 19–20 ms（预处理 13 + 网络 7，GPU 空闲时） |
| 方案 C（换算） | 0.59–0.61 mJ/窗，8 s 约 96 mJ（MAC 约 93%，来自 enc1 全图稠密计数） | 同上 | — |

`estimate_operations` 按单窗统计；enc1 在全图（含 99.5% 空像素）上计 MAC。基线能耗尚未统计，两者还没有同口径对比。

V1 单卡训练 50 轮约 6.6–7.6 小时。
