# 流式 SNN Baseline-V1（设计版本 v1-3）

50 ms 块因果流式分割：每个窗口构造 12 通道输入 → 二维脉冲 U-Net（逐位置独立 LIF，状态跨窗保持）→
读取最后一层复位前膜电位做逐事件分类 → 按原始下标回填 → 调用原 `utils/eval.py` 计算 IoU/ACC/Pd/Fa。

不修改原仓库任何文件，不依赖 spconv / HAIS_OP。模型很小（105,345 个参数），单卡即可。

## 新增文件

| 文件 | 作用 |
|---|---|
| `dataset/stream_windows.py` | 纯 numpy：读取校验 NPZ、切 160 个窗、12 通道输入、训练集统计 |
| `dataset/ev_uav_stream.py` | PyTorch 数据接口：按序列加载、窗口转张量 |
| `dataset/stream_source.py` | 窗口数据来源：numpy 逐窗（原实现）或事件常驻设备、按片段构造（加速） |
| `model/lif2d_stream.py` | LIF 神经元（sigmoid 参数化 tau）、逐通道增益，以及两个对照神经元：graded（有状态+实数）、relu（无状态+实数） |
| `model/evspsegnet_stream.py` | 二维脉冲 U-Net、逐事件读出、逐层增益校准、理论运算量估计 |
| `utils/stream_common.py` | 配置、学习率、TBPTT 分块、子集选择、历史汇总、校准数学、健康门槛 |
| `utils/stream_metrics.py` | 逐窗 TP/FP/FN、滚动/分段 IoU、首次检出延迟 |
| `tools/stream_train_stats.py` | 阶段 1：审计三个划分 + 只用训练集统计 q99 与 pos_weight |
| `tools/summarize_runs.py` | 汇总 log/ 下所有评估结果，按种子求均值输出对比表 |
| `tools/check_stream_execution.py` | 核对加速路径与原实现等价（输入逐位、float64 逻辑、整个划分 IoU） |
| `tools/bench_stream_speed.py` | 训练/评估计时分解与单轮耗时估算 |
| `tools/baseline_energy.py`、`tools/stream_energy.py` | 基线与 SNN 的同口径能耗 |
| `tools/sweep_threshold.py` | 扫判定阈值，按等虚警率比较不同模型 |
| `train_stream_v1.py` | smoke / overfit / train / eval 四种模式 |
| `configs/evisseg_stream_v1.yaml` | 全部超参 |
| `tests/test_stream_*.py` | 单元测试（前两个只需 numpy，`test_stream_torch.py` 需要 torch，CPU 即可） |

## 服务器运行顺序

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV
conda activate evuav
mkdir -p run log
```

### 阶段 1：数据审计与训练集统计（CPU，约几分钟）

```bash
python tools/stream_train_stats.py --config configs/evisseg_stream_v1.yaml
```

生成 `configs/stream_v1_train_stats.json`。检查输出里的 `q99`（12 个正数）、`train_neg_pos_ratio`、
`pos_weight`，以及三个划分的 `empty_windows` / `events_per_window_max`。任何事件越界、p 不是 0/1、
时间戳 >= 8000 都会直接报错并指出文件名。

### 阶段 2：单元测试

```bash
python -m unittest discover -s tests -p "test_stream_*.py" -v
```

### 阶段 3-4：冒烟测试（单卡，约 1-2 分钟）

```bash
CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --mode smoke --save-root log/stream_v1_smoke
```

看 `log/stream_v1_smoke/smoke.json` 与 `calibration.json`：
各层 `firing_rate` 不应为 0、`missing_grads` 为空、`warnings` 为空或可解释、`train_ms_per_window`、`peak_memory_gib`。
若报错 "does not have a deterministic implementation"，把 YAML 中 `deterministic` 改为 `false` 再跑。

### 阶段 5：单序列过拟合三方对照（三个进程可放在不同 GPU 上并行）

```bash
nohup env CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --mode overfit \
  --save-root log/stream_v1_overfit_lif > run/stream_v1_overfit_lif.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 python train_stream_v1.py --mode overfit --neuron relu \
  --save-root log/stream_v1_overfit_relu > run/stream_v1_overfit_relu.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=2 python train_stream_v1.py --mode overfit --state-mode reset_each_window \
  --save-root log/stream_v1_overfit_reset > run/stream_v1_overfit_reset.log 2>&1 &
tail -f run/stream_v1_overfit_lif.log
```

期望 loss 大幅下降、IoU 升到 0.9 左右。三者停在相近位置 → 数据/架构上限；只有 LIF 版本明显低 → 查梯度、读出、增益、索引。

### 阶段 6-7：训练 seed 37

```bash
nohup env CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --mode train \
  --save-root log/stream_v1_seed37 > run/stream_v1_seed37.log 2>&1 &
tail -f run/stream_v1_seed37.log
```

每个 epoch 打印一行（学习率、loss/事件、验证 IoU/ACC、训练子集 IoU、耗时、健康警告数），
详细记录在 `metrics.jsonl`。先看前几个 epoch 的耗时与收敛情况（阶段 6），再决定是否跑满 50 轮。
中断后可加 `--resume` 从 `last.pt` 精确续跑。

### 训练加速与对照实验

配置里的四个开关（`configs/evisseg_stream_v1.yaml` 的 TRAIN 小节，也可用同名命令行参数覆盖）：

| 选项 | 默认 | 说明 |
|---|---|---|
| `execution` | `layer` | `step` 逐窗逐层（原实现）；`layer` 片段内逐层时间并行，数学等价，训练快 6 倍 |
| `input_device` | `gpu` | `cpu` numpy 逐窗构造（原实现）；`gpu` 事件常驻显存按片段构造，输入逐位相同 |
| `train_subset_every` | `5` | 训练子集评估的间隔（只用于诊断，不影响训练） |
| `eval_chunk` | `32` | `layer` 模式下每轮验证一次处理多少个窗口 |

默认是加速路径，每轮 545 s → 73 s，50 轮约 1 小时。**要复现 09-16 那批原实现的数字**，加
`--execution step --input-device cpu --train-subset-every 1`。`--mode eval` 始终走原实现，不受这些开关影响。

三种神经元用于分离"脉冲"与"跨窗记忆"：

```bash
python train_stream_v1.py --mode train --neuron lif      # 有状态 + 二值发放（主模型）
python train_stream_v1.py --mode train --neuron graded   # 有状态 + 实数发放（幅值 = 膜电位）
python train_stream_v1.py --mode train --neuron relu     # 无状态 + 实数
```

`--tbptt-k` 可改梯度回传的窗口数（默认 16 窗 = 800 ms）。

### 评估（同一权重分别跑 carry 与 reset_each_window）

```bash
CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --mode eval \
  --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt --split val
CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --mode eval \
  --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt --split test
```

结果写到 checkpoint 同目录的 `eval_<split>_<checkpoint名>.json`：主指标、分段/滚动 IoU、首次检出延迟、
各层发放率与 tau、单窗耗时（含 CUDA 同步与预热）、理论 MAC/SOP、实测非零输入占比。

跑完若干个后，一条命令出对比表：

```bash
python tools/summarize_runs.py --split test               # 按运行名分组、跨种子求均值
python tools/summarize_runs.py --split test --markdown    # 输出 Markdown，可直接贴进 CLAUDE.md
```

## 口径说明

- 主指标阈值 0.9，IoU/ACC 为所有序列拼接后的全局值，Pd/Fa 原样调用 `roc_update`，与基线 0.6188 一致。
- 延迟是 50 ms 块因果：事件等待 ≤ 50 ms + 单窗计算时间；不能写成"延迟提升 160 倍"。
- SOP 是面向神经形态硬件的理论估计；当前 GPU 实现仍执行稠密卷积。第一层（实数输入）、
  解码器中 ConvT 输出的通道、读出 MLP 按 MAC 计。
- 能耗有两个口径：**稠密**（enc1 在整张画布上计 MAC）与**事件驱动**（enc1 只计非零输入，实测占比 0.09%）。
  与基线比较要用事件驱动口径，因为基线的稀疏卷积本来就只在有事件的位置计算。每 8 秒 = 每窗 × 160。
- 不同模型在同一个阈值下往往落在不同工作点（例如 LIF 与 ReLU）。比较前先用 `tools/sweep_threshold.py`
  扫阈值，在相同虚警率下对齐，否则 IoU 的高低可能只反映工作点差异。
- 参数量约 0.10M，与原 4.0M 模型不对齐。
