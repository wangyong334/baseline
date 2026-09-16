# 流式 SNN Baseline-V1（设计版本 v1-3）

50 ms 块因果流式分割：每个窗口构造 12 通道输入 → 二维脉冲 U-Net（逐位置独立 LIF，状态跨窗保持）→
读取最后一层复位前膜电位做逐事件分类 → 按原始下标回填 → 调用原 `utils/eval.py` 计算 IoU/ACC/Pd/Fa。

不修改原仓库任何文件，不依赖 spconv / HAIS_OP。模型很小（105,345 个参数），单卡即可。

## 新增文件

| 文件 | 作用 |
|---|---|
| `dataset/stream_windows.py` | 纯 numpy：读取校验 NPZ、切 160 个窗、12 通道输入、训练集统计 |
| `dataset/ev_uav_stream.py` | PyTorch 数据接口：按序列加载、窗口转张量 |
| `model/lif2d_stream.py` | LIF 神经元（sigmoid 参数化 tau）、ReLU 对照、逐通道增益 |
| `model/evspsegnet_stream.py` | 二维脉冲 U-Net、逐事件读出、逐层增益校准、理论运算量估计 |
| `utils/stream_common.py` | 配置、学习率、TBPTT 分块、子集选择、历史汇总、校准数学、健康门槛 |
| `utils/stream_metrics.py` | 逐窗 TP/FP/FN、滚动/分段 IoU、首次检出延迟 |
| `tools/stream_train_stats.py` | 阶段 1：审计三个划分 + 只用训练集统计 q99 与 pos_weight |
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

### 评估（同一权重分别跑 carry 与 reset_each_window）

```bash
CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --mode eval \
  --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt --split val
CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --mode eval \
  --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt --split test
```

结果写到 checkpoint 同目录的 `eval_<split>_<checkpoint名>.json`：主指标、分段/滚动 IoU、首次检出延迟、
各层发放率与 tau、单窗耗时（含 CUDA 同步与预热）、理论 MAC/SOP。

## 口径说明

- 主指标阈值 0.9，IoU/ACC 为所有序列拼接后的全局值，Pd/Fa 原样调用 `roc_update`，与基线 0.6188 一致。
- 延迟是 50 ms 块因果：事件等待 ≤ 50 ms + 单窗计算时间；不能写成"延迟提升 160 倍"。
- SOP 是面向神经形态硬件的理论估计；当前 GPU 实现仍执行稠密卷积。第一层（实数输入）、
  解码器中 ConvT 输出的通道、读出 MLP 按 MAC 计。
- 参数量约 0.10M，与原 4.0M 模型不对齐。
