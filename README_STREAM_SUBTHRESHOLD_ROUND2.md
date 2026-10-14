# 亚阈值实验第二轮：横向单独与竞争式纵向编码

本轮固定 `enc2/enc3/enc4` 双仓、seed 37 和所有训练参数，只比较两种亚阈值信息路径：

| 物理 GPU | 配置 | 机制 | 输出目录 |
|---|---|---|---|
| 0 | `configs/evisseg_stream_dual_enc234_residual_horizontal.yaml` | 横向事件残差单独 | `log/stream_dual_enc234_residual_horizontal_seed37` |
| 1 | `configs/evisseg_stream_dual_enc234_residual_competitive_vertical.yaml` | 重新设计的竞争式纵向残差 | `log/stream_dual_enc234_residual_competitive_vertical_seed37` |

横向单独组有意保留上一轮相同的横向实现，用来完成缺失的 2×2 因果对照。上一轮双路模型的
`event_gate_support_frac=1.0`，所以本组不能预设“深层硬支持门已有效筛除背景”；需要根据独立训练结果及
门值分布决定下一步是否重构横向门控。

## 为什么原纵向方案失败

原纵向输入是 `S1 + R1`，其中所有未放电位置的连续正负残差都可进入 enc2。test 表明它比 enc234
基线增加 587 个 TP，却增加 2236 个 FP，IoU 下降 1.656 pp。结构上有三个原因：

1. 没有选择机制：目标边缘、背景事件、传感器噪声获得相同通行权。
2. 没有安全耦合：残差以固定单位尺度直接改变 enc2 的输入分布，模型不能在训练初期保持基线。
3. 连续残差使 enc2 的稀疏脉冲累加变成整幅 MAC，计算代价增加约 16.7%。

## 竞争式稀疏双极残差

当前观测残差仍保持无状态和严格块因果：

```text
R1 = (1-stopgrad(S1)) * clamp(I1/theta, -1, 1)
```

新纵向编码为：

```text
A1       = |R1|
P1       = H(A1-rho_c) * 1[A1 是同通道 3x3 局部峰值]
Z1       = sign(R1) * P1 * g_c
enc2_in  = S1 + Z1

rho_c = rho_min + (rho_max-rho_min)*sigmoid(rho_raw,c)
g_c   = g_max*tanh(g_raw,c)
```

`rho_c` 和 `g_c` 都是逐通道参数。阈值初始为 0.5，范围 `[0.2,0.9]`；`g_c` 范围
`(-0.5,0.5)` 且严格零初始化，所以训练开始时网络与 enc234 基线完全相同。只有损失确认某个通道的
亚阈值证据有用时，该通道才会打开。`Z1` 是稀疏双极码；推理时 `g_c` 可折叠进 enc2 对应输入通道
的卷积核，事件驱动实现可按稀疏加/减累加，不需要原方案的整幅连续 MAC。

评估 JSON 会额外记录 `vertical_signal_nonzero_frac`、正负比例、幅值，以及学习后的阈值和注入幅度。

## 两卡训练

在服务器项目目录运行：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v1
mkdir -p run

nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_horizontal.yaml \
  --mode train --device cuda:0 \
  > run/dual_enc234_residual_horizontal_train.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_competitive_vertical.yaml \
  --mode train --device cuda:0 \
  > run/dual_enc234_residual_competitive_vertical_train.log 2>&1 &
```

查看：

```bash
tail -f run/dual_enc234_residual_horizontal_train.log \
        run/dual_enc234_residual_competitive_vertical_train.log
```

## 正常测试

两组训练完成后并行执行：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_horizontal.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_horizontal_seed37/best_val_iou_seed37.pt \
  > run/dual_enc234_residual_horizontal_test.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_competitive_vertical.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_competitive_vertical_seed37/best_val_iou_seed37.pt \
  > run/dual_enc234_residual_competitive_vertical_test.log 2>&1 &
```

## 同权重因果消融

正常测试完成后，两张卡分别关闭该组新增路径并关闭双仓读取。每张卡内顺序运行，避免同卡并发：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -c '
set -e
python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_horizontal.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_horizontal_seed37/best_val_iou_seed37.pt \
  --subthreshold-ablation skip_off
python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_horizontal.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_horizontal_seed37/best_val_iou_seed37.pt \
  --memory-ablation off
' > run/dual_enc234_residual_horizontal_ablations.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -c '
set -e
python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_competitive_vertical.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_competitive_vertical_seed37/best_val_iou_seed37.pt \
  --subthreshold-ablation vertical_off
python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_competitive_vertical.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_competitive_vertical_seed37/best_val_iou_seed37.pt \
  --memory-ablation off
' > run/dual_enc234_residual_competitive_vertical_ablations.log 2>&1 &
```

同权重关闭会产生分布偏移，它用于判断已训练模型是否实际使用新增路径，不代替独立重训对照。
