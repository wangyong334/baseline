# 亚阈值实验第三轮：竞争式纵向与两种横向路径组合

固定 enc2/enc3/enc4 双仓、seed 37、训练参数和竞争式纵向编码，比较两种横向路径：

| 物理 GPU | 配置 | 横向模式 | 输出目录 |
|---|---|---|---|
| 0 | `configs/evisseg_stream_dual_enc234_residual_competitive_dualpath.yaml` | 原横向门控 | `log/stream_dual_enc234_residual_competitive_dualpath_seed37` |
| 1 | `configs/evisseg_stream_dual_enc234_residual_selective_dualpath.yaml` | 稀疏有界横向门控 | `log/stream_dual_enc234_residual_selective_dualpath_seed37` |

已有竞争式纵向单路结果是共同对照。原横向在独立训练时 `event_gate_support_frac=1.0`，开启路径使
IoU 下降 3.170 pp，主要来自 4,904 个额外 FP。第二组因此同时修复选择性和幅度：

```text
C1 = competitive_code(R1)                       # 每通道阈值 + 3x3 局部竞争，{-1,0,+1}
A  = sum_channel(abs(deep1))
D  = H(A / (MaxPool3x3(A)+eps) - rho_event)     # 深层相对局部峰值支持
G  = 1[C1 非零] * D * sigmoid(linear(deep1_local))
Delta = 0.25 * tanh(project(C1_event * G))
feature_event = U_pre7_event + Delta
```

`rho_event` 可学习，初始 0.8、范围 `[0.5,0.98]`。投影为零初始化，因此两组训练开始时都严格等同
enc234；选择性版本的每维修正严格位于 `[-0.25,0.25]`。横向路径不写入任何跨窗状态。

## 两卡训练

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v1
mkdir -p run

nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_competitive_dualpath.yaml \
  --mode train --device cuda:0 \
  > run/dual_enc234_competitive_dualpath_train.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_selective_dualpath.yaml \
  --mode train --device cuda:0 \
  > run/dual_enc234_selective_dualpath_train.log 2>&1 &
```

## 正常测试

```bash
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_competitive_dualpath.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_competitive_dualpath_seed37/best_val_iou_seed37.pt \
  > run/dual_enc234_competitive_dualpath_test.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_selective_dualpath.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_selective_dualpath_seed37/best_val_iou_seed37.pt \
  > run/dual_enc234_selective_dualpath_test.log 2>&1 &
```

## 同权重消融

每个 checkpoint 均运行 `vertical_off`、`skip_off`、`all_off` 和 `M-off`。前两项给出单路径边际作用，
`all_off` 给出两条新路径相对同一训练所得共享权重的总作用，`M-off` 只关闭双仓读取。

```bash
for pair in \
  '0 competitive_dualpath configs/evisseg_stream_dual_enc234_residual_competitive_dualpath.yaml log/stream_dual_enc234_residual_competitive_dualpath_seed37' \
  '1 selective_dualpath configs/evisseg_stream_dual_enc234_residual_selective_dualpath.yaml log/stream_dual_enc234_residual_selective_dualpath_seed37'
do
  read -r gpu name config output <<< "$pair"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    bash -c "
      set -e
      for ablation in vertical_off skip_off all_off; do
        python -u train_stream_v1.py --config '$config' --mode eval --device cuda:0 \
          --split test --overwrite --checkpoint '$output/best_val_iou_seed37.pt' \
          --subthreshold-ablation \"\$ablation\"
      done
      python -u train_stream_v1.py --config '$config' --mode eval --device cuda:0 \
        --split test --overwrite --checkpoint '$output/best_val_iou_seed37.pt' \
        --memory-ablation off
    " > "run/dual_enc234_${name}_ablations.log" 2>&1 &
done
```

