# Enc234 双仓 + enc1 亚阈值残差双路传递

本实验在 `enc2/enc3/enc4` 双仓模型上增加一个只由当前窗口 enc1 驱动电流生成的残差，处理单物理步流式 SNN 中未破阈响应在层间变成严格零的问题。它不把旧膜电位当成新观测，也不改变 enc1 的二值放电和减阈值软复位。

## 数学定义

enc1 保持原动力学：

```text
I1_t     = gain1(norm1(conv1(X_t)))
U1_pre,t = beta1 * U1_{t-1} + I1_t
S1_t     = H(U1_pre,t - theta)
U1_t     = U1_pre,t - theta * stopgrad(S1_t)
```

新增的当前观测残差为：

```text
R1_t = (1 - stopgrad(S1_t)) * clamp(I1_t / theta, -1, 1)
```

`I1_t=0` 时 `R1_t` 严格为零，因此旧 `U1` 不会被重复传给深层。已经放电的通道由 `S1` 传递，残差置零以免重复计数。

纵向路径使用 `S1+R1` 作为 enc2 输入；enc2 仍输出二值脉冲，enc234 的双仓公式不变。横向路径保留原二值 enc1→dec1 跳连，并只在最终事件坐标读取后加入连续修正：深层 `up1(S6)` 的 3×3 局部活动先生成许可门，没有深层局部活动时浅层残差不能独立影响输出。连续修正不写入 dec1 膜电位，所以不会形成新增跨窗尾迹。

## 两组配置

| 配置 | 纵向残差 | 门控事件跳连 | 输出目录 |
|---|---:|---:|---|
| `configs/evisseg_stream_dual_enc234_residual_vertical.yaml` | 是 | 否 | `log/stream_dual_enc234_residual_vertical_seed37` |
| `configs/evisseg_stream_dual_enc234_residual_dualpath.yaml` | 是 | 是 | `log/stream_dual_enc234_residual_dualpath_seed37` |

二者均为 seed 37、50 epochs、TBPTT 16、50 ms/窗，并继续使用原数据集路径。第二组只比第一组增加157个参数。纵向路径使 enc2 接收实数残差，因此 enc2 按 MAC 计算；事件跳连的线性变换只在实际事件坐标运行。

## 两卡训练

在服务器项目目录 `/media/stephen/nvme0n1/wy_data/EV-UAV-v1` 中运行：

```bash
mkdir -p run

nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_vertical.yaml \
  --mode train --device cuda:0 \
  > run/dual_enc234_residual_vertical_train.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_dualpath.yaml \
  --mode train --device cuda:0 \
  > run/dual_enc234_residual_dualpath_train.log 2>&1 &
```

查看日志：

```bash
tail -f run/dual_enc234_residual_vertical_train.log \
        run/dual_enc234_residual_dualpath_train.log
```

## 测试与同权重消融

```bash
for variant in vertical dualpath; do
  gpu=0
  [ "$variant" = dualpath ] && gpu=1
  nohup env CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python -u train_stream_v1.py \
    --config configs/evisseg_stream_dual_enc234_residual_${variant}.yaml \
    --mode eval --device cuda:0 --split test \
    --checkpoint log/stream_dual_enc234_residual_${variant}_seed37/best_val_iou_seed37.pt \
    > run/dual_enc234_residual_${variant}_test.log 2>&1 &
done
```

组合模型支持同权重消融：

```bash
python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_dualpath.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_dualpath_seed37/best_val_iou_seed37.pt \
  --subthreshold-ablation vertical_off

python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_dualpath.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_dualpath_seed37/best_val_iou_seed37.pt \
  --subthreshold-ablation skip_off

python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234_residual_dualpath.yaml \
  --mode eval --device cuda:0 --split test --overwrite \
  --checkpoint log/stream_dual_enc234_residual_dualpath_seed37/best_val_iou_seed37.pt \
  --subthreshold-ablation all_off
```

评估 JSON 的 `subthreshold_diagnostics` 会保存残差非零率、正负比例、平均幅值、事件门支持比例、门均值和事件修正平均幅值。`all_off` 是同权重干预，不等价于重新训练的 enc234 基线。

## 本地检查

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv-stream-check\Scripts\python.exe -m unittest discover -s tests -p 'test_stream_*.py'
git diff --check
```
