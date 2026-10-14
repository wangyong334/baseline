# V1 双仓流式 SNN：编码器深度对照

2026-10-14 第二轮单层位置对照（enc2 / enc3 / dec3 / dec2）及逐层读取消融见
[README_STREAM_DUAL_PLACEMENT.md](README_STREAM_DUAL_PLACEMENT.md)。下文保留第一轮三组命令。

项目上传到 `/media/stephen/nvme0n1/wy_data/EV-UAV-v1`。数据仍读取
`/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset`，不复制或修改数据。
原始 `configs/evisseg_stream_v1.yaml` 保留；不指定 `dual_memory` 时仍构建原 V1。

## 三组实验与 U-Net 路径

| 物理 GPU | 配置 | 双仓位置 | 新增参数 | 新增 M 状态元素（单序列） |
|---|---|---|---:|---:|
| 1 | `configs/evisseg_stream_dual_enc4.yaml` | enc4 | 96 | 69,696 |
| 2 | `configs/evisseg_stream_dual_enc34.yaml` | enc3、enc4 | 192 | 348,480 |
| 3 | `configs/evisseg_stream_dual_enc234.yaml` | enc2、enc3、enc4 | 240 | 906,048 |

所有组使用相同 V1 卷积、跳连、解码器、软复位、逐事件读出、数据归一化、seed 37、
50 ms × 160 窗、TBPTT 16 窗、50 epochs；只改变双仓部署位置。

| 层 | 特征尺寸 | 原图步幅 | 单窗理论感受野 | 2.6 px 位移对应特征网格 |
|---|---|---:|---:|---:|
| enc1 | 264 × 352 | 1 | 3 × 3 | 2.6 |
| enc2 | 132 × 176 | 2 | 5 × 5 | 1.3 |
| enc3 | 66 × 88 | 4 | 9 × 9 | 0.65 |
| enc4 | 33 × 44 | 8 | 17 × 17 | 0.325 |

感受野按当前代码四个 3×3 卷积、步幅 1/2/2/2 计算；不是实测有效感受野，也不是运动容忍上限。
深层的空间覆盖可能让相邻窗口的目标激活重叠，但不能保证平移不变，更不能保证任意速度下对应正确。
enc4 经 up3 → dec3 → up2 → dec2 → up1 → dec1 影响输出，浅层跳连提供定位细节。
U-Net 的编码/解码尺寸对称不要求新增记忆机制对称；本轮解码器不添加 M。

## 最小机制

每个选中层保留原 LIF 状态 U，增加同尺寸的记忆 M（序列起点均为零）：

```text
beta  = exp(-50 / tau_U)
alpha = exp(-50 / tau_M)
kappa = kappa_max * tanh(kappa_raw)

Q_t     = tanh(I_t / theta)
M_t     = alpha * M_(t-1) + (1-alpha) * Q_t
U_pre,t = beta * U_(t-1) + I_t + (1-beta) * kappa * theta * M_(t-1)
S_t     = H(U_pre,t - theta)
U_t     = U_pre,t - theta * stop_gradient(S_t)
```

- I 是本层卷积、归一化、增益之后的输入电流。写 M 不使用已经被本层 M 增强的 U_pre。
  多层部署时 I 可以包含上游层的历史，但编码器没有从本层返回上游的反馈路径。
- tau_U 保留 V1；tau_M 按通道学习，初始 300 ms，约束在 50–1000 ms。
- kappa 按通道学习，可正可负，上限绝对值为 0.5；raw 初始化为零。
  同种子下原卷积权重初始化不变，初始输出与 V1 一致。读取耦合学离零后，tau_M 才获得非零任务梯度。
- 读取旧 M，当前写入供后续窗口使用。M 不随放电复位；每个 TBPTT 片段内部保留写入梯度，
  只在片段边界递归 detach U/M；新独立序列清空两者。空窗口仍更新状态。
- 不加入空间扩散、速度估计、槽位查询、浅层强泄漏或硬复位。

从零初始化且输入有限时，M 始终在 [-1,1] 内。固定后续输入时，本层初始 M 的差异按 alpha 的乘积衰减。
读取乘上 `(1-beta)`，避免长 tau_U 把固定记忆电流放大成 `1/(1-beta)` 倍的静态偏置。
若本层 U 初始为零、I 始终为零、M 有界，则该记忆路径单独产生的 U 始终低于 0.5 theta，不能自行触发放电。
这是本层、给定条件下的约束，不等于整个网络没有残留响应。

持续背景仍可能被写入，并与当前输入共同触发误报；已有 U 也可能造成短暂残留。
本实现限制直接自我回写与记忆驱动强度，不声称辨别了所有错误历史。
移动目标是否获益、enc4 是否丢失小目标细节，需要下面的对照实验回答。

## 上传后准备（一次）

在原来能运行 V1 的 Python 环境中执行：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v1
conda activate evuav
mkdir -p run log

# 上传原 V1 的同一个统计文件即可；缺失时才重新统计。
if [ ! -f configs/stream_v1_train_stats.json ]; then
  python tools/stream_train_stats.py --config configs/evisseg_stream_v1.yaml
fi

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_stream_*.py'
```

建议先用最深三层组冒烟，覆盖本轮所有双仓位置；冒烟使用独立目录，不产生正式训练权重：

```bash
CUDA_VISIBLE_DEVICES=3 python train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234.yaml --mode smoke \
  --save-root log/stream_dual_enc234_smoke --device cuda:0
```

检查 `smoke.json` 中 missing_grads、非有限值、各层发放率与 kappa。单次更新后 tau_M 还未变化可以是零耦合初始化的正常现象。
若已运行过冒烟，请换一个新的 smoke 输出目录。

## 三卡并行训练

```bash
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc4.yaml --mode train --device cuda:0 \
  > run/dual_enc4_train.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc34.yaml --mode train --device cuda:0 \
  > run/dual_enc34_train.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234.yaml --mode train --device cuda:0 \
  > run/dual_enc234_train.log 2>&1 &
```

设置 CUDA_VISIBLE_DEVICES 后，每个进程内部都使用 cuda:0，即各自可见的那一张物理卡。
输出目录分别为 `log/stream_dual_enc4_seed37`、`log/stream_dual_enc34_seed37`、
`log/stream_dual_enc234_seed37`，互不覆盖。查看日志：

```bash
tail -f run/dual_enc4_train.log run/dual_enc34_train.log run/dual_enc234_train.log
```

中断续训使用同一条训练命令添加 `--resume`，日志重定向改为 `>>`。只能恢复匹配的双仓配置；
不要把 V1 checkpoint 当作新结构的 resume。新结构权重、时间常数、耦合参数与优化器都会保存。
U/M 是序列内在线状态，不写入 checkpoint；训练只在完整 epoch 结束后保存。

## 三卡并行测试（训练结束后执行）

```bash
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc4.yaml --mode eval --device cuda:0 --split test \
  --checkpoint log/stream_dual_enc4_seed37/best_val_iou_seed37.pt \
  > run/dual_enc4_test.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc34.yaml --mode eval --device cuda:0 --split test \
  --checkpoint log/stream_dual_enc34_seed37/best_val_iou_seed37.pt \
  > run/dual_enc34_test.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234.yaml --mode eval --device cuda:0 --split test \
  --checkpoint log/stream_dual_enc234_seed37/best_val_iou_seed37.pt \
  > run/dual_enc234_test.log 2>&1 &
```

每个进程会评估 carry 和 reset_each_window，后者清空所有层 U/M。
JSON 写到对应 checkpoint 目录的 `eval_test_best_val_iou_seed37.json`。
正式测试始终使用逐窗执行；不改变 V1 的阈值 0.9、IoU/ACC/Pd/Fa 口径。

## 记忆是否真正有帮助

正常测试完成后，三组分别再做同权重关闭 M 读取的测试，保留普通 LIF 的 U 历史：

```bash
for pair in '1 enc4' '2 enc34' '3 enc234'; do
  read -r gpu variant <<< "$pair"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u train_stream_v1.py \
    --config "configs/evisseg_stream_dual_${variant}.yaml" --mode eval --device cuda:0 --split test \
    --checkpoint "log/stream_dual_${variant}_seed37/best_val_iou_seed37.pt" \
    --memory-ablation off > "run/dual_${variant}_memory_off_test.log" 2>&1 &
done
```

结果另存为 `eval_test_best_val_iou_seed37_memory_off.json`，不覆盖正常测试。
它回答训练好的模型是否依赖 M；不能代替重新训练的 V1 基线，也不能作为删掉模块后的速度/能耗测试，
因为仍计算 M。若用 `--dump-dir` 导出逐事件预测，正常与 off 两次应指定不同目录。

重点比较：

1. 正常 carry 与 memory_off 的 IoU、召回、虚警，以及早/中/晚时间段指标。
2. 三组的 kappa_abs_mean、memory_tau_mean：是否学会使用 M、是否都顶到范围边界。
3. memory_abs_mean/max/near_bound_frac：用于数值诊断，不能单凭均值证明背景污染被清除。
4. GPU 峰值显存与逐窗耗时。新增状态大小不包含训练计算图；三层组训练显存会更大。

位置对照也同时改变 M 容量，单独这三组不能证明全部增益都来自感受野。
若机制表现有效，再做不同速度/弱信号/目标离开后的残留分析以及多种子重复；本轮先检验可用性。

## 代码与验证范围

- `model/lif2d_stream.py`：双仓神经元、状态树 detach。
- `model/evspsegnet_stream.py`：编码层选择、两种执行路径、状态/额外运算统计。
- `train_stream_v1.py`：配置接入、M/kappa/tau 监控、checkpoint、memory_off 评估。
- `tests/test_stream_dual_memory.py`：三种部署 × 两种解码器、非零读取的前向/梯度等价、
  跨段训练更新、reset、零耦合回归 V1、有界性、未来损失回传与权重恢复。

本地验证使用 CPU 合成序列。实际 EV-UAV 训练效果、4090 显存和吞吐需在服务器验证。
模型运算统计单列额外 M 状态、乘加、除法和 tanh；卷积 MAC/SOP 能耗估算未包含这些新增运算及访存，
不能把原能耗估计直接当作双仓模型完整能耗。
