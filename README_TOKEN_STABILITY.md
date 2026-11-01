# 两段式目标 token 层数值稳定性修复（2026-11-01）

当前主线仍是冻结 E3a，只重训 `configs/object_tokens_v1.yaml` 的目标 token 层。

## 故障与证据

服务器截图：`clip_grad_norm_` 先报告 non-finite norm，随后片段 228 的 loss 非有限。
本地 Python 3.8 / torch 1.9.1 CPU 可复现两条数值风险：

- 原注意力精度 `exp(-width_head(h))` 在输出为 -100 时成为 inf；距离项和梯度失去有限性。
- 有限的 float32 梯度（例如 1e30）也会在原 float32 L2 范数计算中溢出成 inf，裁剪后错误地变为零。

没有服务器故障现场的张量或 checkpoint，不能断言第一次异常一定发生在宽度头。

## 修复

1. 宽度改成正方差的 softplus 参数化，加一像素平方的分辨率下限，再取倒数得到精度。当前步长 8 下初始方差仍为 1（网络单位），注意力和统计量共用该定义。
2. 参考点限制在画布内，预测位移用 tanh 限制在画布尺度内；零位移和小位移附近保留原来的初始化与一阶尺度。
3. 两种训练入口都先以 float64 计算梯度范数，再裁剪。真正含 NaN/Inf 的梯度会在 optimizer.step 前报错，打印片段号与参数名；不跳过异常批次、不污染 Adam 状态。
4. 运行脚本支持 `OUT`、`NAME`，让修复版使用新目录和日志名。

宽度参数的含义已经改变：使用原 E3a 骨干，但目标 token 层从头训练，不加载修复前的 token 权重。冻结骨干、损失权重、Wald 阈值、训练轮数和评测协议保持当前配置。

## 上传清单

上传到服务器项目根目录，保持相对路径：

```text
model/object_tokens.py
utils/train_numerics.py
train_track_head.py
train_end2end.py
tools/run_track_head.sh
tests/test_object_tokens.py
tests/test_train_numerics.py
```

## 本地验证

Python 3.8.20 / torch 1.9.1+cpu：目标 token 层 21 项、梯度稳定性 6 项、端到端入口 6 项、旧航迹头 50 项，共 83 项通过。
包含极端宽度（-1000 到 1000）、大位移、有限大梯度裁剪、NaN 更新拦截，以及冻结训练入口连续 240 次更新（合成缓存，跨片段保留状态）。
这不是服务器真实数据第 228 段的复跑，也没有验证 GPU 精度或完整训练收敛。

## 服务器：卡 2 预检和重跑

预检：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v7-track
conda activate evuav
for test in test_object_tokens.py test_train_numerics.py test_end2end.py test_track_head.py; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p "$test" || break
done
nvidia-smi -L
```

四组测试全部 OK 后，前台冒烟（新目录，不覆盖旧结果）：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v7-track
conda activate evuav
stamp=$(date +%Y%m%d_%H%M%S)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_track_head.py --config configs/object_tokens_v1.yaml \
  --mode train --device cuda:0 --epochs 2 --max-chunks 3 \
  --save-root "log/smoke_tokens_stable_${stamp}"
```

看到 `TRAINING FINISHED` 后，正式训练，结束后脚本自动评估 val 和 test：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v7-track
conda activate evuav
mkdir -p run
name=object_tokens_stable_$(date +%Y%m%d_%H%M%S)
CFG=configs/object_tokens_v1.yaml GPU=2 NAME="$name" OUT="log/${name}_seed37" \
  nohup bash tools/run_track_head.sh > "run/${name}_all.log" 2>&1 &
echo "训练日志：run/${name}_train.log"
```

物理卡 2 经 CUDA_VISIBLE_DEVICES 映射后，在进程内仍写 cuda:0。冒烟只验证入口，正式运行仍需观察梯度、首轮验证与等待统计。
