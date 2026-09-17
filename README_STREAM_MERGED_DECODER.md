# V1 电流合并解码器（merged_decoder）

## 为什么改

原 V1 每个解码阶段是：

```
I = Conv3x3( cat[ ConvT2x2(S_deep), S_skip ] )  -> 增益 -> LIF
```

`ConvT2x2` 的输出是实数，拼接后解码卷积有一半输入是实数，只能按 MAC（实数乘加）计。
按 V1 seed37 实测发放率估算，理论能耗 2.241 mJ，其中 99% 来自 MAC，只比同结构 ANN 省 2.2 倍。

之前试过的「ConvT 后加 LIF」（10 层版本）会增加 195 万个膜电位，冒烟测试时新层沉默，
深层梯度要多穿过三道阈值，并且必须重训。该版本代码已删除。

## 做法

ConvT 与解码卷积之间**没有任何非线性**（无 LIF、无 BN、无 bias），两次线性运算可以精确合成一次：

```
I = ConvT4x4_s2_p1(S_deep) + Conv3x3(S_skip)  -> 增益 -> LIF
```

- 两路突触运算的输入都是 0/1 脉冲，电流在神经元内相加，这正是积分本身
- 仍然是 7 个 LIF、397 万个膜电位，读出头不变
- 与已训练的原版 **数学上严格等价**：低分辨率位置 i 经 ConvT2x2/s2 覆盖高分辨率 2i..2i+1，
  再经 Conv3x3/p1 覆盖 2i-1..2i+2；ConvT4x4/s2/p1 把 i 写到 2i-1..2i+2，覆盖范围完全相同。
  合并核就是该组合对单位冲激的响应（`model/evspsegnet_stream.py: merge_decoder_state_dict`）

与「脉冲相加」不同：SEW-ResNet 的 ADD 把两个脉冲加成 2 再当脉冲往下传；这里相加的是**电流**。

## 两种用法

| | 换算已训练的 V1（推荐先做） | 从零训练 |
|---|---|---|
| 精度 | 与原版相同（数学等价） | 需要实验，参数化方式不同 |
| 耗时 | 换算几分钟 CPU + 评估 | 每个种子约 7–9 小时 |
| 参数 | 由原 105,345 个参数算出，存储为 123,057 个数 | 123,057 个可学习参数 |

### 用法一：换算已训练的 V1 权重

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV
conda activate evuav

# 1. 换算（CPU，不占训练 GPU）。每个种子用验证集前 2 条序列、各 160 窗做两级核对，通过才保存
for s in 37 38 39; do
  python tools/convert_to_merged_decoder.py --checkpoint log/stream_v1_seed$s/best_val_iou_seed$s.pt
done

# 2. 评估换算后的权重（test 和 val），结果写到 log/stream_v1_seed*/eval_<split>_best_val_iou_seed*_merged.json
nohup bash -c '
for s in 37 38 39; do for split in test val; do
  CUDA_VISIBLE_DEVICES=3 python train_stream_v1.py --mode eval --split $split \
      --checkpoint log/stream_v1_seed$s/best_val_iou_seed${s}_merged.pt
done; done; echo ALL MERGED EVAL FINISHED' > run/merged_convert_eval.log 2>&1 &
```

换算工具的两级核对：

- **数学核对**（门槛）：原权重转 float64 后换算，合并核不经 float32 舍入，与原网络在 float64 下逐窗对比，
  必须零脉冲翻转、误差不超过 1e-9，否则不保存
- **部署核对**（只报告）：实际保存的 float32 合并核在 float32 下对比。舍入可能让极少数临界神经元翻转，
  **最终以评估 IoU/ACC/Pd/Fa 与原版是否一致为准**

### 用法二：从零训练

配置 `configs/evisseg_stream_merged_decoder.yaml` 与 `evisseg_stream_v1.yaml` 只差 `merged_decoder: true` 和保存目录，
沿用同一份训练集统计。

```bash
CFG=configs/evisseg_stream_merged_decoder.yaml
CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py --config $CFG --mode smoke \
    --save-root log/stream_merged_decoder_smoke > run/stream_merged_decoder_smoke.log 2>&1
for s in 37 38 39; do
  nohup env CUDA_VISIBLE_DEVICES=$((s-37)) python train_stream_v1.py --config $CFG --mode train \
      --seed $s --save-root log/stream_merged_decoder_seed$s > run/stream_merged_decoder_seed$s.log 2>&1 &
done
```

训练完成后评估：

```bash
python train_stream_v1.py --mode eval --split test --checkpoint log/stream_merged_decoder_seed37/best_val_iou_seed37.pt
```

## 改动清单

| 文件 | 改动 |
|---|---|
| `model/evspsegnet_stream.py` | `merged_decoder` 开关；`SpikingConvBlock` 支持外加电流；`merge_decoder_state_dict`；运算量统计；删除 10 层版本 |
| `train_stream_v1.py` | `build_net` 读取 `merged_decoder`；遇到已删除的 `spiking_decoder` 配置明确报错 |
| `tools/convert_to_merged_decoder.py` | 权重换算 + 两级核对 + 保存报告 |
| `configs/evisseg_stream_merged_decoder.yaml` | 从零训练用配置 |
| `tests/test_stream_merged_decoder.py` | 10 个测试 |

原 V1 配置没有 `merged_decoder` 键，默认 false，前向与提交版逐位一致（有测试覆盖）。

## 口径说明

- **参数量**：换算得到的模型由原 105,345 个参数决定，自由度不变，只是 4×4 核存储的数更多（123,057）；
  论文建议写「训练参数 105,345，部署形式 123,057」。从零训练的版本则是 123,057 个可学习参数。
- **能耗**：用 V1 seed37 实测发放率，原版 2.241 mJ → 电流合并 0.605 mJ。剩余 MAC 的 92% 来自 enc1 的实数计数输入，
  若只按非零输入记账可到约 0.054 mJ（需在论文中写明口径）。SOP 为理论估计，GPU 实现仍为稠密卷积。
- **换算后的 checkpoint 不含优化器状态**（参数形状已变），只用于评估，不能 `--resume`。
- **已删除的 10 层版本权重不能换算**：它的 ConvT 与解码卷积之间有 LIF，换算函数会显式拒绝。

## 验证

```bash
python -m unittest discover -s tests -p "test_stream*.py" 2>&1 | tail -3
```

覆盖：float64 严格等价（LIF / groupnorm_noshift / ReLU × carry / reset）、真实通道数下 float32 电流一致、
所有突触输入为 0/1、读出不变、参数与状态数、运算量统计、非法换算拒绝、原 V1 前向逐位不变、校准、从零训练、换算工具。
