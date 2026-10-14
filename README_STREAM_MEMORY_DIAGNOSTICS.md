# 第三轮：解码末层对照与记忆诊断

两组从头训练：`enc234_dec321` = enc2/3/4 + dec3/2/1；
`enc234_dec32` = enc2/3/4 + dec3/2。唯一结构差异是 dec1 的双仓。
其余配置沿用 seed37、50 epochs。dec1 的 U_pre 直接进入逐事件读出，
它的记忆可以不经过下一层脉冲阈值就影响预测，因此值得独立验证，但不预设它更优。

同步两个新配置、tools/diagnose_stream_memory.py、tools/run_memory_diagnostics.sh，
并保持上一轮模型和训练代码同步。数据路径不变。

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v1
conda activate evuav
mkdir -p run log
for pair in '0 enc234_dec321' '1 enc234_dec32'; do
  read -r gpu variant <<< "$pair"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python -u train_stream_v1.py --config "configs/evisseg_stream_dual_${variant}.yaml" \
    --mode train --device cuda:0 > "run/dual_${variant}_train.log" 2>&1 &
done
# 另外两张卡：现有权重补 M-off、配对零读取、延迟读取、空间均值读取。
nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash tools/run_memory_diagnostics.sh enc234 > run/diagnostics_enc234.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash tools/run_memory_diagnostics.sh enc4 enc34 > run/diagnostics_enc4_enc34.log 2>&1 &
```

训练完成后测试新两组：

```bash
for pair in '0 enc234_dec321' '1 enc234_dec32'; do
  read -r gpu variant <<< "$pair"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    bash -c 'set -e; v="$1"; for a in none off; do
      python -u train_stream_v1.py --config "configs/evisseg_stream_dual_${v}.yaml" \
        --mode eval --device cuda:0 --split test --memory-ablation "$a" \
        --checkpoint "log/stream_dual_${v}_seed37/best_val_iou_seed37.pt"
    done' _ "$variant" > "run/dual_${variant}_test.log" 2>&1 &
done
```

## 诊断口径

- 正式 M-off 使用原评估入口，同时输出 carry/reset、IoU、Pd/Fa。
- 独立诊断逐窗运行两份相同权重：真实记忆与干预记忆，各自演化 U/M；每条新序列清空全部状态和延迟队列。
- zero：读取置零；写入公式不变。应与正式 M-off 的 carry 混淆计数一致。
- delay4：读取原本读取时刻之前 4 个窗口（200ms）的 M；前4窗没有可用历史时读取零。不循环、不读取未来。
- spatial_mean：每通道对当前旧 M 作空间均值后广播，保留缓慢通道偏置、移除空间结构。
- 干预同时作用于 checkpoint 配置的全部双仓层；上游变化可改变下游写入，不声称整网写入轨迹不变。
- 输出目标/背景事件上的概率绝对变化、预测翻转比例，各层全图脉冲翻转比例。
- 在真实分支，按事件坐标所属特征网格采样各通道，统计 |deltaI|/theta、|U_pre-theta|/theta、固定当前 U/输入时直接移除 M 的放电翻转，以及两条完整轨迹的放电翻转。
- 这些统计按事件和通道加权；重复事件重复计数。某背景事件落点的感受野可能包含目标，不能把这些统计当成纯背景特征真值。
- 空窗仍更新；无事件时只做全图脉冲统计。诊断输出 per-sequence TP/FP/FN，支持后续配对分析；不报告诊断过程耗时作为模型性能。
- 延迟干预同时改变历史年龄与内容，空间均值也改变幅值分布；任何一次下降都不能单独证明精确运动对应。
- 本地只有合成数据测试，真实诊断由服务器已有 checkpoint 和数据运行。默认拒绝覆盖；shell脚本跳过已有结果。
