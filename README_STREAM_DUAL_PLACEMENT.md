# 第二轮：编码/解码单层位置消融（2026-10-14）

四组均从头训练，只改变新增 M 所在位置；保留 V1 的软复位、全部 U 历史、卷积、跳连和读出。
同 seed 37、50 epochs、TBPTT 16、50 ms × 160 窗；双仓更新公式和第一轮完全相同。
第一轮 enc4 / enc34 / enc234 配置、权重和结果目录继续保留。

| GPU | 双仓位置 | 配置文件 | 输出目录 | 新增参数 | 新增 M 元素 |
|---|---|---|---|---:|---:|
| 0 | enc2 | configs/evisseg_stream_dual_enc2.yaml | log/stream_dual_enc2_seed37 | 48 | 557,568 |
| 1 | enc3 | configs/evisseg_stream_dual_enc3.yaml | log/stream_dual_enc3_seed37 | 96 | 278,784 |
| 2 | dec3 | configs/evisseg_stream_dual_dec3.yaml | log/stream_dual_dec3_seed37 | 96 | 278,784 |
| 3 | dec2 | configs/evisseg_stream_dual_dec2.yaml | log/stream_dual_dec2_seed37 | 48 | 557,568 |

enc3/dec3 同为 48×66×88，enc2/dec2 同为 24×132×176。比较编码阶段和跳连融合之后保存历史，
控制每对新增状态量与参数量；并不声称两层输入分布或感受野完全相同。
解码双仓位于原解码卷积、归一化、增益之后；不向上采样模块额外插入神经元。

## 上传文件

至少同步 `model/evspsegnet_stream.py`、`model/lif2d_stream.py`、`train_stream_v1.py`、四份新 YAML；
建议同时同步 `tests/test_stream_dual_memory.py`、本文和已修改的工具文件。
原数据与 `configs/stream_v1_train_stats.json` 继续复用，不重新统计、不重新校准已训练 checkpoint。

## 四卡训练

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v1
conda activate evuav
mkdir -p run log

for pair in '0 enc2' '1 enc3' '2 dec3' '3 dec2'; do
  read -r gpu variant <<< "$pair"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python -u train_stream_v1.py \
    --config "configs/evisseg_stream_dual_${variant}.yaml" \
    --mode train --device cuda:0 \
    > "run/dual_${variant}_train.log" 2>&1 &
done
```

独立输出目录不会覆盖第一轮结果。中断续训在匹配命令上加 `--resume`，日志使用 `>>`。

```bash
tail -f run/dual_enc2_train.log run/dual_enc3_train.log run/dual_dec3_train.log run/dual_dec2_train.log
```

每轮自动评估验证集，以验证 IoU 选择最佳权重。不要根据测试集挑 epoch。

## 训练完成：每卡依次执行正常测试和 M-off

下面每张卡只有一个进程任务，每个任务先正常评估、再关闭新增 M 读取；失败则停止该卡后续步骤。
无需再训练。默认沿用阈值 0.9，每次 eval 均报告 carry 和 reset_each_window。

```bash
for pair in '0 enc2' '1 enc3' '2 dec3' '3 dec2'; do
  read -r gpu variant <<< "$pair"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    bash -c '
      set -e
      variant="$1"
      for ablation in none off; do
        python -u train_stream_v1.py \
          --config "configs/evisseg_stream_dual_${variant}.yaml" \
          --mode eval --device cuda:0 --split test \
          --checkpoint "log/stream_dual_${variant}_seed37/best_val_iou_seed37.pt" \
          --memory-ablation "$ablation"
      done
    ' _ "$variant" > "run/dual_${variant}_test.log" 2>&1 &
done
```

每组输出两份 JSON：

- `eval_test_best_val_iou_seed37.json`
- `eval_test_best_val_iou_seed37_memory_off.json`

若需要完整验证集评估，将 `--split test` 改为 `--split val` 并另用日志文件名即可。
正式测试已有结果时默认拒绝覆盖；确定要重复测量才使用 `--overwrite`。

## 第一轮权重的逐层读取消融

新增 `--memory-off-layers`，只改变指定层的读取，不改变写入或普通 U 状态。
只能填写该 checkpoint 实际部署了 M 的层；不允许与 `--memory-ablation off` 同时使用。

例：在已训练三层组中，只关闭 enc4 的 M：

```bash
CUDA_VISIBLE_DEVICES=0 python train_stream_v1.py \
  --config configs/evisseg_stream_dual_enc234.yaml --mode eval --device cuda:0 --split test \
  --checkpoint log/stream_dual_enc234_seed37/best_val_iou_seed37.pt \
  --memory-off-layers enc4
```

结果另存 `eval_test_best_val_iou_seed37_memory_off_enc4.json`。
可用 `--memory-off-layers enc3 enc4` 同时关闭两层，结果后缀按网络顺序排列。
评估 JSON 的 `memory_off_layers` 与每层 `memory_read_enabled` 记录实际开关。
读取消融只解释已训练模型的依赖，不能替代对应位置的重新训练；重置全网 U/M 也不是 M 单独贡献。

## 实现检查

本轮覆盖全部 7 个合法层名，实验只启用表中的四种。逐窗和逐层执行、监控、运算统计都支持解码 M。
测试包含单层与编码/解码混合部署、两种解码器、非零读取下梯度/状态等价、初始化退化 V1、
保存恢复以及实际合成数据的训练/续训/正常评估/全关与逐层关闭读取。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_stream_*.py'
```

本地 CPU 检查不代替服务器真实数据的性能和 GPU 显存验证。
