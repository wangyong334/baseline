# V14：BIR 跨窗膜电位独立重训对照

分支从 `V11@6c1ff2e` 创建。本轮只使用 V5 BIR 基础版；不运行 SDU、VoW、后验判决、环境自适应或通信预算损失。只跑 seed 37 的两次独立训练，不使用已有 checkpoint 续训。

## 对照协议

| 项目 | carry | reset |
|---|---|---|
| 配置 | `configs/v14/bir_carry.yaml` | `configs/v14/bir_reset.yaml` |
| 训练及测试状态 | `carry` | `reset_each_window` |
| 初始化 | seed 37，从头训练 | seed 37，从头训练 |
| 输入 | 50 ms 非重叠窗，5 bins，12 通道 | 相同 |
| 骨干 | V5 BIR / U-Net，7 级量化 | 相同 |
| 训练 | 50 epochs，Adam，原学习率计划 | 相同 |
| 并行 TBPTT | `trainer=streams`，B=4，K=16，accumulate=1 | 相同 |
| 执行 | `stream_layout=greedy`，`execution=layer`，`input_device=gpu` | 相同 |
| 通信损失及额外模块 | 关闭 | 关闭 |
| 选权重 | 各自训练模式下验证集最佳 IoU | 相同规则 |
| 最终评估 | 只评训练时使用的状态模式，阈值 0.9 | 相同规则 |

两组共享训练集统计，分别按同一校准规则校准增益。参数初始化使用同一个种子，校准后的增益允许因状态模式不同而变化。运行器会校验两份配置除 `state_mode` 和 `save_root` 外完全相同。

reset 组保留相同的并行分流和梯度批量，但每窗不读旧膜电位；这不等于把输入的窗内时间 bins 删除。carry 的 TBPTT 边界只 detach 梯度、保留状态数值。不能把同权重临时清零作为这次主对照。

同时比较 IoU、正事件召回（原代码的 ACC）、Pd 和 Fa；IoU 提高不代表目标检出率也提高。最终 JSON 还保留原评估入口的推理计时、延迟和操作量。操作量/理论能耗不是整机功耗实测，训练时间也不等于推理时间。

## 服务器目录和上传

- 新代码及结果：`/media/stephen/nvme0n1/wy_data/EV-UAV-v14`
- 已有调度器及任务日志：`/media/stephen/nvme0n1/wy_data/EV-UAV-v13`
- 数据集：`/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset`
- SFTP：外层 `.vscode/sftp.json` 的默认 profile 为 `lab-v14`，context 为 `EV-UAV`，顶层及该 profile 的 remotePath 都指向新目录。旧 `lab-v13` profile 保留。

先在服务器建立空目录：

```bash
mkdir -p /media/stephen/nvme0n1/wy_data/EV-UAV-v14
```

然后通过 SFTP **上传本地整个 EV-UAV 项目文件夹**到新目录，不能只上传新增的两个配置。SFTP 已排除权重、结果、虚拟环境、参考文献等。数据集使用现有绝对路径，不需要复制。训练统计由首个任务重新生成到本次 release，第二个任务校验来源后复用。

## 加入现有 1 卡池子

确认当前激活的是 `evuav` 环境，上传完成后执行：

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v14 &&
python tools/v14/queue.py enqueue --seeds 37 --release mem_ablation_20261112_r1 --dry-run &&
python tools/v14/queue.py enqueue --seeds 37 --release mem_ablation_20261112_r1
```

只加入 carry、reset 两个任务，绑定物理 GPU 1，串行运行。队列脚本投递到旧 V13 的池子，不另起 dispatcher/worker，不停止或释放已有占位进程。已有 worker 会在开始任务前自动释放固定 20 GiB 占位。

入队前检查数据文件数、依赖、对照配置和已有调度器。入队时保存源码快照及 SHA256，两个任务都从同一快照运行，输出留在新 V14 目录。后续 SFTP 上传不会更改该快照。相同 release 已入队或已有输出时拒绝覆盖；不要反复换 release 名称来重复提交同一实验。

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v14 &&
python tools/v14/queue.py status --release mem_ablation_20261112_r1 --tail 12
```

## 结果位置

```text
EV-UAV-v14/
  run/v14/mem_ablation_20261112_r1/
    release.json
    source/                         固定源码快照
    train_stats.json                两组共用的训练统计
    train_stats_provenance.json
    jobs/<carry|reset>_s37/          运行配置及来源记录
  log/v14/mem_ablation_20261112_r1/
    carry_s37/
    reset_s37/
```

每组包含原训练的 `metrics.jsonl`、`summary.json`、`calibration.json`、最佳权重和 `last.pt`，以及只针对训练模式的 `eval_val_trained_mode.json`、`eval_test_trained_mode.json`。只有训练和评估全部完成才写完成标记；失败记录保留，不自动覆盖重训。

本轮已按用户要求取消 V13 昨日失败任务的打包，不处理旧结果。
