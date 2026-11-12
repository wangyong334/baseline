# V14 重叠观测实验：运行与本地核验

日期：2026-11-12。分支：V14。方法公式与架构见 `V14_OVERLAP_METHOD.md`。
本次是代码交付，尚无本版本完整 50 轮正式训练结果。CLAUDE.md 未改写。

## 固定实验协议

- 主模型：V5 BIR U-Net + 跨窗膜电位 + 原通信预算；100 ms 观测，50 ms 步长。
- 事件仍由原 V5 MLP 批量读出。相同原始事件 ID 配对，34 参数条件修正头只融合一次；首尾单覆盖事件保留。
- 骨干含原读出共 105,777 参数，新增条件头 34 参数；没有事件邻域检索和逐事件递归隐藏状态。
- 从头训练 seed 37，50 epochs；4 条录像流、K=16、分层并行 TBPTT；训练集统计与增益校准重新计算。
- FP32，关闭 CUDA matmul/cuDNN TF32。训练保持 layer；每轮验证选模、最终 val/test 和 profile 均为 step。
- 默认仅排一个 conditional 主模型任务。mean、concat 独立重训配置已备好，不自动占用额外训练时间。
- 固定等待第二次覆盖；本轮不启用高置信提前输出。内部事件多等一个 50 ms 步长。

## 上传与入队

服务器先创建独立目录：

```bash
mkdir -p /media/stephen/nvme0n1/wy_data/EV-UAV-v14-overlap
```

外层 `.vscode/sftp.json` 的默认配置已设为 `lab-v14-overlap`，context 为 `EV-UAV`，远端为上述目录。
通过 SFTP 上传当前项目；至少包括 `train_stream_v1.py`、`model/`、`dataset/`、`utils/`、`tools/` 和 `configs/`。
原配置的连接凭据及其他 profiles 保留；自动保存上传关闭。数据、权重、run/log/Res、References 与 .git 按 ignore 排除。

上传后在服务器执行：

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v14-overlap &&
python tools/v14/overlap_queue.py enqueue \
  --release overlap_20261112_r1 --seeds 37 --variants conditional --dry-run &&
python tools/v14/overlap_queue.py enqueue \
  --release overlap_20261112_r1 --seeds 37 --variants conditional
```

默认 scheduler root 是已有 `/media/stephen/nvme0n1/wy_data/EV-UAV-v13`，只交给其物理 GPU 1。
不启动第二个池子。预检要求既有 dispatcher 活跃且配置正确；如果预检失败，先根据报错处理，不重复入队。
命令冻结源码和配置、拒绝覆盖同 release。服务器数据集仍为 `../datasets/EV-UAV-dataset`。

查看状态：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v14-overlap &&
python tools/v14/overlap_queue.py status --release overlap_20261112_r1 --tail 15
```

正常结束后打包：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v14-overlap &&
python tools/v14/overlap_queue.py pack --release overlap_20261112_r1
```

打包命令打印 DOWNLOAD 路径及 SHA256；通过 SFTP 下载该文件到本地 Res。默认拒绝把未完成任务包装成完成结果；故障诊断可显式加 `--allow-partial`。

## 自动完成哪些步骤

1. 从头训练，每轮验证，按验证集 IoU 保存最佳权重，并保存 last.pt。
2. 用同一最佳 checkpoint 顺序运行完整 val 和 test；输出 IoU、历史 ACC（正事件召回）、Pd、Fa。
3. 同次骨干前向记录 first、second、mean、conditional、calibration_only 五种读出结果、修正与损坏计数、置信度分层、首尾/内部事件指标和条件系数曲线。这是同权重诊断，不冒充独立重训消融。
4. 按输入事件数选低、中、高三个测试录像，各重复两次，只测训练的主读出策略；记录表示准备、逐窗耗时、吞吐、等待延迟、显存及可用的 NVIDIA 板级能量。
5. 验证所有报告属于同一 checkpoint、源码没有变动，写 comparison_result.json 和 complete 状态。

结果目录：`log/v14_overlap/overlap_20261112_r1/conditional_s37/`。
核心文件：`metrics.jsonl`、`summary.json`、`best_val_iou_seed37.pt`、`last.pt`、`eval_val.json`、`eval_test.json`、`profile.json`、`comparison_result.json`。

## 本地核验与限制

- Python 3.8.20 / Torch 1.9.1 CPU：V14 测试 62 项通过；旧流式公共模块回归 230 项通过。
- 集成测试覆盖监督质量、原事件输出覆盖、跨 TBPTT 配对与截断、融合到骨干的梯度、流间状态隔离、时间因果性、模型保存/恢复、训练/评估和入队/打包。
- 实际 RTX 5060 Laptop GPU / Torch 2.11.0：完整空间尺寸、4 流、K16 反传成功，骨干、读出和融合头均有非零梯度；这只是少量 chunk 的流程探针。
- 最终 FP32 逐窗回放：test_021 的 5,638 事件耗时 1.01/0.94 s；test_002 的 25,732 事件耗时 1.07/1.07 s；test_023 的 620,440 事件耗时 1.28/1.26 s。每条原录像均为 8 s；包含读取、索引、表示准备和输出。相同录像重复两次概率逐位一致。
- 上述推理峰值为 0.264 GiB PyTorch allocated，训练探针峰值为 8.59 GiB；不是 nvidia-smi 整个进程显存，也不保证其他环境相同。8 GB Windows 笔记本可能使用共享显存，不能据本地短训练时间外推 4090 全程耗时。
- 这些探针权重只经历少量更新，不能用于精度结论。计时只验证高事件数下流程可执行；正式训练后的速度和能耗应以服务器 profile 为准。能量为整张 GPU 板级积分；存在其他进程时不能归因于单个模型。
- 当前回放预载整录像稀疏输入；网络依然按窗因果执行，但不是实时采集驱动测试。回放延迟不包括整录像预载，完整 wall time 包含。

本轮 GPU 复核发现：批量卷积和逐窗卷积的浮点累加次序在 BIR 量化阈值附近会导致输出差异，关闭 TF32 后探针仍见约 8.7e-4 的最大概率差异。正式评估已统一逐窗，不能声称 CUDA layer 与 step 严格逐位等价。

本地原始核验文件位于 `run/v14_local_check/`（按既有规则不上传）：`gpu_fp32/profile.json`、`gpu_fp32/metrics.jsonl`、`inference_comparison.json`。早期 `training_comparison.json` 是精度策略定稿前的短计时，仅作工程排查，不作为论文性能数据。

已按授权移除 tools/deco 下旧 V11–V13 路径及对应测试，共 45 个跟踪文件；共享骨干、原池子脚本和仍被导入的公共模块保留。没有创建 Git 提交，也没有远程启动任务。
