# V15 依赖驱动代码清理

2026-11-12，根据用户“有 Git 历史，不继续保留无用代码”的明确要求，对当前工作树执行清理。此次清理只删除项目代码、旧配置及对应历史测试，不修改历史研究文档、`CLAUDE.md`、`References/`、`Res/`、数据集、实验权重或服务器进程。

清理前以当前 V15 训练、评估、profile、队列、打包入口，以及仍复用的 `train_stream_v1.py` 和保留基础测试为起点，遍历 Python AST 中的本地导入，得到 29 个文件的依赖闭包。删除候选与这个闭包没有交集。对每个删除路径验证：属于 Git 跟踪文件、没有本地未提交修改、不是符号链接、解析后的绝对路径位于项目根目录内。

本轮最终逐文件删除 331 个 Git 跟踪文件，以及与历史 Python 源文件对应的 250 个生成缓存文件；没有执行递归目录删除。6 份 V5 数值回归测试从 Git 恢复并精简，未计入最终删除数。

| 类别 | 删除数 | 说明 |
|---|---:|---|
| 旧 YAML 配置 | 122 | 旧双仓、对象槽位、轨迹、changepoint、RCL/DECO、V14 等实验配置 |
| 旧 tools 入口 | 109 | 旧后处理、V6–V14 实验训练/评估/排队脚本、被 V15 工作流替代的独立工具 |
| 历史测试 | 34 | 对应废弃模块和旧训练接口的测试 |
| 独立 legacy model | 11 | 原始稀疏基线、MP、对象层、对象token、轨迹头、lookahead、clutter、changepoint、comod、noise_invariant |
| 历史 dataset | 3 | `basedataset.py`、`ev_uav.py`、`noise_aug.py`；当前使用原生流式读取器 |
| 旧 utils | 5 | 不在运行闭包中的聚合、分组、旧损失、旧数值工具、可视化 |
| 根目录旧入口 | 5 | 原始 train/test、MP、end-to-end 轨迹训练入口 |
| 旧稀疏基线扩展 | 42 | 根目录 `lib/hais_ops` 的 C++/CUDA 源码、接口和已跟踪编译产物；References 中的原始基线副本保留 |

完整可审计清单见 [V15_CLEANUP_FILES.json](./V15_CLEANUP_FILES.json)。每个被删源文件记录原路径、字节数和 SHA256，同时记录清理时的 Git HEAD、生成缓存清单和保留的运行依赖闭包。需要历史实现时从 Git 查阅，不在当前发布代码里继续携带副本。

保留内容包括：

- 全部 V15 模型、配置、训练/评估/profile/队列/打包入口和 V15 测试。
- V5 BIR 骨干、神经元、数据表示、通信预算，以及 `comm_peak`、`notransport`、`float_notransport`、`stream_v1` 四份基础配置。
- `tools/stream_train_stats.py`。
- 现有 `tools/final/gpu_pool.sh`、`gpu_queue.sh`、`gpu_hold.py`。清理不修改或运行它们。
- `test_gpu_hold.py`、`test_stream_windows.py`、`test_stream_common_metrics.py` 基础测试。
- 从历史版本保留并精简的 6 份 V5 数值回归：`test_stream_comm.py`、`test_stream_layerwise.py`、`test_stream_multistream.py`、`test_stream_tif.py`、`test_stream_torch.py`、`test_stream_variable_length.py`。覆盖 BIR 前向/梯度等价、批量流与重置边界、通信计数/峰值预算梯度、同样 detach 边界下的 TBPTT 梯度、增益校准和原始 Pd/Fa 口径；旧 trainer CLI、dual/merged 等废弃用例已移除。
- 所有历史统计 JSON/JSONL、结果清单和研究记录。

`model/comod.py`、`model/noise_invariant.py`、`dataset/noise_aug.py` 第一轮仍被旧共享 helper 或数据源导入，故当时保留。随后完成 V5 helper 精简、数据源和骨干中的废弃导入移除，再次确认 V15 依赖后删除，未依靠“默认关闭”来跳过依赖检查。`train_stream_v1.py` 路径保留，但内容现为 V15 使用的 V5 构造、校准和流批次辅助函数，不再包含旧训练 CLI。

最终验证：Python 3.8 / PyTorch 1.9 环境下，现有 140 项测试全部通过，其中包含 69 项 stream 数值/数据测试；三个 V15 正式配置校验通过。发布源文件集合从 41 个减少为 22 个，将其复制到独立临时目录后，训练与 profile 导入成功，queue/run/package/train/profile 五个入口均可直接运行 `--help`，不依赖被删除的旧代码。以上仅为本地代码验证，不表示已完成服务器正式训练，也不代表精度或速度结论。

最终发布冻结使用显式依赖清单，避免 SFTP 服务器残留旧 `.py` 文件被重新打包；r2 独立快照验证见 `Res/v15_r2_release_check.json`。
