# EV-UAV V15：流式 SNN 的延迟特征修正

当前活动代码保留 V5 BIR、跨窗膜电位、通信峰值约束与四流 K=16 的分层并行 TBPTT。在原生 50 ms 不重叠窗口上，用下一窗第一层膜前观测修正旧窗共享特征，再沿用 V5 MLP 给旧窗事件批量出分。修正不写回膜状态。

默认实验：`observation`，seed 37，从头训练 50 轮。`deep` 与 `causal` 为独立重训消融，按需排队。

- [方法、公式与局限](docs/V15_METHOD.md)
- [SFTP 上传、GPU 1 池子、评估和打包](docs/V15_RUN.md)
- [本地审计与实测](docs/V15_LOCAL_CHECKS.md)
- [旧代码清理记录](docs/V15_CLEANUP.md)

训练/评估入口是 `tools/v15/train.py`；自动流水线是 `tools/v15/run_experiment.py`。`train_stream_v1.py` 只保留 V5 数据、校准、分流辅助函数，已不是旧训练 CLI。无需稀疏卷积扩展。

当前上传版本为 `feature_20261112_r2`。本地通过不等于正式精度验证；请用完整验证集选模，再报告测试集结果。

原 EV-UAV 基线的说明及出处保存在 [上游 README](docs/EVUAV_BASELINE_README.md)，完整参考代码仍在本地 `References/同类竞品/EV-UAV` 和 Git 历史中。历史研究记录保留，部分旧文档提到的命令不适用于 V15。
