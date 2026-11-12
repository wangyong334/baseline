# V15 r3：旧版 CUDA 确定性反向修复

2026-11-12，服务器 r2 完成数据准备和增益校准，在第一次 `loss.backward()` 退出：PyTorch 1.9.1 CUDA 的 `upsample_bilinear2d_backward_out_cuda` 不支持已启用的严格确定性模式。任务已被池子派发，故不是排队未启动，也不是显存不足。

此前检查覆盖了 PyTorch 1.9 CPU 和 PyTorch 2.11 CUDA，漏掉了旧版 CUDA 的限制。本次保留 `deterministic: true`、原前向输出、网络参数、损失、V5 carry/BIR/通信预算与四流 K=16 训练协议。

## 修复

- `model/observation_revision.py` 的插值前向仍调用原生双线性插值，`align_corners=False`。整数倍率上采样的反向使用半像素权重的转置：每个输入位置读取有限个输出梯度，再按固定方式加权归约。避免旧 CUDA 的原子 scatter，不构建稠密插值矩阵。数学梯度相同，浮点累加次序不同；不承诺与原反向逐位一致。仅支持本训练所需的一阶导数。
- `tools/v15/runtime_check.py` 在实际运行设备用临时修正模块执行两次 Adam 更新，覆盖非零修正投影、部分/完整 padding、关联和门控梯度。恢复 RNG、确定性和 TF32/cuDNN 标志，不修改正式模型。训练入口在校准前调用，报告 `V15_RUNTIME_BACKWARD_OK` 并保存 `runtime_check.json`。
- 新发布名为 `feature_20261112_r3`，设计标识更新为 r3。冻结清单加入自检文件，共 23 个源文件/配置。r2 失败任务和冻结源码保留，不能通过更新项目根目录修复旧快照；使用新 release 重新入队。

## 验证

- Python 3.8 / PyTorch 1.9.1 CPU：153 项测试执行成功，其中两项 CUDA 专用测试因该环境无 CUDA 跳过；日志 `Res/v15_r3_tests.log`。
- 本地 RTX 5060 Laptop / PyTorch 2.11 CUDA：插值与运行时检查 13 项通过，包含严格确定性反传、重复梯度一致、RNG/标志恢复和两次实际优化更新。
- 双精度 CPU：25 组形状/倍率组合（含单像素轴、倍率 1/2/3/4、不同比例），前向与原生逐值一致，梯度差在 2e-13 内；通过有限差分 gradcheck。完整修正模块的前向逐值相同，参数梯度差在 2e-12 内。
- 真实数据训练探针：四流、K=16、严格确定性，三次更新完成，70.1 秒，梯度/参数有限。完整 `train_000.npz` 回放 39,169 个事件，最终读出数量 39,169。此探针用于兼容性检查，不是精度结果，也不能外推服务器训练时长。结果 `Res/v15_r3_probe/probe.json`。
- 独立复制的 23 文件冻结快照，在 PyTorch 1.9 CPU 完成 CLI 导入、反传自检、两轮合成数据训练、验证选模、最佳权重 val/test 评估。短探针带部分范围标记，不作为正式实验。结果 `Res/v15_r3_release_check/result.json`。

服务器的 PyTorch 1.9 CUDA 无法在本机等同复现；重新入池后的实际设备自检是最后的运行环境验证。本次修复不改变方法设计或新增模型模块。

上传与重新入队命令见 [运行说明](V15_RUN.md)。已有 r2 上传仅需更新模型修正文件和 `tools/v15/{train,run_experiment,runtime_check}.py` 四个文件。
