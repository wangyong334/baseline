# 双卡 FP32 实验版（不是 DDP）

## 范围与限制

新增文件，不修改原始 `train.py`、`test.py`、网络、损失或数据集实现。
保持输入事件上限 700000、batch size 1、FP32 张量、原模型结构及参数名称。
不使用 AMP、GradScaler、重计算或事件分块。浮点内核的 TF32 设置沿用原环境，
不能把“FP32 张量”当成所有底层算子严格禁止 TF32，也不能据此保证论文逐位复现。

默认 split=2：逻辑 GPU 0 放 conv_input、stage 1/2 编码与解码、输出头、损失；
逻辑 GPU 1 放 stage 3/4 编码与解码。split=1 或 3 可改变划分。
只传递边界特征和坐标，不跨卡搬运 spconv indice_dict。
每对下采样和逆卷积留在同一卡上，返回浅层时使用原始 lateral 的本地缓存；
返回边界检查坐标顺序一致。特征 `.to(device)` 不 detach，保留跨卡梯度链。
依据 spconv 2.3.6 的逆卷积缓存契约：
https://github.com/traveller59/spconv/blob/v2.3.6/spconv/pytorch/conv.py

双卡不会自动合成统一的 48GB 显存，不保证最大样本一定放得下，也不保证加速。
本地只做语法与模拟路由测试；真实稀疏算子、跨卡反向、数值一致性、显存均需服务器验证。

## 上传

将本项目上传为单独目录（例如服务器 `EV-UAV-MP`），不要覆盖旧实验目录。
若仅上传增量，需要 `model/evspsegnet_mp.py`、`train_mp.py`、
`configs/evisseg_evuav_mp.yaml`、本说明及可选 `tests/test_mp_routing.py`。
沿用服务器已经能工作的 evuav 环境、spconv 和 HAIS_OP，不必重装。
在项目根目录执行下列命令；YAML 的 DATA.root 已填写此前服务器数据路径，先核对。
`CUDA_VISIBLE_DEVICES=0,1` 选择物理卡 0/1；代码内部称为 cuda:0/1。
不要对模型并行实例再整体调用 `.cuda()` 或 `.to()`。

## 第一步：小样本单卡/双卡一致性检查

```bash
conda activate evuav
mkdir -p run
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
EVUAV_MODE=compare EVUAV_MP_SPLIT=2 \
python -u train_mp.py --config configs/evisseg_evuav_mp.yaml \
2>&1 | tee run/mp_compare_split2.log
```

默认 train_000.npz，只做一次体素化并固定输入。分别执行两次单卡、两次双卡，
检查初始参数、预测、loss、全部参数梯度和前向后的 BN 状态；报告最大绝对误差、
相对 L2 误差、余弦相似度，以及阈值 0.5/0.9 下的预测分歧数。
看到 `FP32 COMPARISON DIAGNOSTIC COMPLETED` 仅表示诊断运行完成，不等于数值已经通过。
先保存完整输出并判断双卡自身重复性和单卡/双卡差异，再决定是否继续 smoke；
不要直接放宽容差。
compare/smoke 不保存权重，也不修改数据。

## 第二步：最大样本连续两步更新

```bash
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
EVUAV_MODE=smoke EVUAV_MP_SPLIT=2 \
python -u train_mp.py --config configs/evisseg_evuav_mp.yaml \
2>&1 | tee run/mp_smoke_split2.log
```

自动扫描训练集，选择原始事件数量最多的样本。当前数据预期 train_096.npz、625178事件。
仍遵守 YAML 的 max_events_num，运行时打印实际事件数，确认没有人为降低上限。
连续两次前向、反向、Adam 更新，检查两个 GPU 都有有限梯度且参数实际改变。
输出每卡 allocated/reserved 峰值；第二步覆盖 Adam 状态已存在的情况。
看到 `DUAL FP32 TWO-STEP SMOKE TEST: OK` 才考虑训练。
每次失败后重新启动进程，不在 OOM 后继续使用同一进程。
可在另一个终端运行 `watch -n 1 nvidia-smi`，它与 PyTorch 显存统计口径不同。

若某张卡 OOM，保存完整报错及各卡显存。split=1 把 stage2 移到第二张卡，
split=3 把 stage3 移到第一张卡；没有哪个划分保证最佳。切换划分后需重新 compare/smoke。
切勿为让检查通过而跳过出错样本。

## 第三步：正式训练（仅在前两步通过后）

```bash
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
EVUAV_MODE=train EVUAV_MP_SPLIT=2 \
python -u train_mp.py --config configs/evisseg_evuav_mp.yaml \
2>&1 | tee run/mp_train_split2.log
```

默认保存到 `log/model_mp_fp32_split2_seed37`，该目录必须尚不存在，避免覆盖实验。
再次实验或不同 split 请先复制配置并给 model_save_root 设置新目录，同时修改 model_path。
命令前台运行；如需断开 SSH，使用已有 tmux 会话。不要同时启动多个相同输出目录的训练。
记录 `run_config.json`、每轮 `metrics.jsonl`，保存 CPU state_dict：
`best_iou_seed37.pt`、`best_loss_seed37.pt`、`last_seed37.pt`。
保存文件是模型权重，不含完整优化器/RNG 恢复状态；当前入口不支持精确断点续训。
保持原 state_dict 参数名称，可使用原单卡 test.py 读取，不需要把测试也改成双卡。

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u test.py --config configs/evisseg_evuav_mp.yaml
```

测试集必须只用于最后评估，不据此选 split、epoch 或阈值。

## 与作者训练入口的明确差异

- 原入口保持不变；新入口始终 FP32、单样本模型并行。
- seed=37、Adam lr=0.001、StepLR(10, 0.1)、50轮、原 STCLoss 和阈值不变。
- 每轮 train()，第40轮起（0基编号）在 eval()+no_grad() 下验证，随后恢复训练模式。
  作者入口验证没有切换 eval()；这里主动修正，避免验证数据更新 BN。这是协议差异，必须披露。
- best_loss 改为最低“每轮平均 batch loss”，不再按偶然最低的单 batch loss。
  best_iou 按验证集 IoU 选择；每轮输出 loss、验证 IoU、每卡峰值显存。
- NPZ 文件排序，DataLoader workers=0（collate 内含 CUDA，不使用配置中的8个worker）。
- 每步前清梯度，释放无用 Python 引用；每轮结束清两卡未使用缓存。
  清缓存只是辅助，不是本方案节省活跃张量显存的机制。
- 每步检查预测、loss、梯度、参数、BN 状态；非有限数立即失败，不替换或跳过。
- 不写 MLflow，使用独立 JSONL 和控制台日志。不要与旧 AMP/校准实验混用输出目录。

## 本地检查

```bash
python -m unittest discover -s tests -p test_mp_routing.py -v
```

此测试使用模拟对象检查三个 split 的设备路由、逆卷积本地缓存、返回坐标及缓存隔离。
不能代替服务器的真实 CUDA 数值/梯度测试。
