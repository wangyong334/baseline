# V16 r4：上传、单个 seed37 任务和结果打包

本轮在现有 V16 分支、服务器 `EV-UAV-v16` 目录运行，只排一个 `full_s37`。
沿用正在运行的 V13 GPU1 池子，不启动第二个调度器，也不停止现有任务。

这轮是**冻结匹配的 r2 骨干后训练小修正头**。骨干已有的 BIR、跨窗状态和通信约束保持；本阶段不重新训练 U-net，不等同于从头训练整个模型。窗口仍为 100/50 ms。200/100 和 400/200 的探针属于独立诊断，不能直接替换这个已匹配的初始化。

## 1. SFTP 上传

本地外层 `.vscode/sftp.json` 已核对：默认 `lab-v16`，context 为 `EV-UAV`，服务器目录为 `/media/stephen/nvme0n1/wy_data/EV-UAV-v16`。

上传以下文件/目录的最新版本：

- `model/evspsegnet_stream.py`
- `model/covariance_revision.py`
- `tools/v16/`（包含新的 `train_covariance.py`）
- `configs/v16/r4/`（三份配置都上传，预检会比较消融协议）

不需要上传本地 `Res` 或缓存。运行前预检会检查依赖、数据、初始化及配置；入池时复制源代码和初始化到独立 release，后续 SFTP 更新不影响已排入的任务。

## 2. 上传后排入现有 GPU1 池子

直接复制整段到服务器 Bash。初始化来自已经完成的 r2 GPU2 seed37，先建立可核对的软链接；遇到已有不同目标时停止，不覆盖。

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
(
set -e
python - <<'PY'
from pathlib import Path
source = Path('log/v16_context/context_20261113_r2_gpu2/full_s37/best_val_iou_seed37.pt').resolve(strict=True)
if not source.is_file() or source.stat().st_size == 0:
    raise SystemExit('初始化权重为空：' + str(source))
link = Path('run/v16_initialization/r4/seed37.pt')
link.parent.mkdir(parents=True, exist_ok=True)
if link.exists() or link.is_symlink():
    if link.resolve(strict=True) != source:
        raise SystemExit('已有初始化路径指向其他权重，请检查：' + str(link))
else:
    link.symlink_to(source)
print('INITIALIZATION:', link, '->', source)
PY
python -B tools/v16/queue_jobs.py enqueue \
  --release context_20261113_r4 \
  --config-dir configs/v16/r4 \
  --init-root run/v16_initialization/r4 \
  --variants full --seeds 37 --dry-run
python -B tools/v16/queue_jobs.py enqueue \
  --release context_20261113_r4 \
  --config-dir configs/v16/r4 \
  --init-root run/v16_initialization/r4 \
  --variants full --seeds 37
)
```

预期出现 `DRY_RUN_OK: 1 training job(s); GPU 1`，随后一条 `QUEUED`。
同名 release 已存在时命令会拒绝重复入池。不要反复改名字排入相同任务。

每个任务依次自动执行：运行环境与反传预检 → 训练集统计与缓存 → 小头训练和验证集选模 → 在线/缓存一致性检查 → 正式验证集评估 → 测试集评估 → 推理时间、显存、板卡功率积分记录。完成标记为 `V16_CONTEXT_FINISHED`；发生任何错误则停止该任务并保留日志。

## 3. 查看 GPU1、当前 release 和池子

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
date &&
nvidia-smi -i 1 &&
python -B tools/v16/queue_jobs.py status \
  --release context_20261113_r4 --tail 25
```

所有池子任务：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13 &&
ls -lh run/queue/pool/pending/ run/queue/gpu1/pending/ run/queue/gpu1/running/ &&
tail -n 8 run/queue/pool.log run/queue/gpu1.log
```

## 4. 完成后打包

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
python -B tools/v16/queue_jobs.py pack --release context_20261113_r4
```

将打印的 `DOWNLOAD:` 文件通过 SFTP 下载到本地 `Res`。打包包含结果、权重、训练历史、配置、统计、训练板卡能量记录、推理性能、预检及源码快照；不包含大的中间特征缓存。未完成或结果哈希不一致会拒绝按完整实验打包。

功率积分是整张 NVIDIA 板卡的测量，不是 CPU/整机功耗，也不是神经形态芯片能耗；共享 GPU 时不可把它归因于单个模型。训练时间需包含统计提取和缓存生成，不能只报告小头优化的几秒钟。
