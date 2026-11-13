# V16 r5 三窗口实验：seed37 / 现有 GPU1 池

代码分支 V16。三个任务使用同一 r2 seed37 骨干初始化，分别运行 100/50、200/100、400/200 ms 的 observation/stride（重叠量等于 stride）。每项自动完成 CUDA 预检、骨干适配、冻结头训练、验证、测试及推理 profiling。只提交 full，不提交其他种子或消融。

本地数学审查和局部探针见 `docs/V16_R5_HOTELLING_AUDIT.md`。当前不能保证三组测试指标增长；尤其不能把新逐录像风险惩罚称为已满足的硬约束。不要改阈值，保持 0.9。

## 1. 服务器目录和 SFTP

```bash
mkdir -p /media/stephen/nvme0n1/wy_data/EV-UAV-v16
```

本地外层 `.vscode/sftp.json` 默认 `lab-v16`：context=`EV-UAV`，remotePath=`/media/stephen/nvme0n1/wy_data/EV-UAV-v16`。上传当前 `model/`、`tools/`、`configs/` 更新（包含所有新 r5 子目录）。不要上传 `run/`、`log/`、`Res/`、本地环境。服务器已运行的发布使用冻结 source；新发布会复制并校验本次源码。

## 2. 上传后依次预检、入池

必须依次运行三个 enqueue，不能并发提交。`--init-root` 是包含 `seed37.pt` 的目录。以下只复用现有 V13 scheduler，不新起池子，不终止显存占位进程。

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
(
set -e
python -B - <<'PY'
import hashlib
from pathlib import Path
root = Path.cwd()
source = root / 'log/v16_context/context_20261113_r2_gpu2/full_s37/best_val_iou_seed37.pt'
expected = 'f0f92d07f44e86a3d2a9e349f72f6f4ae4ae18e074ad4a2de22fbd7143799f8d'
if not source.is_file() or source.stat().st_size == 0:
    raise SystemExit('找不到 r2 seed37 初始权重：' + str(source))
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise SystemExit('初始权重 SHA256 不匹配，停止提交。')
link = root / 'run/v16_initializations/r5/seed37.pt'
link.parent.mkdir(parents=True, exist_ok=True)
if link.exists() or link.is_symlink():
    if link.resolve() != source.resolve():
        raise SystemExit('初始化链接已指向其他文件，停止：' + str(link))
else:
    link.symlink_to(source.resolve())
print('INITIALIZATION_VERIFIED', link)
PY
for w in 100 200 400; do
  python -B tools/v16/queue_jobs.py enqueue \
    --release "context_20261113_r5_w${w}" \
    --config-dir "configs/v16/r5_w${w}" \
    --init-root run/v16_initializations/r5 \
    --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 \
    --seeds 37 --variants full --dry-run
done
for w in 100 200 400; do
  python -B tools/v16/queue_jobs.py enqueue \
    --release "context_20261113_r5_w${w}" \
    --config-dir "configs/v16/r5_w${w}" \
    --init-root run/v16_initializations/r5 \
    --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 \
    --seeds 37 --variants full
done
)
```

成功时应有三个 `DRY_RUN_OK`，随后三个 `QUEUED`；一个 release 对应一个 GPU1 任务。重复执行会拒绝已有 release，不能覆盖继续。如果中途有任务已成功入池，先用下方 status 核对，再仅提交缺失项。

## 3. 查看 1 卡、三个任务、当前日志

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
date &&
nvidia-smi -i 1
for w in 100 200 400; do
  python -B /media/stephen/nvme0n1/wy_data/EV-UAV-v16/tools/v16/queue_jobs.py status \
    --release "context_20261113_r5_w${w}" --tail 18
done
ls -lh /media/stephen/nvme0n1/wy_data/EV-UAV-v13/run/queue/pool/pending/ \
       /media/stephen/nvme0n1/wy_data/EV-UAV-v13/run/queue/gpu1/pending/ \
       /media/stephen/nvme0n1/wy_data/EV-UAV-v13/run/queue/gpu1/running/
```

## 4. 三项完成后打包

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
(
set -e
for w in 100 200 400; do
  python -B tools/v16/queue_jobs.py pack \
    --release "context_20261113_r5_w${w}"
done
)
```

只打包完成结果；核验权重、评估、profile、适配阶段产物及冻结源码。将三个 `DOWNLOAD:` 指向的 `.tgz` 通过 SFTP 下载到本地项目 `Res/`。未完成时不强行打包，已完成的窗口可以单独执行相应 `pack`。
