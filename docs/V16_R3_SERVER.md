# V16 r3：完整模型 seed37，使用现有 GPU1 池

本轮只提交一个 `full_s37` 任务。保留 V5 的 BIR、跨窗膜电位、通信约束，以及四路分流并行 TBPTT；从已完成的 r2 seed37 骨干初始化。最多训练 20 轮，至少 10 轮，验证集 IoU 连续 6 轮无提升时提前结束。初始化状态也参与验证集最佳权重选择，若训练未改善，将保留初始化状态；不能据此声称新模块获得提升。

任务自动执行：真实运行环境前后向预检 → 训练与逐轮验证 → 选择最佳验证权重 → 完整验证集、测试集评估 → 推理性能测量。测试集不用于选模型。不会启动、重启或终止现有池子；任务由已有 GPU1 队列按顺序执行。

## 1. 服务器目录与 SFTP 上传

在服务器执行：

```bash
mkdir -p /media/stephen/nvme0n1/wy_data/EV-UAV-v16
```

当前 SFTP 项目根目录对应 `/media/stephen/nvme0n1/wy_data/EV-UAV-v16`。上传以下源文件更新，保持相对路径：

- `model/anchored_revision.py`
- `tools/v16/` 下的更新文件
- `configs/v16/r3/` 下的三份配置

沿用服务器已有的 V16 项目。不要把本地 `run/`、`log/`、`Res/` 同步覆盖到服务器；尤其不要改动既有发布版本 `run/v16_context/.../source/` 中的冻结源码。新任务入池时会单独冻结这次源码与初始化权重。

## 2. 校验并准备 r2 seed37 初始化权重

下面命令检查指定权重的完整 SHA256，再复制到独立初始化目录。目的文件若已存在且校验一致则直接复用；若内容不同则停止，不覆盖。

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
python -B - <<'PY'
import hashlib
import shutil
from pathlib import Path

root = Path('/media/stephen/nvme0n1/wy_data/EV-UAV-v16')
source = root / 'log/v16_context/context_20261113_r2_gpu2/full_s37/best_val_iou_seed37.pt'
target = root / 'run/v16_initialization/r3/seed37.pt'
expected = 'f0f92d07f44e86a3d2a9e349f72f6f4ae4ae18e074ad4a2de22fbd7143799f8d'

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

if not source.is_file() or source.stat().st_size == 0:
    raise SystemExit('初始化权重缺失或为空：' + str(source))
if digest(source) != expected:
    raise SystemExit('初始化权重 SHA256 不匹配，停止。')
if target.is_symlink():
    raise SystemExit('初始化目的路径是软链接，停止：' + str(target))
if target.exists():
    if not target.is_file() or digest(target) != expected:
        raise SystemExit('初始化目的文件已存在且内容冲突，拒绝覆盖：' + str(target))
    print('INITIALIZATION_REUSED:', target)
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))
    if digest(target) != expected:
        raise SystemExit('复制后 SHA256 不匹配，停止。')
    print('INITIALIZATION_READY:', target)
print('SHA256:', expected)
PY
```

必须看到 `INITIALIZATION_READY` 或 `INITIALIZATION_REUSED` 后再继续。只需要 seed37，无需 seed38 初始化权重。

## 3. 预检并加入现有 1 卡池子

SFTP 上传完成后执行整段：

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
python -B tools/v16/queue_jobs.py enqueue \
  --release context_20261113_r3 \
  --config-dir configs/v16/r3 \
  --variants full --seeds 37 \
  --init-root run/v16_initialization/r3 \
  --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 \
  --dry-run &&
python -B tools/v16/queue_jobs.py enqueue \
  --release context_20261113_r3 \
  --config-dir configs/v16/r3 \
  --variants full --seeds 37 \
  --init-root run/v16_initialization/r3 \
  --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13
```

应看到 `DRY_RUN_OK: 1 training job(s); GPU 1`，然后一个 `QUEUED`。任务显式绑定物理 GPU1，任务内部的 `cuda:0` 是屏蔽其他卡后的逻辑编号。已有 GPU1 任务会先继续执行，不需要另开 worker，也不要重复执行入池命令；同名发布存在时程序会拒绝重复提交。

输出目录：

```text
/media/stephen/nvme0n1/wy_data/EV-UAV-v16/log/v16_context/context_20261113_r3/full_s37
```

## 4. 查看 GPU1、任务状态和当前日志

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
date &&
nvidia-smi -i 1 &&
python -B tools/v16/queue_jobs.py status \
  --release context_20261113_r3 --tail 35
```

`pool/pending` 表示等待现有任务；`gpu1/running; stage=training` 表示正在训练。日志中的 `V16 RUNTIME PREFLIGHT PASSED` 表示运行环境预检通过。最终应为 `gpu1/done; stage=complete`，并有 `V16_CONTEXT_FINISHED`。

查看整个现有 GPU1 池：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13 &&
ls -lh run/queue/pool/pending/ run/queue/gpu1/pending/ run/queue/gpu1/running/ &&
tail -n 12 run/queue/pool.log run/queue/gpu1.log
```

## 5. 完成后打包下载

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
python -B tools/v16/queue_jobs.py pack --release context_20261113_r3
```

此命令要求正式训练、验证、测试和性能测量全部完成，并校验结果、配置、权重及冻结源码。未完成时会停止，不生成冒充完整结果的包。按输出 `DOWNLOAD:` 路径，通过 SFTP 下载对应压缩包到本地 `Res/`。
