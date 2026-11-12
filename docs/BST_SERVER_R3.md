# 从旧 780 任务切换到最终候选 r3

服务器项目为 `/media/stephen/nvme0n1/wy_data/EV-UAV-v13`，现有池子只管理 GPU 1。首轮结果包已经下载归档；以下流程保留池子和 worker，只结束精确匹配的旧训练 Python，再上传并发布新版本。

## 1. 先只上传停止工具

通过现有 SFTP 映射，**仅上传新增文件** `tools/bst/stop_legacy_training.py` 到服务器相同路径。此时先不要覆盖其他 `model/`、`tools/`、`configs/`：旧 780 没有冻结源码，其后续阶段可能读取这些文件。

```bash
conda activate evuav
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13

python tools/bst/stop_legacy_training.py \
  --out log/bst_shared_v5s37_s37_20261111
```

默认仅显示匹配进程，不发送任何信号。匹配条件同时包含当前用户、当前项目的精确工作目录、Python 的 `tools/bst/train.py` 入口、唯一且解析后相同的 `--out`。不依赖截图里的旧 PID；多个匹配或参数歧义会拒绝。

权限兼容修复：枚举时先读取命令行，排除其他程序后才读取候选训练进程的工作目录。无关进程的 `/proc/<pid>/cwd` 不可读不再阻断扫描；无法读取命令行的 PID 单独提示且不发送信号。若候选训练进程无法完整核实，或扫描不完整且没有找到可核实目标，仍明确拒绝。遇到旧版的 `Cannot inspect PID 2810 ... /cwd` 时，重新单独上传更新后的停止工具并重试，无需 sudo。

## 2. 停止旧 trainer 并确认队列交接

```bash
python tools/bst/stop_legacy_training.py \
  --out log/bst_shared_v5s37_s37_20261111 --execute &&
python - <<'PY'
from pathlib import Path
import time

marker = Path("run/queue/gpu1/running/780_bst_shared_v5s37_s37_20261111.sh")
deadline = time.monotonic() + 20
while marker.exists():
    if time.monotonic() >= deadline:
        raise SystemExit("旧 780 任务仍在 running：先检查当前阶段，不要批量上传源码。")
    time.sleep(0.25)
print("QUEUE_HANDOFF_CONFIRMED：旧 780 标记已移出 running，可以上传更新。")
PY

ls -l run/queue/gpu1/running/
tail -n 12 run/queue/gpu1.log
```

`--execute` 会在信号前再次核对 PID 启动时钟、UID、工作目录和完整命令行，只发送一次 SIGTERM，并最多等待 20 秒确认退出；僵尸状态视为已经退出。系统和 Python 支持时使用绑定进程的 pidfd；Python 3.8 使用重复身份校验兼容路径。工具不发 SIGKILL，不结束 pool/worker，也不停止其他用户或项目。

看到 `EXIT_CONFIRMED` 或 `ALREADY_EXITED` 仅代表 trainer 已退出，还必须看到 `QUEUE_HANDOFF_CONFIRMED`，即 worker 已把 `gpu1/running/` 中旧 `780_bst_shared_v5s37_s37_20261111.sh` 移走。上面的第二段最多等待 20 秒，超时以非零状态退出。旧任务因人为停止被 worker 记为 FAILED 是预期状态；已保存的第一轮权重和日志不会被工具删除。首轮之后尚未保存的训练进展不保留。

`NO_MATCH` 只说明没有匹配的活 trainer，**不证明整个旧任务已经结束**：它可能已经进入策略拟合或评估。若旧脚本仍在 running，先查看上述队列日志和当前进程，停止器不会自动扩大范围去杀其他阶段。`REFUSED` 或未确认退出时也先检查现场，不继续上传或执行广泛 `pkill`。

## 3. 上传最终候选更新

确认旧任务退出后，通过 SFTP 上传本次完整 `model/`、`tools/`、`configs/` 更新。必须包含：

- `configs/bst_final.json`；
- `model/bst.py`、`model/bst_window.py`（空证据竞争在该文件中）；
- `tools/bst/data.py`、`training_geometry.py`、`pipeline.py`、`train.py`；
- 本轮更新的策略拟合、评估、解释与队列相关脚本。

新的 `geometry_cache_mb=64` 只约束 CPU 常驻邻居 ID 缓存，保留训练样本和 SGD 批次顺序。新模型需从头训练 SDU/VoW，不将旧第一轮权重改名为新实验。

## 4. 将 37、38 两个种子放进现有池子

```bash
conda activate evuav
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13

python tools/bst/queue_final.py enqueue \
  --release final_20261111_r3 \
  --config configs/bst_final.json --dry-run &&
python tools/bst/queue_final.py enqueue \
  --release final_20261111_r3 \
  --config configs/bst_final.json

ls -l run/queue/pool/pending/ run/queue/gpu1/pending/ run/queue/gpu1/running/
tail -n 12 run/queue/pool.log
tail -n 12 run/queue/gpu1.log
```

`--config configs/bst_final.json` 必须显式保留，队列工具的兼容默认仍是旧 `bst_shared.json`。r3 标签必须尚未使用。两个任务在 GPU 1 串行，使用入队时冻结的源码快照；不要重新启动池子，不执行 `--recover-reboot`，不单独启动 worker。

默认读取相邻 `EV-UAV-v8-rcl41/log/stream_tif_commpeak_msg_seed37/` 和 `seed38/` 下对应的 `best_val_iou_seed*.pt`。这里 seed38 的默认命名与历史实际路径不同：`queue_round2.sh`、`queue_v10.sh` 及历史训练记录明确使用 `log/final/bb_v5_50_seed38/selected_seed38.pt`（V5 TIF、50 ms、carry、seed38，历史选择 epoch36）。若默认路径报缺失，用下面的文件软链接兼容现有队列命令；不复制权重，也不覆盖已有其他文件。

```bash
(
  set -e
  base=/media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
  src="$base/log/final/bb_v5_50_seed38/selected_seed38.pt"
  dst="$base/log/stream_tif_commpeak_msg_seed38/best_val_iou_seed38.pt"
  [ -s "$src" ] || { echo "真实 seed38 权重不存在：$src"; exit 1; }
  python - "$src" <<'PY'
import sys
from tools.bst.queue_final import backbone_seed
if backbone_seed(sys.argv[1]) != 38:
    raise SystemExit("权重内部种子不是 38，未创建链接")
PY
  mkdir -p "$(dirname "$dst")"
  if ! [ "$src" -ef "$dst" ]; then ln -sT "$src" "$dst"; fi
  readlink -f "$dst"
)
```

在新项目根目录执行，成功后重跑上面的 dry-run 和入队命令。当前远端实体是否存在仍由 `test -s` 实际确认；工具继续检查内部种子及 SHA256，链接最终解析到真实权重记录，不会把 37 当作 38。

冻结源码位置：`run/bst_final/final_20261111_r3/source/`。两组输出为 `log/bst_final_final_20261111_r3_v5s37_s37/` 和 `log/bst_final_final_20261111_r3_v5s38_s38/`。实际训练日志会记录几何准备、网络更新、验证与各轮耗时；本地探针速度不是服务器整轮时间保证。

## 5. 完成后汇总和下载

```bash
python tools/bst/summarize_final.py \
  --release run/bst_final/final_20261111_r3/release.json \
  --out Res/final_release_final_20261111_r3

python tools/bst/pack_final.py --release final_20261111_r3 --include-weights
```

下载 `Res/bst_final_final_20261111_r3.tgz` 到本地 Res。未完成、失败和缺失能耗会保留相应状态，不当作完整结果或零能耗。训练权重、验证结果、五路径评估以及计算/板卡能耗应一起分析。
