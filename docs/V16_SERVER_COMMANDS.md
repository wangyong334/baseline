# V16：GPU 1 六作业部署与结果回收

> 本文保留 r1 复现命令。当前 r2 上传、入池与打包请使用 `docs/V16_R2_SERVER_COMMANDS.md`；r2 配置在 `configs/v16/r2/`。

上传项目代码到 `/media/stephen/nvme0n1/wy_data/EV-UAV-v16` 后，在服务器已有 `evuav` 环境执行下面命令。数据保持 `/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset`，调度器仍是 `/media/stephen/nvme0n1/wy_data/EV-UAV-v13` 的现有池。

## 新建目录与上传

先在服务器执行：

```bash
mkdir -p /media/stephen/nvme0n1/wy_data/EV-UAV-v16
```

本地外层 `.vscode/sftp.json` 默认 profile 已切为 `lab-v16`，context 为 `EV-UAV`，remotePath 为上述 V16 目录，uploadOnSave=false。VS Code 执行 **SFTP: Upload Project**；确认使用 `lab-v16`。此次不是恢复旧权重，不需要建立骨干权重软链接。

最少需上传 `model/`、`dataset/`、`utils/`、`tools/v14/`、`tools/v16/`、`configs/`、`train_stream_v1.py`。CLAUDE、Res、run、log、虚拟环境和权重均排除，不上传服务器的运行状态。

默认 release 为 `context_20261112_r1`，按以下顺序发布六项独立训练：

| 顺序 | 配置 | seed |
|---|---|---:|
| 1 | full | 37 |
| 2 | full | 38 |
| 3 | no_revision | 37 |
| 4 | no_revision | 38 |
| 5 | no_context | 37 |
| 6 | no_context | 38 |

三配置只允许 `CONTEXT.revision_variant`、输出目录、统计文件目录不同；种子由 runner 写入独立运行配置。每项都从头训练 50 epoch，4 streams、TBPTT K=16、100 ms 窗/50 ms stride、single rolling carry、V5 BIR 与通信峰值预算 157700 保持固定，训练执行 `layer`，正式评估执行 `step`。CONTEXT 固定 dim=12、hidden=24、heads=3、radius=2、downsample=4。

## 检查并入队

以下整块可复制执行；只有 dry-run 成功才会发布任务。命令不启动、重启或终止调度器。默认只生成 `@gpu1.sh` 作业，等待现有池安排 GPU 1；作业内物理卡 1 映射为 `cuda:0`。

```bash
(
  set -euo pipefail
  conda activate evuav
  cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16
  python -B tools/v16/queue_jobs.py enqueue \
    --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 \
    --config-dir configs/v16 --release context_20261112_r1 --dry-run
  python -B tools/v16/queue_jobs.py enqueue \
    --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 \
    --config-dir configs/v16 --release context_20261112_r1
)
```

dry-run 检查 Python/torch 与训练入口导入、配置、源文件、train/val/test 的 99/24/24 个非空 NPZ、已有调度器进程和停止标记。依赖检查隐藏 CUDA，不运行网络或反向传播，也不创建任务/结果目录。

实际入队先把 `model/`、`dataset/`、`utils/`、`train_stream_v1.py`、`tools/v14/`、`tools/v16/` 和相关配置复制到 `run/v16_context/<release>/source/`，写入逐文件 SHA256；六项作业仅使用冻结源码。上传缺文件、冻结内容变化、同 release 已入队或输出已存在均拒绝继续。每项发布采用独占创建，重复运行上述命令不会重复训练。部分发布失败时保留 manifest 与已发布任务，先查看 STATUS，勿删除目录后重试。

作业取得 GPU 后，依次执行 GPU 反向预检、train、val、test、profile。预检读取该作业的运行配置，报告必须 `passed=true` 才启动正式训练；配置与报告保存在 `run/v16_context/<release>/jobs/<variant>_s<seed>/`。正式训练权重不从预检或旧实验恢复。任一步失败，该项作业标为 failed，不伪造完成结果。

## STATUS

```bash
(
  set -euo pipefail
  conda activate evuav
  cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16
  python -B tools/v16/queue_jobs.py status --release context_20261112_r1 --tail 20
)
```

状态同时显示调度器的 pending/running/done/failed 和每项的 preflight/training/evaluating_val/evaluating_test/profiling/complete/failed。作业日志位于 V13 调度器的 `run/queue/gpu1/logs/<job>.log`。结果目录为 V16 项目的 `log/v16_context/context_20261112_r1/<variant>_s<seed>/`。

val/test 的主结果是对应独立训练 variant，`readout_policy` 为 `full`、`no_revision` 或 `no_context`；`readouts.first/second/main` 是该权重的读出诊断，不能代替不同 variant 独立重训。profiling 使用 test 输入的 3 条录像、2 次重复，记录所选 checkpoint 哈希。

## PACK

全部完成后运行：

```bash
(
  set -euo pipefail
  conda activate evuav
  cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16
  python -B tools/v16/queue_jobs.py pack --release context_20261112_r1
)
```

命令核验六项完成状态、50 epoch 的正式 summary、选定权重与配置/预检/统计/eval/profile 哈希，检查 val/test 的 step 执行与主读出、profiling 权重身份，并要求每项调度器日志存在。压缩包包含权重、运行配置、训练统计/日志、summary、val/test、profile、预检报告、结果汇总、源码 manifest 与冻结源码，不含数据集、SFTP 凭据或缓存。

成功后打印 `DOWNLOAD:` 的完整路径、大小和 SHA256。下载该文件，通常位于 `EV-UAV-v16/Res/v16_context_20261112_r1_<时间>_<标识>.tgz`。打包期间文件仍在变化会明确报错，稍后重试。

若只需回收失败诊断，可显式打部分包：

```bash
(
  set -euo pipefail
  conda activate evuav
  cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16
  python -B tools/v16/queue_jobs.py pack --release context_20261112_r1 --allow-partial
)
```

部分包文件名带 `_partial`，其中 `package_manifest.json` 列明每项缺失结果，不能作为六项完整结果。已标为完整但内容哈希不符的结果仍拒绝打包。若确需独立重跑失败项，应使用一个新的 release，并显式指定该项的 `--variants` 和 `--seeds`；保留原 release 的证据与队列状态。
