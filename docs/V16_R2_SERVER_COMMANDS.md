# V16 r2：沿用现有 GPU 1 池

本地当前分支为 V16。SFTP 默认 lab-v16、context=EV-UAV，目标 `/media/stephen/nvme0n1/wy_data/EV-UAV-v16`，uploadOnSave=false。无需新开池。旧 r1 已运行任务使用冻结 source，新上传文件不会替换它们的源码。

服务器目录（可重复执行）：

```bash
mkdir -p /media/stephen/nvme0n1/wy_data/EV-UAV-v16
```

通过 SFTP: Upload Project 上传。至少上传 `model/`、`dataset/`、`utils/`、`tools/v14/`、`tools/v16/`、`configs/`、`train_stream_v1.py`。不要上传本地 run/log/Res 或环境。新配置在 `configs/v16/r2/`，默认从头训练，无旧权重链接要求。

检查并加入六个任务，全部仅使用物理 GPU 1：

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
python -B tools/v16/queue_jobs.py enqueue \
  --config-dir configs/v16/r2 --release context_20261113_r2 \
  --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 --dry-run &&
python -B tools/v16/queue_jobs.py enqueue \
  --config-dir configs/v16/r2 --release context_20261113_r2 \
  --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13
```

顺序1280 full37、1290 full38、1300 no_revision37、1310 no_revision38、1320 no_context37、1330 no_context38。旧队列剩余任务先执行。每项自动 GPU 预检→训练和逐轮验证→最佳权重验证→测试→速度/功耗评估。最多40轮，至少15轮后按验证 IoU patience8早停。预检通过不等于保证最终精度；某项失败会保留错误日志并由原池继续后续任务。

同 release 不可重复发布；需要重跑时使用新 release 名。源码和配置冻结在 `run/v16_context/context_20261113_r2/`，结果位于 `log/v16_context/context_20261113_r2/`。

查看显卡、六项状态和日志：

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
date && nvidia-smi -i 1 &&
python -B tools/v16/queue_jobs.py status --release context_20261113_r2 --tail 20
```

查看整个池，包括旧任务：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13 &&
ls -l run/queue/pool/pending/ run/queue/gpu1/pending/ run/queue/gpu1/running/ &&
tail -n 12 run/queue/pool.log run/queue/gpu1.log
```

全部完成后打包（校验结果完整性，不修改训练目录）：

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
python -B tools/v16/queue_jobs.py pack --release context_20261113_r2
```

通过 SFTP 下载打印的 DOWNLOAD 文件到本地 Res。若只需当前已完成部分，可显式增加 `--allow-partial`，包内会标明缺失任务；不能把部分包当作六项全部完成。
