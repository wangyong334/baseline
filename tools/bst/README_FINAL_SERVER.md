# 最终两种子任务：GPU 1 池子

> 后续窗口分组模型请使用 [BST_WINDOW_RELEASE.md](../../docs/BST_WINDOW_RELEASE.md) 的 `window_20261111_r2` 命令，并显式传入 `--config configs/bst_window.json`。本页的 r1 命令保留为原平坦均值基线和重启恢复说明；已经运行的池子无需再次恢复。

先通过 SFTP 上传本次代码；服务器项目仍是 `/media/stephen/nvme0n1/wy_data/EV-UAV-v13`。下面命令只管理这个目录的队列，不连接其他机器、不结束其他用户进程。

## 重启后启动并入队

```bash
conda activate evuav
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13

# 必须前台完成恢复，再启动后台池子和添加新任务，避免恢复动作误归档刚添加的任务。
python tools/bst/queue_final.py pool-preflight --recover-reboot
```

仅在上条成功后继续。若检测到仍在运行的调度器或工作进程，工具会保留现场并提示；不要盲目清理。

```bash
mkdir -p run/queue
nohup bash tools/bst/start_pool.sh >> run/queue/pool.log 2>&1 < /dev/null &

# 版本名须全新。两个 BIR 检查点都必须实际存在；缺任意一个则不会发布任务。
python tools/bst/queue_final.py enqueue --release final_20261111_r1 \
  --data-root /media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset \
  --backbone-37 /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41/log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt \
  --backbone-38 /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41/log/stream_tif_commpeak_msg_seed38/best_val_iou_seed38.pt

tail -n 15 run/queue/pool.log
ls -l run/queue/pool/pending run/queue/gpu1/pending run/queue/gpu1/running
tail -n 15 run/queue/gpu1.log
```

可以先把 `enqueue` 命令末尾增加 `--dry-run` 做无写入检查。种子 38 的路径是约定路径，工具会同时核实两份权重存在、非空、内容不同，且 checkpoint.config.seed 分别为 37 和 38；种子字段缺失也会停止。不会假装已远程核实。找不到时把参数改为服务器上真实的种子 38 检查点。

池子严格只管理 GPU 1，任务串行；空闲保留 20 GiB，占位进程由原有 worker 在任务开始前释放。不要另外启动 `gpu_queue.sh`，池子会创建 worker。新 wrapper 通过 `flock` 防止重复启动，同时检查旧调度器。

恢复会将旧的 `pool/pending`、`gpu1/pending`、`gpu1/running` 任务和残留 STOP/PID 文件移入 `run/queue/reboot_archive/<时间>/`，日志原地保留。旧中断任务不会自动续跑。GPU 0/2/3 的队列不改动。

## 两个实验的含义

- BIR 37 + SDU/VoW 37；BIR 38 + SDU/VoW 38。
- 任务固定使用入队时的 Python 解释器绝对路径，运行前复验骨干 SHA 与种子；`environment.json` 记录实际 PyTorch/CUDA/驱动与设备环境。
- 源码冻结在 `run/bst_final/final_20261111_r1/source/`，运行期间继续改工作区不会改变已排队的实验。
- 输出为 `log/bst_final_final_20261111_r1_v5s37_s37/` 和 `log/bst_final_final_20261111_r1_v5s38_s38/`。
- 每组完整导出 train/val/test 先验，再训练 SDU、拟合 VoW，按预先声明的策略评估，最后运行 `final_metrics.py`。不以测试集选择策略。
- `reconsideration/` 是真实录像小样本的同权重机制干预，检查旧证据能否重新审查；它不是重新训练的精度消融实验。
- `COMPLETE` 表示所有阶段成功，`FAILED` 表示该任务失败；已有输出绝不覆盖。中断后需选择新版本，或单独审查后手动续阶段。

## 下载分析包

先查看两个种子的精度、推理时间和板卡能耗表：

```bash
conda activate evuav
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13
python tools/bst/summarize_final.py --release run/bst_final/final_20261111_r1/release.json \
  --out Res/final_release_final_20261111_r1
```

未完成的种子/策略显示 missing 或 partial，缺失能耗显示 NA，不会按零处理；不会在测试集上挑最好的策略。

```bash
python tools/bst/pack_final.py --release final_20261111_r1
```

下载 `Res/bst_final_final_20261111_r1.tgz` 到本地 Res。包含结果、事件输出、日志、配置与冻结源码；默认不含模型权重和先验全缓存。需要权重时增加 `--include-weights`，输出文件名须换新或指定 `--out`。包内 `PACK_STATUS.json` 明确标记两组是否完成，不把未完成结果当完整实验。
