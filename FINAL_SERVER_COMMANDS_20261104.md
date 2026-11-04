# V9-final 服务器训练与测试命令（2026-11-04）

> 方案：`FINAL_PLAN_20261104.md`。实验室服务器目录 `/media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41`（沿用：数据、V5、现有模块三的权重都在这里），conda 环境 `evuav`。
>
> **卡的安排**：在跑的三个任务结束后再开始（卡 2 是别人的，不用）：
> - 卡 0：E1，末行 `GPU0 E1 FINISHED`；
> - 卡 3：读特征自适应 + 因果头，末行 `GPU3 FEATURE NEXT FINISHED`；
> - 卡 1：EV-Flying 读特征，末行 `GPU1 EVF FEATURE FINISHED`。
>
> 每张卡同时只训一个骨干（约 15.5 GB）。

## 0. 上传

- 本地文件：`D:\桌面-D\EV-UAV\EV-UAV\final_v9_code_20261104.tgz`。包里是 V9-final 的全部改动与新增文件，外加 `FINAL_V9_SHA256SUMS`。
- 用 MobaXterm / VS Code 拖到服务器的 `/media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41/`（用 `.tgz` 后缀，sftp 不传 `*.tar.gz`）。
- 包内文件清单（与 `FINAL_V9_SHA256SUMS` 一致）：
  - 代码：`train_stream_v1.py`、`train_track_head.py`、`model/changepoint_layer.py`、`model/lookahead_head.py`、`utils/recording_groups.py`；
  - 工具：`tools/select_checkpoint.py`、`tools/select_backbone.py`、`tools/final_analysis.py`、`tools/b1_delayed_confirm.py`、`tools/paper_tables.py`、`tools/final/*.sh`（8 个）；
  - 配置：`configs/final/*.yaml`（14 个）、`configs/final/paper_manifest.json`、`configs/stream_v1_train_stats_w25.json`；
  - 测试：`tests/test_final_v9.py`；
  - 文档：`FINAL_PLAN_20261104.md`、`FINAL_SERVER_COMMANDS_20261104.md`。

## 1. 解包与预检（前台，约 3 分钟）

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
which python                                         # 应是 evuav 环境里的 python（py3.8、torch 1.9.1）
tar -xzf final_v9_code_20261104.tgz                  # 解出代码、配置、测试与 FINAL_V9_SHA256SUMS
sha256sum -c FINAL_V9_SHA256SUMS                     # 每行都应为 OK
sed -i 's/\r$//' tools/final/*.sh                    # 保险：脚本统一为 LF 换行
python -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -n 3   # 应为 Ran 432 tests 与 OK（本地 432 项）
nvidia-smi -L                                        # 4 张卡，没有 Unknown Error
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

测试数不是 432 时，先看有没有 FAIL / ERROR；没有的话，多半是服务器上多了或少了旧测试文件，把数字发我核对。

## 2. GPU 冒烟（前台，一张空闲的卡，约 5 分钟）

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3       # 换成空闲的卡号
rm -rf log/final/smoke_* run/final/smoke_*
mkdir -p run/final
# ① 25 ms 骨干：显存分组 stream_groups 2，最后应打印 SMOKE TEST FINISHED；记下 peak_memory_gib（应远小于 24）
python -u train_stream_v1.py --config configs/final/bb_v5_25.yaml --mode smoke --device cuda:0 --save-root log/final/smoke_v5_25 2>&1 | grep -E "peak_memory_gib|SMOKE"
# ② B2 前瞻头：2 轮、每轮 3 个片段、每个划分 4 条录像，最后应打印 TRAINING FINISHED
python -u train_track_head.py --config configs/final/b2_lookahead_50.yaml --mode train --device cuda:0 --save-root log/final/smoke_b2 --epochs 2 --max-chunks 3 --max-sequences 4 2>&1 | tail -n 2
# ③ 逐滞后轨迹导出：现有读特征头、1 条 val 录像，最后应打印"结果写入 ..."
python -u train_track_head.py --config configs/final/rcl_feature_50.yaml --mode eval --device cuda:0 --split val --max-sequences 1 --checkpoint log/changepoint_v41_insample_feature_seed37/best_val_iou_seed37.pt --base-checkpoint log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt --trajectory-dir run/final/smoke_traj --tag smoke 2>&1 | tail -n 2
ls run/final/smoke_traj | head -n 3                  # 应有 val_000.npz
rm -rf log/final/smoke_* run/final/smoke_*
```

## 3. 第一批（种子 37，决定主张）：三条队列，错开 30 秒启动

先确认三个在跑的任务都已结束：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
tail -n 1 run/gpu0_e1.log run/gpu3_feature_next.log run/gpu1_v41evf_feature.log   # 三个都应是 FINISHED
mkdir -p run/final
```

再逐条启动（每条单独一行，不用 `A && nohup B &`）：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
GPU=3 nohup bash tools/final/queue_m3.sh > run/final/queue_m3.log 2>&1 &
sleep 30
GPU=0 nohup bash tools/final/queue_m2.sh > run/final/queue_m2.log 2>&1 &
sleep 30
GPU=1 nohup bash tools/final/queue_25ms.sh > run/final/queue_25ms.log 2>&1 &
```

| 队列 | 内容 | 预计 |
|---|---|---|
| 卡 3 `queue_m3.sh` | 冻结 V5 按新口径评估（含 val / test 加噪 4 档）→ B2 两版（只有事件损失 / 加占据监督）训练与全套评估 → 现有读特征头、只读分数头、因果头接入新评估（不重训；读特征头另跑完整模型加噪与 B1）→ EV-Flying 读特征头接入 → 按新协议重训 V5 种子 37（`RETRAIN_V5=0` 可跳过） | 约 19 h |
| 卡 0 `queue_m2.sh` | E1 → E3b → E3a（都按新协议：逐轮存权重、护栏选轮次、val / test 加噪 4 档）→ 按事先定的规则选骨干（`run/final/backbone_choice_seed37.log`）→ 在选定骨干上训练模块三 = 完整模型 | 约 17～18 h |
| 卡 1 `queue_25ms.sh` | V5 @ 25 ms → RCL @ 25 ms（D = 20 窗 = 500 ms） | 约 11～15 h |

查看进度（随时可跑）：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
tail -n 2 run/final/queue_m3.log run/final/queue_m2.log run/final/queue_25ms.log
ls log/final/*/BACKBONE_DONE log/final/*/HEAD_DONE 2>/dev/null          # 已完成的骨干 / 判决头
for f in log/final/*_seed37/metrics.jsonl; do echo "$f $(grep -c . $f)"; done   # 各运行已完成的轮数（共 50）
cat run/final/backbone_choice_seed37.log 2>/dev/null                       # 卡 0 选骨干之后才有
```

- 每个队列的最后一行是 `QUEUE ... FINISHED`；出错时是 `QUEUE ... FAILED at <步骤>`，日志在 `run/final/<运行名>_seed37_<步骤>.log`。
- 某一步失败后，修好问题可以单独重跑那一步。例如只重跑某个骨干的选择与评估：`GPU=0 NAME=bb_e3b_50 TRAIN=0 nohup bash tools/final/backbone.sh > run/final/redo.log 2>&1 &`。

第一批跑完打包（不含权重，含分析 JSON；默认不含轨迹，要含时加 `TRAJ=1`）：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
bash tools/final/pack.sh                             # 生成 final_results_<时间>.tgz，打印大小与文件数
python tools/paper_tables.py --manifest configs/final/paper_manifest.json --out run/final/paper   # 预览表与图（服务器没有中文字体时图里的中文会成方框；正式图我在本地生成）
tar -czf final_paper_preview.tgz run/final/paper     # 表与图另打一个小包
ls -lh final_results_*.tgz final_paper_preview.tgz
```

## 4. 第二批：多种子与 EV-Flying（第一批选出骨干之后）

`M2` 填 `run/final/backbone_choice_seed37.log` 里"选定骨干"的配置名，如 `bb_e3b_50`；选中冻结 V5 时填 `bb_v5_50`。

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
grep "选定骨干" run/final/backbone_choice_seed37.log
GPU=0 SEED=38 M2=bb_e3b_50 nohup bash tools/final/queue_seed.sh > run/final/queue_seed38.log 2>&1 &
sleep 30
GPU=3 SEED=39 M2=bb_e3b_50 nohup bash tools/final/queue_seed.sh > run/final/queue_seed39.log 2>&1 &
sleep 30
GPU=1 M2=bb_e3b_50 nohup bash tools/final/queue_evf.sh > run/final/queue_evf.log 2>&1 &
```

| 队列 | 内容 | 预计 |
|---|---|---|
| `queue_seed.sh`（每个种子一张卡） | V5 与选定骨干的重训 → 选定骨干 + 模块三（完整模型，含加噪）→ V5 + 模块三（含 B1）→ B2 → 因果头；加 `B2OCC=1` 时再跑 B2 + 占据监督（第一批里它若是结构对照的关键一行，就加上） | 每个种子约 20～24 h（+4 h） |
| `queue_evf.sh` | 选定 E3b 时：E3b-EVF（掩码关）+ 模块三；否则直接结束 | 约 8 h |

全部跑完：同第 3 节打包。`paper_manifest.json` 已经列好种子 38 / 39 的路径，主表会自动报均值 ± 标准差。

## 5. 结果回来后我做的事（本地）

- 用 `tools/paper_tables.py` 出表与图；
- 用 `tools/final_analysis.py` 的 JSON 按 FINAL_PLAN §3.5、§4.6 的事先分支判主张；
- 更新 FINAL_PLAN 的结论、PAPER-NOTES、CLAUDE.md。

test 只报告，不用来改任何选择。
