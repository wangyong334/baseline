# V9-final 服务器训练与测试命令（2026-11-04；用户定：先做基础实验，验证最终模型的有效性）

> 方案：`FINAL_PLAN_20261104.md`。
>
> - 实验室服务器目录 `/media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41`（沿用：数据、V5、现有模块三的权重都在这里）；
> - conda 环境 `evuav`；
> - 卡 0 / 1 / 3 可用，卡 2 是别人的不用；
> - 每张卡同时只训一个骨干（约 15.5 GB）。
>
> 本地 sftp 不用改：默认 profile `lab` 的 remotePath 就是上面的目录，`.tgz` 不在忽略列表里。

## 0. 上传（本地 VS Code）

- 在资源管理器里右键 `EV-UAV/final_v9_code_20261104.tgz`，选 "SFTP: Upload File"，会传到服务器的 `EV-UAV-v8-rcl41/` 下。
- 包里是 V9-final 的全部改动与新增文件，共 40 个，外加 `FINAL_V9_SHA256SUMS`：
  - 代码：`train_stream_v1.py`、`train_track_head.py`、`model/changepoint_layer.py`、`model/lookahead_head.py`、`utils/recording_groups.py`；
  - 工具：`tools/select_checkpoint.py`、`tools/select_backbone.py`、`tools/final_analysis.py`、`tools/b1_delayed_confirm.py`、`tools/paper_tables.py`、`tools/final_summary.py`、`tools/final/*.sh`（10 个）；
  - 配置：`configs/final/*.yaml`（14 个）、`configs/final/paper_manifest.json`、`configs/stream_v1_train_stats_w25.json`；
  - 测试：`tests/test_final_v9.py`；
  - 文档：两份 FINAL 文档。

## 1. 先收上一批的结果（前台）

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
tail -n 1 run/gpu0_e1.log run/gpu3_feature_next.log run/gpu1_v41evf_feature.log   # 三个都应是 ... FINISHED
ls -lht *.tgz | head -n 5                            # 上一批的三个结果包（E1、因果头、EV-Flying 读特征），下载回本地 Res/ 发我
ls log/changepoint_v41_insample_feature_causal_seed37/best_val_iou_seed37.pt \
   log/changepoint_v41_insample_feature_seed37/best_val_iou_seed37.pt \
   log/changepoint_v41_insample_seed37/best_val_iou_seed37.pt \
   log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt               # 基础实验要用的四份现有权重，都应存在
```

## 2. 解包、校验、测试、冒烟（前台，约 10 分钟）

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
which python                                         # 应是 evuav 环境里的 python（py3.8、torch 1.9.1）
sha256sum final_v9_code_20261104.tgz                 # 与我给的值核对（本地同一个包）
tar -xzf final_v9_code_20261104.tgz                  # 解出代码、配置、测试与 FINAL_V9_SHA256SUMS
sha256sum -c FINAL_V9_SHA256SUMS | grep -vc ': OK$'  # 应输出 0（全部 OK）
sed -i 's/\r$//' tools/final/*.sh                    # 保险：脚本统一为 LF 换行
python -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -n 3   # 应为 Ran 433 tests 与 OK（本地 433 项）
nvidia-smi -L                                        # 4 张卡，没有 Unknown Error
# GPU 冒烟（卡 3，约 3 分钟）：轨迹导出 + 判决头训练入口
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
rm -rf log/final/smoke_* run/final/smoke_* && mkdir -p run/final
python -u train_track_head.py --config configs/final/rcl_feature_50.yaml --mode eval --device cuda:0 --split val --max-sequences 1 --checkpoint log/changepoint_v41_insample_feature_seed37/best_val_iou_seed37.pt --base-checkpoint log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt --trajectory-dir run/final/smoke_traj --tag smoke 2>&1 | tail -n 2   # 最后应打印"结果写入 ..."
python -u train_track_head.py --config configs/final/rcl_feature_50.yaml --mode train --device cuda:0 --save-root log/final/smoke_head --epochs 2 --max-chunks 3 --max-sequences 4 2>&1 | tail -n 1   # 应打印 TRAINING FINISHED
rm -rf log/final/smoke_* run/final/smoke_*
unset CUDA_VISIBLE_DEVICES
```

测试数不是 433 时，先看有没有 FAIL / ERROR；没有的话，多半是服务器上多了或少了旧测试文件，把数字发我核对。

## 3. 基础实验（种子 37）：验证最终模型的有效性

三条队列，错开 30 秒启动，每条单独一行：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
mkdir -p run/final
GPU=3 nohup bash tools/final/queue_basic_v5.sh > run/final/queue_basic_v5.log 2>&1 &
sleep 30
GPU=0 nohup bash tools/final/queue_basic_e3b.sh > run/final/queue_basic_e3b.log 2>&1 &
sleep 30
GPU=1 nohup bash tools/final/queue_basic_e1.sh > run/final/queue_basic_e1.log 2>&1 &
```

| 卡 | 队列 | 内容 | 回答的问题 | 预计 |
|---|---|---|---|---|
| 0 | `queue_basic_e3b.sh` | E3b 骨干按新协议重训（逐轮存权重、护栏选轮次、val / test 干净与加噪 4 档）→ 在 E3b 上训练模块三（读特征头）→ 全套评估（逐滞后轨迹、正式自适应、不给未来、完整模型加噪、B1） | 完整模型（M1 + M2 + M3）的精度、延迟、鲁棒性 | 约 9～10 h |
| 1 | `queue_basic_e1.sh` | 同上，骨干换成 E1（窄加噪、不带 NISI，模块二的匹配对照）→ 在 E1 上训练模块三 | 模块二是否有用（E3b 对 E1）；模块三是否换个骨干也有效 | 约 8～9 h |
| 3 | `queue_basic_v5.sh` | 冻结 V5 按新口径评估 → V5 上现有三个判决头接入新评估，不重训（读特征头 + B1 + 加噪、只读分数头、因果头）→ E3a 骨干 → 等 E1、E3b 完成后按事先定的规则选骨干，把选中骨干 + 模块三标为完整模型 → 自动生成汇总表 `run/final/summary_basic.md` | 模块一一行；模块三是否必要（对因果头、B1）；最终用哪个骨干 | 约 10～11 h |

查看进度（随时可跑）：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
tail -n 2 run/final/queue_basic_*.log                                       # 三条队列各自到了哪一步
for f in log/final/*_seed37/metrics.jsonl; do echo "$(basename $(dirname $f))：$(grep -c . $f) / 50 轮"; done   # 各训练已完成的轮数
for f in run/final/*_train.log; do echo "== $f"; grep "^epoch" $f | tail -n 1 | cut -c1-150; done           # 各训练最后一轮的 val
ls log/final/*/BACKBONE_DONE log/final/*/HEAD_DONE 2>/dev/null               # 已完成的骨干 / 判决头
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

- 每条队列的最后一行是 `QUEUE BASIC ... FINISHED`；出错时是 `QUEUE BASIC ... FAILED at <步骤>`，日志在 `run/final/<运行名>_seed37_<步骤>.log`。
- 修好问题后可以单独重跑那一步。例如某个骨干只重跑选择与评估：

  ```bash
  GPU=0 NAME=bb_e3b_50 TRAIN=0 nohup bash tools/final/backbone.sh > run/final/redo.log 2>&1 &
  ```

全部跑完（`queue_basic_v5.log` 末行 FINISHED）后，看汇总并打包：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
cat run/final/summary_basic.md                       # 骨干、判决头、选骨干、B1 四张表
bash tools/final/pack.sh                             # 生成 final_results_<时间>.tgz（不含权重与轨迹），打印大小与文件数
ls -lh final_results_*.tgz                           # 下载回本地 Res/ 发我
```

## 4. 第二批、第三批：用 `tools/final/chains.sh` 在每张卡上接着跑（11-04 用户定）

`tools/final/chains.sh` 等每张卡上的基础队列结束，再一个接一个启动下表的队列；选定骨干从 `run/final/backbone_choice_seed37.json` 自动读，某个队列失败不影响后面的。

| 卡 | 等什么 | 接着跑 |
|---|---|---|
| 1 | `queue_basic_e1.log` 末行 FINISHED / FAILED | `queue_25ms` → `queue_evf`（只有选中 E3b 时才真跑） |
| 0 | `queue_basic_e3b.log` 末行 FINISHED / FAILED | `queue_seed` 种子 38（`B2OCC=1`） |
| 3 | `log/final/bb_e3a_50_seed37/BACKBONE_DONE` | `queue_b2_evf` → `queue_seed` 种子 39（`B2OCC=1`） |

```bash
nohup bash tools/final/chains.sh gpu1 > run/final/chain_gpu1.log 2>&1 &
nohup bash tools/final/chains.sh gpu0 > run/final/chain_gpu0.log 2>&1 &
nohup bash tools/final/chains.sh gpu3 > run/final/chain_gpu3.log 2>&1 &
```

下面是原来手动启动各队列的写法，留作参考：

| 队列 | 内容 | 预计 |
|---|---|---|
| `GPU=? nohup bash tools/final/queue_25ms.sh > run/final/queue_25ms.log 2>&1 &` | 25 ms 核心实验：V5 @ 25 ms → RCL @ 25 ms | 约 11～15 h |
| `GPU=? nohup bash tools/final/queue_b2_evf.sh > run/final/queue_b2_evf.log 2>&1 &` | B2 两版（结构对照）→ EV-Flying 读特征头接入 → V5 按新协议重训种子 37 | 约 12 h |
| `GPU=? SEED=38 M2=<选定骨干的配置名> nohup bash tools/final/queue_seed.sh > run/final/queue_seed38.log 2>&1 &` | 种子 38；39 同理（`B2OCC=1` 加跑 B2 + 占据监督） | 每个种子约 20～24 h |
| `GPU=? M2=<选定骨干的配置名> nohup bash tools/final/queue_evf.sh > run/final/queue_evf.log 2>&1 &` | 选中 E3b 时：E3b-EVF（掩码关）+ 模块三 | 约 8 h |

`M2` 填 `run/final/backbone_choice_seed37.log` 里"选定骨干"对应的配置名：`bb_e3b_50`、`bb_e1_50`、`bb_e3a_50`，选中冻结 V5 时填 `bb_v5_50`。

全部跑完后，同第 3 节打包；论文的表和图用 `tools/paper_tables.py --manifest configs/final/paper_manifest.json`，我在本地生成（服务器缺中文字体）。test 只报告，不用来改任何选择。

## 5. V9 优化（11-04 下午，用户批准；不影响正在跑的实验）

改动在 50 ms 下逐位不变（改动前后 63 组参照输出逐位相同，全套 447 项测试 OK），正在跑的链继续用、结果不变；新增的只是三个只评估的工具和 25 ms 完整模型的配置。上传 `v9_opt_20261104.tgz`（右键 SFTP: Upload File），然后：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
sha256sum v9_opt_20261104.tgz                          # 与我给的值核对
tar -xzf v9_opt_20261104.tgz                           # 解出 11 个文件（9 个代码 + 两份文档）与 V9_OPT_SHA256SUMS（几秒钟）
sha256sum -c V9_OPT_SHA256SUMS | grep -vc ': OK$'      # 应输出 0
sed -i 's/\r$//' tools/final/*.sh
python -m unittest tests.test_v9_opt 2>&1 | tail -n 3  # Ran 14 tests ... OK
# 优化 2：E3b 背景估计器消融（卡 1，与 E1 的判决头训练共用显存；等 E3b 骨干完成后自动开始，约 1.5～2 h）
GPU=1 nohup bash tools/final/bg_ablation.sh > run/final/bg_ablation_e3b.log 2>&1 &
# 优化 1：各判决头的提交接口校准（只用 CPU；按顺序等 V5 / E1 / E3b / EV-Flying / 25 ms 的判决头完成）
nohup bash tools/final/commit_cal.sh > run/final/commit_cal.log 2>&1 &
# 优化 3：25 ms 完整模型（卡 1 的链跑完后自动开始；只在规则选中 E3b 时真跑，约 12～16 h）
nohup bash -c 'until grep -q "CHAIN gpu1 FINISHED" run/final/chain_gpu1.log 2>/dev/null; do sleep 300; done; GPU=1 bash tools/final/queue_25ms_full.sh' > run/final/queue_25ms_full.log 2>&1 &
```

查看：`tail -n 3 run/final/bg_ablation_e3b.log run/final/commit_cal.log run/final/queue_25ms_full.log`。打包照旧用 `bash tools/final/pack.sh`（消融与校准的结果都在 log/final 与 run/final/analysis 里）。

## 6. 延迟口径统一 + α 用校准后的含义（11-04 傍晚，用户批准第 2、3 项）

- 改的都是 Python 分析与报告文件，外加一个新脚本 `tools/final/cal_analysis.sh`，不动任何正在运行的 shell 脚本。
- `final_analysis.py` 不加 `--temperature` 时输出与原来逐位相同（只多一个值为 null 的 temperature 字段），本地用真实轨迹核对过。
- 上传 `v9_opt2_20261104.tgz`，然后：

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
sha256sum v9_opt2_20261104.tgz                           # 与我给的值核对
tar -xzf v9_opt2_20261104.tgz                            # 解出文件与 V9_OPT2_SHA256SUMS（几秒钟）
sha256sum -c V9_OPT2_SHA256SUMS | grep -vc ': OK$'       # 应输出 0
sed -i 's/\r$//' tools/final/cal_analysis.sh
python -m unittest tests.test_v9_opt 2>&1 | tail -n 3    # Ran 19 tests ... OK
# 用 val 上拟合的温度重算各判决头的分析（只用 CPU；等 commit_cal 给出温度后依次处理）
nohup bash tools/final/cal_analysis.sh > run/final/cal_analysis.log 2>&1 &
```

- 结果写回 `run/final/analysis/<run>_seed37_{val,test}.json`。新文件是旧文件的超集：旧策略逐位不变，另加校准后的 adaptive_cal / slice_cal / simple_cal。
- 汇总表与论文表有校准结果时优先用它；延迟一律是算法延迟（不含计算），计算时间另列每窗处理 ms。

## 7. 方案 A（11-04 晚，用户定：不要模块二，主模型 = V5 + 判决头）

用 `tools/final/plan_a.sh` 接管还没跑的部分。服务器上正在跑的旧脚本一个字不改。
上传包 `v9_planA_20261104.tgz`（18:45 第三版，sha256 `05d2245c502dea22d98e5d280d94f1932bbf4fa54e036e3c1f827beeb02d5e5f`）里有：
- `tools/select_checkpoint.py`：默认仍是事先定的护栏规则；新增 `--rule plain` / `SELECT_RULE=plain` 和 `--reference-metrics`；
- `tools/final/plan_a.sh`；
- `configs/final/b2_lookahead_25.yaml`；
- `configs/final/reference/bb_v5_25_seed37_v63_record.jsonl`（10-25 那次 V5@25 的逐轮 val 记录，用来核对逐位复现）；
- 两个测试文件。

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41
conda activate evuav
sha256sum v9_planA_20261104.tgz                      # 应为 05d2245c502dea22d98e5d280d94f1932bbf4fa54e036e3c1f827beeb02d5e5f
tar -xzf v9_planA_20261104.tgz
python -m unittest tests.test_final_v9 tests.test_v9_opt 2>&1 | tail -3     # 应为 Ran 42 tests ... OK
bash tools/final/plan_a.sh stop                      # 只预览：带 * 的行是要结束的
DRY=0 bash tools/final/plan_a.sh stop                # 核对无误后执行
nohup bash tools/final/plan_a.sh gpu0 > run/final/plan_a_gpu0.log 2>&1 &
sleep 60
nohup bash tools/final/plan_a.sh gpu1 > run/final/plan_a_gpu1.log 2>&1 &
nohup bash tools/final/plan_a.sh gpu3 > run/final/plan_a_gpu3.log 2>&1 &
nohup bash tools/final/plan_a.sh cpu  > run/final/plan_a_cpu.log 2>&1 &
tail -n 4 run/final/plan_a_*.log                      # 查看进度
```

| 子命令 | 做什么 |
|---|---|
| `stop` | 停掉卡 0 的 `chains.sh gpu0`、`queue_seed.sh`、它下面的 `backbone.sh` 和 E3b 种子 38 的训练；停掉卡 3 的 `chains.sh gpu3` 和 `queue_b2_evf.sh`（正在训的 B2 + 占据监督不动，V5 重训不再跑）。卡 1 不动 |
| `stop25` | 只在改用"只看 val"（`SELECT_RULE=plain`）时用：停掉卡 1 整个旧 25 ms 队列 |
| `gpu0` | 种子 38：V5 已完成就直接用 → RCL（含 B1）→ B2 → EV-Flying 读特征头的新口径评估（不重训） |
| `gpu1` | 等卡 1 旧队列（V5@25 → RCL@25）结束 → 同一份骨干上接 B2@25；并报告 50 轮是否与 10-25 逐位相同 |
| `gpu3` | 等 B2 + 占据监督（种子 37）结束 → 种子 39：V5 → RCL（含 B1）→ B2 |
| `cpu` | 种子 37 的 B2、B2 + 占据监督做提交校准；种子 38 / 39 的判决头在各自队列里校准 |

- 只做干净评估，只报官方合并口径。
- `HEADS="rcl"` / `HEADS25="rcl"` 可去掉 B2；已完成的判决头自动跳过，没完成的旧目录会先移开。
