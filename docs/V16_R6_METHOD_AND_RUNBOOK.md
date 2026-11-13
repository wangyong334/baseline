# V16 r6：dec1 条件协方差读出与 GPU2 独立实验

更新：2026-11-13（Asia/Shanghai）。本文件对应 `configs/v16/r6/`。本轮交付配置和运行说明；**尚未上传或启动服务器 r6，没有 r6 正式 Test 结果**。

r6 固定使用 r2 seed37 骨干、100 ms 窗长 / 50 ms 步长、跨窗状态、dec1 12D 特征与原事件 logit、共享 Fisher 8D 投影、条件协方差白化、有界残差小头，以及 r4 的全局平滑漏检参照约束。仅启动 `full_s37`；其余配置为后续独立消融保留。本轮没有骨干适配阶段。

## 1. 问题、原理与实际主张

重叠窗口 A+B、B+C 为 B 内同一个事件提供前后两次上下文。原流式 SNN 已有跨窗膜电位；补充读出的目的，是利用后来实际到达的上下文改善尚未提交的事件判决。两次读出共享输入及历史，分数不能作为独立证据直接相乘。

Hotelling Observer（HO）以类内协方差组织线性判别。在给定描述子下，记两类均值差为 `d`、等权类内协方差为 `K=(K_0+K_1)/2`，线性检测信噪比的最优方向为 `w=solve(K,d)`，平方可分性为 `d^T solve(K,d)`。这一线性信噪比最优化不要求高斯；解释为仿射对数似然比则需要额外分布假设。它不自动保证最佳 AUC、低误报尾部性能或概率校准。

本实现采用 HO 启发的描述子压缩和条件协方差坐标，再训练非线性残差头。因此称为**HO 启发的协方差条件残差读出**，不称为精确最优 HO 或精确贝叶斯后验。白化只处理估计的类内线性二阶相关，不是语义上从混合特征中剥离重复片段 B，也不证明两次观察独立。

已有原理说明见 [r4 方法](V16_R4_METHOD.md)；本地参考 `References/Hottel/thnov03p0774.pdf` 为 He / Park（2013）的综述 *Model Observers in Medical Imaging Research*，不是最早提出 HO 的论文。以下公式直接对应当前实现。

## 2. 公式与网络的对应

### 2.1 原事件读出与共享通道

对同一个原始事件 ID，在首次和末次覆盖窗口读出：

\[
h_j=u^{\mathrm{pre}}_{\mathrm{dec1},j}(x_e,y_e)\in\mathbb R^{12},\qquad
v_j=[h_j;z_j]\in\mathbb R^{13},\quad j\in\{1,2\}. \tag{1}
\]

`z_j` 仍由原事件 MLP 根据 dec1 特征、极性和窗内时间生成；`h_j` 是复位前特征，不是 MLP 隐层，也不是不含历史的纯当前输入。事件 ID 保证两次读出对应同一待分类事件；不做运动估计或邻域检索。

仅在训练集成对事件上拟合统计：每录像、每类别最多抽 512 个，seed37。合并两次观察后分别求类均值及类内协方差，以两类均值中点 `m` 和类内标准差 `s` 共享标准化：

\[
\widetilde v_j=(v_j-m)\oslash s,\qquad
x_j=P^\top\widetilde v_j\in\mathbb R^8. \tag{2}
\]

`s` 的方差下界为 `1e-8`。`P` 的第一候选轴为标准化空间中的正则化 Fisher 方向，其余候选为按方差降序的类内主方向；经两遍正交化选满 8 轴，退化时用坐标轴补齐。两次观察共用同一个 `P`；此处不启用 `joint_fisher`。

共享空间和联合空间分别使用以下正则化，联合矩阵先整体正则化再分块：

\[
\operatorname{Reg}(C)=0.95C+0.05\operatorname{diag}(C)+\epsilon I,
\quad\epsilon=10^{-5}\max\{\operatorname{mean}(\operatorname{diag}C),10^{-8}\}. \tag{3}
\]

这些不是全样本混合总协方差；类间均值差不会计入类内散布。拟合及求解在 CPU float64 上完成，模型保存固定统计缓冲区，部署通常为 float32。

### 2.2 第二次基准与第一次条件残差

将联合向量按 `[x_2;x_1]` 排列，减去训练背景类联合均值，得到 `[c_2;c_1]`。正则化类内协方差分块为：

\[
\Sigma=\begin{bmatrix}\Sigma_{22}&\Sigma_{21}\\
\Sigma_{12}&\Sigma_{11}\end{bmatrix},\quad
R=\Sigma_{12}\Sigma_{22}^{-1},\quad
S=\Sigma_{11}-R\Sigma_{21}. \tag{4}
\]

记 `L_2 L_2^T=Sigma_22`、`L_r L_r^T=S`，送入小头的坐标为：

\[
q=\begin{bmatrix}L_2^{-1}c_2\\L_r^{-1}(c_1-Rc_2)\end{bmatrix}\in\mathbb R^{16}. \tag{5}
\]

代码在拟合时使用 `solve` / Cholesky，将式（5）存成固定 16×16 变换；**每事件前向没有矩阵求逆**。其线性可分性可作 Schur 分解：

\[
d^\top\Sigma^{-1}d
=d_2^\top\Sigma_{22}^{-1}d_2
+(d_1-Rd_2)^\top S^{-1}(d_1-Rd_2). \tag{6}
\]

这给出“第二次基准＋第一次线性条件补充”的统计解释。正则化下完全重复观察也可能出现正的条件项，因此该项为正不能单独证明新信息。完整白化在保留的 16D 空间中可逆，不增加信息；13→8 压缩则可能损失信息。自由小头也能重新混合这些坐标，实际价值需由消融验证。

### 2.3 有界残差与提交

\[
\delta=2\tanh\!\left(w_o^\top\operatorname{SiLU}(Wq+b)\right),\qquad
z_f=z_2+\delta,\qquad p_f=\sigma(z_f). \tag{7}
\]

`W` 为 8×16，第一层有 8 个偏置，输出层无偏置，共 **144 个可训练参数**。输出层零初始化，起点严格等于 second；修正可正可负，幅度上界为 2，但这不保证分类或召回不变。单次覆盖事件沿用其唯一原读出。

```text
事件流 → 100/50 ms 重叠表示 → 冻结 r2 / V5 BIR SNN U-Net
                                  │ 跨窗膜电位继续传递
                           dec1 u_pre 12D + 原事件 logit
                                  │ 原始事件 ID 配对
                       两次 13D → 共享训练拟合投影，各 8D
                                  │
                   第二次白化坐标 + 第一次条件残差白化，16D
                                  │
                       Linear 16→8 → SiLU → Linear 8→1
                                  │
                           2 tanh → 加回 z_second
                                  │
                           sigmoid → 固定阈值 0.9
```

成对事件在第二次覆盖结束后提交，相对第一次读出多等一个 50 ms 步长。事件发生至最终输出的采集等待取决于事件在窗口中的位置，不能统一称为 50 ms，更不能称为零前瞻；还需加计算和排队时间。

代码对应：式（1）为 `tools/v16/train.py:readout_with_hidden`；式（2）–（6）为 `model/covariance_revision.py:fit_statistics/coordinates`；式（7）为 `forward_encoded`。数据配对与统计抽样见 `tools/v16/train_covariance.py:observations/fit_training_statistics`。

## 3. 固定工作点的训练与选择

设 `a=logit(0.9)=log(9)`，只在训练成对事件上定义类条件均值：

\[
L_{FP}=\mathbb E_{y=0}\operatorname{softplus}(z_f-a),\quad
L_{FN}=\mathbb E_{y=1}\operatorname{softplus}(a-z_f),\quad
c_{ref}=\mathbb E_{y=1}\operatorname{softplus}(a-z_2). \tag{8}
\]

\[
\mathcal L=L_{FP}+\lambda(L_{FN}-c_{ref})+
10^{-3}\{\operatorname{mean}(W^2)+\operatorname{mean}(w_o^2)\}. \tag{9}
\]

参照 `c_ref` 固定，来自同一训练集原 second。乘子以训练集第二次读出对统一 logit 偏移的梯度平衡初始化：

\[
\lambda_0=\operatorname{clip}_{[0,100]}
\frac{\mathbb E_{y=0}\sigma(z_2-a)}{\max(\mathbb E_{y=1}\sigma(a-z_2),10^{-8})},\quad
\lambda\leftarrow\operatorname{clip}_{[0,100]}[\lambda+0.5(L_{FN}-c_{ref})]. \tag{10}
\]

式（10）的更新在每轮结束后用全训练成对事件重算的平滑风险执行。式（8）–（10）对应 `constrained_loss/stationary_multiplier/update_recall_multiplier/fit_cached_head`。这是 **全局 `smooth_fn_reference`**，允许正例之间补偿；不是逐录像约束、二元漏检率保证或非凸优化收敛保证。

协议保持：Adam；batch32768；余弦学习率 0.001→0.0001；梯度裁剪 1；最多 20 轮、最少 8 轮、patience5。全训练成对缓存用于头训练。配置保留 streams4、TBPTT16、原通信预算等兼容字段；本阶段骨干无梯度，不更新通信乘子，也不是再次端到端训练 BIR。

验证用全部事件，固定 0.9。候选相对固定 second 的 IoU / ACC / Pd 必须均不降低、Fa 不增加，零容差；合格且 IoU 严格高于当前最佳才替换。没有合格改善时保留初始零修正。守卫只约束本次验证选择，不能保证 Test 同样不退化。Test 只在训练选模结束后评估，不拟合统计、不调阈值、不选择 checkpoint。

## 4. 配置、消融和已有证据

三份配置相对 r4 仅更改输出/统计路径，并显式加入 `covariance_feature_source: dec1`、`covariance_channel_basis: shared_fisher`。NET `[12,24,48,48]`、TIF levels7、readout_hidden32、carry / single_rolling、原校准均保留；不引入 r5 的 adaptation 或 `record_fn_reference`。

| 配置 | 输入及变换 | 用途 |
|---|---|---|
| `configs/v16/r6/full.yaml` | 两次 dec1＋logit；完整条件白化 | 本轮唯一服务器实验 |
| `configs/v16/r6/no_revision.yaml` | 同投影、同尺度、同容量；用对角标准化替代完整白化 | 后续独立重训头，**仍有修正头** |
| `configs/v16/r6/no_context.yaml` | 两个输入均替换为 second，重新拟合统计并训练 | 后续独立重训头；容量相同，统计秩不同 |

每次正式评估另输出 first / second / main，同权重诊断不等于独立重训消融。

以下是已完成的**本地训练→验证探针**，不是 r6 正式服务器结果。来源：`Res/v16_ho_basis_probe_20261113/REPORT.txt` 与 `consolidated_summary.json`。IoU / ACC / Pd 为 %；Fa 为 ×10⁻⁶。ACC 实际为目标事件召回 `TP/(TP+FN)`；Pd / Fa 沿用项目 FastRoc，Fa 不是背景事件 FP/N。

| 验证方案 | IoU | ACC | Pd | Fa | TP / FP / FN |
|---|---:|---:|---:|---:|---|
| 原 second | 86.805152 | 91.159588 | 91.096178 | 6.319440 | 59715 / 3286 / 5791 |
| dec1＋logit，完整 | 87.098044 | 91.173328 | 91.285174 | 5.546175 | 59724 / 3065 / 5782 |
| 原多层共享投影 | 87.081535 | 91.173328 | 91.159177 | 5.658758 | 59724 / 3078 / 5782 |
| dec1，去完整白化 | 87.078947 | 91.162642 | 91.285174 | 5.652832 | 59717 / 3072 / 5789 |
| dec1，去第一次观察 | 87.008069 | 91.194700 | 91.243175 | 5.839482 | 59738 / 3152 / 5768 |

完整 dec1 相对 second 的 IoU 增加 0.292893 个百分点，TP+9、FP−221。相对原多层只增加 0.016509 点，主要依据是简化而未观察到验证损失。白化独立收益约 0.0191 点；前次观察收益约 0.0900 点，但比无前次观察少 14 个 TP。单骨干、单种子、多次复用验证集存在选择偏差，尚不足以证明强独立贡献，不承诺 90% 或四项 Test 同时提高。

原始描述子由 60D 降至 12D；线性 MAC/配对事件为 `2×13×8 + 16×16 + 16×8 + 8 = 600`，原多层为 1368。两者 q 都为 16D，紧凑缓存不因此缩小。该数不含骨干、gather、归一化、激活、读写或原事件头，不能当作整体计算量/能量降幅。

已有兼容性证据：Python3.8 / Torch1.9.1 CPU 175 项中 173 通过、2 CUDA 项跳过；本地 GPU 在 train_000 / val_000 各前 1 秒上验证了非零头反传、冻结骨干及缓存/在线一致性，最大概率差不超过 5.96×10⁻⁸、0.9 分类差为 0。见同目录 `GPU_VALIDATION.txt`。这些检查不是服务器 Torch1.9 CUDA 或完整数据实验的替代。

## 5. SFTP 上传与初始化

服务器项目：`/media/stephen/nvme0n1/wy_data/EV-UAV-v16`。Python：`/home/stephen/anaconda3/envs/evuav/bin/python3.8`。数据：`/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset`。

在已有 r4 服务器代码基础上，通过现有 SFTP 上传最新：

- `model/covariance_revision.py`
- `tools/v16/` 下 Python 文件，尤其 `train.py`、`train_covariance.py`、`run_experiment.py`、`queue_jobs.py`、`run_standalone.py`
- `configs/v16/r6/` 全部三份 YAML（full 运行也会验证三份协议一致）
- `docs/V16_R6_METHOD_AND_RUNBOOK.md`

已有依赖 `model/evspsegnet_stream.py`、`dataset/` / `utils/` Python 文件、`train_stream_v1.py`、`tools/v14/`、`configs/v14/`、`configs/evisseg_stream_tif_comm_peak.yaml` 和原 `configs/v16/` 应保持当前仓库版本。`source_files` 会将所需 Python 代码、v14/v16 YAML 全部快照并记录 SHA256；只有上传本次差异而缺少已有依赖时，预检会失败。无需上传本地数据、虚拟环境、探针缓存或含凭据的配置。

初始化只使用：

```text
/media/stephen/nvme0n1/wy_data/EV-UAV-v16/log/v16_context/context_20261113_r2_gpu2/full_s37/best_val_iou_seed37.pt
SHA256 f0f92d07f44e86a3d2a9e349f72f6f4ae4ae18e074ad4a2de22fbd7143799f8d
```

保留原 q99、增益及骨干状态；不要用 r4 的融合 checkpoint 代替。独立脚本启动时复制源码与初始化到唯一 release，随后从副本运行，后续 SFTP 更新不会影响该快照。

## 6. GPU2 前台预检与后台启动（整段复制）

本轮 release 为 `context_20261113_r6_gpu2`。本地同名正式输出不存在；服务器是否已有同名目录由 `--dry-run` 和实际启动再次检查，已有即拒绝覆盖。无需池子、调度器或持卡进程；只运行一个 full seed37，完成后退出。

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
(
set -euo pipefail
PY=/home/stephen/anaconda3/envs/evuav/bin/python3.8
RELEASE=context_20261113_r6_gpu2
INIT=log/v16_context/context_20261113_r2_gpu2/full_s37/best_val_iou_seed37.pt
printf '%s  %s\n' f0f92d07f44e86a3d2a9e349f72f6f4ae4ae18e074ad4a2de22fbd7143799f8d "$INIT" | sha256sum -c -
nvidia-smi -i 2
"$PY" -B tools/v16/run_standalone.py run \
  --project-root "$PWD" \
  --data-root /media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset \
  --release "$RELEASE" --config-dir configs/v16/r6 \
  --init-checkpoint "$INIT" --dry-run

# Resolve physical GPU 2 by UUID; each child sees it as logical cuda:0.
GPU_UUID=$("$PY" -B -c 'from tools.v16.run_standalone import gpu2_identity; print(gpu2_identity()["uuid"])')
mkdir -p run/v16_preflight
PREFLIGHT=$(mktemp -d "run/v16_preflight/${RELEASE}.XXXXXX")
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -B -u tools/v16/runtime_preflight.py \
  --config configs/v16/r6/full.yaml --device cuda:0 --out "$PREFLIGHT/runtime.json"

mkdir -p run/v16_context
LAUNCH_LOG=$(mktemp "run/v16_context/${RELEASE}.launch.XXXXXX.log")
nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -B -u tools/v16/run_standalone.py run \
  --project-root "$PWD" \
  --data-root /media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset \
  --release "$RELEASE" --config-dir configs/v16/r6 \
  --init-checkpoint "$INIT" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
printf 'LAUNCH_PID=%s\nLAUNCH_LOG=%s\n' "$!" "$LAUNCH_LOG"
)
```

预期依次出现 `OK`、物理 GPU2 信息、`DRY_RUN_OK: one standalone full/seed37 job on physical GPU 2`、`V16 RUNTIME PREFLIGHT PASSED`，随后打印后台 PID / 启动日志。`nvidia-smi` 展示共享卡使用情况；本脚本不等待空卡、不自动挤出其他任务。外层 dry-run 不创建 CUDA 上下文或正式 release。前台预检只用合成数据检查该配置在服务器的算子、非零梯度和确定性，不训练正式骨干；正式流水线仍在冻结副本内再做一次预检。

自动顺序：源码/权重快照 → GPU 兼容预检 → train-only 统计拟合 → train/val 缓存 → 冻结骨干的头训练与验证选择 → 缓存/在线一致性 → 正式 val → 正式 test → profile → 进程退出。任何阶段失败均保留状态与日志；不会启动池子或持卡程序。

## 7. 状态、打包与回收

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
(
set -e
/home/stephen/anaconda3/envs/evuav/bin/python3.8 -m json.tool \
  run/v16_context/context_20261113_r6_gpu2/standalone.json
tail -n 40 run/v16_context/context_20261113_r6_gpu2/standalone.log
)
```

启动早期若尚无 `standalone.json`，先看启动命令打印的 `LAUNCH_LOG`。最终状态应为 `complete`，日志末尾应有 `V16_CONTEXT_FINISHED` 与 `STANDALONE COMPLETE`；这时再运行：

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v16 &&
/home/stephen/anaconda3/envs/evuav/bin/python3.8 -B tools/v16/run_standalone.py pack \
  --project-root "$PWD" --release context_20261113_r6_gpu2
```

下载输出的 `DOWNLOAD` 路径到本地 `Res/`，同时保留打印的 SHA256。pack 验证完成状态、GPU 身份、配置/源码/初始化/结果哈希；包内包含训练权重、统计、配置、日志、预检、评估和 profile、源码副本，排除大的 NPZ 缓存。失败或未完成任务不能打包成完整实验。

关键文件：

- `log/v16_context/<release>/full_s37/summary.json`、`metrics.jsonl`：完成轮数、选中轮次、是否回退零头。
- 同目录 `eval_val.json`、`eval_test.json`：main / first / second 的四指标与逐录像计数。
- 同目录 `cache_audit.json`、`calibration.json`、`run_config.json`、`covariance_stats.json`：缓存一致性、冻结初始化和训练拟合来源。
- 同目录 `profile.json`：在线推理测量；按 test 输入事件数分位点选 3 条录像（覆盖最少/最多事件数），各 2 次重复，不是完整测试集吞吐。
- `run/v16_context/<release>/jobs/full_s37/config.yaml`、`manifest.json`、`runtime_preflight.json` 与外层 `standalone.json`：实际运行配置、身份和状态。

回收后先核验包和哈希，再分析固定 0.9 的四指标、TP/FP/FN 与逐录像退化、训练/统计/缓存总耗时及在线 profile。若报告“救回/误杀”事件数，还需要原 second 与 main 的逐事件交叉判决；只有 TP/FP 总量之差时，应称净变化，不能据此伪造转移计数。

profile 的可选功率积分属于整张 NVIDIA 板卡，共享时不能独立归因于此模型；不等于整机或神经形态芯片能耗。冻结头训练也需披露此前 r2 骨干训练成本。r6 后续是否补独立消融、其他种子或大窗口，由正式证据和下一阶段安排决定。

## 8. 本次发布检查与旧 r5 结果补充

本地 Python3.8 / Torch1.9.1 CPU 已完成三份配置的协议校验和原 r2 骨干严格加载；头部均为144参数，full / no_revision / no_context 线性 MAC 分别600 / 360 / 600。82份源码配置的隔离冻结副本哈希通过，重复 release 拒绝，本文三段 Bash 通过 `bash -n`。检查记录见 `Res/v16_r6_release_validation_20261113/validation.json`。本地只验证不检查 Linux/GPU 的准备路径，服务器 CLI `--dry-run` 及前台 CUDA 预检尚未执行。

用户随后回收旧 GPU1 池的 r5 100/200/400 ms 结果，分析见 `Res/v16_r5_server_analysis_20261113/REPORT.txt`。其 Val IoU 为86.9169/85.4631/81.9332%，Test为88.6176/84.7217/84.3588%；相对各自second的Test修正增益为−0.1254/−0.1009/0点。400组最终回退零修正，不能解释为学到了有效融合。该轮仍使用多层60D特征、逐录像约束及骨干适配，与本文件r6协议不同。

r5没有提供将大窗口或逐录像约束推进主方案的支持。本次保留此前由train/val确定的r6设置，未根据新看到的Test结果调参；r6正式服务器结果仍待运行。
