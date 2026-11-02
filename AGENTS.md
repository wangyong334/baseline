# 项目交接：EV-UAV 流式 SNN 与双仓记忆实验

> **已过时（10-14 的交接文件，只作历史）**：项目现状看 `CLAUDE.md` 第 1 节，论文线索看 `PAPER-NOTES.md`。
> 10-28 起实验在租用的 3090 上跑：`/hy-tmp/EV-UAV-v1`，分支 v6-4，不用 conda。

最后更新：2026-10-14。本文是本地会话交接文件，不提交 Git，也不通过 SFTP 上传。

## 最新进展（优先于下文第一轮背景）

- 2026-10-15 新增 enc234 短时间梯度对照：`configs/evisseg_stream_dual_enc234_tbptt{4,8}.yaml`。两组仅相对原 enc234 改 `tbptt_k` 和 `save_root`，仍保持 `optimizer_step: per_sequence`；对应最大时间梯度跨度 200/400 ms，U/M 数值状态仍跨完整 160 窗（8 s）。两卡训练、test、M-off 和健壮打包命令见 `README_STREAM_TBPTT_SHORT_HORIZON.md`。这是 TBPTT，不得称为 OTTT、SLTT 或无时间梯度。代码交付时完整 `test_stream_*.py` 共 107 项通过，尚未在服务器训练。

- 第一轮三组已完成并分析，包为 `Res/dual_memory_results_20261014_093222.tar.gz`；分析脚本、原始指标副本与曲线在 `Res/dual_memory_analysis_20261014/`。
- enc4/enc34/enc234 测试 carry IoU 为 82.42%/79.83%/85.19%，验证最佳为 86.47%/85.96%/86.42%。仅单种子，M-off 尚未提供，不能断言 M 独立收益或污染已经解决。
- 用户现有四张空闲 GPU，已授权扩展解码层并安排第二轮：GPU 0 enc2、GPU 1 enc3、GPU 2 dec3、GPU 3 dec2，均只在一个位置加 M，从头训练。enc4 使用第一轮参考，不默认所有组必须包含 enc4。
- 实现已支持全部 7 层的双仓选择和 `--memory-off-layers enc3 enc4` 指定层读取消融。解码层保留原融合卷积/上采样结构，只替换原神经元；状态/监控/运算统计均适配。原第一轮 checkpoint 仍可加载。
- 四卡训练、正常测试与 M-off 顺序执行命令见 `README_STREAM_DUAL_PLACEMENT.md`；四份配置为 `configs/evisseg_stream_dual_{enc2,enc3,dec3,dec2}.yaml`，目录分别为 `log/stream_dual_<层名>_seed37`，不会覆盖第一轮。
- 第二轮目前是代码交付阶段，助手没有连接服务器或启动训练。下面“解码器不添加 M”等描述仅指第一轮，不是对新实验的限制。
- 亚阈值残差训练与 test 已完成；完整报告在 `Res/subthreshold_residual_test_analysis_20261014_173032/ANALYSIS.md`。enc234 / 纵向 / 纵向+横向双路的 test IoU 分别为 85.190% / 83.534% / 85.734%。双路仅单种子提升 0.544 pp，目标时延没有改善，深层硬支持率为 100%，尚不能证明门控或唤醒机制成立。
- 第二轮亚阈值消融显示：横向单路 test IoU 80.051%，同权重关闭横向为 83.221%，横向直接造成 -3.170 pp，增加 1,453 TP 的同时增加 4,904 FP；竞争式纵向单路 82.075%，同权重关闭纵向为 81.817%，直接作用仅 +0.258 pp。对应包为 `Res/subthreshold_round2_results_20261014_215427.tar.gz`。
- 下一轮两组代码已完成但尚未在服务器训练：GPU 0 为竞争式纵向 + 原横向双路；GPU 1 为竞争式纵向 + 稀疏有界横向双路。后者要求浅层竞争峰值与深层局部相对峰值同时成立，并把逐维修正限制在 `[-0.25,0.25]`。配置、训练/test/同权重消融命令见 `README_STREAM_SUBTHRESHOLD_ROUND3.md`。
- 第三轮已完成，包为 `Res/subthreshold_round3_results_20261015_111417.tar.gz`，分析在 `Res/subthreshold_round3_analysis_20261015_111417/ANALYSIS.md`。竞争纵向+原横向 / 选择性横向 test IoU 为 81.693% / 83.673%，均低于 enc234 的85.190%。选择性门把支持率从100%降到62.18%、平均修正从0.381降到0.048，但同权重关闭横向后 IoU 反而从83.673%升到83.747%，横向仍是 -0.073 pp；纵向直接作用 +0.173 pp，两路合计 +0.122 pp。两组 M-off 仅 +0.004/+0.016 pp。当前主模型继续保留 enc234，停止扩大亚阈值组合，下一步优先做真实弱目标分组与多种子验证。

## 先读什么、现在做到哪里

1. 先读本文，再读 `README_STREAM_DUAL_MEMORY.md`，其中有完整机制、三卡训练/测试命令与消融命令。
2. 原 V1 的执行与评估约定在 `README_STREAM_V1.md`；修改前以当前代码、配置为准。
3. 已有真实 EV-UAV 的双仓位置、亚阈值残差训练与 test 结果；新会话优先读上述分析报告，不要重新启动已完成实验。
4. 最新亚阈值代码交付时 `test_stream_*.py` 共 105 项 CPU 测试通过。GPU 结果来自用户服务器；本地没有复跑训练。
5. 当前分支为 `v1-snn`。本轮实现尚未由助手提交；新会话先检查 git status/diff，保留工作区已有修改，不自动切分支、清理或覆盖实验。

## 用户目标与已做决定

用户研究事件流小目标的目标/非目标事件分割，希望从持续在线 SNN 的状态机制中找到能支撑 CVPR 论文的贡献。任务输出是逐事件分类，不要擅自改成框检测。

关注的两个问题：

- 长期状态可能包含错误历史、背景响应，持续影响后续感知；保留、遗忘、更新、读取需要可检验的机制。
- 目标在运动，但膜电位与记忆绑定特征图坐标；旧位置残留和新位置无法利用历史可能同时发生。

用户最终选择：**双仓记忆 + 中深层特征的有限平移容忍**，先实现简洁基础方案再实验。
已放弃槽位记忆库、按特征检索历史的方案；不要在没有新证据或用户要求时重新引入。
本轮也不引入光流/速度估计、运动补偿、高斯扩散、浅层强制快泄漏、硬复位、依放电活动遗忘等额外机制。

用户非常重视批判性：不能把“有界”等同于“记忆正确”，不能用“模型会学会判别”代替机制依据，
不能保证 CVPR 接收、任意运动对齐、消除全部污染，也不能把增加记忆本身当作已证明的创新。

## 目录、环境与同步

- 本地仓库：`D:\桌面-D\EV-UAV\EV-UAV`。
- 服务器项目：`/media/stephen/nvme0n1/wy_data/EV-UAV-v1`。
- 数据集：`/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset`，下有 train/val/test；路径不变。
- 服务器沿用原 V1 的 `evuav` 环境；代码保持原 torch 1.9 接口兼容方式，未在本地实测该旧版本。
- 本地验证 Python：`.venv-stream-check/Scripts/python.exe`，CPU 环境；不要误称本地测试用了 4090。
- 用户自己上传代码、运行训练。未经要求不要连接服务器、重启任务或修改正在运行的实验。
- SFTP 配置位于外层 `D:\桌面-D\EV-UAV\.vscode\sftp.json`，context 为 `EV-UAV`，remotePath 为上述新项目路径。
  配置包含连接凭据，不要把原文、密码或其他认证信息写入报告/文档/日志。
- `.gitignore` 的 `AGENTS.md` 与 SFTP ignore 的 `**/AGENTS.md` 应保留。本文仅在本机交接，实验 YAML 和实现代码仍应正常提交/同步。

## V1 实际结构与时间约定

每条序列 8 秒，按 50 ms 切为 160 个连续窗口。每窗包含 5 个时间 bin，组成 12 个输入通道：
整窗正/负极性计数 2 通道 + 5 bins × 2 极性。**bin 是通道，不是额外 SNN 时间步**。
每窗每个神经元只更新一次；状态跨窗传递。输入使用训练集 q99 归一化统计。

原图 260×346，右下补齐到 264×352；channels 为 [12,24,48,48]。

| 层 | 尺寸 | 相对原图步幅 | 当前单窗理论感受野 |
|---|---|---|---|
| enc1 | 264×352 | 1 | 3×3 |
| enc2 | 132×176 | 2 | 5×5 |
| enc3 | 66×88 | 4 | 9×9 |
| enc4 | 33×44 | 8 | 17×17 |

enc4 → up3/dec3 → up2/dec2 → up1/dec1；跳连来自 enc3/enc2/enc1。
读出头用 dec1 复位前膜电位及事件 p、t_local 逐事件分类。全网共 7 组神经元状态。
本轮解码器与跳连不变，不因 U-Net 尺寸对称而对称添加记忆。

感受野大小不是实测有效感受野或最大可容忍位移。2.6 px 位移在 enc2/3/4 分别对应 1.3/0.65/0.325 网格；
这是说明尺度换算的示例，不是已经测得的数据集速度。下采样不保证平移不变；更深层也可能损失小目标细节。

## 当前实现的双仓机制（以代码为准）

在选中编码层，把单状态 U 改成 (U,M)。I 为本层卷积、归一化、增益后的电流；theta 是原阈值：

```text
beta  = exp(-dt_ms / tau_U)
alpha = exp(-dt_ms / tau_M)
kappa = kappa_max * tanh(kappa_raw)

Q_t     = tanh(I_t / theta)
M_t     = alpha * M_prev + (1-alpha) * Q_t
U_pre,t = beta * U_prev + I_t + (1-beta) * kappa * theta * M_prev
S_t     = H(U_pre,t - theta)
U_t     = U_pre,t - theta * S_t.detach()
```

- 保留 V1 的减阈值软复位与代理梯度。M 不随放电复位。
- tau_U 沿用 V1：初始 200 ms，范围 50–2000 ms；tau_M 按通道学习，初始 300 ms，范围 50–1000 ms。
- kappa 按通道学习，可正可负，kappa_max=0.5，kappa_raw 初始为零；新增参数不消耗随机数，原卷积初始化不变。
- 初始输出等同 V1，读取耦合先学习；耦合为零时 tau_M 的任务梯度为零不代表实现断梯度。
- 写入不使用本层被 M 增强的 U_pre，切断直接自我回写。多层部署时 I 仍可能含上游层历史，不能说它是完全无状态的新观测。
- `(1-beta)` 限制长 tau_U 对记忆偏置的静态放大。在本层零 U、始终零 I、有界 M 条件下，新增路径单独不足以触发脉冲。
  这不是整个网络稳定性/无误报的保证；持续背景仍可能被写入，旧 U 也可能有残留。
- M 从零初始化时有界，固定本层后续输入时初始 M 差异衰减；不要将这个局部性质扩展为任意完整网络轨迹的收缩证明。

状态由调用方显式传入/返回，不缓存在神经元内。TBPTT 片段内可反传，边界递归 detach U/M 并保留数值。
新独立序列清空状态；空窗照常更新。reset_each_window 忽略两仓全部历史。

## 正在运行的最终三组（不要退回早先 enc1/2/3 分组）

| 物理 GPU | 双仓层 | 配置 | 输出目录 |
|---|---|---|---|
| 1 | enc4 | configs/evisseg_stream_dual_enc4.yaml | log/stream_dual_enc4_seed37 |
| 2 | enc3+enc4 | configs/evisseg_stream_dual_enc34.yaml | log/stream_dual_enc34_seed37 |
| 3 | enc2+enc3+enc4 | configs/evisseg_stream_dual_enc234.yaml | log/stream_dual_enc234_seed37 |

三组 seed 37、50 epochs、TBPTT 16、每完整序列更新优化器、execution=layer、input_device=gpu、eval_chunk=32。
除 dual_memory 与输出目录外，三份配置与原 V1 相同。新增参数分别为 96/192/240。
每序列新增 M 元素数分别为 69,696 / 348,480 / 906,048；不含训练计算图开销。
各进程 CUDA_VISIBLE_DEVICES 指定物理卡，进程内使用 cuda:0。

训练日志：`run/dual_<enc4|enc34|enc234>_train.log`。
权重：各输出目录的 `best_val_iou_seed37.pt`、`last.pt`。续训用匹配配置加 `--resume`，不要将旧 V1 checkpoint 直接作为双仓 resume。
checkpoint 保存参数、优化器和配置，不保存序列内 U/M；仅在 epoch 结束时保存。

## 下次会话：收到结果后的工作顺序

1. 先确认每组训练是否完成、是否有报错/NaN、最佳 epoch、实际配置及执行路径。不要因为平均指标下降立即改机制。
2. 优先读取各组 `metrics.jsonl`、`summary.json`、`calibration.json`、`run_config_train.json`；
   以及 `eval_test_best_val_iou_seed37.json` 和原 V1 同口径结果。用户若只提供截图，可先分析可见数据，缺失信息明确说明。
3. 比较验证曲线、最佳验证指标、训练子集与验证差距、测试 IoU/ACC/Pd/Fa、早/中/晚片段指标、时延、峰值显存。
   沿用原评估阈值 0.9；若比较等虚警或调阈值，阈值选择使用验证集，避免在测试集挑最优超参。
4. 读取 tau 监控中的 memory_tau_mean/min/max、kappa_abs_mean/min/max，以及 layers 中
   memory_abs_mean/max/near_bound_frac。确认 M 是否被使用、有无饱和或时间常数顶边。
   不把平均 M 幅度小或平均发放率低当作背景污染已经解决的证据。
5. 做同权重 `--memory-ablation off`：只关新增 M 读取，普通 LIF 的 U 历史保留。
   结果另存 `eval_test_best_val_iou_seed37_memory_off.json`；和普通 carry 比较，回答已训练模型是否受益于 M。
   该消融不等于重新训练的 V1，也不是移除模块的速度测试，因为仍计算 M。
6. 普通 eval 本身输出 carry 与 reset_each_window；后者清空全网 U/M，不能用于单独归因 M。
7. 如果初步有效，再设计弱信号、目标离开后的背景误报残留、速度/位移分组、不同种子等机制验证。
   分组需真实标签/轨迹与明确口径，不能把预测身份、脉冲或匹配置信度直接当作真值。
8. 三组同时改变位置、层数和状态容量，不能只凭排名声称感受野是唯一原因。用证据决定下一轮最小改动。

当前只研究 EV-UAV 这轮实验。早先讨论过 EV-Flying，但没有在本轮代码或实验中验证其数据特点；不要凭聊天假定只有单目标、只有目标有长轨迹或背景都静止。

## 主要代码入口与检查

- `model/lif2d_stream.py`：StreamingDualMemoryLIF2d、递归 detach_states、state_tensors。
- `model/evspsegnet_stream.py`：dual_memory.layers 选择、forward/forward_chunk、M 监控、状态/运算计数。
- `train_stream_v1.py`：build_net、train_sequence、LayerMonitor、tau_statistics、checkpoint、memory_ablation。
- `tools/check_stream_execution.py`：包括 M 在内的前向/状态/梯度等价检查。
- `tools/stream_energy.py`：双仓额外算术与 tanh 单列，未折算进原卷积 MAC/SOP 能耗；不要当成完整能耗。
- `tests/test_stream_dual_memory.py`：三组位置、两种解码器、非零读取下的执行等价、训练更新、reset、初始化、梯度与完整入口检查。

本地检查：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv-stream-check\Scripts\python.exe -m unittest discover -s tests -p 'test_stream_*.py'
git diff --check
```

新会话应在有代码改动、失败或新的未解决问题时运行相关检查，避免无改动反复跑整套测试。
用户希望快速、直接完成授权任务，不需要反复确认已经定下来的方案。汇报区分代码事实、局部性质、假设与实验结果。

## 2026-10-14：亚阈值深度阻断实验

双仓位置实验已经完成，暂定使用 enc234。已有测试 IoU：V1历史82.258、enc2 82.953、enc3 82.880、enc4 82.419、dec2 83.393、dec3 80.260、enc34 79.833、enc234 85.190、enc234+dec32 82.500、enc234+dec321 81.244。enc234 的同权重 M-off 约85.213，说明测试收益不能归因于活动的 M 读取；论文不得写成双仓已被充分证明。

本轮新增“当前观测亚阈值残差”，完整说明与命令见 `README_STREAM_SUBTHRESHOLD_RESIDUAL.md`。核心定义：

```text
R1_t = (1 - stopgrad(S1_t)) * clamp(I1_t / theta, -1, 1)
```

其中 `I1_t` 是 enc1 卷积、归一化、逐通道增益后的当前驱动电流。R 不使用 U_pre，所以当前电流为零时严格为零，不会把旧膜电位重播为新证据。enc1 的原 LIF 放电与减阈值软复位完全不变。

- 纵向路径：enc2 接收 `S1+R1`，之后 enc2/3/4 仍为 enc234 双仓二值 LIF。
- 横向路径：原 enc1→dec1 二值跳连保留。连续 R 只在事件坐标采样后修正读出特征；门由 `up1(S6)` 的3×3局部深层活动产生，无深层局部活动时严格关闭。修正不写入 dec1 状态。
- 横向新增157个参数，投影零初始化；纵向无新参数。纵向使 enc2 卷积按连续MAC计，约占全网稠密卷积等价运算5.68%。
- 配置一：`configs/evisseg_stream_dual_enc234_residual_vertical.yaml`，输出 `log/stream_dual_enc234_residual_vertical_seed37`。
- 配置二：`configs/evisseg_stream_dual_enc234_residual_dualpath.yaml`，输出 `log/stream_dual_enc234_residual_dualpath_seed37`。
- 两张卡先训练纵向单独与纵向+横向组合；已有 enc234 作为基线。若组合优于纵向，下一轮补“横向单独”才能计算完整2×2交互项。
- eval 支持 `--subthreshold-ablation vertical_off|skip_off|all_off`；这是同权重干预，不等于重训基线。
- eval JSON 新增 `subthreshold_diagnostics`，包含残差密度/正负/幅值、事件门支持率、门均值和修正幅值。
- 新测试：`tests/test_stream_subthreshold_residual.py`，覆盖无输入不重播、纵向作用、深层门约束、all-off精确基线、逐窗/分块状态/输出/梯度等价、训练更新和运算量。
- 本轮完整 `test_stream_*.py` 共99项已通过，`git diff --check` 也通过。

### 亚阈值残差 test 结论

- 原始包：`Res/subthreshold_residual_test_20261014_173032.tar.gz`；解包与派生报告：`Res/subthreshold_residual_test_analysis_20261014_173032/`。
- enc234 基线 / 纵向 / 双路 carry IoU 为 85.190% / 83.534% / 85.734%。纵向比基线多 587 TP，但多 2236 FP，IoU 下降 1.656 pp；当前纵向方案应淘汰。
- 双路比基线多 1119 TP、多 814 FP，IoU 提升 0.544 pp。0–15 窗提升 2.712 pp，但平均/中位/P90 目标时延分别慢 7.20/2.17/2.17 ms，Pd 低 0.612 pp；不能声称解决唤醒滞后。
- 双路 `event_gate_support_frac=1.0`，说明当前深层硬支持掩码对所有被评估事件都开放，没有形成选择性门控。`dec1` always-on 比例升到 24.158%，carry-reset 差为18.858 pp，存在读出捷径与更强状态依赖风险。
- 两个残差版本的估算 event-driven MAC 比基线高约16.7%，主要因为连续值进入 enc2 使该卷积成为稠密 MAC；横向头仅多157参数。
- 下一步优先对同一 dualpath checkpoint 运行 `skip_off / vertical_off / all_off`，另做 memory-off；再根据消融决定是否重训横向单独版本。补 gate 分布及目标/背景条件诊断，并用真实目标首次出现对齐时延。至少补3个种子后再形成论文主张。

### 亚阈值第二轮（两卡，代码已完成，等待训练）

- 完整设计、训练、测试及消融命令见 `README_STREAM_SUBTHRESHOLD_ROUND2.md`。
- GPU 0：`configs/evisseg_stream_dual_enc234_residual_horizontal.yaml`，训练横向事件修正单独版本，输出 `log/stream_dual_enc234_residual_horizontal_seed37`。该组保留上一轮横向实现，仅用于补齐路径因果对照；已知上一轮深层硬支持率为100%，不能预设门控有效。
- GPU 1：`configs/evisseg_stream_dual_enc234_residual_competitive_vertical.yaml`，训练竞争式稀疏纵向版本，输出 `log/stream_dual_enc234_residual_competitive_vertical_seed37`。
- 新纵向机制把连续 `R1` 编码为超过逐通道可学习阈值且为3×3局部峰值的稀疏双极码，再乘逐通道有界耦合 `g=0.5*tanh(raw)`。12个阈值+12个耦合，共新增24参数。耦合严格零初始化，初始前向等同enc234；推理时耦合可折叠到enc2卷积核。
- 日志/评估新增 `vertical_signal_*` 稀疏率与正负幅值，以及 `subthreshold_parameters` 的阈值、耦合范围和正负通道比例。竞争式运算统计按实测稀疏码密度计SOP，池化/比较/访存仍未折算。
- 完整 `test_stream_*.py` 102项通过，`git diff --check`通过。不要重新运行旧的连续纵向配置；第二轮使用上述两个新配置。
