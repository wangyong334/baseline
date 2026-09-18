# CLAUDE.md — EV-UAV 流式 SNN 研究

> 记录这个项目做过的事、当前进展和可继续的方向，用于持续迭代创新和写论文。
> 每完成一次迭代，更新第 1、3、4、7 节。最后更新：2026-09-18，分支 `research/windowed-streaming-snn-v1` @ `5b316a9`。

## 1. 概况与进展

**目标**：以 ICCV 2025 论文 EV-SpSegNet（事件相机反无人机微小目标分割）为基线，做因果、低延迟、脉冲驱动的流式 SNN，形成论文。

**已完成（按时间）**
1. 基线工程（09-05 ~ 09-14）：双卡训练、线性学习率、基线 v2 实验协议
2. 早期探索 SNN v0 / 静态量化：只把基线里两处激活换成发放率 LIF，不是完整 SNN，已归档在 `main` 分支
3. 流式 SNN V1（v1-1，09-16）：test IoU 0.8363（3 种子），参数 0.105M，50 ms 块因果
4. 结果核验（v1-2，09-16）：确认 V1 没有评估错误或信息泄漏；发现原基线代码的时间下采样 bug
5. 基线 K5 修复（09-16）：基线 test IoU 从 0.6188 提到 0.8011，V1 的领先缩小到 +0.035
6. 解码器全脉冲化（方案 C 电流合并，09-17）：对已训练 V1 严格等价换算，指标不变，每窗理论能耗 2.24 → 0.60 mJ
7. 方案 C 从零训练（09-17/18，三种子）：test 0.8401 ± 0.024，与换算版 0.8363 持平（此前单种子看到的差距是噪声）
8. 训练加速 A（09-17）：GPU 构造输入 + 片段内逐层时间并行，数学等价，每轮 545 → 73 s（7.5 倍），50 轮 7.6 h → 1.0 h
9. 批量实验（09-18）：ReLU 对照、方案 C 从零、TBPTT 32/64 各自三种子或单种子跑完并评估；基线与 SNN 同口径能耗统计完成

**正在跑（09-18 下午，结果待回收）**
- `stream_v1_graded_fast_seed{37,38,39}`：graded 对照（有状态 + 实数发放）。ReLU 对照同时去掉了脉冲与跨窗状态，这一组只去掉二值化，
  用来判断 ReLU 相对 LIF 少的那 0.10 检出率来自哪一项。graded 接近 LIF -> 二值化几乎无代价，SNN 主线成立；graded 明显更好 -> 二值化有代价，要按权衡来写
- `stream_v1_fast_seed{40,41}`：给"加速路径是否影响精度"补种子
- 逐事件概率导出（`log/verify/{v1,relu}_s3?_test`、`baseline_k5_s37_test`）+ `tools/sweep_threshold.py` 扫阈值：
  0.9 这个固定阈值下 LIF 与 ReLU 落在不同工作点，必须在等虚警率下重新比较
- 结果回收后要做：更新第 7 节表格、决定论文主线写法

**待办**
- LIF 在序列开头 0.8 s 明显吃亏（分段 IoU 前 16 窗：LIF 0.73–0.77，ReLU 0.83），状态预热有代价；可试可学习的初始膜电位
- 膜电位极端化：|U| 超过 10 倍阈值的元素在 dec3 占 55–63%，输出层 14–19% 的位置几乎每窗发放；可能与"记忆可被替代"有关
- 加速路径的 test 落差（-0.029，val 只差 0.008）等 s40/s41 确认；在此之前论文主结果用原实现的三种子

**主要发现**
- 原基线代码三次时间下采样都漏掉 `t%4==2` 的时间片，这部分事件 IoU 只有 0.27–0.32；修复后基线大幅提升
- V1 原解码器可以精确改写为两路脉冲电流之和，所有突触输入变成脉冲，能耗降 73%，精度不变
- 跨窗记忆可以被替代：从一开始用 reset 训练，test 只比 carry 低 0.016（种子间标准差 0.013）；但 carry 训练出的模型推理时去掉状态会明显掉点（V1 s37：val 0.8756 → 0.6964，test 0.8226 → 0.6892），说明网络学会了使用状态，只是状态没有带来不可替代的增益
- V1 的脉冲层普遍工作在膜电位极端区：原 V1 s37 末轮（val）|U_pre| 超过 10 倍阈值的元素占比 enc3 34%、enc4 33%、dec3 55%、dec2 31%，输出层 dec1 有 19% 的位置几乎每窗发放（按位置统计，正负号未区分）。神经元多处在"一直发放"或"深度抑制"两端，这可能也是状态可被替代的原因之一（推测）
- 单窗耗时里预处理约 13 ms（64%），网络约 7 ms；剩余理论能耗约 93% 来自 enc1 的实数输入
- V1 训练一轮约 540 s：训练约 410 s（25.7 ms/窗）+ 评估约 125 s（23%），GPU 利用率约 10%
- 训练加速 A 计时（seed37 权重，1 号卡）：numpy 逐窗预处理 10.2 ms/窗 → GPU 片段构造 0.12 ms/窗；训练 26.9 → 4.1 ms/窗（6.5 倍，峰值显存 1.2 → 1.7 GiB）；带监控的验证 19.8 → 1.8 ms/窗。估算一轮 566 → 约 75 s（50 轮约 1 h），原实现的估算比实测高 5–19%。网络前反向改成 16x16 小输入后耗时只下降 4–20%，说明大部分耗时不随输入尺寸增长（不能据此精确拆分调度与 GPU 计算）
- 加速 A 与原实现等价：输入逐位相同；float64 下 logits、u_pre、膜电位误差为 0，梯度 5e-15，一次更新后参数 6e-15，零脉冲翻转；关闭 TF32 的 float32 前向也逐位相同。默认 TF32 下整条序列概率最大差 0.04–0.07（舍入沿 160 窗累积），但 0.9 阈值判定没有变化，整个验证集 IoU 均为 0.8756
- 非零输入元素占比（12 通道 × 264×352 画布）：test 0.093%，val_000 0.050%；enc1 按事件驱动口径计时 MAC 约为稠密口径的千分之一
- TBPTT 16 窗意味着梯度只回传 800 ms，而 τ 上限是 2000 ms，这可能是跨窗记忆弱的原因之一（tau1000 变体更好也与此一致）
- 加速重训 seed37 学到的 τ 各层均值 162–266 ms、最大 297 ms，都停在初值 200 ms 附近，没有变长
- **ReLU 对照（三种子）**：test IoU 0.8435 ± 0.005，反而高于 LIF（原实现 0.8363、加速 0.8072）；但 Pd 只有 0.8517（LIF 0.948–0.950），Fa 6.5e-6（LIF 1.0–1.2e-5），即 ReLU 更保守：漏掉更多目标、虚警更少。能耗 781 mJ/8s，是方案 C（7.85 mJ）的 100 倍。脉冲的价值体现在目标级检出率与能耗，而不是逐事件 IoU
- **ReLU 与 LIF 在 val 和 test 上排序相反**：ReLU val 0.8353（最低）但 test 0.8435（最高）；LIF val-test 落差 +0.025~+0.047，ReLU 为 -0.008。两个划分各 24 条序列，排序不稳定，比较要谨慎
- **加长 TBPTT 无效**：32 窗 test 0.8320、64 窗 0.7795（明显更差）；学到的 τ 几乎不变（16 窗 152–309 ms，64 窗 183–320 ms）；但对状态的依赖大幅上升（reset 评估 IoU：16 窗 0.64、32 窗 0.56、64 窗 0.18）。说明跨窗记忆弱不是被 800 ms 的梯度截断卡住的，这个猜想被排除
- **同口径能耗（每 8 秒，test）**：基线 K5 稀疏 ANN 174.1 mJ（99.8% 来自稀疏卷积，解码器 conv_up_t3/t4/m3 占 59%）；方案 C 事件驱动口径 7.85 mJ（22 倍优势）、稠密口径 96.4 mJ（1.8 倍）；原解码器 V1 270–358 mJ，比基线还贵；ReLU 781 mJ。膜电位衰减乘法未计入，上界约 2.9 mJ/8s
- 加速重训 s37 复现了上面"状态"与"极端区"两条：val carry 0.8649、reset 0.7184；|U_pre| 超过 10 倍阈值的占比 enc3 33%、enc4 32%、dec3 63%、dec2 30%、dec1 14%，dec1 有 17.5% 的位置几乎每窗发放；同一权重 reset 评估时只有 4–7%

## 2. 基线：EV-SpSegNet

### 2.1 原论文

Chen et al., *Event-based Tiny Object Detection: A Benchmark Dataset and Baseline*, ICCV 2025（arXiv 2506.23575，PDF 在 `References/`）。

- **数据集 EV-UAV**：DAVIS346（346×260），147 条序列（train 99 / val 24 / test 24），每条 8 秒，逐事件标注。目标平均 6.8×5.4 像素，全部小于 32²，覆盖多种光照和场景。标注方法：按时间窗累积成帧画框，沿时间扩成 3D 框，框内事件即目标事件。NPZ 包含 `ev`、`evs_norm`（含 label 与 target_id）、`ev_loc`（整数 x, y, t）。
- **核心观察**：运动小目标在 (x, y, t) 点云中形成连续细长曲线，背景和噪声是离散的点或面，所以靠时空连续性而不是外观来判别。
- **网络**：体素化（1 px × 1 px × 1 ms，特征是事件归一化坐标与极性的均值）→ 3D 稀疏 U-Net（spconv）。核心模块 GDSCA = 逐点稀疏卷积 → 分组膨胀稀疏卷积（膨胀率 1–4）→ 稀疏 SE → Patch Attention。编码器下采样 stride [2,2,4]；逐体素 Sigmoid 输出，再映射回事件。
- **STC Loss**：用预测在 k×k×τ 邻域内的和算出时空相关性权重，加权 BCE，保留有连续轨迹支撑的事件、压制孤立噪声。论文 k=3、τ=5、γ=2，没有类别平衡。
- **评估**（`utils/eval.py`，阈值 0.9）：IoU 为全部序列拼接后的正类 IoU；ACC 实际是正类召回；Pd 按 50 ms 帧逐目标统计；Fa 为虚警 8 连通域数除以（帧数 × 像素数）。
- **论文结果**：IoU 55.18，ACC 65.02，Pd 77.53，Fa 1.63e-4，参数 4.0M，处理 8 秒数据 35.9 ms。
- **作者自陈局限**：静止或慢速目标不产生事件会漏检；未来方向是多模态。

### 2.2 我们的修订

| 提交 | 内容 |
|---|---|
| `64290b8` | 双卡模型并行训练 `train_mp.py`、`model/evspsegnet_mp.py` |
| `8768fbf` | 线性学习率 1e-3 → 1e-4 |
| `9e7db5f` / `0f1a424` | **基线 v2 协议**：每轮验证、种子可配置、末 5 轮均值统计（`utils/aggregate.py`）、STC γ 参数化；两组学习率对照后采用仓库学习率（`configs/evisseg_evuav_baseline_v2_repolr.yaml`） |
| `7081aa6` | **K5 修复**：conv2/3/4 与 inv_conv2/3/4 的时间核 3 → `[3,3,5]`。当前 `model/evspsegnet.py` 已是 K5，旧 K3 权重不兼容 |

**K5 修复的依据**：下采样 `kernel=3, stride=4, padding=1` 时，`t%4==2` 的输入位置不会被任何输出引用。验证过程：按 `t%4` 分组，原基线在余数 2 上明显塌陷，而 V1（没有 mod-4 结构）四组持平，3 种子 × val/test 全部复现；作者自己在 `basemodel.py` 的 `Downsample_block` 里写的就是 `[3,3,5]`，但没有被调用；改成 K5 后重训，四组持平，整体 IoU 提升 18 个点。核 5 同时加大了感受野和参数量，所以提升不能全部归因于补洞（`[3,3,4]` 可以分离，未做）。

### 2.3 基线仍存在的问题（潜在切入点）

- **Patch Attention 实际没有跨 patch 交互**：`unsqueeze(0)` 后序列长度为 1，注意力退化为逐体素线性变换（池化、反池化和残差仍在）。未修，修了可能进一步抬高基线。
- **参数量与论文不符**：发布配置 `width: 12` 下，K3 实际约 94 万参数，K5 约 108 万，论文写的是 4.0M。
- **代码与论文不一致**：学习率（代码 1e-3，论文 1e-2）、STC γ（代码 1，论文 2）。
- **`utils/eval.py` 的口径问题**：Pd/Fa 分帧用严格不等号，排除约 2% 的边界事件；帧数少算 1。为了与论文可比，没有修改。

## 3. 流式 SNN V1

### 3.1 设计（v1-3）

- **数据**：8 秒切成 160 个 50 ms 窗，每窗构造 12 通道稠密计数图（整窗正/负极性 + 5 个 10 ms bin × 正/负），`clip(log1p(C)/q99, 0, 3)` 归一化，q99 来自训练集。
- **神经元**：LIF，`U_pre = β·U + I`，超过阈值 1.0 发放并软复位（减阈值）；代理梯度 `1/(1+|v|)²`；τ 用 sigmoid 约束在 [50, 2000] ms，逐通道可学（初值 200 ms）。每窗只更新一次，膜电位跨窗保持。训练前逐层校准增益，使校准数据上的正电流 q99 对齐阈值。
- **网络**：7 层 2D 脉冲 U-Net（通道 12/24/48/48，分辨率 264×352 → 33×44），每层 `Conv3×3 → 增益 → LIF`，无 bias、无 BN。跳连传脉冲，读出取最后一层复位前膜电位 U_pre，与事件的极性 p、窗内时间 t_local 一起经 MLP 逐事件输出。参数 105,345。
- **损失与训练**：`BCEWithLogits`，pos_weight = min(负/正, 30)（没有用 STC Loss）；TBPTT 每 16 窗一段，一条序列一次更新；50 轮，学习率 1e-3 → 1e-4。
- **评估**：主指标调用原 `utils/eval.py`；同一权重分别以 carry（保持状态）和 reset_each_window（每窗清零）评估；另有逐窗 IoU、首次检出延迟、各层发放率、理论 MAC/SOP 运算量。

### 3.2 迭代记录

| 版本 | 改动 | 原因 | 结果 |
|---|---|---|---|
| v1-1 `8ac25ce` | 初版流式 SNN | 方案一"因果流式脉冲分割"：原方法全是离线的，要等满 8 秒 | test IoU 0.8363；reset 变体 0.8201；tau1000 变体 0.8498 |
| v1-2 `9bb2a0e` | 核验与诊断工具（`verify_predictions`、`diagnose_gap`） | V1 比基线高 21 个点，结果反常 | 数据对齐、指标实现、读出消融均无问题；定位到基线 bug |
| 基线 K5 `7081aa6` | 修复基线时间下采样 | 同上 | 基线 0.8011，V1 在 test 上领先 +0.035，val 上 +0.013 |
| 10 层 LIF（已删除） | 三个 ConvT 后各加增益 + LIF | 原解码器 ConvT 输出实数，占理论能耗 99% | 状态增加 195 万，冒烟时新层几乎不发放，需要重训；放弃，无训练结果 |
| **方案 C** `6cc2d31` | 电流合并解码器 `I = ConvT4×4_s2_p1(S_deep) + Conv3×3(S_skip)`，以及权重换算工具 | 同上；另外比较过最近邻上采样（需重训，能耗不更省） | 与原 V1 数学等价，可直接换算：3 种子指标差 ≤ 2e-4，每窗能耗 2.24 → 0.60 mJ，状态数不变 |
| 方案 C 从零训练 | 用合并形式直接训练 | 看这种参数化直接训练是否更好（函数空间更大） | **三种子 test 0.8401 ± 0.024，与换算版 0.8363 持平**（s37 单种子的 -0.011 是噪声）。以下为 s37 的诊断，仍有参考价值：val 0.8588 / test 0.8120。前 12 轮收敛更快（第 8 轮达 0.75，原 V1 第 12 轮），第 25 轮后停在约 0.85，25 轮中 23 轮低于原 V1（末 5 轮 0.8535 对 0.8638）。召回与 Pd 略高、Fa 高约 18%，IoU 被虚警拉低；训练子集 IoU 与原 V1 相同（末 5 轮 0.950 对 0.949），差距出在泛化；深层和输出层更饱和（enc4 发放率 0.19 对 0.04，dec3 膜电位过大 73% 对 55%，dec1 几乎每窗发放的位置 36% 对 19%），输出层过度活跃与虚警偏多一致 |
| 训练加速 A `6838d49` `5b316a9` | GPU 构造输入（事件常驻显存、查表归一化，与 numpy 输入逐位相同）；片段内逐层时间并行 `forward_chunk`（卷积按 T 个窗口批量算，只有膜电位按时间递推）；训练子集评估可隔轮 | 训练 7–9 h、GPU 利用率 10%，实验排不过来 | 等价核对通过（float64 零误差；默认 TF32 下整个验证集 IoU 同为 0.8756）。每轮 545 → 73 s（7.5 倍），50 轮 7.6 h → 1.0 h，显存 1.2 → 1.7 GiB。三种子 test 0.8072，比原实现低 0.029（val 只差 0.008），疑似噪声，补种子中。`--mode eval` 仍走原实现 |
| ReLU 对照 | `--neuron relu`，无状态、实数输出 | 决定论文以 SNN 还是轻量因果流式为主线 | 三种子 test 0.8435 ± 0.005（高于 LIF），但 Pd 只有 0.8517（LIF 0.948）、Fa 更低、能耗 781 mJ/8s（方案 C 的 100 倍）。脉冲的价值在检出率与能耗，不在逐事件 IoU |
| TBPTT 32 / 64 | `--tbptt-k`，梯度回传 1.6 s / 3.2 s | 检验"跨窗记忆弱是否因为梯度只回传 800 ms" | 32 窗 test 0.8320、64 窗 0.7795（更差）；τ 几乎不变；对状态依赖上升（reset IoU 0.64 → 0.56 → 0.18）。猜想被排除 |
| graded 对照 | `--neuron graded`，有状态 + 实数发放（幅值 = 膜电位，阈值处等于 1） | ReLU 同时去掉了脉冲与状态，需要分离两者 | 进行中（三种子） |

补充分析：
- **稀疏卷积**：输入像素占用仅 0.5%，但卷积扩散加下采样后深层接近稠密；GPU 上收益小，所以没有采用。
- **是否真 SNN**：V1 属于直接训练的 SNN（代理梯度 + BPTT + 有状态 LIF）。第一层实数输入、末层读膜电位都是常见做法；原解码器的实数输入已由方案 C 解决。

### 3.3 模块状态

| 模块 | 当前 | 可继续迭代 |
|---|---|---|
| 输入表征 | 12 通道计数图，50 ms，5 bin | enc1 实数输入是剩余能耗主项；窗长和 bin 数没做过消融；预处理可挪到 GPU |
| 神经元 / 时序 | LIF，可学 τ | 跨窗记忆几乎没起作用 |
| 编码器 | 4 层脉冲卷积 | — |
| 解码器 | 全脉冲（方案 C，换算） | 从零训练不优于换算（seed37 低约 0.01），原因未查明 |
| 读出 | U_pre + p + t_local 的 MLP | 分析显示不是瓶颈 |
| 损失 | BCE + pos_weight | 可引入因果版 STC 时空先验 |
| 脉冲必要性 | ReLU 对照已完成 | graded 对照进行中；还需按等虚警率比较（阈值扫描） |

## 4. 创新方向（持续更新）

| 方向 | 状态 | 说明 |
|---|---|---|
| 因果流式分割（时间轴 → SNN 时间步） | ✅ V1 | 50 ms 块因果，逐事件输出 |
| 解码器全脉冲（电流合并） | ✅ 方案 C | 严格等价，能耗降 73% |
| 发现并修复基线实现缺陷 | ✅ | 可作为复现性发现 |
| 脉冲必要性（同结构 ReLU 对照） | ✅ 有结论但需细化 | ReLU 的 test IoU 更高、Pd 低 0.10、能耗高 100 倍；论文主线应是"低能耗 + 高检出"，不是 IoU。还需"有状态不发放"对照分离脉冲与记忆 |
| 强化跨窗时序记忆 | 部分排除 | 更长 TBPTT（32/64）无效，已排除；剩下：循环连接、读出引入历史、按事件时间衰减、抑制膜电位极端化 |
| 输入层脉冲化 / enc1 能耗 | ✅ 口径已定 | 事件驱动口径下 enc1 只剩 0.09% 的非零输入，方案 C 每 8 s 从 96.4 降到 7.85 mJ；若要硬件严格成立，仍可把输入编码成脉冲 |
| 因果版 STC 时空相关性先验 | 候选 | 把论文的时空连续性思想做成跨窗形式 |
| 低延迟工程 | 待做 | 预处理挪 GPU，规范计时 |
| 窗长 / 时间分箱设计 | 候选 | 50 ms / 5 bin 尚未消融，可考虑自适应窗 |
| 静止 / 慢速目标 | 远期 | 论文自陈局限，多模态或记忆机制 |
| 公平比较 | 需要 | 参数量不对齐（0.105M 对 0.94M）；基线 Patch Attention 未修 |
| 同口径能效对比 | ✅ 已完成 | 160 步是 160 段新数据，应按每 8 秒比较；基线稀疏卷积只在事件上计算，SNN 的 enc1 在全图上计 MAC。`baseline_energy.py` 测基线实际连接数，`stream_energy.py` 给出 SNN 稠密 / 事件驱动两种口径；决定论文能否主张节能 |
| 可并行训练的脉冲神经元 | 候选 | 训练时整段并行、推理时递推（去复位或线性递推，参考 PSN / 脉冲 SSM）；训练变快后可做完整 160 窗 BPTT，同时针对跨窗记忆弱 |
| 膜电位 / 发放率约束 | 候选 | V1 大量膜电位超过 10 倍阈值、输出层部分位置几乎每窗发放；方案 C 从零训练更严重且泛化更差。可试发放率或膜电位幅度正则、膜电位裁剪、自适应阈值；可能同时改善泛化、降低 SOP、让状态携带更多历史信息 |

## 5. 代码地图

**原作者代码**：`train.py` / `test.py`（原入口）、`configs/configs.py` + `evisseg_evuav.yaml`、`dataset/ev_uav.py` + `basedataset.py`（NPZ 读取、体素化）、`model/evspsegnet.py`（主网络，已含 K5）、`model/basemodel.py`（GDBlock 等）、`utils/stcloss.py`、`utils/eval.py`（评估，可比性基准）、`lib/hais_ops/`（CUDA 体素化扩展）。

**基线工程**：`train_mp.py`（双卡入口，环境变量 `EVUAV_MODE` / `EVUAV_MP_SPLIT` / `EVUAV_SEED` / `EVUAV_SAVE_ROOT`）、`model/evspsegnet_mp.py`、`configs/evisseg_evuav_mp*.yaml` 与 `baseline_v2*.yaml`、`utils/aggregate.py`（多种子汇总）、`README_MODEL_PARALLEL.md`、`README_LINEAR_EXPERIMENT.md`。

**流式 SNN V1**

| 文件 | 作用 |
|---|---|
| `train_stream_v1.py` | 入口 `--mode smoke/overfit/train/eval`；TBPTT 训练、逐窗推理、评估、发放率监控、计时 |
| `dataset/stream_windows.py` | 纯 numpy：读 NPZ、切窗、12 通道计数、归一化、按下标回填、训练集统计 |
| `dataset/ev_uav_stream.py` | 按序列加载、窗口转张量 |
| `dataset/stream_source.py` | 窗口数据来源：numpy 逐窗（原实现）或设备端按片段构造（`input_device`） |
| `model/lif2d_stream.py` | LIF、代理梯度、逐通道增益；两个对照神经元：graded（有状态+实数）、relu（无状态+实数） |
| `model/evspsegnet_stream.py` | 网络（`merged_decoder` 开关；逐窗 `forward` / 片段逐层 `forward_chunk`）、权重合并换算、增益校准、运算量估计 |
| `utils/stream_common.py`、`utils/stream_metrics.py` | 配置、学习率、TBPTT 分段、校准数学、健康检查；逐窗与延迟指标 |
| `configs/evisseg_stream_v1.yaml`、`evisseg_stream_merged_decoder.yaml` | V1、方案 C 配置（后者只多 `merged_decoder: true` 与保存目录） |
| `tests/test_stream_*.py` | 单元测试 |
| `README_STREAM_V1.md`、`README_STREAM_MERGED_DECODER.md` | 运行说明、方案 C 原理 |

**工具**：`tools/stream_train_stats.py`（训练集统计）、`dump_baseline_predictions.py`（导出基线逐事件预测并打印原指标）、`verify_predictions.py`（多方法对齐与指标核验）、`diagnose_gap.py`（按 t%4 分组、读出上限、eval 口径量化）、`convert_to_merged_decoder.py`（V1 → 方案 C 换算，含等价核对）、`check_stream_execution.py`（加速选项与原实现的等价核对）、`bench_stream_speed.py`（训练/评估计时分解）、`baseline_energy.py`（基线稀疏卷积实际运算量与能耗）、`stream_energy.py`（SNN 每 8 秒能耗的两种口径与基线对比）、`sweep_threshold.py`（扫判定阈值，按等虚警率比较不同模型）。

**其他**：`outputs/`（V1 示意图、合并等价性独立验证）、`References/`（论文 PDF）、V1 架构讲解页 <https://claude.ai/artifact/TYCEZjySwtjWbeFnRR9xWS>（对应原版解码器）。

**分支**：`research/windowed-streaming-snn-v1`（当前）；`main` = `archive/activation-v0`（SNN v0、静态量化）。

## 6. 服务器与命令习惯

- 服务器 `/media/stephen/nvme0n1/wy_data/EV-UAV`，conda 环境 `evuav`，Python 3.8 + torch 1.9.1，4 × RTX 4090；数据在 `/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset/{train,val,test}`。
- **代码同步用 SFTP**（本地改完上传），服务器上不用 git；SFTP 不会删除服务器上的旧文件，删除需要单独 `rm`。
- 长任务用 `nohup ... > run/<名称>.log 2>&1 &`，再用 `tail -f` 查看；输出放 `log/<名称>/`，逐事件预测 dump 放 `log/verify/`。
- 代码需兼容 Python 3.8 / torch 1.9（例如 1.9 的 `torch.testing.assert_close` 默认比较 stride）。本地没有 GPU 和数据，本地测试用 `.venv-stream-check/Scripts/python.exe -m unittest discover -s tests -p "test_stream*.py"`。

### 流式 V1 / 方案 C：训练 → 验证集 → 测试集

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV && conda activate evuav
CFG=configs/evisseg_stream_merged_decoder.yaml   # 原 V1：configs/evisseg_stream_v1.yaml
RUN=stream_merged_decoder; S=37; GPU=0

python -m unittest discover -s tests -p "test_stream*.py" 2>&1 | tail -3

# 冒烟
CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --config $CFG --mode smoke \
    --save-root log/${RUN}_smoke > run/${RUN}_smoke.log 2>&1

# 训练（每轮在 val 上评估，保存 best_val_iou_seed$S.pt；中断后加 --resume 续训）
nohup env CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --config $CFG --mode train \
    --seed $S --save-root log/${RUN}_seed$S > run/${RUN}_seed$S.log 2>&1 &
tail -f run/${RUN}_seed$S.log

# 验证集、测试集（carry 与 reset 各评一遍；可加 --dump-dir log/verify/${RUN}_s${S}_<split>）
for split in val test; do
  CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --mode eval --split $split \
      --checkpoint log/${RUN}_seed$S/best_val_iou_seed$S.pt
done

# 已训练的 V1 换算成方案 C（输出 *_merged.pt，之后同样用 --mode eval 评估）
CUDA_VISIBLE_DEVICES=3 python tools/convert_to_merged_decoder.py --device cuda:0 \
    --checkpoint log/stream_v1_seed$S/best_val_iou_seed$S.pt

# 汇总所有评估结果
python - <<'EOF'
import json, glob
E = lambda op: (op["mac"] * 4.6e-12 + op["sop"] * 0.9e-12) * 1e3
for f in sorted(glob.glob("log/*/eval_*_best_val_iou_seed*.json")):
    d = json.load(open(f)); c, r = d["carry"], d["reset_each_window"]
    print("%-72s IoU %.4f ACC %.4f Pd %.4f Fa %.2e | reset %.4f | %.3f mJ" % (
        f, c["iou"], c["acc"], c["pd"], c["fa"], r["iou"], E(c["operations"])))
EOF
```

### 训练加速与能耗：核对 → 计时 → 能耗 → 重训

```bash
CK=log/stream_v1_seed37/best_val_iou_seed37.pt; GPU=1
# 等价核对（输入逐位、float64 逻辑、float32 数值、整个验证集 IoU），最后一行应为 EXECUTION CHECK PASSED
CUDA_VISIBLE_DEVICES=$GPU python tools/check_stream_execution.py --checkpoint $CK --split val --full-split     > run/check_stream_execution_s37.log 2>&1
# 计时分解（四种组合 × 训练/评估，前向/反向，小输入对照只看规模变化；加速以重训日志 epoch_seconds 为准）
CUDA_VISIBLE_DEVICES=$GPU python tools/bench_stream_speed.py --checkpoint $CK     --reference-metrics log/stream_v1_seed37/metrics.jsonl > run/bench_stream_speed_s37.log 2>&1
# 能耗：基线实际运算量（GPU）→ SNN 两种口径并对比（CPU）
CUDA_VISIBLE_DEVICES=$GPU python tools/baseline_energy.py --config configs/evisseg_evuav_baseline_v2_repolr.yaml     --checkpoint log/baseline_k5_repolr_seed37/best_iou_seed37.pt --split test > run/baseline_energy_s37.log 2>&1
python tools/stream_energy.py --config configs/evisseg_stream_v1.yaml --split test     --eval-json log/stream_v1_seed37/eval_test_best_val_iou_seed37.json log/stream_v1_seed37/eval_test_best_val_iou_seed37_merged.json     --baseline-json log/baseline_k5_repolr_seed37/energy_test_best_iou_seed37.json
# 用加速选项重训（与 log/stream_v1_seed37 的曲线和耗时对照）
nohup env CUDA_VISIBLE_DEVICES=$GPU python train_stream_v1.py --config configs/evisseg_stream_v1.yaml --mode train     --seed 37 --execution layer --input-device gpu --train-subset-every 5     --save-root log/stream_v1_fast_seed37 > run/stream_v1_fast_seed37.log 2>&1 &
```

### 对照实验与阈值扫描（09-18）

```bash
V1=configs/evisseg_stream_v1.yaml
launch() {   # 用法：launch <GPU> <运行名> <种子> [其他参数...]
  local gpu=$1 name=$2 seed=$3; shift 3
  nohup env CUDA_VISIBLE_DEVICES=$gpu python train_stream_v1.py --mode train --seed $seed       --execution layer --input-device gpu --train-subset-every 5 "$@"       --save-root log/${name}_seed$seed > run/${name}_seed$seed.log 2>&1 &
  sleep 5
}
launch 0 stream_v1_graded_fast 37 --config $V1 --neuron graded      # 有状态 + 实数发放
launch 3 stream_v1_fast        40 --config $V1                      # 加速路径补种子

# 导出逐事件概率（阈值扫描用），再按等虚警率比较
for S in 37 38 39; do
  CUDA_VISIBLE_DEVICES=3 python train_stream_v1.py --mode eval --split test --overwrite       --checkpoint log/stream_v1_seed$S/best_val_iou_seed$S.pt --dump-dir log/verify/v1_s${S}_test
done
python tools/sweep_threshold.py --target-fa 1e-5 6.5e-6     --dump-dir log/verify/v1_s3?_test log/verify/relu_s3?_test log/verify/baseline_k5_s37_test     --out log/energy/threshold_sweep_test.json
```

### 基线 K5（双卡）：训练 → 验证集 → 测试集

```bash
S=37
nohup env CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  EVUAV_MODE=train EVUAV_MP_SPLIT=2 EVUAV_SEED=$S EVUAV_SAVE_ROOT=log/baseline_k5_repolr_seed$S \
  python -u train_mp.py --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
  > run/baseline_k5_repolr_seed$S.log 2>&1 < /dev/null &
tail -n 30 -f run/baseline_k5_repolr_seed$S.log

# 验证集、测试集完整指标（只用原仓库代码，同时导出逐事件预测）
for split in val test; do
  CUDA_VISIBLE_DEVICES=0 python tools/dump_baseline_predictions.py \
      --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
      --checkpoint log/baseline_k5_repolr_seed$S/best_iou_seed$S.pt \
      --split $split --out-dir log/verify/baseline_k5_s${S}_$split
done
```

## 7. 指标记录

阈值 0.9；ACC 为正类召回；流式模型为 carry 模式；"a / b / c"对应种子 37 / 38 / 39。

### 测试集

| 版本 | 参数 | IoU | ACC | Pd | Fa |
|---|---|---|---|---|---|
| 论文 Tab.2 | 4.0M（论文值） | 0.5518 | 0.6502 | 0.7753 | 1.63e-4 |
| 官方权重 | — | 0.5844 | 0.6785 | 0.7848 | 8.49e-6 |
| 基线 v2（K3，有 bug） | ~0.94M | 0.5894 / 0.6243 / 0.6428 → **0.6188** | 0.694 | 0.806 | 1.79e-5 |
| 基线 K5 | ~1.08M | 0.8143 / 0.7946 / 0.7944 → **0.8011** | 0.8307 / 0.8085 / 0.8060 | — | — |
| **流式 V1** | 105,345 | 0.8226 / 0.8372 / 0.8492 → **0.8363** | 0.9183 / 0.9085 / 0.9190 | 0.9459 / 0.9478 / 0.9505 | 1.15e-5 / 9.75e-6 / 1.01e-5 |
| V1 reset 训练（s37） | 105,345 | 0.8201 | 0.8751 | 0.8705 | 5.62e-6 |
| V1 tau1000（s37） | 105,345 | 0.8498 | 0.9240 | 0.9503 | 1.05e-5 |
| **V1 换算为方案 C** | 训练 105,345 / 存储 123,057 | 0.8226 / 0.8372 / 0.8493 | 同 V1 | 0.9459 / 0.9476 / 0.9505 | 同 V1 |
| 方案 C 从零训练 | 123,057 | 0.8120 / 0.8539 / 0.8546 → **0.8401** | 0.9080 | 0.9396 | 9.72e-6 |
| V1 加速重训 | 105,345 | 0.8045 / 0.7858 / 0.8313 → **0.8072** | 0.9001 | 0.9499 | 1.23e-5 |
| **ReLU 对照（无状态）** | 105,345 | 0.8477 / 0.8382 / 0.8446 → **0.8435** | 0.9008 | 0.8517 | 6.49e-6 |
| V1 TBPTT 32（s37） | 105,345 | 0.8320 | 0.9128 | 0.9457 | 1.20e-5 |
| V1 TBPTT 64（s37） | 105,345 | 0.7795 | 0.8354 | 0.9040 | 1.05e-5 |

### 验证集

| 版本 | IoU | ACC | Pd | Fa |
|---|---|---|---|---|
| 基线 K5 | 0.8592 / 0.8479 / 0.8386 | 0.8780 / 0.8700 / 0.8549 | — | — |
| **流式 V1** | 0.8756 / 0.8544 / 0.8551 → 0.8617 | 0.9221 / 0.8968 / 0.9128 | 0.9246 / 0.9045 / 0.9238 | 7.29e-6 / 6.92e-6 / 7.80e-6 |
| V1 reset 训练（s37） | 0.8152 | 0.8897 | 0.8362 | 9.38e-6 |
| V1 tau1000（s37） | 0.8667 | 0.9181 | 0.9217 | 7.86e-6 |
| V1 加速重训（s37，第 33 轮） | 0.8649 | 0.9271 | 0.9273 | 8.88e-6 |
| V1 换算为方案 C | 同 V1 | 同 V1 | 同 V1 | 同 V1 |
| 方案 C 从零训练 | 0.8588 / 0.8524 / 0.8641 → 0.8584 | 0.9080 | — | — |
| V1 加速重训 | 0.8649 / 0.8515 / 0.8446 → 0.8537 | — | — | — |
| ReLU 对照（无状态） | 0.8393 / 0.8272 / 0.8393 → 0.8353 | — | — | — |
| V1 TBPTT 32 / 64（s37） | 0.8468 / 0.7681 | — | — | — |

推理时去掉状态（reset 评估，test 三种子均值）：原 V1 0.6734、加速 0.6445、方案 C 从零 0.6664、TBPTT32 0.5622、TBPTT64 0.1766；ReLU 无状态，两者相同。

分段 IoU（test，前 16 窗 / 中间 / 末 16 窗）：原 V1 0.769/0.846/0.835，加速 0.727/0.826/0.807，ReLU 0.828/0.857/0.828，方案 C 从零 0.776/0.853/0.848。LIF 在序列开头明显低，ReLU 没有这个问题——状态预热是有代价的。

### 效率

| 版本 | 理论能耗（每 50 ms 窗；MAC 4.6 pJ / AC 0.9 pJ，实测发放率） | 膜电位状态 | 单窗耗时 |
|---|---|---|---|
| 流式 V1 | 2.24 mJ/窗，8 s 约 358 mJ（MAC 占 99%） | 7 层，397 万 | 约 19–20 ms（预处理 13 + 网络 7，GPU 空闲时） |
| 方案 C（换算 / 从零） | 0.60 mJ/窗；8 s 稠密 96.4 mJ、事件驱动 **7.85 mJ** | 同上 | — |
| ReLU 对照 | 8 s 781 mJ（全部按 MAC 计） | 无状态 | — |
| 基线 K5（稀疏 ANN，实际连接数） | 8 s **174.1 mJ**（MAC 3.79e10，99.8% 稀疏卷积；解码器 conv_up_t3/t4/m3 占 59%） | 无 | 论文称 8 s 数据 35.9 ms |
| 方案 C 从零训练（s37） | val 0.617 mJ/窗（MAC 约 90%）；test 待补 | 同上 | — |

`estimate_operations` 按单窗统计；enc1 在全图（含 99.5% 空像素）上计 MAC。基线能耗尚未统计，两者还没有同口径对比。

V1 单卡训练 50 轮约 6.6–7.6 小时。加速 A（layer + gpu，训练子集每 5 轮）重训实测每轮约 73 s，50 轮约 1 小时。
