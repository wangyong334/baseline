# BST-SNN 服务器训练与评估

本文件对应 2026-11-11 的共享证据版本。当前可运行主流程是：复用已训练的 BIR 骨干，训练新的共享 SDU，再拟合 VoW，最后以真实提前退出执行器评估。它不是已经完成的全网联合训练或完整服务器性能结论。

BIR 保留有界量化记忆的 SNN 骨干；新的 SDU / VoW 使用连续状态 MLP 和一次性离散提交，属于混合读出。本轮没有把读出层改造成全脉冲或神经形态计算。

研究主线、概率定义、解释边界与消融要求见 [方法说明](docs/BST_METHOD.md)。旧 V11/V13 代码不需要删除，可以继续作对照。

## 1. 上传内容与环境

保留原项目环境，上传新文件和必要的现有依赖：

- `model/bst.py`
- `tools/bst/` 下的所有 `.py` 与 `run.sh`
- `configs/bst_shared.json`
- `tests/test_bst*.py`
- `docs/BST_METHOD.md`、`docs/figures/bst/`、本文
- 原项目的 `model/lif2d_stream.py`、`model/evspsegnet_stream.py`、数据集与指标代码仍是依赖；不要只上传几个新文件到空目录。

新核心使用 Python 3.8 / PyTorch 1.9 可用接口。训练还需要 NumPy、SciPy；画图另需 Matplotlib。服务器沿用已有 `evuav` 环境。按项目约定仅使用物理 GPU 1，映射后脚本内设备为 `cuda:0`。

```bash
conda activate evuav
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v13
export CUDA_VISIBLE_DEVICES=1
python -c "import sys, torch, numpy, scipy; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())"
python -m unittest discover -s tests -p 'test_bst*.py'
```

不要因此重启现有作业、修改正在运行的旧脚本或占用其他卡。所有命令均由用户在服务器执行；本地实现阶段未连接服务器。

## 2. 准备冻结 BIR 的初始分数

本轮用同一份 BIR 权重为 train / val / test 导出初始分数，之后 SDU 与 VoW 训练不再重复跑骨干。优先使用新导出器，保留骨干 SHA-256、完整录像时长和完成状态。不同 BIR 权重或不同窗口长度的分数不能混在一个目录中。

设置下面的工作路径。其中 `BST_BACKBONE` 应填写你选定的真实 BIR 权重文件，不能用示例文件名代替。

```bash
export BST_DATA=/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset
export BST_Z0=log/bst_priors_seed37
export BST_RUN=log/bst_shared_seed37_20261111
export BST_BACKBONE=/absolute/path/to/your/selected_BIR_checkpoint.pt
```

若尚无相应缓存，执行：

```bash
python tools/bst/export_priors.py --checkpoint "$BST_BACKBONE" --data-root "$BST_DATA" --split train --out "$BST_Z0/z0_train" --device cuda:0
python tools/bst/export_priors.py --checkpoint "$BST_BACKBONE" --data-root "$BST_DATA" --split val --out "$BST_Z0/z0_val" --device cuda:0
python tools/bst/export_priors.py --checkpoint "$BST_BACKBONE" --data-root "$BST_DATA" --split test --out "$BST_Z0/z0_test" --device cuda:0
```

这里只导出 test 的初始预测，不用 test 标签训练或选择参数。目录结构应为：

```text
BST_Z0/
  z0_train/train_*.npz
  z0_val/val_*.npz
  z0_test/test_*.npz
```

每个缓存必须与原始 NPZ 的事件行顺序一致。新加载器检查长度、时间窗与标签对齐；原始事件及邻居观测保持完整。每个 `z0_*` 目录还包含 `prior_provenance.json`；新导出器只跳过已经核验同一身份的完整文件，可以安全续导出。已有非空目录缺少身份文件或身份冲突时会拒绝写入，应使用新目录。

旧版完整先验文件仍可由加载器读取，但必须由用户确认其骨干来源。旧文件缺少录像完整长度时，使用配置的 `recording_n_windows` 补充；当前标准配置是 160 个 50 ms 窗口。不能把最后一个事件的窗口当成录像终点，否则末尾空窗口会丢失，训练轨迹与在线强制提交时间不一致。新缓存里的长度和配置冲突时会直接拒绝。

## 3. 训练共享 SDU

```bash
python tools/bst/train.py --config configs/bst_shared.json --data-root "$BST_DATA" --z0-root "$BST_Z0" --out "$BST_RUN" --device cuda:0
```

主要输出：

- `model.pt`：由配置指定的独立验证指标选择的模型；当前默认是固定出生区间策略的加权 IoU 代理估计，同时记录前缀 BCE。
- `last.pt`：最后一轮模型。
- `training.json`：训练与验证记录。
- `manifest.json`：实际录像名单、配置和先验路径。

脚本默认保留所有正类训练查询，并按配置均匀抽样背景查询，使用已知包含概率的倒数校正损失；先选查询，再构造其邻域。抽样不删除来源观测，不用于测试阶段，也不是对允许出生的事件设置固定比例上限。

`background_budget` 是每条录像的背景训练查询预算，`0` 表示不抽样；`validation_background_budget` 只影响选择模型时的验证代理估计。最终官方评估必须覆盖全部事件。所有这些参数都保存在配置中，默认值不是从某个测试结果挑出的“最优常数”。

验证录像按固定名单拆成 `policy_fit` 和 `selection` 两部分。前者用于拟合 VoW，后者用于选择 SDU 与对照策略；名单存入模型。不要把 test 放进其中任何一部分。

短流程检查可以单独使用新目录：

```bash
python tools/bst/train.py --config configs/bst_shared.json --data-root "$BST_DATA" --z0-root "$BST_Z0" --out log/bst_smoke_seed37 --device cuda:0 --epochs 1 --background-budget 256 --limit-train 2 --limit-val 4
```

这是工程探针，不能作为论文主表结果；完整实验不要保留 `--limit-*`。

## 4. 拟合 VoW

```bash
python tools/bst/fit_policy.py --checkpoint "$BST_RUN/model.pt" --data-root "$BST_DATA" --z0-root "$BST_Z0" --out "$BST_RUN/policy.pt" --device cuda:0
```

仅使用模型清单中的 `policy_fit` 录像。SDU 保持冻结，先产生合法前缀的训练轨迹，再执行共享等待头的拟合策略迭代。监督是冻结策略的实际代价回报，不是用未来真值逐事件选择最佳停止时刻。

输出 `policy.pt` 与同名 JSON 记录拟合配置、录像、代价参数、目标、拟合误差及纠错/误伤样本数。策略绑定具体 SDU 权重；重新训练 SDU 后应重新拟合，不能混用旧策略。

该步骤同时准备可选出生头。正式配置的默认主路径为 `band` 出生 + 学习提交，显式使用 `policy.band_kind="martingale"`，即 V11 的出生区间形式。这里把它作为启发式准入规则；未证明当前学习概率满足真实后验鞅假设，因此不宣称它保证实际翻转风险小于 α。`learned` 出生用于独立对照。低训练误差不等于已经证明等待有价值。

旧 checkpoint 未记录 `band_kind` 时保留 `sym`，以复现此前对称区间行为。2026-11-11 的第一轮本地小样本探针使用这一旧对称规则，不是正式默认 V11 区间的实验结果。

## 5. 先做验证集对照，再报告 test

先在 `selection` 验证录像上比较。输出目录须为空，避免覆盖旧结果。

```bash
python tools/bst/evaluate.py --checkpoint "$BST_RUN/model.pt" --data-root "$BST_DATA" --z0-root "$BST_Z0" --split val --subset selection --out "$BST_RUN/val_fixed" --device cuda:0 --birth band --commit fixed
python tools/bst/evaluate.py --checkpoint "$BST_RUN/model.pt" --policy "$BST_RUN/policy.pt" --data-root "$BST_DATA" --z0-root "$BST_Z0" --split val --subset selection --out "$BST_RUN/val_vow" --device cuda:0 --birth band --commit learned
python tools/bst/evaluate.py --checkpoint "$BST_RUN/model.pt" --policy "$BST_RUN/policy.pt" --data-root "$BST_DATA" --z0-root "$BST_Z0" --split val --subset selection --out "$BST_RUN/val_birth_vow" --device cuda:0 --birth learned --commit learned
```

固定主配置后报告完整 test，例如默认主路径：

```bash
python tools/bst/evaluate.py --checkpoint "$BST_RUN/model.pt" --policy "$BST_RUN/policy.pt" --data-root "$BST_DATA" --z0-root "$BST_Z0" --split test --out "$BST_RUN/test_vow" --device cuda:0 --birth band --commit learned
```

评估直接执行逐窗口的待判队列，已提交查询不再进入 SDU。它不先计算每个查询的完整截止轨迹再假装提前退出。每条录像保存分数、标签、提交原因和等待步数，`summary.json` 汇总官方 IoU / ACC / Pd / Fa 与运行统计。

当前官方指标固定使用报告分数阈值 0.9。两个评估入口会拒绝不同的配置阈值，避免模型动作与指标统计采用不同工作点；内部贝叶斯风险阈值仍按 `logit(0.9) - log_w` 映射。

`--commit arrival` 表示“允许进入 SDU 后，在首次历史重审后立即提交”，不是裸 BIR；裸 BIR 已在每次评估的 `baseline` 字段中单独报告。`--birth all` 是所有事件都初始化 SDU 的昂贵对照，不能与默认区间规则混称。

## 6. 完整 BIR → SDU → VoW 在线推理

`evaluate.py` 复用初始分数缓存；`infer.py` 则在每个时间窗口实际执行事件表征、冻结 BIR、共享 SDU 和提交策略，不需要 `--z0-root`。必须使用生成训练先验时的同一份 BIR 权重，不能因为两份权重具有相同的 `pos_weight` 就认为可以替换。新模型清单保存先验身份，在线入口核对骨干 SHA-256；旧清单缺少来源时会明确警告，并输出 `provenance_verified: false`。

```bash
python tools/bst/infer.py --backbone "$BST_BACKBONE" --checkpoint "$BST_RUN/model.pt" --policy "$BST_RUN/policy.pt" --data-root "$BST_DATA" --split val --subset selection --birth band --commit learned --out "$BST_RUN/online_val_vow" --device cuda:0
```

主配置固定后，将 `--split val --subset selection` 改为 `--split test` 并使用新的输出目录，执行完整测试集在线推理。输入录像加载不计入前向时间；逐窗口事件表征、BIR、索引、共享缓存、SDU 与 VoW 均在在线执行路径中。`summary.json` 的 `timing_scope` 明确这一范围，写盘与指标计算不计入。`backbone_seconds` 是其中的 BIR 子阶段，不应与总时间重复相加。

`--limit` 和 `--max-windows` 仅用于工程探针，会在输出中标为 `probe`；它们产生的短录像结果不能当作完整测试指标。BIR 仍使用原有稠密 PyTorch 卷积实现，通信减少没有自动变成 GPU 卷积跳过。

已有完整先验缓存时，也可以使用串联脚本执行预先声明的固定等待、学习提交、学习出生对照：

```bash
DATA_ROOT="$BST_DATA" Z0_ROOT="$BST_Z0" OUT="$BST_RUN" DEVICE=cuda:0 bash tools/bst/run.sh
```

这条命令需要全新的 `BST_RUN`，会完成 SDU 训练、VoW 拟合以及各对照的 val/test 评估，拒绝覆盖已有权重。若设置 `BACKBONE="$BST_BACKBONE"`，脚本会先通过新导出器核对身份并补齐分数缓存。脚本生成 test 对照不意味着允许依据 test 选择配置，论文主策略与调参仍应由 `selection` 验证集确定。脚本不会自动执行 `infer.py`，需要端到端耗时结果时应另跑上面的在线命令。

## 7. 结果如何解释

- 内部 `posterior_probability=σ(ℓ)` 是学习得到的概率估计；官方 `report_probability=σ(ℓ+log_w)` 继续使用阈值 0.9。两者不能混画。
- `runtime` 中的实际查询更新、邻居读取和来源编码数可以验证计算是否减少；同一轮的时延与显存结果才可用来宣称真实效率收益。
- `evaluate.py` 的计时是使用预计算 BIR 分数的 SDU/VoW/索引/缓存开销；`infer.py` 才包含逐窗口事件表征和骨干前向。两者都不含文件读取，应按输出记录的范围报告。
- `mean_extra_wait_ms` 是额外观察等待；完整算法延迟还包括初始窗口结束等待。实时部署另有计算和排队时间。
- 当前模型没有严格风险上界。不要把“与截止标签一致”“预测正确”“以后永不翻转”写成同一保证。
- 服务器主实验至少比较 V11、共享 SDU 固定等待、共享 SDU + VoW；同一骨干、输入、阈值与可见时间范围下比较，额外报告密集录像和目标邻近背景。

本地已运行的检查、负结果与证据路径见 [验证记录](docs/BST_VALIDATION.md)。它们用于判断实现是否可信和定位瓶颈，不代替完整服务器实验。

## 8. 架构图

可编辑 SVG 与 PNG 位于 `docs/figures/bst/`：

| 文件 | 内容 |
|---|---|
| `01_overall.svg` / `.png` | 主架构与两级判决路径 |
| `02_bir.svg` / `.png` | BIR U-Net 与有界状态／量化汇报 |
| `03_shared_evidence.svg` / `.png` | 跨查询共享编码与有限历史缓存 |
| `04_sdu.svg` / `.png` | 重新赋权、重新赋值与总修正替换 |
| `05_vow.svg` / `.png` | 停止风险、等待代价与策略拟合 |
| `06_real_event_explanation.svg` / `.png` | 真实单事件的信念、等待代价和重审贡献诊断 |

若修改结构，先改生成脚本，再重新生成：

```bash
python tools/bst/draw_architecture.py --out docs/figures/bst
```

`--pdf` 可另存矢量 PDF。图中不包含未经实验证实的精度、加速倍数或风险保证；图与模型发生变化时，应同步核对方法公式。

## 9. 单事件证据与等待诊断

可以导出原始事件行号对应的完整诊断轨迹，例如：

```bash
python tools/bst/explain.py --checkpoint "$BST_RUN/model.pt" --policy "$BST_RUN/policy.pt" --data "$BST_DATA/val/val_001.npz" --prior "$BST_Z0/z0_val/val_001.npz" --event-id 21507 --out "$BST_RUN/explanations/val_001_event_21507.json" --device cpu
```

JSON 保存每个合法前缀的分数、总修正、邻居权重、重新赋值后的有符号证据、归一化贡献、旧邻居贡献变化，以及停止风险、等待代价与策略动作。贡献之和应重构当前总修正；它们是模型内部的代数分解，不能解释为真实因果影响、同源概率或已验证的单事件似然比。

诊断会强制读入指定查询，不应用出生门；即使提交策略已经决定停止，也会继续生成合法后续前缀供分析。这些提交后的步骤属于离线诊断，不是在线执行器继续计算，不能用作提前退出的耗时结果。

将输出绘成四面板诊断图：

```bash
python tools/bst/plot_explanation.py --trace "$BST_RUN/explanations/val_001_event_21507.json" --out "$BST_RUN/explanations/val_001_event_21507"
```

本地已核对示例为 [06_real_event_explanation.png](docs/figures/bst/06_real_event_explanation.png)，另有同名可编辑 SVG。五张架构图由 `draw_architecture.py` 生成，第六张实测诊断图由 `plot_explanation.py` 生成。
