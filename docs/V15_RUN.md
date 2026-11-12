# V15 上传、GPU 1 入队和结果下载

本流程使用独立服务器目录 `EV-UAV-v15`，复用 `EV-UAV-v13` 已经运行的池子。不会启动第二个池子，不改 V14 实验，不停止任何服务器进程。默认只训练 **observation，seed 37**。

当前 r3 修复了服务器 PyTorch 1.9 CUDA 在严格确定性模式下拒绝双线性插值反向的问题。原插值前向不变，反向改为固定邻域加权归约。保留 deterministic=true。先前的 r2 已失败；使用下面的新 release 重新入队，上传不会修改 r2 的冻结快照。

已有 r2 完整上传时，只需更新 `model/observation_revision.py`、`tools/v15/train.py`、`tools/v15/run_experiment.py`，并新增上传 `tools/v15/runtime_check.py`。SFTP 仍选择 lab-v15；无须重建服务器目录或重启池子。训练进程在校准前自动执行两次小规模 GPU 参数更新自检，成功打印 `V15_RUNTIME_BACKWARD_OK`，报告存入 `runtime_check.json`。

## 1. 新建服务器目录并上传

在服务器终端执行：

```bash
mkdir -p /media/stephen/nvme0n1/wy_data/EV-UAV-v15
```

本地 SFTP 选择 **lab-v15** 配置，确认 `remotePath` 为：

```text
/media/stephen/nvme0n1/wy_data/EV-UAV-v15
```

上传本次 V15 代码。至少完整上传以下文件和目录，尤其不能漏掉修改后的骨干文件：

```text
train_stream_v1.py
model/
dataset/
utils/
tools/stream_train_stats.py
tools/v15/
configs/evisseg_stream_tif_comm_peak.yaml
configs/v15/
```

新目录不需要复制旧项目的 `log/`、`run/`、`Res/` 或数据集。数据仍从原来的独立数据目录读取。不要把本地 SFTP 配置文件上传到服务器。

服务器先做文件检查：

```bash
conda activate evuav
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v15
python - <<'PY'
from tools.v15.run_experiment import ROOT, source_files, validate_configs
from tools.v15.queue_jobs import dependency_check
print('CONFIGS:', sorted(validate_configs(ROOT)))
print('SOURCE_FILES:', len(source_files(ROOT)))
print(dependency_check(ROOT))
PY
```

这会验证三个配置、全部运行依赖和 PyTorch 导入。入队时还会检查数据集 99/24/24 份录像、现有调度器、STOP 标记和重复实验。

## 2. 只把主实验加入现有池子

```bash
conda activate evuav &&
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v15 &&
python tools/v15/queue_jobs.py enqueue \
  --release feature_20261112_r3 \
  --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 \
  --modes observation --seeds 37 --dry-run &&
python tools/v15/queue_jobs.py enqueue \
  --release feature_20261112_r3 \
  --scheduler-root /media/stephen/nvme0n1/wy_data/EV-UAV-v13 \
  --modes observation --seeds 37
```

默认任务名为 `1200_v15_feature_feature_20261112_r3_observation_s37`；如池子存在更高编号，编号会顺延。任务以 `@gpu1.sh` 发布，只允许在物理 GPU 1 执行，程序中对应 `cuda:0`。GPU 1 上已有任务时排队等待。

出现 `DRY_RUN_OK` 后才真正入队。出现 `QUEUED` 表示已加入队列，**不代表训练完成**。如果脚本报告找不到唯一的现有调度器，请检查原池子，不要再启动一个池子。

入队时会把源代码和配置冻结到：

```text
run/v15/feature_20261112_r3/source/
```

发布仅封存显式列出的 23 个运行依赖/配置，服务器残留的旧实验文件不进入快照。入队前、冻结后、任务启动时均检查源文件 SHA256。之后通过 SFTP 上传到项目根目录，不会改变已冻结的运行版本。同名 release 不允许重复入队或覆盖中途结果；需要改模型重跑时使用新的 release 名。

## 3. 自动执行内容

一个任务会依次完成：

1. 校准前执行实际设备上的严格确定性反传自检；生成本次数据表示的训练集统计量和增益校准。验证/测试集仅做格式审计，不参与统计量估计。
2. 从头训练；保持 V5 BIR、跨窗状态、通信预算、4 条并行流与 K=16 的 TBPTT。每轮验证集选优，保存 `last.pt` 和最佳验证权重。
3. 用最佳权重按严格逐窗 FP32 执行完整验证集、完整测试集，输出 IoU、ACC、Pd、Fa 和模块诊断。
4. 在选出的 3 份测试录像上重复 2 次测速，比较同一已训练骨干的原始读出与 V15 最终读出，保存时延、吞吐、内存、算量和可用的板卡能耗采样。
5. 检查工件及权重 SHA256，全部成功后写入 `comparison_result.json`，标记 `complete`。

这里的 ACC 延续项目原定义，是正事件召回率 `TP/(TP+FN)`，不是总体分类准确率。测速中的“原始读出”是同权重的计算开销对照，不代替独立重训的精度消融。额外修正模块是 ANN 计算，不计入 BIR 通信预算；板卡能耗采样不是全系统能耗。

主实验输出目录：

```text
log/v15/feature_20261112_r3/observation_s37/
```

训练统计量使用此任务自己的 `train_stats.json`，不要求提前上传旧权重或旧统计文件。打包包含训练来源清单和统计量 SHA256。

## 4. 查看运行情况

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v15
python tools/v15/queue_jobs.py status --release feature_20261112_r3 --tail 20
```

输出同时显示调度队列位置和实际阶段：`not_started`、`training`、`evaluating_val`、`evaluating_test`、`profiling`、`complete` 或 `failed`。`complete` 才表示整条流程完成；仅训练结束或显存占用变小不能作为完成依据。

## 5. 完成后打包并通过 SFTP 下载

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV-v15
python tools/v15/package_results.py --release feature_20261112_r3
```

脚本打印 `DOWNLOAD:`、大小和 SHA256。通过 SFTP 将该 `.tgz` 下载到本地项目的 `Res/`，再进行指标分析。

默认打包会拒绝尚未完成的任务。包内包括冻结源代码、实际配置、数据文件元数据、训练记录、最佳/末轮权重、完整评估和诊断、profile、调度日志及文件 SHA256。不会包含完整数据集、特征缓存、逐事件预测数组或 SFTP 凭据。

如果任务失败，需要下载已有产物排错，可明确选择部分打包：

```bash
python tools/v15/package_results.py --release feature_20261112_r3 --allow-partial
```

此时文件名带 `_partial`，包内也标记 `complete: false`，不能当成完整实验结果。若文件正在写入，打包会失败并提示稍后重试，避免下载不一致的权重/指标组合。

## 6. 后续独立重训消融

主实验确认值得继续后，可用新的 release 只加入 deep 和 causal 两个对照。它们从头独立训练，仍保留同样的 V5、通信约束和训练设置：

```bash
python tools/v15/queue_jobs.py enqueue \
  --release feature_controls_20261112_r1 \
  --modes deep causal --seeds 37 --dry-run &&
python tools/v15/queue_jobs.py enqueue \
  --release feature_controls_20261112_r1 \
  --modes deep causal --seeds 37
```

这段不会重复加入主实验。全部任务仍在既有物理 GPU 1 队列中顺序执行。
