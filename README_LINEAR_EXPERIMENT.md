# 第三次实验：减慢学习率衰减

这是单因素诊断实验，不是论文所述 0.01 到 0.001 的配置。
FP32、split=2、seed=37、batch_size=1、50 轮、事件上限 700000 均不变。
从头初始化，不加载之前的权重；沿用 epoch >= 40 的验证/最佳权重候选范围。

## 上传

将更新的 train_mp.py 和 configs/evisseg_evuav_mp_linear.yaml 上传到服务器项目对应位置。
旧配置未增加 lr_schedule 时仍使用 StepLR(step_size=10, gamma=0.1)。
tests/test_mp_lr.py 是不依赖 torch 的策略测试，可选上传。

## 线性计划

零基 epoch e 的实际训练学习率为：

    lr(e) = 0.001 + (0.0001 - 0.001) * e / 49

epoch 0 为 0.001；epoch 49 为 0.0001。每轮内部学习率固定。
最后一轮之后 next_lr 保持 0.0001，不继续下降或出现负值。
metrics.jsonl 中 lr 是本轮使用值，next_lr 是下一轮计划值。
run_config.json 保存解析后的 lr_policy。

## 服务器启动

```bash
conda activate evuav
cd /media/stephen/nvme0n1/wy_data/EV-UAV
mkdir -p run

# 确认上传了新版脚本和新配置；初次运行要求权重目录不存在。
grep -n 'def linear_epoch_lr' train_mp.py
grep -E 'lr:|lr_schedule:|lr_end:|model_save_root:' configs/evisseg_evuav_mp_linear.yaml

(
set -e
test ! -e log/model_mp_fp32_linear_seed37
test ! -e run/mp_fp32_linear_seed37.log
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  CUBLAS_WORKSPACE_CONFIG=:16:8 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  EVUAV_MODE=train EVUAV_MP_SPLIT=2 \
  python -u train_mp.py --config configs/evisseg_evuav_mp_linear.yaml \
  > run/mp_fp32_linear_seed37.log 2>&1 < /dev/null &
echo $! > run/mp_fp32_linear_seed37.pid
)
tail -n 30 -f run/mp_fp32_linear_seed37.log
```

若目录或日志已存在，上述启动段停止，不删除或覆盖旧实验。请另设配置、输出和日志名称。
看到 LR policy 中 name=linear 与 EPOCH_START epoch=0 lr=0.001 后确认生效。
Ctrl+C 仅退出 tail，nohup 训练继续。

## 测试与限制

```bash
python -B -m unittest discover -s tests -p test_mp_lr.py -v
```

本地检查不替代真实 GPU 数值/训练验证。不要将未来指标改善预先归因于调度。
本次没有增加每轮验证、BN 重校准、训练恢复或多种子功能，避免同时改变实验因素。
保存文件仍是模型权重，不含完整 optimizer/scheduler/RNG，不能据此声称无缝续训。
