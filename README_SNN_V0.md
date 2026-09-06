# SNN v0: local rate-code hybrid ablation

This is the preparatory control for target-preserving spike budgets. It has no
target gate, budget penalty, motion memory, or energy-saving claim yet.

## Exact computation

Only `conv3.1.2` and `conv4.1.2` (final post-act ReLUs) are replaced. GDBlock
internal ReLUs, input stem, attention, decoder and sigmoid head remain ANN.
Each replacement integrates the SAME ReLU feature current for T=4 local steps:

    v = beta * v + relu(x)
    s = (v >= threshold)
    v = v - threshold * stop_gradient(s)
    output = threshold * mean_t(s)

beta=0.9, threshold=1.0; surrogate derivative = 1/(1+abs(v-threshold))^2.
The output is bounded by threshold; therefore amplitude clipping and rate
quantization are deliberate confounds to diagnose (saturation is recorded).
The next layer receives a floating-point rate, NOT a sequence of binary spikes.
The whole U-Net runs ONCE per sample. BN updates once. No state survives a
module call, and no repeated sparse-convolution coordinate alignment is assumed.
T is encoding time, NOT physical event time. This cannot be described as a full
SNN, a causal streaming detector, or demonstrated sparse execution.
ANN control uses unchanged ReLU and ignores T. A later clipped/quantized ANN
control is needed before attributing effects to neuronal memory.

## Training protocol

From scratch, seed configurable (default 37), Adam, batch 1, FP32, 700000-event
cap, 50 epochs, inclusive linear LR .001 -> .0001, eval-mode validation at
epochs 40..49, threshold .9 and original STC/metric implementation.
Use `EVUAV_VARIANT=ann` for a same-entry control with separate EVUAV_RUN_DIR.
Epoch-boundary recovery includes Adam and RNG state; see README_SNN_V0_RETRAIN.md.
New train/overfit directories are exclusive. Foreground metrics now accumulate
on CPU without CUDA masked assignment; metric definitions are unchanged.
Each run saves source copies/hashes, YAML, actual config, activity and metrics.
Activity statistics are sampled every 20 training updates and are NOT whole-run
energy measurements or target/background-separated statistics. Timing includes
finite checks/synchronization/diagnostics; it is NOT pure model inference time.
Memory is measured using both devices' PyTorch peaks; it excludes some external
allocator use. Multi-step local activations can INCREASE training memory.

## Server: first run only the tests and smoke check

Upload the new model files, entry, config, test, and this README through SFTP.
Do not replace the environment or rebuild HAIS.

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV
conda activate evuav
mkdir -p run
set -o pipefail
python -m unittest discover -s tests -p test_lif_rate.py -v

CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
EVUAV_MODE=smoke EVUAV_VARIANT=snn EVUAV_MP_SPLIT=2 \
python -u train_snn_v0.py --config configs/evisseg_evuav_snn_v0.yaml \
2>&1 | tee run/snn_v0_smoke.log
```

Default smoke selects the largest raw training sample, does two Adam updates,
checks finite outputs/state/gradients and actual updates on both GPUs. Override
with `EVUAV_SAMPLE=train_000.npz` for a smaller first debugging sample.
Successful completion does not imply convergence or generalization.

## Fixed-sample overfit check (after smoke passes)

```bash
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
EVUAV_MODE=overfit EVUAV_VARIANT=snn EVUAV_SAMPLE=train_000.npz \
EVUAV_OVERFIT_STEPS=100 \
python -u train_snn_v0.py --config configs/evisseg_evuav_snn_v0.yaml \
2>&1 | tee run/snn_v0_overfit.log
```

Check loss decreases substantially; inspect saturation and silent fractions.
100 completed steps alone is NOT an overfit pass. Repeat with an explicit new
`EVUAV_RUN_DIR=log/...` (same data/config) to avoid overwriting earlier results.

## Full training (only after reviewing diagnostics)

Run the regression tests in README_SNN_V0_RETRAIN.md first. A full validation
preflight runs before training and restores model/RNG state afterwards.

```bash
mkdir -p run
nohup env CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
EVUAV_MODE=train EVUAV_VARIANT=snn EVUAV_MP_SPLIT=2 \
python -u train_snn_v0.py --config configs/evisseg_evuav_snn_v0.yaml \
> run/snn_v0_train.log 2>&1 < /dev/null &
echo "Training PID: $!"
tail -n 30 -f run/snn_v0_train.log
```

Ctrl+C stops tail only, not the background training process.

For matched ANN control add `EVUAV_VARIANT=ann` instead of snn and
`EVUAV_RUN_DIR=log/ann_control_v0_seed37`. Seed and SNN parameters are in YAML;
copy config and set distinct output paths for every variant/seed.

## Test only after variant selection on validation

The original test.py constructs ANN and would silently treat these weights as
ANN because LIF has no parameters. Always use the new entry for SNN checkpoints.
It requires adjacent run_config.json and verifies the encoding configuration.

```bash
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
EVUAV_MODE=test EVUAV_VARIANT=snn \
python -u train_snn_v0.py --config configs/evisseg_evuav_snn_v0.yaml \
2>&1 | tee run/snn_v0_test.log
```

If output directory was overridden, set EVUAV_CHECKPOINT to its best_iou file.
Retain accompanying run_config.json. Test still requires two GPUs.

## Validation limits

Development host has no PyTorch/CUDA/spconv. Python 3.8 syntax is checked locally;
the six tensor tests must execute on the server (a skipped test is not a pass).
GPU sparse forwarding/backpropagation, largest-sample memory, accuracy and cost
remain unverified until the server checks complete.
