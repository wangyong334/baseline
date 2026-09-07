# Static quantization control (not a proposed novel SNN)

This is the next controlled ablation after continuous clipping. It changes only
`conv3.1.2` and `conv4.1.2`. Forward is
`floor(4*clip(x,0,1)+0.5)/4`, with levels 0, .25, .5, .75, 1 and half-up ties.
Backward is a straight-through estimator inside 0<x<1 and zero elsewhere,
matching the continuous clip control's local derivative mask, including boundaries.
The full training gradients can still differ because downstream activations differ.

There are no trainable activation parameters, membrane states or physical time
steps. These are the SAME possible output levels as T4 LIF, NOT the same mapping
from input current to output. There is no claim of GPU speedup or energy reduction.
The tensors and convolutions remain FP32; this is not integer quantized inference.

Run from the project root in the original conda environment:

```bash
bash -n run_quant_v0.sh
mkdir -p log
quant_dir=$(mktemp -d log/quant_v0_XXXXXX)
export EVUAV_QUANT_EXPERIMENT="$quant_dir"
nohup bash run_quant_v0.sh > "$quant_dir/queue.log" 2>&1 < /dev/null &
echo "PID: $!; results: $quant_dir"
tail -n 30 -f "$quant_dir/queue.log"
```

The queue requires the original successful SNN seed37 config at
`log/snn_v0_localrate_T4_seed37_retry1/config.yaml`. It first runs four CPU unit
tests and a two-GPU two-step smoke check, then trains seeds 37,38,39 sequentially.
Any failure stops the queue. No test-set evaluation or parameter search runs.
All three keep 50 epochs, the same LR schedule and validation epochs 40..49.
Directories are exclusive. Download the entire experiment directory afterwards.

Training uses `EVUAV_VARIANT=quant`; metadata rejects testing these weights as
`ann`, `clip` or `snn`. Future evaluation must use the same variant, saved config
and `EVUAV_CHECKPOINT`, never the original `test.py`.

Scope for the next decision: compare existing ANN/SNN/clip with this control on
validation, report all seeds, and do not chase one failing test video. This does
not establish novelty or a target-preserving compute-budget mechanism. That
candidate still requires a specified temporal representation, a measurable
compute-saving operation, matched-budget baselines and literature verification.

Local validation: Python 3.8 syntax only; PyTorch absent, so tensor tests skip
locally. Server queue runs these tests with PyTorch before proceeding. Updating
the shared runner changes its source hash: old source-strict recovery checkpoints
need their original code snapshot; ordinary existing weight evaluation is retained.
