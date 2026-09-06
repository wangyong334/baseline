# SNN v0 validation fix and retraining

The SNN entry now computes streaming foreground counts on CPU. IoU is
TP/(TP+FP+FN); seg_acc is foreground recall TP/(TP+FN), NOT overall accuracy.
The threshold stays 0.9. Original utils/eval.py and its Pd/Fa logic are unchanged.
Validation no longer uses its CUDA masked assignment. Invalid labels/predictions
or a whole split without foreground fail explicitly. No training hyperparameters,
LIF dynamics, BN momentum or validation checkpoint-selection schedule changed.

Before the first training epoch, the full validation split is evaluated. Success
prints VALIDATION_PREFLIGHT_OK. Model state and Python/NumPy/PyTorch/CUDA RNG
states are restored afterwards. Initial IoU may be zero; this diagnostic does not
select a best checkpoint. Formal checkpoint selection remains epochs 40..49.

## Upload and start (nohup + tail)

Upload train_snn_v0.py, utils/semantic_cpu.py, tests/test_semantic_cpu.py,
tests/test_snn_recovery.py and this document. Existing model/config remain valid.

```bash
cd /media/stephen/nvme0n1/wy_data/EV-UAV
conda activate evuav
python -m unittest discover -s tests -p test_semantic_cpu.py -v
python -m unittest discover -s tests -p test_snn_recovery.py -v
```

Only start if both commands report actual test passes (not skips).
The run directory must NOT exist; use a new output name for each attempt.

```bash
mkdir -p run
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  CUBLAS_WORKSPACE_CONFIG=:16:8 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  EVUAV_MODE=train EVUAV_VARIANT=snn EVUAV_MP_SPLIT=2 \
  EVUAV_RUN_DIR=log/snn_v0_localrate_T4_seed37_retry1 \
  python -u train_snn_v0.py --config configs/evisseg_evuav_snn_v0.yaml \
  > run/snn_v0_train_seed37_retry1.log 2>&1 < /dev/null &
echo "Training PID: $!"
tail -n 30 -f run/snn_v0_train_seed37_retry1.log
```

Ctrl+C exits tail, not training. Inspect the log for preflight success, no
traceback, and finally TRAINING FINISHED. With the output override, later testing
must set EVUAV_CHECKPOINT to this run's best_iou_seed37.pt.

## Recovery

last_seed37.pt and recovery_last.pt are atomically replaced BEFORE validation.
recovery_last.pt includes model, optimizer, RNG, epoch and pending-validation flag.
After successful validation, recovery is committed again with pending=false.
A mid-epoch interruption restarts from the last committed epoch boundary.
Best loss/IoU weights are saved separately. These files do not protect against
deleting the entire run directory. This is not a bitwise GPU reproducibility claim.

To resume a NEW run produced by this version, use the same nohup command with
EVUAV_RESUME=log/snn_v0_localrate_T4_seed37_retry1/recovery_last.pt and a NEW
EVUAV_RUN_DIR, e.g. log/snn_v0_localrate_T4_seed37_resume1. Rename the shell log too.
Keep the original run's best weights next to recovery_last.pt: they are copied
to the resumed directory. Source hashes, config and GPU split must match; changes
are rejected. Old weights-only checkpoints cannot restore Adam/RNG.

## Local checks

Five NumPy metric tests execute on the development host. Three checkpoint tests
need PyTorch and are skipped there; run them on the server. Sparse CUDA forward,
full validation, dual-device recovery and the 50-epoch run require server testing.
