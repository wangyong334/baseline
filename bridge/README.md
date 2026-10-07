# bridge

Equivalence checks between SPEED and the legacy V1-V3 code / original benchmark evaluation. Imports both sides;
not part of the SPEED release.

- `test_legacy_data.py` – SPEED data layer vs `dataset/stream_windows.py`
- `check_readers.py` – full-dataset gate (all EV-UAV recordings vs the legacy loader; EV-Flying counts)
- `legacy_eval.py` – runs the original `utils/eval.py` and the legacy latency on SPEED streams
- `test_legacy_eval.py`, `check_eval.py` – SPEED metrics vs the original evaluation (synthetic; full EV-UAV val/test gate)
- `convert_dumps.py`, `test_convert_dumps.py` – legacy `--dump-dir` outputs -> SPEED result files
- `energy_full_system.py` – reproduces the legacy `tools/energy_v2` total, then adds the two per-pixel heads and the
  sparse decision layer (full-system energy of V2-1)
- `check_frozen.py` – stage-1 regression gate: frozen V2-1 / V3 dumps -> SPEED results -> SPEED evaluator must equal the
  frozen evaluation JSON for every readout; also reports the event publish latency (new definition)
