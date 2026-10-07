# bridge

Equivalence checks between SPEED and the legacy V1-V3 code / original benchmark evaluation. Imports both sides;
not part of the SPEED release.

- `test_legacy_data.py` – SPEED data layer vs `dataset/stream_windows.py`
- `check_readers.py` – full-dataset gate (all EV-UAV recordings vs the legacy loader; EV-Flying counts)
- `legacy_eval.py` – runs the original `utils/eval.py` and the legacy latency on SPEED streams
- `test_legacy_eval.py`, `check_eval.py` – SPEED metrics vs the original evaluation (synthetic; full EV-UAV val/test gate)
- `convert_dumps.py`, `test_convert_dumps.py` – legacy `--dump-dir` outputs -> SPEED result files
