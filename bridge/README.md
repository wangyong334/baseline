# bridge

Equivalence checks between SPEED and the legacy V1-V3 code / original benchmark evaluation. Imports both sides;
not part of the SPEED release.

- `test_legacy_data.py` – SPEED data layer vs `dataset/stream_windows.py`
- `check_readers.py` – full-dataset gate (all EV-UAV recordings vs the legacy loader; EV-Flying counts)
