# SPEED

Streaming Predictive Event-Evidence Detector: causal, per-event segmentation of small moving objects in event-camera
streams with spiking neural networks.

Work in progress. This folder is self-contained: it imports nothing from the surrounding repository.

## Layout

- `speed/data/` – unified event streams (integer microseconds), dataset readers, dataset cards and split statistics
- `speed/data/cards/` – one card per dataset (sensor, splits, label conventions) plus generated `*.stats.json`
- `speed/eval/` – per-event result files and the benchmark metrics (IoU / ACC / Pd / Fa as in the EV-UAV
  benchmark, for any sensor size; publish and first-detection latency)
- `tools/` – command-line tools
- `tests/` – unit tests (`python -m unittest discover -s tests` from this folder)

## Dataset statistics

```
python tools/dataset_card.py --card evflying --root /path/to/evflying
python tools/dataset_card.py --card evuav --root /path/to/EV-UAV-dataset
```

Statistics are computed on the training split only.

## Evaluation

Every method writes one result file per recording (`speed/eval/results.py`: event probabilities, optional decisions and
publish times). All metrics come from the same evaluator:

```
python tools/evaluate.py --card evuav --root /path/to/EV-UAV-dataset --split test --results runs/<method>/test
```
