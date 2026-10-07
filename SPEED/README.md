# SPEED

Streaming Predictive Event-Evidence Detector: causal, per-event segmentation of small moving objects in event-camera
streams with spiking neural networks.

Work in progress. This folder is self-contained: it imports nothing from the surrounding repository.

## Layout

- `speed/data/` – unified event streams (integer microseconds), dataset readers, dataset cards and split statistics
- `speed/data/cards/` – one card per dataset (sensor, splits, label conventions) plus generated `*.stats.json`
- `speed/eval/` – per-event result files, the benchmark metrics (IoU / ACC / Pd / Fa as in the EV-UAV
  benchmark, for any sensor size; publish and first-detection latency) and full-system energy accounting
- `speed/viz/` – 3-D event-stream figures (background grey, target red, zoom box; methods side by side with GT)
- `speed/core/` – the skeleton: clock and device-resident event blocks, network composition, system (causal
  streaming inference), training loop, build-from-config
- `speed/slots/` – one package per slot; each holds the base variant (frozen V2-1 / V3 behaviour) and later the
  V4 variants: `representation`, `neuron`, `backbone`, `heads`, `transport`, `verify`, `publish`, `loss`
- `configs/` – system configs (`base_evuav.yaml` = base variant of every slot)
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

## Energy

`speed/eval/energy.py` sums operation counts per window over every stage of a system (front end, backbone, output
heads, decision layer): MAC 4.6 pJ, AC 0.9 pJ, transcendental functions reported at 1 / 10 / 20 MAC. Results are given
per window, per second and per clip.

## Figures

```
python tools/plot_trajectory.py --card evuav --root /path/to/EV-UAV-dataset --split test     --recordings test/test_003.npz test/test_012.npz --results Ours=runs/a/test K5=runs/k5/test --out fig.png
```

The zoom box covers the busiest 0.5 s of ground-truth target events and is drawn as an inset in every panel.

## System and slots

A system is a chain of slots, each chosen by `kind` in the config:
clock -> representation -> network (neuron, backbone, transport, heads) -> verifier -> readouts, trained with a loss
and a training method. The base variants reproduce the frozen V2-1 network and the V2-1 / V3 readouts bit for bit
(checked against the legacy code in `../bridge`). A new idea is a new variant of one slot; an ablation swaps one slot.

```
python tools/train.py --config configs/base_evuav.yaml --root /path/to/EV-UAV-dataset --out runs/base_s37 --device cuda:1
python tools/infer.py --checkpoint runs/base_s37/best.pt --root /path/to/EV-UAV-dataset --split test     --out runs/base_s37/test --evaluate --device cuda:1
```
