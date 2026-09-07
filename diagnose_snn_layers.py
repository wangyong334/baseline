"""Read-only layer tracing and event projections for fixed checkpoint pairs."""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from model.lif_rate import LIFRate
from model.evspsegnet_mp import evspsegnet_mp
from train_snn_v0 import cfg, setup, make_dataset, cpu_state, evspsegnet_snn_v0
from diagnose_snn_splits import scores
from utils.semantic_cpu import ForegroundMetrics


def summary(value):
    if hasattr(value, 'features'):
        value = value.features
    if not torch.is_tensor(value):
        return None
    x = value.detach().float()
    if not x.numel():
        return {'shape': list(x.shape), 'count': 0}
    if not torch.isfinite(x).all().item():
        raise RuntimeError('Nonfinite feature encountered')
    flat = x.reshape(-1)
    stride = max(1, (flat.numel() + 99999) // 100000)
    sampled = flat[::stride].cpu().numpy()
    return dict(shape=list(x.shape), min=x.min().item(), max=x.max().item(),
                mean=x.mean().item(), abs_mean=x.abs().mean().item(),
                zero_fraction=(x == 0).float().mean().item(),
                quantile_stride=stride,
                sampled_p01_p50_p99=np.quantile(sampled, [.01, .5, .99]).tolist())


def projection(path, locs, labels, probabilities, width, height):
    x, y = locs[:, 1].astype(int), locs[:, 2].astype(int)
    if ((x < 0) | (x >= width) | (y < 0) | (y >= height)).any():
        raise ValueError('Coordinates outside configured image dimensions')
    target, predicted = labels == 1, probabilities >= .9
    masks = [np.ones(len(x), dtype=bool), target, predicted,
             predicted & ~target, target & ~predicted]
    titles = ['All events', 'Ground truth', 'Prediction >=0.9', 'False positives', 'False negatives']
    panels = []
    for title, mask in zip(titles, masks):
        counts = np.zeros((height, width), dtype=np.int64)
        np.add.at(counts, (y[mask], x[mask]), 1)
        # Binary occupancy, not event counts: overlapping events share pixels.
        tile = np.zeros((height + 30, width, 3), dtype=np.uint8)
        tile[30:][counts > 0] = 255
        cv2.putText(tile, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        panels.append(tile)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise IOError('Could not save projection')


def main():
    if torch.cuda.device_count() != 2 or cfg.batch_size != 1:
        raise RuntimeError('Requires exactly two visible GPUs and batch_size=1')
    torch.cuda.set_device(0)
    variant = os.environ.get('EVUAV_VARIANT', 'snn')
    if variant not in ('ann', 'snn'):
        raise ValueError('EVUAV_VARIANT must be ann or snn')
    seed = int(os.environ.get('EVUAV_DIAG_SEED', getattr(cfg, 'seed', 37)))
    setup(seed)
    checkpoint = Path(os.environ['EVUAV_CHECKPOINT'])
    split = int(os.environ.get('EVUAV_MP_SPLIT', '2'))
    if variant == 'snn':
        meta = json.loads((checkpoint.parent / 'run_config.json').read_text())
        if meta['variant'] != 'snn':
            raise ValueError('Expected SNN checkpoint')
        for key in ('seed', 'width', 'input_channel', 'snn_stages', 'snn_steps',
                    'snn_beta', 'snn_threshold', 'root', 'res', 'whole_t'):
            if getattr(cfg, key) != meta['config'][key]:
                raise ValueError('Config mismatch: ' + key)
        if seed != cfg.seed:
            raise ValueError('SNN diagnostic seed differs from config')
        split = meta['split']
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    root = Path(os.environ['EVUAV_LAYER_OUTPUT'])
    root.mkdir(parents=True, exist_ok=False)
    net = (evspsegnet_snn_v0(cfg, split=split) if variant == 'snn'
           else evspsegnet_mp(cfg, split=split))
    net.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
    net.eval()
    before = cpu_state(net)
    current_sample = [None]
    sequence = [0]
    with (root / 'layers.jsonl').open('x') as stream:
        def emit(record):
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()

        def hook(name):
            def trace(module, inputs, output):
                sequence[0] += 1
                row = dict(sample=current_sample[0], order=sequence[0], layer=name,
                           type=type(module).__name__, input=summary(inputs[0]), output=summary(output))
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    row['bn'] = dict(running_mean=summary(module.running_mean),
                                     running_var=summary(module.running_var),
                                     weight=summary(module.weight), bias=summary(module.bias), eps=module.eps,
                                     effective_scale=summary(module.weight / torch.sqrt(module.running_var + module.eps)))
                if isinstance(module, LIFRate):
                    row['activity'] = dict(module.stats)
                emit(row)
            return trace

        handles = []
        for name, module in net.named_modules():
            if (isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, LIFRate))
                    or name in ('conv3.1.2', 'conv4.1.2')
                    or 'Conv' in type(module).__name__ and hasattr(module, 'weight')
                    or name in ('conv_input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')):
                handles.append(module.register_forward_hook(hook(name)))
        dataset = make_dataset('test')
        records = []
        try:
            with torch.no_grad():
                for name in ('test_008.npz', 'test_000.npz', 'test_004.npz'):
                    current_sample[0], sequence[0] = name, 0
                    batch = dataset.custom_collate([dataset[dataset.file_list.index(name)]])
                    if hasattr(net, 'diagnostics'):
                        net.diagnostics(True)
                    preds, voxel = net(batch['voxel_ev'])
                    probabilities = preds[batch['p2v_map'].long().cuda(0)].reshape(-1).cpu().numpy()
                    labels = batch['seg_label'].reshape(-1).cpu().numpy()
                    locs = batch['locs'].cpu().numpy()
                    metric = ForegroundMetrics(.9)
                    metric.update(probabilities, labels)
                    np.savez_compressed(str(root / name), locs=locs, labels=labels, probabilities=probabilities)
                    projection(root / (Path(name).stem + '.png'), locs, labels, probabilities, *cfg.res)
                    row = dict(sample=name, metrics=scores(metric),
                               activity=net.spike_stats() if hasattr(net, 'spike_stats') else {})
                    records.append(row)
                    print(json.dumps(row, allow_nan=False), flush=True)
                    del batch, preds, voxel, probabilities, labels, locs
        finally:
            for handle in handles:
                handle.remove()
    for name, value in net.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[name]):
            raise RuntimeError('Model state changed: ' + name)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed on disk')
    (root / 'summary.json').write_text(json.dumps(dict(seed=seed, variant=variant,
        split=split, config=vars(cfg), checkpoint=str(checkpoint),
        sha256=digest, samples=records, model_state_unchanged=True,
        note='PNG: full-window pixel occupancy, not event counts or physical frames; '
             'quantiles use deterministic strided samples; no calibration or optimization.'), indent=2))
    print('LAYER_DIAGNOSIS_DONE', flush=True)


if __name__ == '__main__':
    main()
