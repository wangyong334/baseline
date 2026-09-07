"""Read-only full-split evaluation; never calibrates BN or saves weights."""
import hashlib
import json
import os
from pathlib import Path

import torch
from train_snn_v0 import (cfg, setup, make_dataset, make_loader, finite,
                          cpu_state, clear_unused, emit, evspsegnet_snn_v0)
from utils.semantic_cpu import ForegroundMetrics


def scores(metric):
    tp, fp, fn, tn = metric.tp, metric.fp, metric.fn, metric.tn
    def ratio(a, b):
        return a / b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=ratio(tp, tp + fp + fn), recall=ratio(tp, tp + fn),
                precision=ratio(tp, tp + fp))


def main():
    if torch.cuda.device_count() != 2 or cfg.batch_size != 1:
        raise RuntimeError('Expose exactly two GPUs; batch_size must be 1')
    torch.cuda.set_device(0)
    setup(cfg.seed)
    checkpoint = Path(os.environ['EVUAV_CHECKPOINT'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = json.loads((checkpoint.parent / 'run_config.json').read_text())
    if metadata['variant'] != 'snn':
        raise RuntimeError('SNN checkpoint required')
    for key in ('seed', 'width', 'input_channel', 'snn_stages', 'snn_steps',
                'snn_beta', 'snn_threshold', 'root', 'whole_t', 'res'):
        if metadata['config'][key] != getattr(cfg, key):
            raise RuntimeError('Config mismatch: ' + key)
    net = evspsegnet_snn_v0(cfg, split=metadata['split'])
    net.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
    net.eval()
    initial = cpu_state(net)
    emit(dict(status='START', seed=cfg.seed, checkpoint=str(checkpoint), sha256=digest,
              note='Fixed threshold 0.9; event confusion counts, not official Pd/Fa'))
    with torch.no_grad():
        for split in ('val', 'test'):
            dataset = make_dataset(split)
            total = ForegroundMetrics(0.9)
            count = 0
            for index, batch in enumerate(make_loader(dataset, False)):
                net.diagnostics(True)
                preds, voxel = net(batch['voxel_ev'])
                finite(preds, split + ' predictions')
                mapping = batch['p2v_map'].long().cuda(0)
                probabilities = preds[mapping].reshape(-1).cpu().numpy()
                labels = batch['seg_label'].reshape(-1).cpu().numpy()
                metric = ForegroundMetrics(0.9)
                metric.update(probabilities, labels)
                total.update(probabilities, labels)
                emit(dict(kind='sample', seed=cfg.seed, split=split, index=index,
                          sample=dataset.file_list[index], events=len(labels),
                          metrics=scores(metric), activity=net.spike_stats()))
                count += 1
                del preds, voxel, mapping, probabilities, labels, batch
            emit(dict(kind='split_summary', seed=cfg.seed, split=split,
                      samples=count, metrics=scores(total)))
            for name, value in net.state_dict().items():
                if not torch.equal(value.detach().cpu(), initial[name]):
                    raise RuntimeError('Evaluation changed model state: ' + name)
            clear_unused()
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed on disk during diagnosis')
    emit(dict(status='DIAGNOSIS_DONE', seed=cfg.seed, model_state_unchanged=True))


if __name__ == '__main__':
    main()
