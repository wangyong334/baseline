"""Train/val-only r5 recording-risk audit on the immutable old r2 backbone.

This is not the two-stage r5 experiment: backbone adaptation is deliberately
omitted. Each head is newly initialized and fits its own training statistics.
Existing compact caches are reused only after exact transform-buffer equality.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from tools.v16 import train as tr
from tools.v16.train_covariance import (observations, load_cache, export_cache,
                                        fit_cached_head, cached_evaluation)
from utils.stream_common import load_flat_config


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def macro(result):
    return {name: float(np.mean([r['readouts']['main'][name] for r in result['per_record']]))
            for name in ('iou', 'acc', 'pd', 'fa')}


@torch.no_grad()
def training_audit(head, rows, cfg):
    boundary = float(np.log(cfg['threshold']/(1.-cfg['threshold'])))
    result = []
    for row in rows:
        z2 = torch.from_numpy(row['z2'])
        main = z2.clone()
        ids = np.flatnonzero(row['paired'])
        for start in range(0, len(ids), cfg['head_batch_size']):
            take = ids[start:start+cfg['head_batch_size']]
            main[take] = head.forward_encoded(torch.from_numpy(row['q'][take]), z2[take])
        positive = torch.from_numpy(row['label']) > .5
        paired_positive = positive & torch.from_numpy(row['paired'])
        baseline_fn = int(((z2 < boundary) & positive).sum())
        current_fn = int(((main < boundary) & positive).sum())
        n_positive = int(positive.sum())
        reference = float(F.softplus(boundary-z2[paired_positive]).mean()) if paired_positive.any() else None
        risk = float(F.softplus(boundary-main[paired_positive]).mean()) if paired_positive.any() else None
        result.append({'record': row['record'], 'positive_events': n_positive,
                       'baseline_fn': baseline_fn, 'fn': current_fn,
                       'baseline_recall': 1.-baseline_fn/n_positive if n_positive else None,
                       'recall': 1.-current_fn/n_positive if n_positive else None,
                       'baseline_fp': int(((z2 >= boundary) & ~positive).sum()),
                       'fp': int(((main >= boundary) & ~positive).sum()),
                       'reference_paired_smooth_fn': reference, 'paired_smooth_fn': risk,
                       'constraint_gap': risk-reference if risk is not None else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'Res/v16_r5_cached_probe37')
    parser.add_argument('--statistics-device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError('Refuse overwriting an existing probe: ' + str(args.out))
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = ROOT/'Res/v16_r4_final_verified37'
    baseline_cfg = json.loads((source/'run_config.json').read_text('utf8'))['config']
    cfg = load_flat_config(ROOT/'configs/v16/r5_w100/full.yaml')
    cfg.update(root=baseline_cfg['root'], init_checkpoint=baseline_cfg['init_checkpoint'],
               stats_path=str(source/'train_stats.json'), backbone_adaptation=False,
               constraint='record_fn_reference', _training_scope={'limit_records': None, 'max_chunks': None})
    source_files = ('tools/v16/probe_r5_cached.py', 'tools/v16/train_covariance.py',
                    'tools/v16/train.py', 'model/covariance_revision.py')
    protocol = {'scope': 'fixed old r2 representation risk audit; not the full two-stage r5 experiment',
                'seed': 37, 'threshold': .9, 'train_records': 99, 'val_records': 24,
                'statistics_refitted': True, 'trainable_parameters_reinitialized': True,
                'backbone_updates': 0, 'test_access': False, 'head_device': 'cpu', 'threads': 4,
                'statistics_device': args.statistics_device,
                'source_sha256': {p: tr.sha256(ROOT/p) for p in source_files},
                'backbone_checkpoint': cfg['init_checkpoint'],
                'backbone_sha256': tr.sha256(cfg['init_checkpoint']), 'config': cfg}
    write(args.out/'protocol.json', protocol)
    tr.seed_everything(37, True)
    gpu = torch.device(args.statistics_device)
    net, unused = tr.make_models(cfg, gpu)
    del unused
    checkpoint = tr.load_checkpoint(cfg['init_checkpoint'], torch.device('cpu'))
    net.load_state_dict(checkpoint['model'], strict=True)
    net.eval().requires_grad_(False)
    q99 = np.asarray(checkpoint['stats']['q99'], np.float32)
    train, val = tr.Records(cfg, 'train'), tr.Records(cfg, 'val')
    if (len(train), len(val)) != (99, 24):
        raise RuntimeError('This preregistered probe requires all 99 train and 24 validation records')
    samples = [[] for _ in range(5)]
    rng = np.random.RandomState(37)
    extraction_start = time.perf_counter()
    print('GPU_STATISTICS_START train99 only', flush=True)
    for j in range(len(train)):
        seq = train[j]
        ids = []
        for y in (0, 1):
            candidates = np.flatnonzero((seq.label == y) & (seq.coverage_count == 2))
            ids.extend(rng.choice(candidates, min(512, len(candidates)), replace=False).tolist())
        ids = np.asarray(ids, np.int64)
        extracted = observations(net, seq, cfg, q99, gpu, ids)
        for target, values in zip(samples, extracted + (seq.label[ids],)):
            target.append(values)
        if (j+1) % 10 == 0 or j+1 == len(train):
            print('TRAIN_STATISTICS %d/%d %.1fs' % (j+1, len(train), time.perf_counter()-extraction_start), flush=True)
    values = [np.concatenate(parts) for parts in samples]
    del samples
    prepared = []
    for variant, old_source in (('full', source), ('no_context', ROOT/'Res/v16_r4_local_no_context37')):
        local_cfg = dict(cfg, revision_variant=variant)
        tr.seed_everything(37, True)
        unused, head = tr.make_models(local_cfg, torch.device('cpu'))
        del unused
        old = tr.load_checkpoint(old_source/'best_val_iou_seed37.pt', torch.device('cpu'))
        if not all(torch.equal(weight, old['model'][name]) for name, weight in checkpoint['model'].items()):
            raise RuntimeError('Cached backbone differs for ' + variant)
        if not np.array_equal(q99, np.asarray(old['stats']['q99'], np.float32)):
            raise RuntimeError('Cached input normalization differs for ' + variant)
        destination = args.out/variant
        destination.mkdir()
        refit_start = time.perf_counter()
        report = head.fit_statistics(*values)
        report.update(training_records=99, split='train', seconds=time.perf_counter()-refit_start)
        write(destination/'covariance_stats.json', report)
        equal = {name: bool(torch.equal(buffer, old['fusion'][name])) for name, buffer in head.named_buffers()}
        cache_source = old_source/'cache'
        if not all(equal.values()):
            print('REFITTED_TRANSFORM_CHANGED regenerating cache', variant, equal, flush=True)
            cache_source = destination/'cache'
            cache_source.mkdir()
            head.to(gpu)
            export_cache(net, head, train, local_cfg, q99, gpu, cache_source/'train')
            export_cache(net, head, val, local_cfg, q99, gpu, cache_source/'val')
            head.cpu()
        if int(head.output.weight.count_nonzero()) != 0:
            raise RuntimeError('Probe head is not zero initialized')
        provenance = {'variant': variant, 'refitted_buffers_equal_existing_cache': equal,
                      'cache_source': str(cache_source), 'cache_reused': all(equal.values()),
                      'cache_manifests_sha256': {s: tr.sha256(cache_source/s/'manifest.json') for s in ('train', 'val')},
                      'source_checkpoint_sha256': tr.sha256(old_source/'best_val_iou_seed37.pt'),
                      'source_checkpoint_parameters_used': False, 'test_access': False}
        write(destination/'provenance.json', provenance)
        torch.save({'fusion': head.state_dict(), 'config': local_cfg}, str(destination/'initial.pt'))
        prepared.append((variant, head, local_cfg, cache_source, destination, provenance))
        del old
    extraction_seconds = time.perf_counter()-extraction_start
    del net, checkpoint, values, train
    if gpu.type == 'cuda':
        torch.cuda.empty_cache()
    print('GPU_RELEASED; fitting two zero-initialized heads on CPU', flush=True)
    r4 = json.loads((source/'selected_validation.json').read_text('utf8'))
    summary = {'protocol': protocol, 'statistics_and_cache_seconds': extraction_seconds,
               'r4_selected': {'metrics': r4['metrics'], 'macro': macro(r4), 'selected_epoch': 19}, 'variants': {}}
    for variant, head, local_cfg, cache_source, destination, provenance in prepared:
        print('HEAD_FIT_START', variant, flush=True)
        fit_start = time.perf_counter()
        rows = load_cache(cache_source/'train', torch.device('cpu'), channels=head.channels)
        validation = load_cache(cache_source/'val', torch.device('cpu'), channels=head.channels)
        selection = {}
        def save(epoch, optimizer, validation_result, record, improved, stopped, multiplier):
            if epoch == -1:
                write(destination/'initial_validation.json', validation_result)
            if improved:
                selection.update(epoch=epoch+1, validation=copy.deepcopy(validation_result),
                                 training=copy.deepcopy(record))
                torch.save({'fusion': copy.deepcopy(head.state_dict()), 'config': local_cfg,
                            'selected_epoch': epoch+1, 'rate_multiplier': multiplier, 'provenance': provenance},
                           str(destination/'selected.pt'))
                write(destination/'selected_validation.json', validation_result)
            if record is not None:
                with (destination/'metrics.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
        history = fit_cached_head(head, rows, validation, val, local_cfg, torch.device('cpu'), save)
        selected = tr.load_checkpoint(destination/'selected.pt', torch.device('cpu'))
        head.load_state_dict(selected['fusion'], strict=True)
        verified = cached_evaluation(head, validation, val, local_cfg, torch.device('cpu'))
        if verified['metrics'] != selection['validation']['metrics']:
            raise RuntimeError('Selected checkpoint does not reproduce selected metrics')
        audit = training_audit(head, rows, local_cfg)
        write(destination/'selected_training_records.json', audit)
        item = {'metrics': verified['metrics'], 'readouts': verified['readouts'], 'macro': macro(verified),
                'selected_epoch': selection['epoch'], 'fallback_to_zero': selection['epoch'] == 0,
                'epochs_completed': len(history), 'fit_and_audit_seconds': time.perf_counter()-fit_start,
                'optimization_seconds': sum(r['train_seconds'] for r in history),
                'selected_training': selection['training'], 'train_records': len(rows),
                'val_records': len(validation), 'test_access': False, 'backbone_updates': 0,
                'train_record_fn_violations_at_selected': sum(r['constraint_gap'] > 0 for r in audit if r['constraint_gap'] is not None),
                'train_macro_recall_delta': float(np.mean([r['recall']-r['baseline_recall'] for r in audit if r['recall'] is not None]))}
        write(destination/'summary.json', item)
        summary['variants'][variant] = item
        write(args.out/'summary.json', summary)
        print('HEAD_FIT_COMPLETE', variant, 'epoch', selection['epoch'], 'metrics', verified['metrics'], flush=True)
        del rows, validation, selected, history
    summary['total_seconds'] = time.perf_counter()-started
    write(args.out/'summary.json', summary)
    print('PROBE_COMPLETE %.1fs' % summary['total_seconds'], flush=True)


if __name__ == '__main__':
    main()
