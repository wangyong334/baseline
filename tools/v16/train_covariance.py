"""Frozen-backbone, train-only statistical fitting and cached r4 training.

The immutable V5-family backbone is evaluated chronologically once per export.
Only compact paired descriptors are reused across epochs. Validation never fits
statistics or the operating threshold. Python 3.8 / torch 1.9 compatible.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def constrained_loss(logits, labels, multiplier, threshold=.9, reference_fn=0.):
    """Lagrangian for a *smooth* false-negative constraint at fixed threshold.

    The reference is an immutable training-set statistic, never a validation
    calibration target. The same softplus FN risk drives the dual update.
    """
    boundary = math.log(threshold / (1. - threshold))
    margin = logits - boundary
    positive, negative = labels > .5, labels <= .5
    zero = logits.sum() * 0.
    fp = F.softplus(margin[negative]).mean() if negative.any() else zero
    fn = F.softplus(-margin[positive]).mean() if positive.any() else zero
    return fp + float(multiplier) * (fn - float(reference_fn))


def four_metric_noninferiority(metrics, reference):
    """A validation selection guard, not a population/test-set guarantee."""
    keys = ('iou', 'acc', 'pd', 'fa')
    if not all(math.isfinite(float(row[key])) for row in (metrics, reference) for key in keys):
        return False
    return (all(metrics[key] >= reference[key] for key in ('iou', 'acc', 'pd'))
            and metrics['fa'] <= reference['fa'])


def update_recall_multiplier(multiplier, smooth_fn, reference_fn, dual_lr):
    # Upper clipping is a numerical guard; it is not a constrained-convergence
    # theorem for this nonlinear model. Binary FNR is diagnostic only.
    return float(np.clip(multiplier + dual_lr * (smooth_fn - reference_fn), 0., 100.))


@torch.no_grad()
def observations(net, seq, cfg, q99, device, selected=None):
    """Read two observations at the exact original IDs, including singletons."""
    from tools.v16 import train as tr
    source = tr.OverlapWindowSource(seq, cfg, q99, device)
    selected = np.arange(seq.n_events) if selected is None else np.asarray(selected, dtype=np.int64)
    lookup = np.full(seq.n_events, -1, np.int64)
    lookup[selected] = np.arange(len(selected))
    feature_source = tr.covariance_feature_source(cfg)
    dim = tr.covariance_feature_dim(cfg)
    h1, h2 = np.zeros((len(selected), dim), np.float32), np.zeros((len(selected), dim), np.float32)
    z1, z2 = np.zeros(len(selected), np.float32), np.zeros(len(selected), np.float32)
    hits = np.zeros((len(selected), 2), np.int8)
    states = None
    for k in range(seq.n_windows):
        x, ev, _, ids = source.window(k)
        _, states, info = net(x, ev, states, return_features=True, skip_readout=True,
                              feature_taps=feature_source == 'multilevel')
        rows = np.flatnonzero(lookup[ids] >= 0)
        if not len(rows):
            continue
        index = torch.as_tensor(rows, device=device, dtype=torch.long)
        events = {key: value[index] for key, value in ev.items() if key in ('b', 'y', 'x', 'p', 't_local')}
        events['t'] = torch.zeros_like(events['b'])
        context = tuple(f.unsqueeze(0) for f in info['context_features']) if 'context_features' in info else None
        logits, hidden = tr.readout_with_hidden(net, info['features'].unsqueeze(0), events,
                                              context, source=feature_source)
        values = torch.cat([logits[:, None], hidden], 1).cpu().numpy()
        event_ids, dest = ids[rows], lookup[ids[rows]]
        for slot, times, zs, hs in ((0, seq.first_window, z1, h1), (1, seq.last_window, z2, h2)):
            use = times[event_ids] == k
            zs[dest[use]] = values[use, 0]
            hs[dest[use]] = values[use, 1:]
            hits[dest[use], slot] += 1
    if not np.all(hits == 1):
        raise RuntimeError('Observation identity/coverage failed: ' + seq.name)
    return z1, z2, h1, h2


@torch.no_grad()
def fit_training_statistics(net, head, dataset, cfg, q99, device):
    samples = [[] for _ in range(5)]
    inventory = []
    start = time.perf_counter()
    rng = np.random.RandomState(int(cfg['seed']))
    for i in range(len(dataset)):
        seq = dataset[i]
        ids = []
        for y in (0, 1):
            candidates = np.flatnonzero((seq.label == y) & (seq.coverage_count == 2))
            ids.extend(rng.choice(candidates, min(512, len(candidates)), replace=False).tolist())
        ids = np.asarray(ids, dtype=np.int64)
        obs = observations(net, seq, cfg, q99, device, ids)
        for output, values in zip(samples, obs + (seq.label[ids],)):
            output.append(values)
        inventory.append({'record': seq.name, 'pairs': len(ids), 'positive': int(seq.label[ids].sum())})
        if (i+1) % 10 == 0 or i+1 == len(dataset):
            print('STATISTICS train %d/%d %.1fs' % (i+1, len(dataset), time.perf_counter()-start), flush=True)
    values = [np.concatenate(rows) for rows in samples]
    result = head.fit_statistics(*values)
    result.update(split='train', sample_policy='up to 512 per class per recording, paired only',
                  records=inventory, seconds=time.perf_counter()-start)
    return result


@torch.no_grad()
def export_cache(net, head, dataset, cfg, q99, device, destination):
    from tools.v16.train import write_json
    destination.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rows = []
    for i in range(len(dataset)):
        seq = dataset[i]
        z1, z2, h1, h2 = observations(net, seq, cfg, q99, device)
        q = np.zeros((seq.n_events, 2 * head.channels), np.float32)
        paired = seq.coverage_count == 2
        ids = np.flatnonzero(paired)
        for lo in range(0, len(ids), 65536):
            take = ids[lo:lo+65536]
            inputs = [torch.as_tensor(a[take], device=device) for a in (z1, z2, h1, h2)]
            q[take] = head.encode_pair(*inputs).cpu().numpy()
        path = destination / (Path(seq.name).stem + '.npz')
        np.savez(str(path), q=q, z1=z1, z2=z2, label=seq.label.astype(np.uint8), paired=paired)
        rows.append({'record': seq.name, 'events': seq.n_events, 'pairs': int(paired.sum()), 'bytes': path.stat().st_size})
        print('CACHE %s %d/%d %.1fs' % (destination.name, i+1, len(dataset), time.perf_counter()-start), flush=True)
    result = {'records': rows, 'seconds': time.perf_counter()-start, 'complete': True}
    write_json(destination / 'manifest.json', result)
    return result


def load_cache(directory, device, channels=None):
    rows = []
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
    if manifest.get('complete') is not True:
        raise ValueError('Incomplete feature cache')
    width = None if channels is None else 2 * int(channels)
    seen = set()
    for entry in manifest['records']:
        name, n, pairs = entry['record'], entry['events'], entry['pairs']
        if (not isinstance(name, str) or Path(name).name != name or
                Path(name).stem in seen or isinstance(n, bool) or not isinstance(n, int) or n < 0 or
                isinstance(pairs, bool) or not isinstance(pairs, int) or not 0 <= pairs <= n):
            raise ValueError('Invalid compact cache manifest entry')
        seen.add(Path(name).stem)
        with np.load(str(directory / (Path(name).stem + '.npz')), allow_pickle=False) as f:
            row = {key: f[key] for key in f.files}
        valid = set(row) == {'q', 'z1', 'z2', 'label', 'paired'}
        if valid:
            q = row['q']
            valid = (q.ndim == 2 and q.shape[0] == n and q.shape[1] > 0 and
                     q.shape[1] % 2 == 0 and q.dtype == np.float32 and np.isfinite(q).all())
            if valid:
                width = q.shape[1] if width is None else width
                valid = q.shape[1] == width
            valid = valid and all(row[key].shape == (n,) for key in ('z1', 'z2', 'label', 'paired'))
            if valid:
                valid = (all(row[key].dtype == np.float32 and np.isfinite(row[key]).all()
                             for key in ('z1', 'z2')) and
                         row['label'].dtype.kind in 'buif' and
                         np.isfinite(row['label']).all() and
                         np.isin(row['label'], (0, 1)).all() and
                         row['paired'].dtype == np.bool_ and int(row['paired'].sum()) == pairs)
        if not valid:
            raise ValueError('Invalid compact cache: ' + name)
        row['record'] = name
        rows.append(row)
    if not rows:
        raise ValueError('Empty feature cache')
    return rows


def _validation_reference_signature(rows, sequences, cfg):
    """Bind cached fixed-view metrics to the data and their exact metric policy."""
    digest = hashlib.sha256()
    policy = {key: cfg[key] for key in ('threshold', 'pd_detT', 'correct_thresh', 'height', 'width_px')}
    digest.update(json.dumps(policy, sort_keys=True, allow_nan=False).encode('utf8'))
    for row, seq in zip(rows, sequences):
        digest.update(seq.name.encode('utf8'))
        for values in (row['z1'], row['z2'], seq.label, seq.t, seq.target_id, seq.x, seq.y):
            array = np.ascontiguousarray(values)
            digest.update(str((array.dtype.str, array.shape)).encode('ascii'))
            digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


@torch.no_grad()
def cached_evaluation(head, rows, dataset, cfg, device, reference=None):
    from tools.v16.train import metric_row, FastRoc, window_confusion
    if len(rows) != len(dataset):
        raise ValueError('Validation cache record count mismatch')
    sequences = [dataset[i] for i in range(len(rows))]
    if any(seq.name != row['record'] for row, seq in zip(rows, sequences)):
        raise ValueError('Validation cache record identity mismatch')
    signature = _validation_reference_signature(rows, sequences, cfg)
    if reference is not None:
        if (reference.get('reference_signature') != signature or
                len(reference.get('per_record', [])) != len(rows) or
                any(record.get('record') != seq.name
                    for record, seq in zip(reference['per_record'], sequences))):
            raise ValueError('Validation reference does not match fixed views or metric policy')
    modes = ('first', 'second', 'main') if reference is None else ('main',)
    counts = {name: np.zeros(3, np.int64) for name in modes}
    rocs = {name: FastRoc(cfg['pd_detT'], cfg['correct_thresh'], cfg['height'], cfg['width_px']) for name in counts}
    records = []
    head.eval()
    for i, (row, seq) in enumerate(zip(rows, sequences)):
        z = torch.as_tensor(row['z2'], device=device)
        main = z.clone()
        ids = np.flatnonzero(row['paired'])
        for lo in range(0, len(ids), 65536):
            take = ids[lo:lo+65536]
            index = torch.as_tensor(take, device=device)
            main[index] = head.forward_encoded(torch.as_tensor(row['q'][take], device=device), z[index])
        probabilities = {'main': torch.sigmoid(main).cpu().numpy()}
        if reference is None:
            probabilities.update(first=torch.sigmoid(torch.as_tensor(row['z1'], device=device)).cpu().numpy(),
                                 second=torch.sigmoid(z).cpu().numpy())
        fixed = {} if reference is None else {
            name: dict(reference['per_record'][i]['readouts'][name]) for name in ('first', 'second')}
        record = {'record': seq.name, 'readouts': fixed}
        for name in modes:
            p = probabilities[name]
            count = np.asarray(window_confusion(p, seq.label, np.float32(cfg['threshold']))[:3])
            counts[name] += count
            roc = FastRoc(cfg['pd_detT'], cfg['correct_thresh'], cfg['height'], cfg['width_px'])
            roc.update(seq.t, p, seq.target_id, seq.label, seq.x, seq.y, cfg['threshold'])
            # Each update is recording-local. Aggregate its integer numerators
            # and denominators, preserving FastRoc's original frame boundaries.
            for field in ('obj_num', 'correct_num', 'false_num', 'frame_num'):
                setattr(rocs[name], field, getattr(rocs[name], field) + getattr(roc, field))
            record['readouts'][name] = metric_row(count, roc)
        records.append(record)
    readouts = {} if reference is None else {name: dict(reference['readouts'][name])
                                           for name in ('first', 'second')}
    readouts.update({name: metric_row(counts[name], rocs[name]) for name in modes})
    return {'readouts': readouts,
            'metrics': metric_row(counts['main'], rocs['main']), 'per_record': records,
            'threshold': cfg['threshold'], 'split': 'val', 'cache_execution': 'chronological step backbone',
            'reference_signature': signature}


def stationary_multiplier(z2, labels, threshold=.9):
    """Train-reference bias stationarity, not validation calibration."""
    boundary = math.log(threshold / (1.-threshold))
    positive = labels > .5
    fp_gradient = torch.sigmoid(z2[~positive] - boundary).mean()
    fn_gradient = torch.sigmoid(boundary - z2[positive]).mean()
    return float((fp_gradient / fn_gradient.clamp_min(1e-8)).clamp(0., 100.))


@torch.no_grad()
def record_class_means(values, labels, record_ids, n_records):
    """Statistical class means per recording; absent classes have zero mean.

    CPU reductions avoid CUDA scatter_add nondeterminism on PyTorch 1.9.
    These immutable statistics are never part of the training gradient path.
    """
    if (values.dim() != 1 or labels.shape != values.shape or record_ids.shape != values.shape
            or record_ids.dtype != torch.long or int(n_records) < 1):
        raise ValueError('Record risks require aligned vectors and integer record IDs')
    if record_ids.numel() and (int(record_ids.min()) < 0 or int(record_ids.max()) >= int(n_records)):
        raise ValueError('Record risk ID is outside the recording inventory')
    destination = values.device
    values, labels, record_ids = values.cpu(), labels.cpu(), record_ids.cpu()
    results = []
    for selected in (labels > .5, labels <= .5):
        count = values.new_zeros(int(n_records))
        total = values.new_zeros(int(n_records))
        index = record_ids[selected]
        count.scatter_add_(0, index, torch.ones_like(values[selected]))
        total.scatter_add_(0, index, values[selected])
        results.append(((total / count.clamp_min(1.)).to(destination), count.to(destination)))
    return results


def record_loss_weights(positive_counts, negative_counts, n_events):
    """Weights whose uniform event-batch mean is an equal-record class mean."""
    weights = []
    for counts in (positive_counts, negative_counts):
        active = counts > 0
        if not bool(active.any()):
            raise ValueError('Record-constrained training requires both classes')
        weights.append(torch.where(active, float(n_events) / (active.sum() * counts.clamp_min(1.)),
                                   torch.zeros_like(counts)))
    return tuple(weights)


def record_constrained_loss(logits, labels, record_ids, multipliers, positive_weights,
                            negative_weights, reference_fn, threshold=.9):
    """Unbiased event-batch objective for separate recording FN constraints.

    Both class risks average recordings equally. Each recording's FN risk is
    compared only with its immutable second-readout training risk, so another
    recording cannot cancel that constraint. The surrogate is not binary FNR.
    """
    boundary = math.log(threshold / (1. - threshold))
    positive = labels > .5
    margin = logits - boundary
    fp = negative_weights[record_ids] * (~positive).to(logits.dtype) * F.softplus(margin)
    fn = (positive_weights[record_ids] * positive.to(logits.dtype) *
          multipliers[record_ids] * F.softplus(-margin))
    active = positive_weights > 0
    reference = (multipliers[active] * reference_fn[active]).mean()
    return (fp + fn).mean() - reference


@torch.no_grad()
def record_stationary_multipliers(z2, labels, record_ids, n_records, threshold=.9):
    """Derive each initial dual from its baseline uniform-logit-shift gradients."""
    boundary = math.log(threshold / (1. - threshold))
    (fn_gradient, positive_counts), _ = record_class_means(
        torch.sigmoid(boundary - z2), labels, record_ids, n_records)
    _, (fp_gradient, negative_counts) = record_class_means(
        torch.sigmoid(z2 - boundary), labels, record_ids, n_records)
    n_positive = int((positive_counts > 0).sum())
    n_negative = int((negative_counts > 0).sum())
    if not n_positive or not n_negative:
        raise ValueError('Record-constrained training requires both classes')
    value = (float(n_positive) / n_negative) * fp_gradient / fn_gradient.clamp_min(1e-8)
    return torch.where(positive_counts > 0, value.clamp(0., 100.), torch.zeros_like(value))


def update_record_multipliers(multipliers, smooth_fn, reference_fn, dual_lr):
    """One independent dual update per recording using the primal FN risk."""
    if not (multipliers.shape == smooth_fn.shape == reference_fn.shape):
        raise ValueError('Record multiplier and risk shapes must match')
    if not all(bool(torch.isfinite(value).all()) for value in (multipliers, smooth_fn, reference_fn)):
        raise ValueError('Record multiplier and risks must be finite')
    return (multipliers + float(dual_lr) * (smooth_fn - reference_fn)).clamp(0., 100.)


def fit_cached_head(head, rows, validation, val_dataset, cfg, device, callback=None, resume=None):
    """Train fixed compact channels; return immutable-reference guarded history."""
    fields = {key: torch.as_tensor(np.concatenate([r[key][r['paired']] for r in rows]), device=device)
              for key in ('q', 'z2', 'label')}
    q, z2, labels = fields['q'], fields['z2'], fields['label'].float()
    positive = labels > .5
    if not positive.any() or positive.all():
        raise ValueError('Training pairs require both classes')
    boundary = math.log(cfg['threshold'] / (1.-cfg['threshold']))
    reference_fnr = float((z2[positive] < boundary).float().mean())
    reference_fn = float(F.softplus(boundary - z2[positive]).mean())
    constraint = cfg.get('constraint', 'smooth_fn_reference')
    if constraint not in ('smooth_fn_reference', 'record_fn_reference'):
        raise ValueError('Unknown cached head constraint: ' + str(constraint))
    by_record = constraint == 'record_fn_reference'
    record_ids, record_reference, record_fp_reference = None, None, None
    positive_weights, negative_weights = None, None
    if by_record:
        record_ids = torch.cat([torch.full((int(r['paired'].sum()),), j, dtype=torch.long, device=device)
                                for j, r in enumerate(rows)])
        cpu_record_ids = record_ids.cpu()
        (record_reference, positive_counts), _ = record_class_means(
            F.softplus(boundary-z2), labels, record_ids, len(rows))
        _, (record_fp_reference, negative_counts) = record_class_means(
            F.softplus(z2-boundary), labels, record_ids, len(rows))
        positive_weights, negative_weights = record_loss_weights(positive_counts, negative_counts, len(q))
        multiplier = record_stationary_multipliers(z2, labels, record_ids, len(rows), cfg['threshold'])
    else:
        multiplier = stationary_multiplier(z2, labels, cfg['threshold'])
    def serial_multiplier(value):
        return value.detach().cpu().tolist() if by_record else float(value)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg['lr'])
    initial_val = cached_evaluation(head, validation, val_dataset, cfg, device)
    reference_val = initial_val['readouts']['second']
    if resume is None and initial_val['metrics'] != reference_val:
        raise ValueError('r4 requires a zero-residual initial head as the feasible fallback')
    best = initial_val['metrics']['iou']
    best_epoch, history, start = -1, [], 0
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        start = int(resume['epoch']) + 1
        history = resume['_history']
        best, best_epoch = resume['best_val_iou'], resume['_best_epoch']
        if by_record:
            multiplier = torch.as_tensor(resume.get('rate_multiplier', multiplier), device=device, dtype=z2.dtype)
            if (tuple(multiplier.shape) != (len(rows),) or not bool(torch.isfinite(multiplier).all())
                    or bool(((multiplier < 0) | (multiplier > 100.)).any())):
                raise ValueError('Invalid resumed recording multipliers')
        else:
            multiplier = resume.get('rate_multiplier', multiplier)
    if callback is not None and resume is None:
        callback(-1, optimizer, initial_val, None, True, False, serial_multiplier(multiplier))
    batch_size = int(cfg.get('head_batch_size', 32768))
    # Identical shuffled sample order on CPU and CUDA. Numerical kernels may
    # still differ; this does not promise bitwise cross-device reproducibility.
    rng = torch.Generator(device='cpu').manual_seed(int(cfg['seed']))
    for epoch in range(start, int(cfg['epochs'])):
        begin = time.perf_counter()
        head.train()
        lr = (cfg['lr_end'] + .5 * (cfg['lr'] - cfg['lr_end']) *
              (1. + math.cos(math.pi * epoch / max(cfg['epochs'] - 1, 1))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = torch.randperm(len(q), generator=rng).to(device)
        multiplier_used = multiplier.clone() if by_record else multiplier
        total_loss, batches = 0., 0
        for lo in range(0, len(q), batch_size):
            index = order[lo:lo+batch_size]
            logits = head.forward_encoded(q[index], z2[index])
            if by_record:
                loss = record_constrained_loss(logits, labels[index], record_ids[index], multiplier,
                                               positive_weights, negative_weights, record_reference, cfg['threshold'])
            else:
                loss = constrained_loss(logits, labels[index], multiplier, cfg['threshold'], reference_fn)
            loss = loss + float(cfg['covariance_regularizer']) * head.variance_regularizer()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite cached head objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg['grad_clip']))
            if not math.isfinite(float(norm)):
                raise RuntimeError('Nonfinite cached head gradient')
            optimizer.step()
            total_loss += float(loss.detach()); batches += 1
        head.eval()
        train_fn, train_fp, smooth_fn_sum = 0, 0, 0.
        if by_record:
            record_fn_sums = torch.zeros(len(rows), dtype=z2.dtype)
            record_fp_sums = torch.zeros(len(rows), dtype=z2.dtype)
        with torch.no_grad():
            for lo in range(0, len(q), batch_size):
                s = head.forward_encoded(q[lo:lo+batch_size], z2[lo:lo+batch_size])
                y = positive[lo:lo+batch_size]
                train_fn += int(((s < boundary) & y).sum())
                train_fp += int(((s >= boundary) & ~y).sum())
                smooth_fn_sum += float(F.softplus(boundary - s[y]).sum())
                if by_record:
                    ids, cpu_y = cpu_record_ids[lo:lo+batch_size], y.cpu()
                    record_fn_sums.scatter_add_(0, ids[cpu_y], F.softplus(boundary-s[y]).cpu())
                    record_fp_sums.scatter_add_(0, ids[~cpu_y], F.softplus(s[~y]-boundary).cpu())
        fnr = train_fn / float(positive.sum())
        smooth_fn = smooth_fn_sum / float(positive.sum())
        if by_record:
            record_fn = record_fn_sums.to(device) / positive_counts.clamp_min(1.)
            record_fp = record_fp_sums.to(device) / negative_counts.clamp_min(1.)
            multiplier = update_record_multipliers(multiplier, record_fn, record_reference, cfg['recall_dual_lr'])
        else:
            multiplier = update_recall_multiplier(multiplier, smooth_fn, reference_fn, cfg['recall_dual_lr'])
        seconds = time.perf_counter()-begin
        val = cached_evaluation(head, validation, val_dataset, cfg, device, reference=initial_val)
        metrics = val['metrics']
        eligible = four_metric_noninferiority(metrics, reference_val)
        improved = bool(eligible and metrics['iou'] > best)
        if improved:
            best, best_epoch = metrics['iou'], epoch
        stopped = bool(cfg.get('early_stop_patience', 0) and epoch+1 >= cfg.get('min_epochs', 1)
                       and epoch-best_epoch >= cfg['early_stop_patience'])
        record = {'epoch': epoch, 'lr': lr, 'train_seconds': seconds, 'epoch_seconds': time.perf_counter()-begin,
                  'loss_per_batch': total_loss / max(batches, 1), 'updates': batches,
                  'train_fp': train_fp, 'train_fn': train_fn, 'train_fnr': fnr,
                  'reference_train_fnr': reference_fnr, 'train_smooth_fn': smooth_fn,
                  'reference_train_smooth_fn': reference_fn, 'constraint_gap': smooth_fn-reference_fn,
                  'constraint': constraint, 'rate_multiplier_used': serial_multiplier(multiplier_used),
                  'rate_multiplier': serial_multiplier(multiplier), 'val_iou': metrics['iou'], 'val_acc': metrics['acc'],
                  'val_readouts': val['readouts'], 'eligible': bool(eligible), 'selected': improved,
                  'best_val_iou': best, 'best_epoch': best_epoch, 'backbone_updates': 0}
        if by_record:
            record['record_risks'] = [
                {'record': row['record'], 'positive_events': int(positive_counts[j]),
                 'negative_events': int(negative_counts[j]),
                 'smooth_fn': float(record_fn[j]), 'reference_smooth_fn': float(record_reference[j]),
                 'smooth_fp': float(record_fp[j]), 'reference_smooth_fp': float(record_fp_reference[j]),
                 'constraint_gap': float(record_fn[j]-record_reference[j]),
                 'multiplier_used': float(multiplier_used[j]), 'multiplier': float(multiplier[j])}
                for j, row in enumerate(rows)]
            active = positive_counts > 0
            gaps = record_fn[active]-record_reference[active]
            record.update(record_fn_mean=float(record_fn[active].mean()),
                          reference_record_fn_mean=float(record_reference[active].mean()),
                          record_fp_mean=float(record_fp[negative_counts > 0].mean()),
                          record_fn_worst_gap=float(gaps.max()),
                          record_fn_violations=int((gaps > 0).sum()))
        history.append(record)
        if callback is not None:
            callback(epoch, optimizer, val, record, improved, stopped, serial_multiplier(multiplier))
        multiplier_text = ('%.5f..%.5f' % (float(multiplier.min()), float(multiplier.max()))
                           if by_record else '%.5f' % multiplier)
        print('HEAD EPOCH %d/%d %.2fs IoU %.6f recall %.6f FP %d eligible=%s lambda=%s' %
              (epoch+1, cfg['epochs'], record['epoch_seconds'], metrics['iou'], metrics['acc'],
                metrics['fp'], eligible, multiplier_text), flush=True)
        if stopped:
            break
    return history


def train_cached(cfg, args):
    """Include preprocessing and training in the optional board-energy audit."""
    # Reject an existing experiment before the finally block can write an
    # energy report into it. The runner also checks this, but direct CLI calls
    # must preserve completed and failed experiment artifacts alike.
    out = Path(args.out)
    entries = list(out.iterdir()) if out.exists() else []
    adaptation_only = (cfg.get('backbone_adaptation') is True and len(entries) == 1 and
                       entries[0].name == 'backbone_adaptation' and entries[0].is_dir())
    if entries and not adaptation_only:
        raise ValueError('Output already exists; choose a new release')
    from tools.v16.profile_inference import BoardPower
    from tools.v16.train import write_json
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)
    meter = BoardPower(device)
    meter.start()
    start = time.perf_counter()
    completed = False
    try:
        result = _train_cached(cfg, args)
        completed = True
        return result
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        energy = meter.finish(start, end)
        energy.update(completed=completed, wall_seconds=end-start,
                      phases='statistics extraction, compact cache generation, head fitting and parity checks')
        if Path(args.out).is_dir():
            write_json(Path(args.out) / 'training_energy.json', energy)


def validate_initial_q99(values, expected, compare_values=True):
    """Reject malformed inherited normalization even in a limited-data probe."""
    q99 = np.asarray(values, dtype=np.float32)
    target = np.asarray(expected, dtype=np.float32)
    if q99.ndim != 1 or q99.shape != target.shape or not np.isfinite(q99).all():
        raise ValueError('Initial normalization must have the finite per-channel q99 shape')
    if compare_values and not np.allclose(q99, target, atol=1e-6, rtol=0):
        raise ValueError('Training-set representation normalization mismatch')
    return q99


def _train_cached(cfg, args):
    from tools.v16 import train as tr
    if not cfg.get('freeze_backbone', False):
        raise ValueError('r4 statistical coordinates require a frozen backbone; no silent stale-covariance fine-tuning')
    if not cfg.get('init_checkpoint'):
        raise ValueError('Covariance fitting requires a backbone with the matched observation window and stride')
    if args.max_chunks is not None:
        raise ValueError('Cached r4 uses --limit-records/--epochs for probes, not --max-chunks')
    out, device = Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        raise ValueError('r4 restart must use a new release; cache-only recovery is not implemented')
    if any((out / name).exists() for name in ('run_config.json', 'cache', 'last.pt')):
        raise ValueError('Output already exists; choose a new release')
    tr.seed_everything(int(cfg['seed']), bool(cfg['deterministic']))
    dataset, validation = tr.Records(cfg, 'train'), tr.Records(cfg, 'val')
    if args.limit_records:
        dataset.paths = dataset.paths[:args.limit_records]
        validation.paths = validation.paths[:args.limit_records]
    cfg['_training_scope'] = {'limit_records': args.limit_records, 'max_chunks': None}
    from tools.v16.queue_jobs import source_files
    cfg['_source_sha256'] = source_files(tr.ROOT)
    stats = tr.compute_overlap_stats(dataset.paths, cfg)
    initial = tr.load_checkpoint(cfg['init_checkpoint'], torch.device('cpu'))
    source_cfg = initial['config']
    for key in ('seed', 'channels', 'neuron', 'state_mode', 'window_ms', 'stride_ms', 'time_bins',
                'input_clip', 'height', 'width_px', 'pad_height', 'pad_width', 'norm', 'v_threshold',
                'tau_init_ms', 'tau_min_ms', 'tau_max_ms', 'readout_hidden'):
        if source_cfg.get(key) != cfg.get(key):
            raise ValueError('Initial backbone configuration mismatch: ' + key)
    if tr.tif_settings(source_cfg.get('tif')) != tr.tif_settings(cfg.get('tif')):
        raise ValueError('Initial BIR settings mismatch')
    q99 = validate_initial_q99(initial['stats']['q99'], stats['q99'],
                               compare_values=not args.limit_records)
    stats['q99'] = q99.tolist()
    net, head = tr.make_models(cfg, device)
    net.load_state_dict(initial['model'], strict=True)
    net.eval().requires_grad_(False)
    comm = tr.CommBudget.from_config(cfg, net)
    comm.mu = float(initial.get('comm_mu', 0.))
    if not math.isfinite(comm.mu) or comm.mu < 0:
        raise ValueError('Invalid initial communication multiplier')
    del initial
    initialization = {'checkpoint': cfg['init_checkpoint'], 'sha256': tr.sha256(cfg['init_checkpoint']),
                      'design': source_cfg.get('revision_architecture'), 'frozen': True}
    tr.write_json(out / 'run_config.json', {'design': tr.design_name(cfg), 'config': cfg,
                   'initialization': initialization, 'backbone_parameters': tr.count_parameters(net),
                   'fusion_parameters': tr.count_parameters(head), 'torch': torch.__version__,
                   'source': cfg['_source_sha256'], 'training': 'frozen V5-family backbone, cached head only',
                   'communication': 'inherited weights and budget; frozen backbone has no new communication update'})
    tr.write_json(out / 'train_stats.json', stats)
    tr.write_json(out / 'calibration.json', initialization)
    begin = time.perf_counter()
    report = fit_training_statistics(net, head, dataset, cfg, q99, device)
    tr.write_json(out / 'covariance_stats.json', report)
    cache = out / 'cache'
    cache.mkdir()
    export_cache(net, head, dataset, cfg, q99, device, cache / 'train')
    export_cache(net, head, validation, cfg, q99, device, cache / 'val')
    preprocessing_seconds = time.perf_counter()-begin
    rows = load_cache(cache / 'train', device, channels=head.channels)
    val_rows = load_cache(cache / 'val', device, channels=head.channels)
    # Compare cache and actual online forward before optimizing. This catches a
    # training/export feature path mismatch rather than trusting identical code.
    first = validation[0]
    online, _, _ = tr.infer_record(net, head, first, cfg, q99, device, modes=['main'], step=True)
    reference = torch.sigmoid(torch.as_tensor(val_rows[0]['z2'], device=device)).cpu().numpy()
    error = float(np.max(np.abs(online['main']-reference)))
    if error > 2e-6:
        raise RuntimeError('Online/cache initial parity failed: %.8g' % error)
    tr.write_json(out / 'cache_audit.json', {'initial_max_probability_error': error,
                  'statistics_split': 'train', 'threshold': cfg['threshold'],
                  'preprocessing_seconds': preprocessing_seconds, 'test_access': False})
    saved_history, best_value = [], [-float('inf')]
    def save(epoch, optimizer, val, record, improved, stopped, multiplier):
        if epoch == -1:
            best_value[0] = val['metrics']['iou']
            tr.write_json(out / 'initial_validation.json', val)
        if record is not None:
            saved_history.append(record)
            with (out / 'metrics.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        if improved:
            best_value[0] = val['metrics']['iou']
        payload = {'design': tr.design_name(cfg), 'model': net.state_dict(), 'fusion': head.state_dict(),
                   'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_val_iou': best_value[0],
                   'config': cfg, 'stats': stats, 'comm_mu': comm.mu, 'rate_multiplier': multiplier,
                   'initialization': initialization}
        def write_checkpoint(path):
            temp = path.with_suffix('.tmp')
            torch.save(payload, str(temp)); os.replace(str(temp), str(path))
        if improved or epoch == -1:
            write_checkpoint(out / ('best_val_iou_seed%d.pt' % cfg['seed']))
            tr.write_json(out / 'selected_validation.json', val)
        if epoch >= 0:
            write_checkpoint(out / 'last.pt')
            complete = epoch+1 == cfg['epochs'] or stopped
            tr.write_json(out / 'summary.json', {'design': tr.design_name(cfg), 'epochs_completed': epoch+1,
                'epochs_requested': cfg['epochs'], 'best_val_iou': best_value[0],
                'best_epoch': record['best_epoch']+1, 'early_stopped': stopped,
                'complete': complete and args.limit_records is None, 'probe_only': args.limit_records is not None,
                'training_scope': cfg['_training_scope'], 'seed': cfg['seed'], 'revision_variant': cfg['revision_variant'],
                'train_seconds_total': sum(r['train_seconds'] for r in saved_history),
                'epoch_seconds_total': sum(r['epoch_seconds'] for r in saved_history),
                'preprocessing_seconds': preprocessing_seconds, 'backbone_updates': 0,
                'stopping_rule': 'maximize validation IoU only when IoU/recall/Pd/Fa all noninferior to immutable second; zero-head fallback; test never used',
                'constraint': 'smooth false-negative risk relative to immutable training second-readout risk'})
    fit_cached_head(head, rows, val_rows, validation, cfg, device, save)
    # A nonzero trained head must also agree with the actual online feature path;
    # zero-initialization alone would only test the second-view baseline.
    selected = tr.load_checkpoint(out / ('best_val_iou_seed%d.pt' % cfg['seed']), device)
    head.load_state_dict(selected['fusion'])
    head.eval()
    online, _, _ = tr.infer_record(net, head, first, cfg, q99, device, modes=['main'], step=True)
    row = val_rows[0]
    logits = torch.as_tensor(row['z2'], device=device).clone()
    with torch.no_grad():
        ids = np.flatnonzero(row['paired'])
        for lo in range(0, len(ids), 32768):
            take = ids[lo:lo+32768]
            index = torch.as_tensor(take, device=device)
            logits[index] = head.forward_encoded(torch.as_tensor(row['q'][take], device=device), logits[index])
        cached = logits.sigmoid().cpu().numpy()
    error_after = float(np.max(np.abs(online['main']-cached)))
    differing_labels = int(np.count_nonzero((online['main'] >= np.float32(cfg['threshold'])) !=
                                            (cached >= np.float32(cfg['threshold']))))
    tr.write_json(out / 'cache_audit.json', {'initial_max_probability_error': error,
                  'selected_max_probability_error': error_after, 'selected_label_disagreements': differing_labels,
                  'selected_epoch': int(selected['epoch'])+1, 'statistics_split': 'train',
                  'threshold': cfg['threshold'], 'preprocessing_seconds': preprocessing_seconds, 'test_access': False})
    if error_after > 2e-5 or differing_labels:
        raise RuntimeError('Selected online/cache parity failed: error=%g labels=%d' % (error_after, differing_labels))
