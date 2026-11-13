"""Train-only alternating statistics for a from-scratch Hotelling readout.

Backbone optimization uses one immutable set of observer moments per epoch.
The current backbone is then replayed on training recordings before validation,
so a saved checkpoint contains weights and moments from the same model version.
No validation/test observations or labels enter this module. Python 3.8.
"""
import hashlib
import json
from pathlib import Path
import time

import torch


def tensor_state_sha256(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf8'))
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def backbone_sha256(net):
    return tensor_state_sha256(net.state_dict())


def statistics_sha256(head):
    return tensor_state_sha256(dict(head.named_buffers()))


def statistics_path(out, epoch):
    name = 'initial.json' if epoch < 0 else 'epoch_%03d.json' % (epoch + 1)
    return Path(out) / 'hotelling_statistics' / name


@torch.no_grad()
def refresh_statistics(net, head, dataset, cfg, q99, device, out, epoch, split):
    """Replay only train, preserving calibrated parameters and caller modes."""
    if split != 'train':
        raise ValueError('Hotelling statistics may only fit the training split')
    if cfg.get('covariance_feature_source') != 'dec1':
        raise ValueError('Hotelling fitting requires dec1 descriptors')
    paths = getattr(dataset, 'paths', None)
    if paths is not None and any(Path(path).parent.name != 'train' for path in paths):
        raise ValueError('Hotelling statistics dataset contains a non-training path')
    from tools.v16.train_covariance import fit_training_statistics
    from tools.v16.train import write_json
    started = time.perf_counter()
    model_identity = backbone_sha256(net)
    calibration = {name: value.detach().clone() for name, value in head.named_parameters()}
    previous_net_mode, previous_head_mode = net.training, head.training
    net.eval()
    head.eval()
    try:
        report = fit_training_statistics(net, head, dataset, cfg, q99, device)
    finally:
        net.train(previous_net_mode)
        head.train(previous_head_mode)
    if backbone_sha256(net) != model_identity:
        raise RuntimeError('Training-statistics replay changed backbone parameters or buffers')
    if any(not torch.equal(calibration[name], value.detach()) for name, value in head.named_parameters()):
        raise RuntimeError('Statistics refresh changed trainable Hotelling calibration')
    report.update(epoch=int(epoch) + 1, split='train', backbone_sha256=model_identity,
                  statistics_sha256=statistics_sha256(head), test_access=False,
                  train_only=True, feature_source='dec1', seconds=time.perf_counter() - started,
                  refresh_policy='after each backbone epoch, before validation and checkpoint selection')
    target = statistics_path(out, epoch)
    write_json(target, report)
    write_json(Path(out) / 'hotelling_stats_latest.json', report)
    if epoch < 0:
        write_json(Path(out) / 'hotelling_stats_initial.json', report)
    print('HOTELLING STATISTICS epoch %d train-only %.2fs' % (epoch + 1, report['seconds']), flush=True)
    return report


def resume_statistics(net, head, out, completed_epoch):
    """Restore the report bound to last.pt, ignoring an uncommitted newer replay."""
    from tools.v16.train import write_json, load_checkpoint
    out = Path(out)
    path = statistics_path(out, completed_epoch)
    report = json.loads(path.read_text(encoding='utf8'))
    if (report.get('epoch') != completed_epoch + 1 or report.get('split') != 'train' or
            report.get('backbone_sha256') != backbone_sha256(net) or
            report.get('statistics_sha256') != statistics_sha256(head)):
        raise ValueError('Resumed Hotelling statistics do not match the saved backbone and observer')
    selected = json.loads((out / 'hotelling_stats_selected.json').read_text(encoding='utf8'))
    best_paths = list(out.glob('best_val_iou_seed*.pt'))
    if len(best_paths) != 1:
        raise ValueError('Resumed Hotelling run must have one selected checkpoint')
    best = load_checkpoint(best_paths[0], torch.device('cpu'))
    buffers = {name: best['fusion'][name] for name, _ in head.named_buffers()}
    if (int(best['epoch']) > completed_epoch or selected.get('epoch') != int(best['epoch']) + 1 or
            selected.get('backbone_sha256') != tensor_state_sha256(best['model']) or
            selected.get('statistics_sha256') != tensor_state_sha256(buffers)):
        raise ValueError('Selected Hotelling checkpoint/statistics are inconsistent with the committed resume epoch')
    write_json(Path(out) / 'hotelling_stats_latest.json', report)
    return report


def write_training_audit(net, head, out, epoch, best_epoch, complete):
    """Record scratch provenance and model/statistics identities without rerunning data."""
    from tools.v16.train import write_json, sha256
    out = Path(out)
    initial = json.loads((out / 'initialization.json').read_text(encoding='utf8'))
    latest = json.loads((out / 'hotelling_stats_latest.json').read_text(encoding='utf8'))
    selected = json.loads((out / 'hotelling_stats_selected.json').read_text(encoding='utf8'))
    current_hash = backbone_sha256(net)
    if latest['backbone_sha256'] != current_hash or latest['statistics_sha256'] != statistics_sha256(head):
        raise ValueError('Hotelling latest statistics are stale at checkpoint audit')
    if selected['epoch'] != best_epoch + 1:
        raise ValueError('Hotelling selected statistics refer to the wrong validation epoch')
    report = {'from_scratch': True, 'initial_checkpoint': None, 'complete': bool(complete),
              'epoch': epoch + 1, 'selected_epoch': best_epoch + 1,
              'initial_backbone_sha256': initial['calibrated_backbone_sha256'],
              'last_backbone_sha256': current_hash,
              'selected_backbone_sha256': selected['backbone_sha256'],
              'backbone_changed_from_initial': current_hash != initial['calibrated_backbone_sha256'],
              'selected_backbone_changed_from_initial': selected['backbone_sha256'] != initial['calibrated_backbone_sha256'],
              'statistics_split': 'train', 'test_access_during_training': False,
              'statistics_refreshed_after_final_epoch': latest['epoch'] == epoch + 1,
              'latest_statistics_sha256': latest['statistics_sha256'],
              'selected_statistics_sha256': selected['statistics_sha256'],
              'statistics_artifact_sha256': {
                  path.relative_to(out).as_posix(): sha256(path)
                  for path in sorted((out / 'hotelling_statistics').glob('*.json'))}}
    write_json(out / 'hotelling_training_audit.json', report)
    return report
