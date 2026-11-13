"""Runtime contracts for immutable covariance validation and cache artifacts."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from tools.v16 import train as tr
from tools.v16 import train_covariance as runtime


class EncodedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = 0.

    def forward_encoded(self, q, z2):
        return z2 + self.delta * q[:, 0]


def validation_fixture():
    cfg = dict(threshold=.9, pd_detT=50, correct_thresh=.25, height=8, width_px=10)
    rows, sequences = [], []
    for i in range(2):
        t = np.array([0, 1, 2, 49, 50, 51, 74, 99, 100, 101, 149, 150, 199, 200, 249])
        label = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1], np.uint8)
        z2 = np.array([3., 3., 1., 3., 4., 3., 2.1, 3., 1., 2.3, -1., 3., 2.3, 3., 1.], np.float32)
        name = 'val_%03d.npz' % i
        q = np.zeros((len(t), 4), np.float32)
        q[:, 0] = np.where(np.arange(len(t)) % 2, 1., -1.)
        paired = np.ones(len(t), np.bool_)
        paired[[0, -1]] = False
        rows.append(dict(record=name, q=q, z1=z2-.4+i*.2, z2=z2,
                         label=label, paired=paired))
        sequences.append(SimpleNamespace(name=name, t=t, label=label,
                         target_id=label.astype(np.float64),
                         x=(np.arange(len(t)) + i) % cfg['width_px'],
                         y=np.arange(len(t)) % cfg['height']))
    return cfg, rows, sequences


def independently_accumulated_metrics(head, rows, sequences, cfg):
    """Keep the original per-record plus global FastRoc update as the oracle."""
    names = ('first', 'second', 'main')
    counts = {name: np.zeros(3, np.int64) for name in names}
    global_roc = {name: tr.FastRoc(cfg['pd_detT'], cfg['correct_thresh'],
                                 cfg['height'], cfg['width_px']) for name in names}
    records = []
    for row, seq in zip(rows, sequences):
        z2 = torch.as_tensor(row['z2'])
        main = z2.clone()
        ids = np.flatnonzero(row['paired'])
        main[ids] = head.forward_encoded(torch.as_tensor(row['q'][ids]), z2[ids])
        scores = dict(first=torch.sigmoid(torch.as_tensor(row['z1'])).numpy(),
                      second=torch.sigmoid(z2).numpy(), main=torch.sigmoid(main).numpy())
        result = dict(record=seq.name, readouts={})
        for name in names:
            p = scores[name]
            count = np.asarray(tr.window_confusion(p, seq.label, cfg['threshold'])[:3])
            counts[name] += count
            roc = tr.FastRoc(cfg['pd_detT'], cfg['correct_thresh'], cfg['height'], cfg['width_px'])
            for meter in (roc, global_roc[name]):
                meter.update(seq.t, p, seq.target_id, seq.label, seq.x, seq.y, cfg['threshold'])
            result['readouts'][name] = tr.metric_row(count, roc)
        records.append(result)
    return dict(readouts={name: tr.metric_row(counts[name], global_roc[name]) for name in names},
                metrics=tr.metric_row(counts['main'], global_roc['main']), per_record=records)


class CovarianceRuntimeTests(unittest.TestCase):
    def test_cached_fixed_views_and_roc_aggregation_match_original_metrics_exactly(self):
        cfg, rows, sequences = validation_fixture()
        head = EncodedHead()
        expected = independently_accumulated_metrics(head, rows, sequences, cfg)
        original_update = tr.FastRoc.update
        with mock.patch.object(tr.FastRoc, 'update', autospec=True, side_effect=original_update) as update:
            initial = runtime.cached_evaluation(head, rows, sequences, cfg, torch.device('cpu'))
        self.assertEqual(update.call_count, 3 * len(rows))
        for key, value in expected.items():
            self.assertEqual(initial[key], value)
        # Only the learned main path changes; singleton fallback stays fixed.
        head.delta = 1.1
        expected = independently_accumulated_metrics(head, rows, sequences, cfg)
        before = copy.deepcopy(initial)
        with mock.patch.object(tr.FastRoc, 'update', autospec=True, side_effect=original_update) as update:
            cached = runtime.cached_evaluation(head, rows, sequences, cfg, torch.device('cpu'),
                                               reference=initial)
        self.assertEqual(update.call_count, len(rows))
        for key, value in expected.items():
            self.assertEqual(cached[key], value)
        self.assertNotEqual(cached['metrics'], initial['metrics'])
        self.assertEqual(initial, before)
        # Serialized initial validation is also a usable reference.
        restored = json.loads(json.dumps(initial, allow_nan=False))
        self.assertEqual(runtime.cached_evaluation(head, rows, sequences, cfg, torch.device('cpu'),
                                                   reference=restored), cached)

    def test_cached_reference_rejects_changed_data_or_metric_policy(self):
        cfg, rows, sequences = validation_fixture()
        head = EncodedHead()
        reference = runtime.cached_evaluation(head, rows, sequences, cfg, torch.device('cpu'))
        for change in ('z1', 'z2', 'label', 't', 'target_id', 'x', 'y', 'threshold', 'pd_detT'):
            other_cfg, other_rows, other_sequences = copy.deepcopy((cfg, rows, sequences))
            if change in ('z1', 'z2'):
                other_rows[0][change][1] += .5
            elif change in ('threshold', 'pd_detT'):
                other_cfg[change] = .8 if change == 'threshold' else 25
            else:
                value = getattr(other_sequences[0], change)
                value[1] = 1-value[1] if change == 'label' else value[1]+1
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'reference'):
                runtime.cached_evaluation(head, other_rows, other_sequences, other_cfg,
                                          torch.device('cpu'), reference=reference)

    def test_refused_rerun_never_starts_meter_or_overwrites_existing_energy(self):
        from tools.v16 import profile_inference
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            energy = output / 'training_energy.json'
            energy.write_bytes(b'{"completed": true, "board_joules": 12345}\n')
            before = energy.read_bytes()
            with mock.patch.object(profile_inference, 'BoardPower') as meter, \
                    mock.patch.object(runtime, '_train_cached') as training:
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    runtime.train_cached({}, SimpleNamespace(out=str(output), device='cpu'))
            meter.assert_not_called()
            training.assert_not_called()
            self.assertEqual(energy.read_bytes(), before)

    def test_initial_q99_rejects_broadcast_shape_and_nonfinite_probe_values(self):
        expected = np.arange(12, dtype=np.float32) + 1
        np.testing.assert_array_equal(runtime.validate_initial_q99(expected.tolist(), expected), expected)
        for value in (expected.reshape(1, 12), expected[:1],
                      np.r_[np.nan, expected[1:]], np.r_[np.inf, expected[1:]]):
            for compare in (False, True):
                with self.subTest(compare=compare, shape=value.shape), self.assertRaises(ValueError):
                    runtime.validate_initial_q99(value, expected, compare_values=compare)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            runtime.validate_initial_q99(expected + 1, expected)
        np.testing.assert_array_equal(runtime.validate_initial_q99(expected + 1, expected,
                                                                  compare_values=False), expected+1)

    def test_cache_loader_rejects_invalid_fields_before_training(self):
        _, fixture, _ = validation_fixture()
        row = {key: value.copy() for key, value in fixture[0].items() if key != 'record'}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            entry = dict(record='val_000.npz', events=len(row['q']), pairs=int(row['paired'].sum()))
            (cache / 'manifest.json').write_text(json.dumps(dict(complete=True, records=[entry])), encoding='utf8')

            def write(values):
                np.savez(str(cache / 'val_000.npz'), **values)

            write(row)
            loaded = runtime.load_cache(cache, torch.device('cpu'), channels=2)
            self.assertEqual(loaded[0]['record'], entry['record'])
            for key in row:
                np.testing.assert_array_equal(loaded[0][key], row[key])
            with self.assertRaises(ValueError):
                runtime.load_cache(cache, torch.device('cpu'), channels=3)
            for change in ('missing', 'q_width', 'q_nan', 'z1_length', 'z2_nan',
                           'label_value', 'paired_dtype', 'pairs_count'):
                values = copy.deepcopy(row)
                if change == 'missing':
                    del values['z1']
                elif change == 'q_width':
                    values['q'] = values['q'][:, :3]
                elif change == 'q_nan':
                    values['q'][1, 0] = np.nan
                elif change == 'z1_length':
                    values['z1'] = values['z1'][:-1]
                elif change == 'z2_nan':
                    values['z2'][1] = np.nan
                elif change == 'label_value':
                    values['label'][1] = 2
                elif change == 'paired_dtype':
                    values['paired'] = values['paired'].astype(np.int64)
                else:
                    values['paired'][0] = True
                write(values)
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Invalid compact cache'):
                    runtime.load_cache(cache, torch.device('cpu'), channels=2)


if __name__ == '__main__':
    unittest.main()
