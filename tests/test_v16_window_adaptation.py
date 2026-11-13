"""Window changes require explicit adaptation, not frozen weight relabelling."""
import copy
import unittest

import numpy as np
import torch

from tools.v16 import train as tr
from tools.v16 import runtime_preflight as preflight
from tools.v16.run_experiment import adaptation_runtime_config, load_yaml, flatten


class WindowAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.raw = load_yaml(tr.ROOT / 'configs/v16/r5_w100/full.yaml')
        self.source = flatten(self.raw)
        self.checkpoint = {'config': self.source, 'stats': {'q99': [1.] * 12}, 'comm_mu': .1}

    def stage(self, window):
        raw = load_yaml(tr.ROOT / ('configs/v16/r5_w%d/full.yaml' % window))
        return flatten(adaptation_runtime_config(raw, 'probe'))

    def test_only_explicit_adaptation_allows_new_window_and_normalization(self):
        for window in (100, 200, 400):
            cfg = self.stage(window)
            q99 = np.ones(12) * (1 if window == 100 else 2)
            changed, mu = tr.validate_backbone_initialization(self.checkpoint, cfg, q99)
            self.assertEqual(changed, window != 100)
            self.assertEqual(mu, .1)
        with self.assertRaisesRegex(ValueError, 'window mismatch'):
            tr.validate_backbone_initialization(self.checkpoint,
                dict(self.stage(200), initialization_policy='matched'), np.ones(12))

    def test_adaptation_never_bypasses_model_seed_or_malformed_stats(self):
        for patch in ({'seed': 38}, {'channels': [8, 16, 32, 32]},
                      {'freeze_backbone': True}, {'revision_variant': 'full'}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                tr.validate_backbone_initialization(self.checkpoint,
                    dict(self.stage(200), **patch), np.ones(12))
        bad = copy.deepcopy(self.checkpoint)
        bad['stats']['q99'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'normalization'):
            tr.validate_backbone_initialization(bad, self.stage(200), np.ones(12))
        with self.assertRaisesRegex(ValueError, 'normalization'):
            tr.validate_backbone_initialization(self.checkpoint, self.stage(100), np.ones(12)*2)

    def test_checkpoint_load_cannot_overwrite_new_clock(self):
        old, _ = tr.make_models(self.stage(100), torch.device('cpu'))
        new, _ = tr.make_models(self.stage(400), torch.device('cpu'))
        new.load_state_dict(old.state_dict(), strict=True)
        cells = [m for m in new.modules() if hasattr(m, 'dt_ms')]
        self.assertTrue(cells)
        self.assertTrue(all(m.dt_ms == 200 for m in cells))

    def test_real_backward_and_event_coverage_at_each_timebase(self):
        torch.set_num_threads(2)
        for window in (100, 200, 400):
            result = preflight.check(self.stage(window), torch.device('cpu'))
            self.assertTrue(result['passed'])
            self.assertEqual(result['training']['committed_events'], 480)
            self.assertEqual(result['stride_ms'], window//2)


if __name__ == '__main__':
    unittest.main()
