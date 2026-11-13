"""Feature selection preserves V5 scores and online/cache observation identity."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from tests.test_v14_training import synthetic_record
from tools.v16 import train as tr
from tools.v16 import train_covariance as cached
from tools.v16 import runtime_preflight as preflight
from tools.v16.run_experiment import (adaptation_runtime_config, flatten,
                                     load_yaml, validate_covariance_protocol)


class CovarianceFeatureSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.raw = load_yaml(tr.ROOT / 'configs/v16/r4/full.yaml')
        self.cfg = flatten(self.raw)
        self.cfg.update(height=8, width_px=8, pad_height=8, pad_width=8,
                        total_ms=300, n_windows=5, eval_chunk=2, timing_warmup_windows=0)
        self.device = torch.device('cpu')
        self.q99 = np.full(12, np.log1p(2), dtype=np.float32)
        self.seq = synthetic_record(37)

    def models(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        tr.seed_everything(37, True)
        net, head = tr.make_models(cfg, self.device)

        def factory():
            source = tr.OverlapWindowSource(self.seq, cfg, self.q99, self.device)

            def windows():
                for k in range(self.seq.n_windows):
                    x, ev, _, _ = source.window(k)
                    yield x, ev

            yield windows()

        tr.calibrate_gains(net, factory, .99, 32, 1, .05, 20., 4037)
        net.eval()
        return net, head

    def test_all_descriptor_sources_preserve_exact_original_readout_logits(self):
        net, _ = self.models()
        source = tr.OverlapWindowSource(self.seq, self.cfg, self.q99, self.device)
        x, ev, _, _ = source.chunk(0, self.seq.n_windows)
        _, _, info = net.forward_chunk(x, ev, None, return_features=True,
                                       skip_readout=True, feature_taps=True)
        features, context = info['features'], info['context_features']
        values = features[ev['t'], ev['b'], :, ev['y'], ev['x']]
        original = net._readout(values, ev)
        for name, dimension in (('mlp', 32), ('dec1', 12), ('multilevel', 60)):
            logits, hidden = tr.readout_with_hidden(net, features, ev, context, source=name)
            self.assertTrue(torch.equal(logits, original), name)
            self.assertEqual(tuple(hidden.shape), (len(original), dimension))
            if name == 'dec1':
                self.assertTrue(torch.equal(hidden, values))
        old_logits, old_hidden = tr.readout_with_hidden(net, features, ev)
        self.assertTrue(torch.equal(old_logits, original))
        self.assertEqual(old_hidden.shape[1], 32)
        _, legacy_context = tr.readout_with_hidden(net, features, ev, context)
        _, explicit_context = tr.readout_with_hidden(net, features, ev, context, source='multilevel')
        self.assertTrue(torch.equal(legacy_context, explicit_context))
        with self.assertRaisesRegex(ValueError, 'feature taps'):
            tr.readout_with_hidden(net, features, ev, source='multilevel')
        with self.assertRaisesRegex(ValueError, 'feature source'):
            tr.readout_with_hidden(net, features, ev, source='unknown')

    def test_model_factory_defaults_and_basis_are_explicit_without_changing_backbone(self):
        defaults = dict(self.cfg)
        variants = (defaults, dict(defaults, covariance_feature_source='multilevel',
                                   covariance_channel_basis='shared_fisher'),
                    dict(defaults, covariance_feature_source='dec1',
                         covariance_channel_basis='joint_fisher'))
        states = []
        for cfg, dimension, basis in zip(variants, (60, 60, 12),
                                         ('shared_fisher', 'shared_fisher', 'joint_fisher')):
            tr.seed_everything(37, True)
            net, head = tr.make_models(cfg, self.device)
            self.assertEqual(head.feature_dim, dimension)
            self.assertEqual(head.channel_basis, basis)
            self.assertEqual(tr.covariance_feature_dim(cfg), dimension)
            states.append((net.state_dict(), head.state_dict()))
        for name, value in states[0][0].items():
            self.assertTrue(all(torch.equal(value, state[0][name]) for state in states[1:]), name)
        for name, value in states[0][1].items():
            self.assertTrue(torch.equal(value, states[1][1][name]), name)

    def test_protocol_checks_source_basis_and_actual_projection_capacity(self):
        for source, maximum in (('multilevel', 61), ('dec1', 13)):
            cfg = dict(self.cfg, covariance_feature_source=source)
            for basis, minimum in (('shared_fisher', 1), ('joint_fisher', 3)):
                for channels in (minimum, maximum):
                    validate_covariance_protocol(dict(cfg, covariance_channel_basis=basis,
                                                      covariance_channels=channels))
            for channels in (0, maximum + 1, True, 2.5):
                with self.subTest(source=source, channels=channels), self.assertRaises(ValueError):
                    validate_covariance_protocol(dict(cfg, covariance_channels=channels))
            with self.assertRaisesRegex(ValueError, 'joint_fisher'):
                validate_covariance_protocol(dict(cfg, covariance_channel_basis='joint_fisher',
                                                  covariance_channels=2))
        for patch in ({'covariance_feature_source': 'mlp'}, {'covariance_channel_basis': 'unknown'}):
            with self.assertRaises(ValueError):
                validate_covariance_protocol(dict(self.cfg, **patch))

    def test_nonzero_head_online_and_exported_cache_match_for_each_source(self):
        for feature_source, dimension in (('multilevel', 60), ('dec1', 12)):
            cfg = dict(self.cfg, covariance_feature_source=feature_source,
                       covariance_channel_basis='joint_fisher')
            net, head = self.models(cfg)
            with mock.patch.object(net, 'forward', wraps=net.forward) as forward:
                obs = cached.observations(net, self.seq, cfg, self.q99, self.device)
            self.assertTrue(all(call.kwargs['feature_taps'] == (feature_source == 'multilevel')
                                for call in forward.call_args_list))
            self.assertEqual(obs[2].shape, (self.seq.n_events, dimension))
            self.assertEqual(obs[3].shape, (self.seq.n_events, dimension))
            paired = self.seq.coverage_count == 2
            head.fit_statistics(*[value[paired] for value in obs], self.seq.label[paired])
            with torch.no_grad():
                head.output.weight.normal_(0., .04)
            with mock.patch.object(net, 'forward', wraps=net.forward) as forward:
                online, _, _ = tr.infer_record(net, head, self.seq, cfg, self.q99,
                                               self.device, step=True)
            self.assertTrue(all(call.kwargs['feature_taps'] == (feature_source == 'multilevel')
                                for call in forward.call_args_list))
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                destination = Path(directory) / 'cache'
                cached.export_cache(net, head, [self.seq], cfg, self.q99, self.device, destination)
                row = cached.load_cache(destination, self.device, channels=head.channels)[0]
                expected = row['z2'].copy()
                with torch.no_grad():
                    expected[paired] = head.forward_encoded(torch.from_numpy(row['q'][paired]),
                                                            torch.from_numpy(row['z2'][paired])).numpy()
                expected = torch.sigmoid(torch.from_numpy(expected)).numpy()
            np.testing.assert_allclose(online['main'], expected, rtol=0, atol=2e-6)
            for name, logits in zip(('first', 'second'), obs[:2]):
                np.testing.assert_allclose(online[name], torch.sigmoid(torch.from_numpy(logits)).numpy(),
                                           rtol=0, atol=2e-6)
            self.assertGreater(float(np.max(np.abs(online['main'] - online['second']))), 1e-6)
            with mock.patch.object(net, 'forward', wraps=net.forward) as forward:
                diagnostics, _, _ = tr.infer_record(net, head, self.seq, cfg, self.q99,
                                                    self.device, modes=['first', 'second'], step=True)
            self.assertTrue(all(not call.kwargs['feature_taps'] for call in forward.call_args_list))
            for name in diagnostics:
                np.testing.assert_array_equal(diagnostics[name], online[name])

    def test_adaptation_uses_mlp_even_when_covariance_options_are_present(self):
        raw = load_yaml(tr.ROOT / 'configs/v16/r5_w100/full.yaml')
        raw['CONTEXT'].update(covariance_feature_source='dec1', covariance_channel_basis='joint_fisher')
        cfg = flatten(adaptation_runtime_config(raw, 'probe'))
        self.assertEqual(tr.readout_feature_source(cfg), 'mlp')
        _, head = tr.make_models(cfg, self.device)
        self.assertEqual(head.feature_dim, 32)

    def test_dec1_joint_basis_runs_real_preflight_training_and_online_inference(self):
        cfg = dict(self.cfg, covariance_feature_source='dec1', covariance_channel_basis='joint_fisher')
        with contextlib.redirect_stdout(io.StringIO()):
            result = preflight.check(cfg, self.device)
        self.assertTrue(result['passed'])
        self.assertEqual(result['training']['updates'], 2)
        self.assertEqual(result['training']['committed_events'], 480)
        self.assertEqual(result['synthetic_training_statistics']['raw_channels'], 13)
        self.assertEqual(result['synthetic_training_statistics']['channel_basis'], 'joint_fisher')


if __name__ == '__main__':
    unittest.main()
