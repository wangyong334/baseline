"""Window-matched adaptation and frozen covariance runner contracts."""
import copy
import io
from pathlib import Path
import shutil
import tarfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import torch

from tests import test_v16_queue as legacy


ROOT, queue, runner = legacy.ROOT, legacy.queue, legacy.runner


class V16R5PipelineTests(unittest.TestCase):
    plan = legacy.V16QueueTests.plan
    enqueue = legacy.V16QueueTests.enqueue
    args = legacy.V16QueueTests.args
    pack = legacy.V16QueueTests.pack

    def setUp(self):
        legacy.V16QueueTests.setUp(self)
        self.release = 'context_r5_window_test'
        for window in (100, 200, 400):
            relative = Path('configs/v16/r5_w%d' % window)
            (self.project / relative).mkdir(parents=True)
            for variant in queue.VARIANTS:
                shutil.copyfile(str(ROOT / relative / (variant + '.yaml')),
                                str(self.project / relative / (variant + '.yaml')))
        self.initial = self.project / 'initial'
        self.initial.mkdir()
        torch.save({'config': {'seed': 37, 'window_ms': 100, 'stride_ms': 50},
                    'model': {'weight': torch.ones(1)}, 'stats': {'q99': [1.] * 12}},
                   str(self.initial / 'seed37.pt'))

    def r5_args(self, window=200):
        args = self.args()
        args.config_dir = 'configs/v16/r5_w%d' % window
        args.init_checkpoint = self.initial / 'seed37.pt'
        args.init_sha256 = runner.sha256(args.init_checkpoint)
        return args

    def execute(self, command, source):
        legacy.V16QueueTests.fake_execute(self, command, source)
        if '--mode' not in command or command[command.index('--mode') + 1] != 'train':
            return
        out = Path(command[command.index('--out') + 1])
        raw = runner.load_yaml(Path(command[command.index('--config') + 1]))
        runner.write_json(out / 'summary.json', {'epochs_completed': raw['TRAIN']['epochs'],
                          'complete': True, 'probe_only': False, 'best_epoch': 3})
        if out.name != 'backbone_adaptation':
            return
        checkpoint = {'config': runner.flatten(raw), 'epoch': 2, 'model': {'weight': torch.ones(1)},
                      'stats': {'q99': [1.] * 12}}
        torch.save(checkpoint, str(out / 'best_val_iou_seed37.pt'))
        torch.save(checkpoint, str(out / 'last.pt'))
        for name in ('calibration.json', 'initialization.json', 'initial_validation.json', 'run_config.json'):
            runner.write_json(out / name, {'fixture': True})
        (out / 'metrics.jsonl').write_text('{"epoch": 0, "updates": 1}\n', encoding='utf8')

    def test_three_window_configs_keep_v5_and_scale_time_reporting(self):
        base = runner.load_yaml(self.project / 'configs/v14/overlap_conditional.yaml')
        expected = {100: (159, [[0, 15], [64, 79], [144, 158]], 20),
                    200: (79, [[0, 7], [32, 39], [72, 78]], 10),
                    400: (39, [[0, 3], [16, 19], [36, 38]], 5)}
        for window, (n, segments, warmup) in expected.items():
            configs = runner.validate_configs(self.project, 'configs/v16/r5_w%d' % window)
            for variant, raw in configs.items():
                self.assertEqual(raw['NET'], base['NET'])
                self.assertEqual(raw['CALIB'], base['CALIB'])
                self.assertEqual((raw['DATA']['window_ms'], raw['DATA']['stride_ms'], raw['DATA']['n_windows']),
                                 (window, window // 2, n))
                self.assertEqual(raw['EVAL']['segments'], segments)
                self.assertEqual(raw['EVAL']['timing_warmup_windows'], warmup)
                self.assertEqual(raw['CONTEXT']['revision_variant'], variant)
                self.assertEqual(raw['TRAIN']['constraint'], 'record_fn_reference')
                self.assertTrue(raw['TRAIN']['freeze_backbone'])
                self.assertTrue(raw['TRAIN']['backbone_adaptation'])

    def test_explicit_window_family_rejects_disabled_adaptation_or_mismatched_window(self):
        path = self.project / 'configs/v16/r5_w200/full.yaml'
        original = runner.load_yaml(path)
        import yaml
        for section, key, value in (('TRAIN', 'backbone_adaptation', False),
                                    ('DATA', 'window_ms', 100), ('DATA', 'stride_ms', 50),
                                    ('DATA', 'n_windows', 159),
                                    ('TRAIN', 'constraint', 'smooth_fn_reference')):
            changed = copy.deepcopy(original)
            changed[section][key] = value
            path.write_text(yaml.safe_dump(changed), encoding='utf8')
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_configs(self.project, 'configs/v16/r5_w200')
        path.write_text(yaml.safe_dump(original), encoding='utf8')
        # A new directory name cannot silently opt into another time base.
        other = self.project / 'configs/v16/unregistered_window'
        shutil.copytree(str(path.parent), str(other))
        with self.assertRaises(ValueError):
            runner.validate_configs(self.project, 'configs/v16/unregistered_window')

    def test_constraint_enum_and_head_batch_size_are_validated_before_training(self):
        cfg = runner.flatten(runner.load_yaml(self.project / 'configs/v16/r5_w100/full.yaml'))
        for constraint in ('smooth_fn_reference', 'record_fn_reference'):
            runner.validate_covariance_protocol(dict(cfg, constraint=constraint))
        for value in (0, -1, True, 3.5, '32', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'head_batch_size'):
                runner.validate_covariance_protocol(dict(cfg, head_batch_size=value))
        with self.assertRaisesRegex(ValueError, 'constraint'):
            runner.validate_covariance_protocol(dict(cfg, constraint='test_tuned'))

    def test_adaptation_config_uses_only_second_readout_and_trainable_backbone(self):
        raw = runner.runtime_config('full', 37, self.release, self.data, self.project,
                                    self.project, 'configs/v16/r5_w400', self.initial / 'seed37.pt')
        out = Path(raw['TRAIN']['save_root']) / 'backbone_adaptation'
        stage = runner.adaptation_runtime_config(raw, out)
        self.assertEqual(stage['CONTEXT'], {'revision_architecture': 'evidence', 'revision_variant': 'second',
                                          'evidence_hidden': 8, 'evidence_kappa': 1.})
        self.assertEqual(stage['NET'], raw['NET'])
        self.assertEqual(stage['CALIB'], raw['CALIB'])
        self.assertEqual(stage['DATA']['window_ms'], 400)
        self.assertFalse(stage['TRAIN']['freeze_backbone'])
        self.assertEqual(stage['TRAIN']['fusion_loss'], 'bce')
        self.assertEqual(stage['TRAIN']['initialization_policy'], 'window_adaptation')
        self.assertEqual((stage['TRAIN']['epochs'], stage['TRAIN']['min_epochs'],
                          stage['TRAIN']['early_stop_patience']), (20, 8, 5))
        self.assertEqual((stage['TRAIN']['lr'], stage['TRAIN']['lr_end']), (1e-4, 1e-5))
        self.assertTrue(stage['TRAIN']['validate_initial'])
        self.assertEqual(stage['TRAIN']['selection_rule'], 'four_metric_noninferiority')
        self.assertEqual(stage['TRAIN']['recall_tolerance'], 0.)
        self.assertEqual(stage['TRAIN']['pd_tolerance'], 0.)
        self.assertNotIn('constraint', stage['TRAIN'])

    def test_runner_adapts_then_trains_head_and_only_then_accesses_test(self):
        args = self.r5_args()
        seen = []
        def execute(command, source):
            seen.append((list(command), runner.load_yaml(Path(command[command.index('--config') + 1]))))
            self.execute(command, source)
        with mock.patch.object(runner, 'ROOT', self.project), \
                mock.patch.object(runner, 'execute', side_effect=execute), redirect_stdout(io.StringIO()):
            result = runner.run(args)
        self.assertEqual(len(seen), 7)
        self.assertTrue(seen[0][0][3].endswith('runtime_preflight.py'))
        self.assertEqual(seen[1][0][-2:], ['--mode', 'train'])
        self.assertTrue(seen[2][0][3].endswith('runtime_preflight.py'))
        self.assertEqual(seen[3][0][-2:], ['--mode', 'train'])
        self.assertEqual(seen[4][0][-2:], ['--split', 'val'])
        self.assertEqual(seen[5][0][-2:], ['--split', 'test'])
        self.assertTrue(seen[6][0][3].endswith('profile_inference.py'))
        for command, cfg in seen[:2]:
            self.assertNotIn('--split', command)
            self.assertEqual(cfg['CONTEXT']['revision_variant'], 'second')
        adapted = result['backbone_adaptation']['checkpoint']
        for _, cfg in seen[2:]:
            self.assertEqual(cfg['TRAIN']['init_checkpoint'], adapted['path'])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['initialization_chain'][0]['sha256'], args.init_sha256)
        self.assertEqual(result['initialization']['sha256'], adapted['sha256'])
        self.assertEqual(result['backbone_adaptation']['checkpoint']['selected_epoch'], 3)
        self.assertFalse(result['backbone_adaptation']['test_access'])
        self.assertGreaterEqual(result['elapsed_seconds'], result['backbone_adaptation']['elapsed_seconds'])
        for name in ('initialization.json', 'calibration.json', 'train_stats.json', 'summary.json',
                     'metrics.jsonl', 'last.pt', 'best_val_iou_seed37.pt'):
            relative = (Path(result['out']) / 'backbone_adaptation' / name).relative_to(self.project).as_posix()
            self.assertIn(relative, result['artifact_sha256'])

    def test_adaptation_failure_records_failed_manifest_and_prevents_head_training(self):
        args = self.r5_args()
        with mock.patch.object(runner, 'ROOT', self.project), \
                mock.patch.object(runner, 'execute', side_effect=ValueError('adaptation preflight failed')) as execute:
            with self.assertRaisesRegex(ValueError, 'preflight'):
                runner.run(args)
        self.assertEqual(execute.call_count, 1)
        meta = self.project / 'run/v16_context' / self.release / 'jobs/full_s37'
        manifest = runner.read_json(meta / 'manifest.json')
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['backbone_adaptation']['status'], 'preflight')

    def test_wrong_window_adapted_checkpoint_stops_before_head(self):
        def execute(command, source):
            self.execute(command, source)
            if '--mode' in command and command[-1] == 'train':
                out = Path(command[command.index('--out') + 1])
                path = out / 'best_val_iou_seed37.pt'
                try:
                    checkpoint = torch.load(str(path), weights_only=False)
                except TypeError:
                    checkpoint = torch.load(str(path))
                checkpoint['config']['window_ms'] = 100
                torch.save(checkpoint, str(path))
        with mock.patch.object(runner, 'ROOT', self.project), \
                mock.patch.object(runner, 'execute', side_effect=execute) as execute, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'window_ms'):
                runner.run(self.r5_args())
        self.assertEqual(execute.call_count, 2)

    def test_initial_validation_checkpoint_is_a_valid_adaptation_fallback(self):
        raw = runner.runtime_config('full', 37, self.release, self.data, self.project,
                                    self.project, 'configs/v16/r5_w100', self.initial / 'seed37.pt')
        stage = runner.adaptation_runtime_config(raw, self.project / 'adaptation')
        checkpoint = self.initial / 'fallback.pt'
        torch.save({'config': runner.flatten(stage), 'epoch': -1, 'model': {'weight': torch.ones(1)},
                    'stats': {'q99': [1.] * 12}}, str(checkpoint))
        result = runner.validate_adapted_checkpoint(checkpoint, stage, {'epochs_completed': 8, 'best_epoch': 0})
        self.assertEqual(result['selected_epoch'], 0)

    def test_complete_queue_package_contains_both_stage_weights_and_provenance(self):
        plan = self.plan(config_dir='configs/v16/r5_w200', variants=('full',),
                         seeds=(37,), init_root=self.initial)
        self.enqueue(plan)
        args = self.r5_args()
        args.init_checkpoint = Path(plan['initializations']['37']['path'])
        with mock.patch.object(runner, 'ROOT', self.project), \
                mock.patch.object(runner, 'execute', side_effect=self.execute), redirect_stdout(io.StringIO()):
            result = runner.run(args)
        job = plan['jobs'][0]
        log = self.scheduler / 'run/queue/gpu1/logs' / (job['name'] + '.log')
        log.parent.mkdir(parents=True)
        log.write_text('completed both stages', encoding='utf8')
        package = self.pack()
        with tarfile.open(str(package)) as archive:
            names = set(archive.getnames())
        prefix = 'log/v16_context/%s/full_s37/' % self.release
        for suffix in ('best_val_iou_seed37.pt', 'backbone_adaptation/best_val_iou_seed37.pt',
                       'backbone_adaptation/last.pt', 'backbone_adaptation/calibration.json',
                       'backbone_adaptation/initialization.json', 'backbone_adaptation/initial_validation.json',
                       'backbone_adaptation/train_stats.json', 'backbone_adaptation/metrics.jsonl',
                       'backbone_adaptation/summary.json', 'backbone_adaptation/adaptation_result.json'):
            self.assertIn(prefix + suffix, names)
        self.assertIn('original_initialization', result)


if __name__ == '__main__':
    unittest.main()
