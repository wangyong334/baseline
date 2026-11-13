"""Standalone GPU-2 orchestration without a pool or real GPU workload."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tests import test_v16_r5_pipeline as fixtures
from tools.v16 import run_standalone as standalone


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        fixtures.V16R5PipelineTests.setUp(self)
        self.release = 'context_standalone_gpu2_test'
        shutil.copyfile(str(fixtures.ROOT / 'tools/v16/run_standalone.py'),
                        str(self.project / 'tools/v16/run_standalone.py'))

    def plan(self):
        plan = standalone.prepare_plan(self.project, self.data, self.release,
                                       'configs/v16/r5_w200', self.initial / 'seed37.pt',
                                       runtime_checks=False)
        plan['runtime'] = {'gpu': {'index': 2, 'uuid': 'GPU-physical-two', 'name': 'test GPU'}}
        return plan

    def stream(self, command, source, environment, log):
        def value(flag):
            return command[command.index(flag) + 1]
        args = argparse.Namespace(project_root=Path(value('--project-root')), data_root=Path(value('--data-root')),
                                  release=value('--release'), config_dir=value('--config-dir'),
                                  variant=value('--variant'), seed=int(value('--seed')), device=value('--device'),
                                  init_checkpoint=Path(value('--init-checkpoint')), init_sha256=value('--init-sha256'))
        Path(log).write_text('standalone fixture log\n', encoding='utf8')
        with mock.patch.object(fixtures.runner, 'ROOT', Path(source)), \
                mock.patch.object(fixtures.runner, 'execute',
                                  side_effect=lambda cmd, root: fixtures.V16R5PipelineTests.execute(self, cmd, root)), \
                mock.patch.dict(os.environ, environment), redirect_stdout(io.StringIO()):
            fixtures.runner.run(args)

    def complete(self):
        plan = self.plan()
        with mock.patch.object(standalone, 'stream_runner', side_effect=self.stream), redirect_stdout(io.StringIO()):
            result = standalone.run(plan)
        return plan, result

    def test_plan_is_read_only_and_command_is_one_fixed_gpu2_full_seed37_runner(self):
        plan = self.plan()
        self.assertFalse((self.project / 'run').exists())
        self.assertFalse((self.scheduler / 'run').exists())
        self.assertEqual(plan['physical_gpu'], 2)
        self.assertNotIn('scheduler_root', plan)
        command = standalone.runner_command(plan)
        for flag, expected in (('--variant', 'full'), ('--seed', '37'), ('--device', 'cuda:0'),
                               ('--config-dir', 'configs/v16/r5_w200')):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertEqual(Path(command[3]).parent.parent.parent, Path(plan['source']))
        self.assertEqual(command[command.index('--init-checkpoint')+1], plan['initializations']['37']['path'])
        self.assertFalse(any('hold' in item or 'gpu_pool' in item or 'gpu_queue' in item for item in command))
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            environment = standalone.child_environment(plan)
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'GPU-physical-two')
        self.assertEqual(environment['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
        self.assertEqual(environment['PYTHONDONTWRITEBYTECODE'], '1')

    def test_freeze_copies_source_and_initial_weights_and_rejects_same_release(self):
        plan = self.plan()
        manifest = standalone.freeze(plan)
        snapshot = Path(plan['source']) / 'model/context_revision.py'
        initial = Path(plan['initializations']['37']['path'])
        snapshot_bytes, initial_bytes = snapshot.read_bytes(), initial.read_bytes()
        (self.project / 'model/context_revision.py').write_text('later SFTP upload\n', encoding='utf8')
        (self.initial / 'seed37.pt').write_bytes(b'later initialization file')
        self.assertEqual(snapshot.read_bytes(), snapshot_bytes)
        self.assertEqual(initial.read_bytes(), initial_bytes)
        standalone.common.verify_source(manifest)
        with self.assertRaises(FileExistsError):
            standalone.freeze(plan)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.plan()

    def test_plan_preserves_virtualenv_python_symlink(self):
        executable = str(self.project / 'venv/bin/python')
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if str(path) == executable:
                return self.project / 'system/python'
            return original_resolve(path, *args, **kwargs)

        with mock.patch.object(standalone.sys, 'executable', executable), \
                mock.patch.object(Path, 'resolve', resolve):
            plan = self.plan()
        self.assertEqual(plan['python'], executable)
        self.assertEqual(standalone.runner_command(plan)[0], executable)

    def test_changed_source_before_freeze_fails_without_starting_runner(self):
        plan = self.plan()
        (self.project / 'model/context_revision.py').write_text('changed\n', encoding='utf8')
        with mock.patch.object(standalone, 'stream_runner') as stream:
            with self.assertRaisesRegex(ValueError, 'changed while freezing'):
                standalone.run(plan)
        stream.assert_not_called()
        state = json.loads((Path(plan['source']).parent / 'standalone.json').read_text(encoding='utf8'))
        self.assertEqual(state['status'], 'failed')

    def test_success_runs_frozen_pipeline_once_and_never_creates_queue_files(self):
        plan = self.plan()
        with mock.patch.object(standalone, 'stream_runner', side_effect=self.stream) as stream, \
                redirect_stdout(io.StringIO()):
            result = standalone.run(plan)
        self.assertEqual(stream.call_count, 1)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['cuda_visible_devices'], 'GPU-physical-two')
        self.assertEqual(result['source_sha256'], plan['files'])
        state = json.loads((Path(plan['source']).parent / 'standalone.json').read_text(encoding='utf8'))
        self.assertEqual(state['status'], 'complete')
        self.assertFalse((self.project / 'run/queue').exists())
        self.assertFalse((self.scheduler / 'run').exists())

    def test_runner_failure_is_recorded_and_cannot_be_packaged_as_complete(self):
        plan = self.plan()
        with mock.patch.object(standalone, 'stream_runner', side_effect=subprocess.CalledProcessError(7, ['runner'])), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                standalone.run(plan)
        state = json.loads((Path(plan['source']).parent / 'standalone.json').read_text(encoding='utf8'))
        self.assertEqual(state['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'not complete'):
            standalone.pack(self.project, self.release)

    def test_pack_contains_both_stages_initial_weights_source_and_standalone_log(self):
        plan, result = self.complete()
        cache = Path(plan['out']) / 'cache'
        cache.mkdir()
        (cache / 'unused.npz').write_bytes(b'large cache excluded')
        with redirect_stdout(io.StringIO()):
            archive = standalone.pack(self.project, self.release)
        with tarfile.open(str(archive)) as handle:
            names = set(handle.getnames())
            package = json.load(handle.extractfile('package_manifest.json'))
        self.assertTrue(package['complete'])
        self.assertEqual(package['physical_gpu'], 2)
        self.assertEqual(package['execution'], 'standalone')
        for path in (Path(plan['out']) / 'best_val_iou_seed37.pt',
                     Path(plan['out']) / 'backbone_adaptation/best_val_iou_seed37.pt',
                     Path(plan['out']) / 'backbone_adaptation/initialization.json',
                     Path(plan['initializations']['37']['path']), Path(plan['log']),
                     Path(plan['source']) / 'tools/v16/run_standalone.py'):
            self.assertIn(path.relative_to(self.project).as_posix(), names)
        self.assertFalse(any(name.endswith('.npz') or '/queue/' in name or '.vscode' in name for name in names))
        self.assertEqual(package['checkpoint_sha256'], result['checkpoint_sha256'])

    def test_pack_rejects_damaged_adaptation_artifacts(self):
        plan, _ = self.complete()
        artifact = Path(plan['out']) / 'backbone_adaptation/last.pt'
        artifact.write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, 'completed artifact is changed'):
            standalone.pack(self.project, self.release)
        artifact.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or empty'):
            standalone.pack(self.project, self.release)

    def test_pack_rejects_result_from_a_different_gpu(self):
        plan, result = self.complete()
        result['cuda_visible_devices'] = 'GPU-other-board'
        standalone.common.write_json(Path(plan['out']) / 'comparison_result.json', result)
        with self.assertRaisesRegex(ValueError, 'result GPU identity'):
            standalone.pack(self.project, self.release)

    def test_gpu_probe_resolves_only_physical_two_without_cuda_context(self):
        with mock.patch.object(standalone.subprocess, 'check_output', return_value='2, GPU-card-two, NVIDIA RTX\n') as query:
            identity = standalone.gpu2_identity()
        self.assertEqual(identity, {'index': 2, 'uuid': 'GPU-card-two', 'name': 'NVIDIA RTX'})
        self.assertEqual(query.call_args[0][0][:3], ['nvidia-smi', '-i', '2'])
        with mock.patch.object(standalone.subprocess, 'check_output', return_value='1, GPU-card-one, NVIDIA RTX\n'):
            with self.assertRaisesRegex(ValueError, 'physical GPU 2'):
                standalone.gpu2_identity()

    def test_run_requires_verified_gpu_before_creating_release(self):
        plan = self.plan()
        plan.pop('runtime')
        with self.assertRaisesRegex(ValueError, 'verified physical GPU 2'):
            standalone.run(plan)
        self.assertFalse((self.project / 'run').exists())

    def test_run_rejects_a_different_physical_gpu_before_creating_release(self):
        plan = self.plan()
        plan['runtime']['gpu']['index'] = 1
        with self.assertRaisesRegex(ValueError, 'verified physical GPU 2'):
            standalone.run(plan)
        self.assertFalse((self.project / 'run').exists())

    def test_stream_logs_child_output_and_propagates_nonzero_exit(self):
        log = self.project / 'child.log'
        with redirect_stdout(io.StringIO()) as output:
            standalone.stream_runner([sys.executable, '-B', '-u', '-c', "print('pipeline completed')"],
                                     self.project, dict(os.environ), log)
        self.assertIn('pipeline completed', output.getvalue())
        self.assertIn('pipeline completed', log.read_text(encoding='utf8'))
        with redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError) as error:
            standalone.stream_runner([sys.executable, '-B', '-u', '-c', 'raise SystemExit(4)'],
                                     self.project, dict(os.environ), self.project / 'failed.log')
        self.assertEqual(error.exception.returncode, 4)


if __name__ == '__main__':
    unittest.main()
