"""Random-initialized direct-HO deployment and complete-result contracts."""
import argparse
import copy
import io
from pathlib import Path
import shutil
import tarfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tests import test_v16_queue as legacy

ROOT, queue, runner = legacy.ROOT, legacy.queue, legacy.runner


class HotellingPipelineTests(unittest.TestCase):
    def setUp(self):
        legacy.V16QueueTests.setUp(self)
        self.release = 'context_r7_hotelling_scratch_test'
        directory = self.project / 'configs/v16/r7_hotelling'
        directory.mkdir(parents=True)
        for variant in queue.VARIANTS:
            shutil.copyfile(str(ROOT/'configs/v16/r7_hotelling'/(variant+'.yaml')),
                            str(directory/(variant+'.yaml')))
        for name in ('model/hotelling_readout.py', 'tools/v16/train_hotelling.py'):
            (self.project/name).write_text('# included source fixture\n', encoding='utf8')

    def plan(self, **kwargs):
        options = dict(config_dir='configs/v16/r7_hotelling', seeds=(37,), variants=('full',),
                       runtime_checks=False)
        options.update(kwargs)
        return queue.prepare_plan(self.project,self.scheduler,self.data,self.release,**options)

    def test_protocol_is_fifty_epoch_random_training_with_matched_ablations(self):
        configs=runner.validate_configs(self.project,'configs/v16/r7_hotelling')
        for variant,raw in configs.items():
            flat=runner.flatten(raw)
            self.assertEqual(flat['revision_variant'],variant)
            self.assertTrue(flat['from_scratch'])
            self.assertEqual((flat['epochs'],flat['early_stop_patience']),(50,0))
            self.assertEqual(flat['threshold'],.9)
            self.assertEqual(flat['covariance_feature_source'],'dec1')
            self.assertFalse(flat['freeze_backbone'])
            self.assertFalse(flat['backbone_adaptation'])
            self.assertEqual(flat['selection_rule'],'val_iou')

    def test_warm_start_freezing_adaptation_and_shortened_formal_run_are_rejected(self):
        raw=runner.load_yaml(self.project/'configs/v16/r7_hotelling/full.yaml')
        flat=runner.flatten(raw)
        for change in ({'from_scratch':False},{'freeze_backbone':True},{'backbone_adaptation':True},
                       {'require_initialization':True},{'init_checkpoint':'old.pt'},
                       {'initialization_policy':'window_adaptation'},{'adaptation_epochs':20},
                       {'hotelling_shrinkage':float('nan')},{'hotelling_shrinkage':False},
                       {'threshold':.8},{'fusion_loss':'rate_constrained'},
                       {'selection_rule':'four_metric_noninferiority'}, {'epochs':8},
                       {'early_stop_patience':5}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                runner.validate_hotelling_protocol(dict(flat,**change),formal=True)
        with self.assertRaisesRegex(ValueError,'forbids --init-root'):
            self.plan(init_root=self.project/'old_checkpoint')
        with self.assertRaisesRegex(ValueError,'forbids --init-checkpoint'):
            runner.runtime_config('full',37,self.release,self.data,self.project,self.project,
                                  'configs/v16/r7_hotelling',self.project/'old.pt')
        self.assertFalse((self.scheduler/'run').exists())

    def test_one_gpu1_job_contains_no_checkpoint_or_pool_management(self):
        plan=self.plan()
        self.assertEqual(len(plan['jobs']),1)
        self.assertEqual(plan['physical_gpu'],1)
        self.assertEqual(plan['initializations'],{})
        for name in ('model/hotelling_readout.py','tools/v16/train_hotelling.py'):
            self.assertIn(name,plan['files'])
        script=queue.job_script(plan,plan['jobs'][0],Path(plan['source']).parent/'release.json')
        self.assertIn('--seed 37',script)
        self.assertIn('CUDA_VISIBLE_DEVICES=1',script)
        for forbidden in ('--init-checkpoint','--resume','gpu_hold','kill ', 'nohup'):
            self.assertNotIn(forbidden,script)

    def fake_execute(self,command,source):
        self.commands.append(command)
        legacy.V16QueueTests.fake_execute(self,command,source)
        if '--mode' in command and command[command.index('--mode')+1]=='train':
            out=Path(command[command.index('--out')+1])
            for name in ('hotelling_stats_initial.json','hotelling_stats_latest.json',
                         'hotelling_stats_selected.json','initialization.json','hotelling_training_audit.json'):
                runner.write_json(out/name,{'mode':'random','statistics_split':'train'})
            for epoch in range(1,51):
                runner.write_json(out/'hotelling_statistics'/('epoch_%03d.json'%epoch),{'epoch':epoch})

    def complete(self):
        plan=self.plan()
        with redirect_stdout(io.StringIO()):
            queue.enqueue(plan)
        args=argparse.Namespace(project_root=self.project,data_root=self.data,release=self.release,
                               config_dir='configs/v16/r7_hotelling',variant='full',seed=37,
                               device='cpu',init_checkpoint=None,init_sha256=None)
        self.commands=[]
        with mock.patch.object(runner,'ROOT',Path(plan['source'])), \
                mock.patch.object(runner,'execute',side_effect=self.fake_execute),redirect_stdout(io.StringIO()):
            result=runner.run(args)
        log=self.scheduler/'run/queue/gpu1/logs'/(plan['jobs'][0]['name']+'.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text('completed fixture\n',encoding='utf8')
        return plan,result

    def test_complete_pipeline_hashes_all_refits_and_packs_provenance(self):
        plan,result=self.complete()
        self.assertTrue(result['trained_from_scratch'])
        self.assertIsNone(result['initialization'])
        self.assertEqual(len(self.commands),5)
        self.assertFalse(any('--init-checkpoint' in command for command in self.commands))
        self.assertEqual(len([n for n in result['artifact_sha256'] if '/hotelling_statistics/' in n]),50)
        with redirect_stdout(io.StringIO()):
            archive=queue.pack(self.project,self.release)
        with tarfile.open(str(archive)) as handle:
            names=handle.getnames()
        self.assertTrue(any(n.endswith('/hotelling_stats_selected.json') for n in names))
        self.assertTrue(any(n.endswith('/hotelling_training_audit.json') for n in names))
        self.assertFalse(any('/initialization/seed37.pt' in n for n in names))
        out=Path(plan['jobs'][0]['out'])
        (out/'hotelling_stats_selected.json').write_text('{}',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'changed'):
            queue.pack(self.project,self.release)

    def test_completion_rejects_missing_per_epoch_refit(self):
        plan,_=self.complete()
        out=Path(plan['jobs'][0]['out'])
        (out/'hotelling_statistics/epoch_050.json').unlink()
        with self.assertRaisesRegex(ValueError,'every epoch'):
            runner.hotelling_artifacts(out)


if __name__=='__main__':
    unittest.main()
