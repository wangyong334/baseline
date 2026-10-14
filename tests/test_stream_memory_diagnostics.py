import unittest
import torch
from tools.diagnose_stream_memory import ReadIntervention, diagnose_sequence
from test_stream_dual_memory import cell, net
from test_stream_layerwise import synthetic_sequence, CFG, Q99
from train_stream_v1 import run_sequence


class DiagnosticTests(unittest.TestCase):
    def test_zero_read_matches_switch_and_preserves_write(self):
        a, b = cell(), cell()
        with torch.no_grad():
            a.kappa_raw.fill_(.8)
            b.load_state_dict(a.state_dict())
        b.memory_read_enabled = False
        h = a.register_forward_hook(ReadIntervention('zero', 2))
        sa = sb = None
        for _ in range(6):
            x = torch.randn(1, 3, 2, 2, dtype=torch.float64)
            oa, sa, ua = a(x, sa)
            ob, sb, ub = b(x, sb)
            self.assertTrue(torch.equal(ua, ub))
            self.assertTrue(torch.equal(sa[1], sb[1]))
        h.remove()

    def test_delayed_read_is_past_only_write_unchanged(self):
        n = cell()
        with torch.no_grad():
            n.kappa_raw.fill_(.8)
        hook = ReadIntervention('delay', 2)
        history = []
        for k in range(5):
            x = torch.zeros(1, 3, 2, 2, dtype=torch.float64)
            state = (x, torch.full_like(x, k / 10.))
            ordinary = n(x, state)
            result = hook(n, (x, state), ordinary)
            expected = torch.zeros_like(x) if k < 2 else history[k-2]
            self.assertTrue(torch.equal(result[2], n._coefficients(x)[2] * expected))
            self.assertTrue(torch.equal(result[1][1], ordinary[1][1]))
            history.append(state[1])

    def test_end_to_end_all_interventions(self):
        model = net(('enc2', 'enc3', 'enc4', 'dec3', 'dec2', 'dec1'), nonzero=True).eval()
        seq = synthetic_sequence()
        cfg = dict(CFG, input_device='cpu', threshold=.9)
        for mode in ['zero', 'delay', 'spatial_mean']:
            result = diagnose_sequence(model, seq, cfg, Q99, torch.device('cpu'), mode, 2)
            self.assertIn('dec1/target/read_abs_over_theta', result['statistics'])
            self.assertEqual(result['counts']['real'][0] + result['counts']['real'][2], int((seq.label > .5).sum()))
            self.assertFalse(model.dec1.neuron._forward_hooks)
            if mode == 'zero':
                model.set_memory_read_enabled(False)
                with torch.no_grad():
                    _, confusion = run_sequence(model, seq, dict(cfg, execution='step'), Q99,
                                                torch.device('cpu'), 'carry')
                self.assertEqual(result['counts']['intervention'], confusion.sum(axis=0)[:3].tolist())
                model.set_memory_read_enabled(True)


if __name__ == '__main__':
    unittest.main()
