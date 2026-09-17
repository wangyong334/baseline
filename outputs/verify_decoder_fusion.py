"""Standalone scheme-C audit; does not modify production model or checkpoints.

Run from repository root with a PyTorch environment.
"""
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.evspsegnet_stream import EvSpSegNetStream

torch.set_num_threads(2)


def fuse(up, deep):
    # PyTorch ConvT weight [in, mid, a, b], Conv weight [out, mid, i, j].
    # q = 2*n + a - i + 1 = 2*n + fused_index - padding(1).
    kernel = up.new_zeros(up.shape[0], deep.shape[0], 4, 4)
    for a in range(2):
        for b in range(2):
            for i in range(3):
                for j in range(3):
                    kernel[:, :, a - i + 2, b - j + 2] += (
                        up[:, :, a, b] @ deep[:, :, i, j].T)
    return kernel


def local_checks(dtype):
    worst = 0.
    for cin, mid, cout, h, w in [(2, 3, 4, 1, 1), (4, 2, 3, 3, 5), (3, 4, 2, 8, 6)]:
        up = torch.randn(cin, mid, 2, 2, dtype=dtype) * .1
        conv = torch.randn(cout, mid + 2, 3, 3, dtype=dtype) * .1
        merged = fuse(up, conv[:, :mid])
        # Test continuous input too: the equality is linear, not limited to spikes.
        for corner in (None, (0, 0), (h - 1, w - 1)):
            x = torch.randn(1, cin, h, w, dtype=dtype)
            if corner is not None:
                x.zero_()
                x[:, :, corner[0], corner[1]] = 1
            skip = torch.randn(1, 2, h * 2, w * 2, dtype=dtype)
            original = F.conv2d(torch.cat([F.conv_transpose2d(x, up, stride=2), skip], 1),
                                conv, padding=1)
            result = (F.conv_transpose2d(x, merged, stride=2, padding=1)
                      + F.conv2d(skip, conv[:, mid:], padding=1))
            error = (original - result).abs().max().item()
            worst = max(worst, error)
            torch.testing.assert_close(original, result,
                                       atol=2e-6 if dtype == torch.float32 else 1e-12,
                                       rtol=2e-5 if dtype == torch.float32 else 1e-12)
    return worst


def fused_forward(net, kernels, x, events, states):
    prev = states if states is not None else [None] * 7
    values, next_states, spikes, potentials = [], [], [], []
    for index, name in enumerate(('enc1', 'enc2', 'enc3', 'enc4')):
        s, state, u, _ = getattr(net, name)(x, prev[index])
        values.append(s)
        next_states.append(state)
        spikes.append(s)
        potentials.append(u)
        x = s
    for index, (up, dec, skip) in enumerate((('up3', 'dec3', values[2]),
                                            ('up2', 'dec2', values[1]),
                                            ('up1', 'dec1', values[0])), 4):
        block = getattr(net, dec)
        mid = getattr(net, up).out_channels
        current = F.conv_transpose2d(x, kernels[up], stride=2, padding=1)
        current = current + F.conv2d(skip, block.conv.weight[:, mid:], padding=1)
        x, state, u = block.neuron(block.gain(block.norm(current)), prev[index])
        next_states.append(state)
        spikes.append(x)
        potentials.append(u)
    feat = u[events['b'], :, events['y'], events['x']]
    extra = torch.stack([events['p'], events['t_local']], 1)
    return net.readout(torch.cat([feat, extra], 1)).squeeze(1), next_states, spikes, potentials


@torch.no_grad()
def audit(dtype):
    torch.manual_seed(103)
    net = EvSpSegNetStream().to(dtype).eval()
    # Synthetic positive weights produce active paths throughout the entire network.
    # These are NOT trained weights and cannot validate dataset metrics.
    for name, p in net.named_parameters():
        if name.endswith('weight'):
            p.abs_()
    other = copy.deepcopy(net)
    kernels = {up: fuse(getattr(other, up).weight,
                        getattr(other, dec).conv.weight[:, :getattr(other, up).out_channels])
               for up, dec in (('up3', 'dec3'), ('up2', 'dec2'), ('up1', 'dec1'))}
    events = dict(b=torch.zeros(32, dtype=torch.long), y=torch.randint(0, 264, (32,)),
                  x=torch.randint(0, 352, (32,)), p=torch.randint(0, 2, (32,)).to(dtype),
                  t_local=torch.rand(32, dtype=dtype))
    a = b = None
    logit_error = state_error = potential_error = 0.
    flips = total = nonzero = 0
    for k in range(12):
        x = torch.rand(1, 12, 264, 352, dtype=dtype) * .4
        if k in (4, 5):
            x.zero_()
        ya, a, info = net(x, events, a, collect=True)
        yb, b, sb, ub = fused_forward(other, kernels, x, events, b)
        logit_error = max(logit_error, (ya - yb).abs().max().item())
        for aa, bb, sa, ss, ua, uu in zip(a, b, info['spikes'], sb, info['u_pre'], ub):
            state_error = max(state_error, (aa - bb).abs().max().item())
            potential_error = max(potential_error, (ua - uu).abs().max().item())
            flips += int((sa != ss).sum())
            total += sa.numel()
            nonzero += int(sa.sum())
    return dict(dtype=str(dtype), local_max_error=local_checks(dtype), windows=12,
                logit_max_error=logit_error, state_max_error=state_error,
                pre_reset_max_error=potential_error, spike_flips=flips,
                compared_spikes=total, emitted_spikes=nonzero)


if __name__ == '__main__':
    results = [audit(torch.float64), audit(torch.float32)]
    result_path = Path(__file__).with_suffix('.json')
    result_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
