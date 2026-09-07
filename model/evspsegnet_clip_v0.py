"""Continuous clipped-ReLU control at the SAME two sites as local-rate SNN."""
from torch import nn
from model.evspsegnet_mp import evspsegnet_mp


def install_clipped_activations(net, stages):
    stages = list(stages)
    if not stages or len(set(stages)) != len(stages) or any(s not in (1, 2, 3, 4) for s in stages):
        raise ValueError('Expected unique stages in 1..4')
    sites = []
    for stage in stages:
        blocks = list(getattr(net, 'conv%d' % stage).children())
        block = blocks[-1]
        if not isinstance(block._modules.get('2'), nn.ReLU):
            raise RuntimeError('Unexpected terminal activation layout')
        block._modules['2'] = nn.Hardtanh(min_val=0.0, max_val=1.0, inplace=False)
        sites.append('conv%d.%d.2' % (stage, len(blocks) - 1))
    return sites


class evspsegnet_clip_v0(evspsegnet_mp):
    def __init__(self, cfg, split=2):
        super().__init__(cfg, split)
        if cfg.snn_threshold != 1.0:
            raise ValueError('This fixed-cap control requires reference snn_threshold=1')
        self.activation_sites = install_clipped_activations(self, cfg.snn_stages)
