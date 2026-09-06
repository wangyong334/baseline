"""Partial local-rate spiking ablation; all other ANN modules are unchanged."""
from torch import nn
from model.evspsegnet_mp import evspsegnet_mp
from model.lif_rate import LIFRate


class evspsegnet_snn_v0(evspsegnet_mp):
    def __init__(self, cfg, split=2):
        super().__init__(cfg, split)
        stages = list(cfg.snn_stages)
        if not stages or len(set(stages)) != len(stages) or any(s not in (1, 2, 3, 4) for s in stages):
            raise ValueError("snn_stages must contain unique stages in 1..4")
        self.spike_sites = []
        # Replace only the terminal activation of each stage's final post-act
        # block. Internal GDBlock activations and downsample activations stay ANN.
        for stage in stages:
            block = list(getattr(self, "conv%d" % stage).children())[-1]
            if not isinstance(block._modules.get("2"), nn.ReLU):
                raise RuntimeError("Unexpected encoder activation layout")
            block._modules["2"] = LIFRate(cfg.snn_steps, cfg.snn_beta, cfg.snn_threshold)
            self.spike_sites.append("conv%d.%d.2" % (stage, len(list(getattr(self, "conv%d" % stage).children())) - 1))

    def diagnostics(self, enabled):
        for module in self.modules():
            if isinstance(module, LIFRate):
                module.collect = enabled
                module.stats = {}

    def spike_stats(self):
        return {name: dict(module.stats) for name, module in self.named_modules()
                if isinstance(module, LIFRate) and module.stats}
