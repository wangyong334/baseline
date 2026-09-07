"""Static quantization control at the same activation sites as SNN/clip v0."""
from model.evspsegnet_mp import evspsegnet_mp
from model.quant_rate import install_quantized_activations


class evspsegnet_quant_v0(evspsegnet_mp):
    def __init__(self, cfg, split=2):
        super().__init__(cfg, split)
        if cfg.snn_steps != 4 or cfg.snn_threshold != 1.0:
            raise ValueError('This fixed five-level control matches only T4 threshold1 output levels')
        self.activation_sites = install_quantized_activations(self, cfg.snn_stages)
