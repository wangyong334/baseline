"""Slot 4 base variant: spiking U-Net with the current-merged decoder (V1 scheme C), three stride-2 downsamplings.

    enc1  Conv3x3 Cin->c1 s1   enc2  c1->c2 s2   enc3  c2->c3 s2   enc4  c3->c4 s2
    dec3  ConvT4x4 s2 (enc4 spikes) + Conv3x3 (enc3 spikes)   two synaptic currents summed in the neuron
    dec2  ConvT4x4 s2 (dec3 spikes) + Conv3x3 (enc2 spikes)
    dec1  ConvT4x4 s2 (dec2 spikes) + Conv3x3 (enc1 spikes)   -> u7 = pre-reset membrane potential of dec1
Every conv is bias-free and every synaptic input except enc1's is a 0/1 spike. Each conv is followed by a learnable
per-channel gain (calibrated once on training data) and the neuron of slot 3.

forward_dense        one step through all layers (streaming, calibration)
forward_dense_chunk  layer by layer over a chunk of T steps (convs batched over T*B, neurons recur over T);
                     identical results because no layer feeds back in time.
"""
import numpy as np
import torch
import torch.nn as nn

from speed.slots.neuron.lif import ChannelGain

LAYER_NAMES = ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")
DOWNSAMPLE = 8


class SpikingConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, neuron_factory):
        super(SpikingConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gain = ChannelGain(out_ch)
        self.neuron = neuron_factory(out_ch)
        self.out_ch = out_ch

    def forward(self, x, state, extra_current=None):
        current = self.conv(x)
        if extra_current is not None:
            current = current + extra_current
        spikes, new_state, u_pre = self.neuron(self.gain(current), state)
        return spikes, new_state, u_pre, current

    def forward_steps(self, x, state, carry, keep_u_pre, extra_current=None):
        steps, batch = int(x.shape[0]), int(x.shape[1])
        current = self.conv(x.reshape((steps * batch,) + tuple(x.shape[2:])))
        if extra_current is not None:
            current = current + extra_current
        current = self.gain(current)
        current = current.view((steps, batch) + tuple(current.shape[1:]))
        return self.neuron.forward_steps(current, state, carry, keep_u_pre)


class MergedUNet(nn.Module):
    def __init__(self, in_channels, channels, neuron_factory):
        super(MergedUNet, self).__init__()
        c1, c2, c3, c4 = (int(c) for c in channels)
        block = lambda i, o, s: SpikingConvBlock(i, o, s, neuron_factory)  # noqa: E731
        upsample = lambda i, o: nn.ConvTranspose2d(i, o, 4, stride=2, padding=1, bias=False)  # noqa: E731
        # Construction order is kept identical to the legacy network (same parameter initialisation per seed).
        self.enc1 = block(in_channels, c1, 1)
        self.enc2 = block(c1, c2, 2)
        self.enc3 = block(c2, c3, 2)
        self.enc4 = block(c3, c4, 2)
        self.up3, self.dec3 = upsample(c4, c3), block(c3, c3, 1)
        self.up2, self.dec2 = upsample(c3, c2), block(c2, c2, 1)
        self.up1, self.dec1 = upsample(c2, c1), block(c1, c1, 1)
        self.out_channels = c1
        self.register_buffer("gain_calibrated", torch.tensor(False))

    @property
    def v_threshold(self):
        return self.enc1.neuron.v_threshold

    def blocks(self):
        return [getattr(self, name) for name in LAYER_NAMES]

    @staticmethod
    def _previous(states, carry):
        prev = states if (carry and states is not None) else [None] * 7
        if len(prev) != 7:
            raise ValueError("expected 7 membrane states, got %d" % len(prev))
        return prev

    def forward_dense(self, x, states, carry=True, collect=False):
        prev = self._previous(states, carry)
        s1, n1, u1, i1 = self.enc1(x, prev[0])
        s2, n2, u2, i2 = self.enc2(s1, prev[1])
        s3, n3, u3, i3 = self.enc3(s2, prev[2])
        s4, n4, u4, i4 = self.enc4(s3, prev[3])
        s5, n5, u5, i5 = self.dec3(s3, prev[4], self.up3(s4))
        s6, n6, u6, i6 = self.dec2(s2, prev[5], self.up2(s5))
        s7, n7, u7, i7 = self.dec1(s1, prev[6], self.up1(s6))
        info = None
        if collect:
            info = {"spikes": [s1, s2, s3, s4, s5, s6, s7], "u_pre": [u1, u2, u3, u4, u5, u6, u7],
                    "current": [i1, i2, i3, i4, i5, i6, i7]}
        return u7, [n1, n2, n3, n4, n5, n6, n7], info

    def forward_dense_chunk(self, x, states, carry=True, collect=False):
        if x.dim() != 5:
            raise ValueError("chunk input must be [T,B,C,H,W]")
        prev = self._previous(states, carry)
        steps, batch = int(x.shape[0]), int(x.shape[1])
        flat = lambda seq: seq.reshape((steps * batch,) + tuple(seq.shape[2:]))  # noqa: E731
        s1, n1, u1 = self.enc1.forward_steps(x, prev[0], carry, collect)
        s2, n2, u2 = self.enc2.forward_steps(s1, prev[1], carry, collect)
        s3, n3, u3 = self.enc3.forward_steps(s2, prev[2], carry, collect)
        s4, n4, u4 = self.enc4.forward_steps(s3, prev[3], carry, collect)
        s5, n5, u5 = self.dec3.forward_steps(s3, prev[4], carry, collect, self.up3(flat(s4)))
        s6, n6, u6 = self.dec2.forward_steps(s2, prev[5], carry, collect, self.up2(flat(s5)))
        s7, n7, u7 = self.dec1.forward_steps(s1, prev[6], carry, True, self.up1(flat(s6)))
        info = None
        if collect:
            info = {"spikes": [s1, s2, s3, s4, s5, s6, s7], "u_pre": [u1, u2, u3, u4, u5, u6, u7]}
        return u7, [n1, n2, n3, n4, n5, n6, n7], info

    def operations(self, height, width, firing_rates, input_density):
        """Synaptic operations per step: enc1 is driven by real-valued inputs (MAC, event-driven = dense * density),
        every other conv by spikes (AC = firing rate of its input * dense count)."""
        h, w = int(height), int(width)
        size = {"enc1": h * w, "enc2": (h // 2) * (w // 2), "enc3": (h // 4) * (w // 4), "enc4": (h // 8) * (w // 8)}
        conv_ops = lambda conv, hw: conv.out_channels * conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1] * hw  # noqa: E731,E501
        layers = [{"layer": "enc1", "mac": conv_ops(self.enc1.conv, size["enc1"]) * float(input_density), "ac": 0.0}]
        for name, src in (("enc2", "enc1"), ("enc3", "enc2"), ("enc4", "enc3")):
            layers.append({"layer": name, "mac": 0.0,
                           "ac": firing_rates[src] * conv_ops(getattr(self, name).conv, size[name])})
        for up, dec, deep, skip, hw in (("up3", "dec3", "enc4", "enc3", size["enc3"]),
                                        ("up2", "dec2", "dec3", "enc2", size["enc2"]),
                                        ("up1", "dec1", "dec2", "enc1", size["enc1"])):
            conv_t = getattr(self, up)
            d_up = conv_t.in_channels * conv_t.out_channels * conv_t.kernel_size[0] * conv_t.kernel_size[1] * (hw // 4)
            layers.append({"layer": up, "mac": 0.0, "ac": firing_rates[deep] * d_up})
            layers.append({"layer": dec, "mac": 0.0, "ac": firing_rates[skip] * conv_ops(getattr(self, dec).conv, hw)})
        return layers


def gains_from_positive_samples(samples, positive_counts, v_threshold, quantile, min_positive, gain_min, gain_max,
                                eps=1e-6):
    """Per-channel gain = v_th / q-quantile of positive pre-gain currents (layer quantile for sparse channels)."""
    n_ch = len(samples)
    nonempty = [s for s in samples if s is not None and len(s)]
    pooled = np.concatenate(nonempty) if nonempty else np.zeros(0)
    q_layer = float(np.quantile(pooled, quantile)) if pooled.size else 0.0
    gains = np.ones(n_ch, dtype=np.float64)
    info = {"q_layer": q_layer, "q_channel": [], "fallback": [], "raw_gain": []}
    for c in range(n_ch):
        s = samples[c]
        use_layer = int(positive_counts[c]) < int(min_positive) or s is None or len(s) == 0
        q = q_layer if use_layer else float(np.quantile(s, quantile))
        raw = 1.0 if q < eps else float(v_threshold) / q
        gains[c] = min(max(raw, float(gain_min)), float(gain_max))
        info["q_channel"].append(q)
        info["fallback"].append(bool(use_layer))
        info["raw_gain"].append(raw)
    return gains, info


def calibrate_gains(net, sequence_factory, quantile, samples_per_channel, min_positive, gain_min, gain_max, seed,
                    carry=True):
    """One-off layer-by-layer gain calibration on training data (gain = v_th / q-quantile of positive current).
    sequence_factory() yields sequences; each sequence yields input tensors [B,C,H,W] in time order."""
    if bool(net.gain_calibrated):
        raise RuntimeError("gains are already calibrated")
    generator = torch.Generator().manual_seed(int(seed))
    reports = []
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for layer_index, block in enumerate(net.blocks()):
            n_ch = block.out_ch
            samples = [[] for _ in range(n_ch)]
            positive_counts = np.zeros(n_ch, dtype=np.int64)
            for sequence in sequence_factory():
                states = None
                for x in sequence:
                    _, states, info = net.forward_dense(x, states, carry, collect=True)
                    current = info["current"][layer_index]
                    for c in range(n_ch):
                        channel = current[:, c]
                        values = channel[channel > 0]
                        n = int(values.numel())
                        positive_counts[c] += n
                        if n == 0:
                            continue
                        if n > samples_per_channel:
                            pick = torch.randint(0, n, (samples_per_channel,), generator=generator)
                            values = values[pick.to(values.device)]
                        samples[c].append(values.float().cpu().numpy())
            merged = [np.concatenate(s) if s else np.zeros(0) for s in samples]
            gains, detail = gains_from_positive_samples(merged, positive_counts, net.v_threshold, quantile,
                                                        min_positive, gain_min, gain_max)
            block.gain.set_gain(torch.from_numpy(gains))
            reports.append(dict({"layer": LAYER_NAMES[layer_index], "positive_counts": positive_counts.tolist(),
                                 "gain": gains.tolist()}, **detail))
    net.gain_calibrated.fill_(True)
    net.train(was_training)
    return reports
