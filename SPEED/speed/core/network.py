"""The learnable part of a system: backbone (slot 4, built with the neuron of slot 3) + heads (slot 6), with the state
transport of slot 5 applied between steps.

forward_chunk(inputs [T,B,C,H,W], states) -> (outputs {name: [T,B,...]}, states, info)
    time-parallel path when the transport is the identity (convs batched over the chunk), step-by-step otherwise;
    both give the same result for the identity transport (force_stepwise exists for that check).
"""
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, backbone, heads, transport):
        super(Network, self).__init__()
        self.backbone = backbone
        self.heads = heads
        self.transport = transport

    @property
    def v_threshold(self):
        return self.backbone.v_threshold

    def blocks(self):
        return self.backbone.blocks()

    def forward_step(self, x, states, carry=True, collect=False):
        u, states, info = self.backbone.forward_dense(x, states, carry, collect)
        outputs = self.heads(u)
        return outputs, self.transport.apply(states, outputs), info

    def forward_chunk(self, inputs, states, carry=True, collect=False, force_stepwise=False):
        steps, batch = int(inputs.shape[0]), int(inputs.shape[1])
        if self.transport.is_identity and not force_stepwise:
            u, states, info = self.backbone.forward_dense_chunk(inputs, states, carry, collect)
            flat = self.heads(u.reshape((steps * batch,) + tuple(u.shape[2:])))
            outputs = {name: value.view((steps, batch) + tuple(value.shape[1:])) for name, value in flat.items()}
            return outputs, states, info
        per_step, infos = [], []
        for t in range(steps):
            out, states, info = self.forward_step(inputs[t], states, carry, collect)
            per_step.append(out)
            infos.append(info)
        outputs = {name: torch.stack([o[name] for o in per_step]) for name in per_step[0]}
        info = None
        if collect:
            info = {key: [torch.stack([i[key][layer] for i in infos]) for layer in range(len(infos[0][key]))]
                    for key in ("spikes", "u_pre")}
        return outputs, states, info

