"""A system = clock (1) + representation (2) + network (3-6) + verifier (7) + readouts (8).

run_stream(stream) processes one recording causally, chunk by chunk:
    representation.encode -> network.forward_chunk -> per step: verifier.step, every readout.step
and returns ({readout name: (prob float32 [N], publish_us int64 [N])} in file order, steps); publish_us is the end
of the step at which the readout published the event. The verifier only runs when a readout needs it.
"""
import torch

from speed.core.clock import EventBlocks, canvas_size


class System(object):
    def __init__(self, clock, representation, network, verifier, readouts, canvas_multiple=8, chunk_steps=32,
                 threshold=0.9):
        self.clock, self.representation, self.network = clock, representation, network
        self.verifier, self.readouts = verifier, readouts
        self.canvas_multiple, self.chunk_steps, self.threshold = int(canvas_multiple), int(chunk_steps), float(threshold)

    def canvas(self, stream):
        return canvas_size(stream.height, stream.width, self.canvas_multiple)

    def to(self, device):
        self.representation.to(device)
        self.network.to(device)
        return self

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def dtype(self):
        return next(self.network.parameters()).dtype

    def readout_names(self):
        names = []
        for r in self.readouts:
            names += [r.name] if hasattr(r, "name") else ["%s%d" % (r.prefix, d) for d in r.delays]
        return names

    def run_stream(self, stream, carry=True, monitor=None, readouts=None):
        """Causal inference over a whole stream (call under torch.no_grad() with the network in eval mode)."""
        readouts = self.readouts if readouts is None else readouts
        device, dtype = self.device, self.dtype
        H, W = self.canvas(stream)
        steps = self.clock.partition(stream)
        blocks = EventBlocks(stream, steps, H, W, device, dtype)
        rep_state = self.representation.init_state(1, H, W, device, dtype)
        verify = self.verifier is not None and any(getattr(r, "needs_verifier", False) for r in readouts)
        ver_state = self.verifier.init_state(1, H, W, device, dtype) if verify else None
        for r in readouts:
            r.begin(stream.n_events)
        states, prev_log_g = None, None
        n = steps.n_steps
        for start in range(0, n, self.chunk_steps):
            end = min(start + self.chunk_steps, n)
            blk = blocks.block(start, end)
            rep_state, inputs, aux = self.representation.encode(rep_state, blk)
            outputs, states, info = self.network.forward_chunk(inputs, states, carry, collect=monitor is not None)
            if monitor is not None:
                monitor.update(info)
            ev = blk["events"]
            logits = outputs["mark"][ev["t"], ev["b"], 0, ev["y"], ev["x"]]
            # One sigmoid per chunk, then sliced: on CPU the vectorised and scalar-tail paths of exp can differ
            # in the last bit, so the result must not depend on where a step's events start in the array.
            net_prob = torch.sigmoid(logits).float().cpu().numpy()
            offset = 0
            for t, k in enumerate(range(start, end)):
                count = int(steps.bounds[k + 1] - steps.bounds[k])
                sl = slice(offset, offset + count)
                if verify:
                    ver_state = self.verifier.step(ver_state, aux["total"][t], aux["mu0"][t], prev_log_g)
                ctx = {"k": k, "idx": blk["idx"][offset:offset + count], "b": ev["b"][sl], "y": ev["y"][sl],
                       "x": ev["x"][sl], "logits": logits[sl], "prob": net_prob[sl], "total": aux["total"][t],
                       "verifier": ver_state}
                for r in readouts:
                    r.step(ctx)
                prev_log_g = outputs["log_g"][t]
                offset += count
        results = {}
        for r in readouts:
            r.flush()
            results.update(r.results(self.threshold))
        return {name: (prob, steps.t_end_us[when]) for name, (prob, when) in results.items()}, steps

