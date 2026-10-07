"""Slot 5 base variant: no state transport (memory stays where it was written, as in V1-V3).

A transport variant moves the carried network state between two steps, e.g. along predicted motion. Its interface:
    apply(states, outputs) -> states      states: list of carried tensors; outputs: head outputs of the step just done
is_identity tells the network that it may run the time-parallel (layer-by-layer) path.
"""


class NoTransport(object):
    is_identity = True

    def apply(self, states, outputs):
        return states
