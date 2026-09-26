"""Read-only V2-2 contract probes; no model weights or evaluation defaults change.

Run with the project's PyTorch environment.  The first, second and fourth
probes inject controlled hypothesis states to isolate readout behaviour; they
are not real-data experiments or estimates of recoverable IoU.  The third
probe runs a synthetic target through the actual bank update/end lifecycle.
The report records observed behaviour without asserting that these behaviours
explain the complete real-data performance gap.
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model.attribution_readout import AttributionReadout
from model.target_hypotheses import F64, HypothesisBank, distribution, gaussian_template
from tests.test_stream_v2_attribution import H, W, make_window, run, track


def controlled_hypothesis(bank, support=4, radius=1.5):
    h = bank._new(0, torch.tensor([20., 30.], dtype=F64),
                  torch.zeros(2, dtype=F64), "audit")
    h.status, h.confirmed_at, h.support = "confirmed", 0, support
    h.S = gaussian_template(7, radius)
    h.meas = {j: (h.anchor.clone(), 10.) for j in (0, 1, 2)}
    bank.alive.append(h)
    return h


def broad_snapshot(bank, h):
    bank.snapshots.setdefault(0, {})[h.hid] = (h.anchor.clone(), 3., h.S.clone())


def entry_at(readout, y=20, x=30):
    return readout._register(0, 0, torch.tensor([y]), torch.tensor([x]), None, None)


def probe_future_gate():
    bank = HypothesisBank(H, W)
    h = controlled_hypothesis(bank)
    broad_snapshot(bank, h)
    reader = AttributionReadout(bank, [2], variants=())
    entry = entry_at(reader, x=36)
    ys, xs = entry["ys"], entry["xs"]
    qd, phi = bank.density(h, 0, 2, ys, xs)
    q0 = bank.snapshot_density(0, h.hid, ys, xs)
    snap = bank.snapshot_state(0, h.hid)
    old_peak = distribution(snap[2], snap[0], snap[1],
                            torch.tensor([20]), torch.tensor([30]))
    ungated = torch.clamp(torch.log((qd + reader.eps) / (q0 + reader.eps)),
                         -reader.cap, reader.cap)
    actual = reader._readout(entry, 2)
    return dict(past_phi=float(q0 / old_peak), future_phi=float(phi),
                gate_rel=bank.p["gate_rel"], q0=float(q0), qd=float(qd),
                ungated_ratio=float(ungated), actual_delta=float(actual["attr"]),
                actual_case=int(actual["case"]))


def probe_unsupported_winner():
    bank = HypothesisBank(H, W)
    old = controlled_hypothesis(bank, support=4, radius=2.)
    broad_snapshot(bank, old)
    reader = AttributionReadout(bank, [2], variants=())
    entry = entry_at(reader)
    before = reader._readout(entry, 2)
    weak = controlled_hypothesis(bank, support=1, radius=.5)
    broad_snapshot(bank, weak)
    after = reader._readout(entry, 2)
    return dict(eligible_only_delta=float(before["attr"]),
                with_ineligible_hyp_delta=float(after["attr"]),
                actual_case=int(after["case"]))


def probe_ended_history():
    rng = np.random.default_rng(4)
    windows = [make_window(rng, track(k)) for k in range(8)]
    bank = HypothesisBank(H, W)
    run(bank, windows)
    reader = AttributionReadout(bank, [5], variants=())
    ys, xs, _, _, g, mu = windows[6]
    entry = reader._register(6, 6, ys, xs, mu, g)
    for k in (8, 9, 10):
        ys, xs, ps, _, g, mu = make_window(rng, None, n_bg=0)
        bank.step(k, ys, xs, ps, g, mu)
    # Same historical observations and cutoff; only bank retention differs.
    retained = copy.deepcopy(bank)
    retained_reader = AttributionReadout(retained, [5], variants=())
    with_history = retained_reader._readout(entry, 11)
    ys, xs, ps, _, g, mu = make_window(rng, None, n_bg=0)
    bank.step(11, ys, xs, ps, g, mu)
    actual = reader._readout(entry, 11)
    return dict(alive_before=len(retained.alive), alive_after=len(bank.alive),
                retained_history_nonzero=int((with_history["attr"] != 0).sum()),
                retained_history_maxabs=float(with_history["attr"].abs().max()),
                actual_nonzero=int((actual["attr"] != 0).sum()))


def probe_merge_reference():
    bank = HypothesisBank(H, W)
    old = controlled_hypothesis(bank, support=4)
    controlled_hypothesis(bank, support=5)
    # Identical current distributions; only the older ID had a confirmed
    # snapshot at the queried window. Merge ranking favours the other ID.
    center, sigma, shape = bank.state(old, 0, 2)
    bank.snapshots[0] = {old.hid: (center.clone(), sigma, shape.clone())}
    reader = AttributionReadout(bank, [2], variants=())
    entry = entry_at(reader)
    before = reader._readout(entry, 2)
    bank._merge(1)
    bank._merge(2)
    after = reader._readout(entry, 2)
    return dict(before_delta=float(before["attr"]), before_case=int(before["case"]),
                after_delta=float(after["attr"]), after_case=int(after["case"]),
                surviving_id=bank.alive[0].hid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional JSON report path")
    args = parser.parse_args()
    report = dict(python=sys.version.split()[0], torch=torch.__version__,
                  scope="Controlled readout probes, not real-data efficacy tests",
                  future_gate=probe_future_gate(),
                  unsupported_winner=probe_unsupported_winner(),
                  ended_history=probe_ended_history(),
                  merge_reference=probe_merge_reference())
    content = json.dumps(report, indent=2)
    print(content)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            stream.write(content + "\n")


if __name__ == "__main__":
    main()
