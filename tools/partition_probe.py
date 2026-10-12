"""Temporal partition probe for a trained streaming SNN V1 checkpoint (inference only, same weights).

The same event stream is processed under different temporal partitions and everything is compared at common
physical checkpoints (every 200 ms):

    A (system)    pieces of 25 / 50 / 100 / 200 ms or random (>= 25 ms, inside each 200 ms block); every piece is
                  re-encoded exactly as the system encodes a window (counts, 5 time bins, q99 normalisation,
                  piece-relative t_local)
    B (SNN only)  pieces of 50 / 100 / 200 ms or random multiples of 50 ms; the input currents of the 50 ms windows
                  (the trained encoding) are summed with their exact decay exp(-(t_end - t_j)/tau) before the first
                  neuron, so the input information is identical and only the SNN update partition changes

Every LIF uses beta = exp(-dt/tau) with the piece's own dt. Emission rule (--variant), cap = max(1, round(dt / 50 ms)):
    base       binary spike, reset by v_th (the trained neuron)
    reset      binary spike, reset by min(demand, cap) * v_th (own state discharged, downstream still gets one spike)
    multi      emits n = min(demand, cap) spikes, reset by n * v_th
    multi_zoh  multi + delivery correction: spike currents into a step longer than 50 ms are scaled per postsynaptic
               channel as if the step's spikes were spread evenly over its 50 ms sub-steps and decayed exactly
demand = floor(max(u_pre, 0) / v_th). With cap 1 at <= 50 ms every variant equals base there, so the 50 ms
reference is the trained model in all variants (A25 is skipped for non-base variants: identical to base).

Teacher-forced message oracles (diagnostic only, B partitions only): every layer's spikes inside a piece are the 50 ms
reference's; each membrane is updated once for the whole piece, V_end = exp(-dt/tau) V_start + input - own reset,
where input and own reset are computed from a message of those spikes:
    tf_exact          all sub-step spikes, each decayed exactly to the piece end (equals the reference)
    tf_count_end      spike count only, delivered at the piece end
    tf_count_uniform  spike count only, spread evenly over the piece (per-channel delivery factor)
    tf_timing         count + decay-weighted count sum_j exp(-(t_end - t_j) / tau_ref), tau_ref = tau_init; each
                      receiving channel rebuilds its own exact decay weights by a least-squares fit on the two
                      (exact for pieces of <= 2 sub-steps, so B100 must equal the reference)
The first layer's input is the exact B current in every oracle.

Reported per partition:
    event IoU / recall / false events at threshold 0.9 (each event labelled at the end of its own piece)
    fair subset: events in the last 25 ms (A) / 50 ms (B) before each 200 ms checkpoint, labelled at the same
                 physical time in every partition: IoU, mean |p - p_ref|, decision flips
    state disagreement at the checkpoints, per layer: ||S - S_ref|| / ||S_ref|| and cosine (reference = 50 ms)
    spikes per neuron per physical second, fraction of |u_pre| > 10 v_th, per layer
    spike multiplicity: spikes of every neuron in every 200 ms block, joint with the reference's count (0..4)
    latent demand: fraction of neuron-steps with demand >= 2, mean (demand - emitted) per neuron-step

    python tools/partition_probe.py --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt \
        --root /path/to/EV-UAV-dataset --split test --out log/partition_probe/s37 [--variant base] [--max-clips N] \
        [--selfcheck]
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset.stream_windows import count_channels, load_npz_events, normalize_counts  # noqa: E402
from model.evspsegnet_stream import EvSpSegNetStream  # noqa: E402
from model.lif2d_stream import StreamingLIF2d  # noqa: E402

BLOCK_MS = 200
LAYERS = ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")
B_GROUPINGS = ((4,), (1, 3), (3, 1), (2, 2), (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 1, 1, 1))
EMIT_VARIANTS = ("base", "reset", "multi", "multi_zoh")
TF_VARIANTS = ("tf_exact", "tf_count_end", "tf_count_uniform", "tf_timing")
VARIANTS = EMIT_VARIANTS + TF_VARIANTS
QMAX = 8                                                # block spike counts above this are pooled in the last bin


def build_net(cfg, device):
    net = EvSpSegNetStream(
        in_channels=2 + 2 * int(cfg["time_bins"]), channels=tuple(cfg["channels"]), neuron=cfg["neuron"],
        norm=cfg["norm"], state_mode=cfg["state_mode"], readout_hidden=int(cfg["readout_hidden"]),
        dt_ms=float(cfg["window_ms"]), tau_init_ms=float(cfg["tau_init_ms"]), tau_min_ms=float(cfg["tau_min_ms"]),
        tau_max_ms=float(cfg["tau_max_ms"]), v_threshold=float(cfg["v_threshold"]),
        merged_decoder=cfg.get("merged_decoder", False))
    return net.to(device)


def set_dt(net, dt_ms):
    for m in net.modules():
        if isinstance(m, StreamingLIF2d):
            m.dt_ms = float(dt_ms)


def partition(kind, value, total_ms, rng):
    """-> list of (start_ms, end_ms) covering [0, total_ms), with a boundary at every multiple of BLOCK_MS."""
    pieces = []
    for b0 in range(0, total_ms, BLOCK_MS):
        if kind == "uniform":
            cuts = list(range(b0, b0 + BLOCK_MS, int(value))) + [b0 + BLOCK_MS]
        elif kind == "randA":
            while True:
                k = int(rng.randint(1, 5))
                inner = sorted(rng.choice(np.arange(25, BLOCK_MS - 24), k - 1, replace=False).tolist()) if k > 1 else []
                cuts = [b0] + [b0 + c for c in inner] + [b0 + BLOCK_MS]
                if min(b - a for a, b in zip(cuts[:-1], cuts[1:])) >= 25:
                    break
        elif kind == "randB":
            grouping = B_GROUPINGS[int(rng.randint(len(B_GROUPINGS)))]
            cuts = [b0] + [b0 + 50 * c for c in np.cumsum(grouping)]
        else:
            raise ValueError(kind)
        pieces += list(zip(cuts[:-1], cuts[1:]))
    return pieces


class Clip(object):
    def __init__(self, seq, cfg, q99, device, dtype):
        self.x, self.y, self.t, self.p = seq.x, seq.y, seq.t, seq.p
        self.label = seq.label > 0.5
        self.cfg, self.q99, self.device, self.dtype = cfg, q99, device, dtype
        self.n = int(self.t.shape[0])
        self.total_ms = int(cfg["window_ms"]) * int(cfg["n_windows"])

    def select(self, a, b):
        return np.flatnonzero((self.t >= a) & (self.t < b))

    def encode(self, idx, a, b):
        """Input tensor of the piece [a, b) exactly as the system encodes a window of length b - a."""
        cfg = self.cfg
        bins = int(cfg["time_bins"])
        local = (self.t[idx] - a).astype(np.int64)
        inner = np.minimum((local * bins) // int(b - a), bins - 1)
        counts = count_channels(self.x[idx], self.y[idx], self.p[idx], inner, bins, int(cfg["pad_height"]),
                                int(cfg["pad_width"]))
        x = normalize_counts(counts, self.q99, float(cfg["input_clip"]))
        return torch.from_numpy(x).unsqueeze(0).to(self.device, self.dtype)

    def events(self, idx, t_local):
        dev = self.device
        return {"b": torch.zeros(len(idx), dtype=torch.long, device=dev),
                "y": torch.from_numpy(self.y[idx]).long().to(dev), "x": torch.from_numpy(self.x[idx]).long().to(dev),
                "p": torch.from_numpy(self.p[idx].astype(np.float32)).to(dev),
                "t_local": torch.from_numpy(t_local.astype(np.float32)).to(dev)}


def spike_cap(dt_ms):
    """Most spikes per step: 1 at the trained 50 ms clock, about dt / 50 ms for longer steps."""
    return max(1, int(math.floor(dt_ms / 50.0 + 0.5)))


def delivery_gain(neuron, dt_ms):
    """Per-channel factor [1, C, 1, 1] for spike currents into a step longer than 50 ms, else None.

    The m = dt / 50 sub-steps' share of the step's spikes, decayed exactly to the step end, relative to delivering
    them all at the end: (1 - exp(-dt / tau)) / (m * (1 - exp(-50 / tau)))."""
    if dt_ms <= 50:
        return None
    tau = neuron.tau()
    k = (1.0 - torch.exp(-dt_ms / tau)) / ((dt_ms / 50.0) * (1.0 - torch.exp(-50.0 / tau)))
    return k.view(1, -1, 1, 1)


def neuron_step(neuron, current, state, dt_ms, variant):
    """One LIF step under the probe's emission rule -> (output, new_state, u_pre, demand)."""
    th = neuron.v_threshold
    if state is None:
        u_pre = current
    else:
        beta = torch.exp(-dt_ms / neuron.tau()).to(current.dtype).view(1, -1, 1, 1)
        u_pre = beta * state + current
    demand = torch.floor(torch.clamp(u_pre, min=0.0) / th)
    fired = (u_pre >= th).to(u_pre.dtype)
    if variant == "base":
        out, discharged = fired, fired
    elif variant == "reset":
        out, discharged = fired, torch.clamp(demand, max=spike_cap(dt_ms))
    else:
        out = torch.clamp(demand, max=spike_cap(dt_ms))
        discharged = out
    return out, u_pre - th * discharged, u_pre, demand


def block_forward(block, x, state, dt_ms, variant, extra=None, spike_input=True):
    """SpikingConvBlock.forward with the probe's neuron step (conv -> + extra -> norm -> gain -> neuron)."""
    current = block.conv(x)
    if extra is not None:
        current = current + extra
    current = block.gain(block.norm(current))
    if variant == "multi_zoh" and spike_input:
        k = delivery_gain(block.neuron, dt_ms)
        if k is not None:
            current = current * k.to(current.dtype)
    return neuron_step(block.neuron, current, state, dt_ms, variant)


def run_layers(net, s1, prev, dt_ms, variant):
    """Layers 2-7 of EvSpSegNetStream.forward given the first layer's output."""
    def f(block, x, state, extra=None):
        return block_forward(block, x, state, dt_ms, variant, extra)

    s2, n2, u2, d2 = f(net.enc2, s1, prev[1])
    s3, n3, u3, d3 = f(net.enc3, s2, prev[2])
    s4, n4, u4, d4 = f(net.enc4, s3, prev[3])
    if net.merged_decoder:
        s5, n5, u5, d5 = f(net.dec3, s3, prev[4], net.up3(s4))
        s6, n6, u6, d6 = f(net.dec2, s2, prev[5], net.up2(s5))
        s7, n7, u7, d7 = f(net.dec1, s1, prev[6], net.up1(s6))
    else:
        s5, n5, u5, d5 = f(net.dec3, torch.cat([net.up3(s4), s3], 1), prev[4])
        s6, n6, u6, d6 = f(net.dec2, torch.cat([net.up2(s5), s2], 1), prev[5])
        s7, n7, u7, d7 = f(net.dec1, torch.cat([net.up1(s6), s1], 1), prev[6])
    return ([s2, s3, s4, s5, s6, s7], [n2, n3, n4, n5, n6, n7], [u2, u3, u4, u5, u6, u7],
            [d2, d3, d4, d5, d6, d7])


def layer_current(net, i, s):
    """Synaptic current of layer i (1..6 = enc2..dec1) from presynaptic tensors s[0..6] (any linear spike message)."""
    extra = None
    if i == 1:
        blk, x = net.enc2, s[0]
    elif i == 2:
        blk, x = net.enc3, s[1]
    elif i == 3:
        blk, x = net.enc4, s[2]
    elif net.merged_decoder:
        if i == 4:
            blk, x, extra = net.dec3, s[2], net.up3(s[3])
        elif i == 5:
            blk, x, extra = net.dec2, s[1], net.up2(s[4])
        else:
            blk, x, extra = net.dec1, s[0], net.up1(s[5])
    elif i == 4:
        blk, x = net.dec3, torch.cat([net.up3(s[3]), s[2]], 1)
    elif i == 5:
        blk, x = net.dec2, torch.cat([net.up2(s[4]), s[1]], 1)
    else:
        blk, x = net.dec1, torch.cat([net.up1(s[5]), s[0]], 1)
    c = blk.conv(x)
    if extra is not None:
        c = c + extra
    return blk.gain(blk.norm(c))


def fit_two(w, g):
    """Per-channel least squares alpha + beta * g(k) ~ w(k, c) over sub-steps k -> (alpha [C], beta [C])."""
    gc = g - g.mean()
    beta = (gc.view(-1, 1) * (w - w.mean(0))).sum(0) / (gc * gc).sum()
    return w.mean(0) - beta * g.mean(), beta


def joint_hist(q_run, q_ref):
    """[5, QMAX + 1] counts of (reference spikes, this partition's spikes) per neuron in one 200 ms block."""
    i = q_ref.round().long().clamp(0, 4).flatten()
    j = q_run.round().long().clamp(0, QMAX).flatten()
    return torch.bincount(i * (QMAX + 1) + j, minlength=5 * (QMAX + 1)).view(5, QMAX + 1).cpu().numpy()


class Run(object):
    """One partition of one clip: state, per-event probabilities, statistics."""

    def __init__(self, name, mode, pieces, clip, net, variant="base", teacher=None, record=False):
        self.name, self.mode, self.pieces, self.clip, self.net = name, mode, pieces, clip, net
        self.variant = variant
        self.emit = variant if variant in EMIT_VARIANTS else "base"     # emission rule of non-forced steps
        self.teacher = teacher                                          # 50 ms run whose spikes are forced
        self.record = record
        self.trace = []                                                 # (start_ms, spikes per layer) per step
        self.tau_ref = float(clip.cfg["tau_init_ms"])
        self.states = [None] * 7
        self.prob = np.full(clip.n, np.nan, np.float32)
        self.spikes = np.zeros(7)
        self.sat = np.zeros(7)
        self.demand2 = np.zeros(7)                                      # per-step fraction with demand >= 2
        self.deficit = np.zeros(7)                                      # per-step mean (demand - emitted)
        self.block = [None] * 7                                         # spike counts since the last checkpoint
        self.steps = 0
        self.cursor = 0

    def take_block(self):
        out, self.block = self.block, [None] * 7
        self.trace = []
        return out

    def advance_to(self, t_end):
        """Process all pieces ending at or before t_end."""
        while self.cursor < len(self.pieces) and self.pieces[self.cursor][1] <= t_end:
            a, b = self.pieces[self.cursor]
            self.step(a, b)
            self.cursor += 1

    def b_current(self, a, b):
        """First-layer current of a B piece: the 50 ms window currents summed with their exact decay."""
        net, clip = self.net, self.clip
        current = None
        beta50 = torch.exp(-50.0 / net.enc1.neuron.tau()).view(1, -1, 1, 1)
        for w0 in range(a, b, 50):
            w_idx = clip.select(w0, w0 + 50)
            block = net.enc1
            i_w = block.gain(block.norm(block.conv(clip.encode(w_idx, w0, w0 + 50))))
            k = (b - (w0 + 50)) // 50
            i_w = i_w * beta50.to(i_w.dtype) ** k
            current = i_w if current is None else current + i_w
        return current

    def step(self, a, b):
        if self.teacher is not None:
            return self.forced_step(a, b)
        net, clip = self.net, self.clip
        dt = b - a
        set_dt(net, dt)
        idx = clip.select(a, b)
        prev = self.states
        if self.mode == "A":                                            # event counts, not spikes: no delivery factor
            s1, n1, u1, d1 = block_forward(net.enc1, clip.encode(idx, a, b), prev[0], dt, self.emit,
                                           spike_input=False)
            t_local = (clip.t[idx] - a) / float(b - a)
        else:                                                           # B: exact sum of the 50 ms currents
            s1, n1, u1, d1 = neuron_step(net.enc1.neuron, self.b_current(a, b), prev[0], dt, self.emit)
            t_local = ((clip.t[idx] - a) % 50) / 50.0
        spikes, news, upres, demands = run_layers(net, s1, prev, dt, self.emit)
        spikes, news, upres, demands = [s1] + spikes, [n1] + news, [u1] + upres, [d1] + demands
        if self.record:
            self.trace.append((a, spikes))
        self.finish(idx, t_local, spikes, news, upres, demands)

    def finish(self, idx, t_local, spikes, news, upres, demands):
        """Per-event readout from the last layer's u_pre, statistics, new state."""
        net, clip = self.net, self.clip
        if len(idx):
            ev = clip.events(idx, t_local)
            feat = upres[6][ev["b"], :, ev["y"], ev["x"]]                    # [N, C], as in the model
            logits = net._readout(feat, ev)
            self.prob[idx] = torch.sigmoid(logits).float().cpu().numpy()
        th = float(net.enc1.neuron.v_threshold)
        for i in range(7):
            self.spikes[i] += float(spikes[i].sum())
            self.sat[i] += float((upres[i].abs() > 10.0 * th).float().mean())
            self.demand2[i] += float((demands[i] >= 2).float().mean())
            self.deficit[i] += float((demands[i] - spikes[i]).mean())
            self.block[i] = spikes[i] if self.block[i] is None else self.block[i] + spikes[i]
        self.states = news
        self.steps += 1

    def forced_step(self, a, b):
        """Teacher-forced macro step (diagnostic): the spikes of every layer in [a, b) are the 50 ms reference's;
        each membrane is updated once for the whole piece from the variant's message of those spikes."""
        net, clip = self.net, self.clip
        sub = [s for t0, s in self.teacher.trace if a <= t0 < b]       # m sub-steps x 7 layers
        m = len(sub)
        if m * 50 != b - a:
            raise RuntimeError("teacher trace does not cover piece [%d, %d)" % (a, b))
        exact = self.variant == "tf_exact" or m == 1
        timing = self.variant == "tf_timing" and not exact
        th = float(net.enc1.neuron.v_threshold)
        k = torch.arange(1, m + 1, dtype=torch.float64, device=sub[0][0].device)
        g = torch.exp(-(m - k) * 50.0 / self.tau_ref)                   # timing message weights
        n = [sum(sub[j][i] for j in range(m)) for i in range(7)]
        msg = [sum(float(g[j]) * sub[j][i] for j in range(m)) for i in range(7)] if timing else None
        blocks = (net.enc1, net.enc2, net.enc3, net.enc4, net.dec3, net.dec2, net.dec1)
        news, upres, demands = [], [], []
        for i, blk in enumerate(blocks):
            tau = blk.neuron.tau().double()
            w = torch.exp(-(m - k).view(-1, 1) * 50.0 / tau.view(1, -1))   # [m, C] exact decay to the piece end

            def chan(v):
                return v.to(n[i].dtype).view(1, -1, 1, 1)

            if exact:
                d_own = th * sum(chan(w[j]) * sub[j][i] for j in range(m))
            elif self.variant == "tf_count_end":
                d_own = th * n[i]
            elif self.variant == "tf_count_uniform":
                d_own = th * chan(w.mean(0)) * n[i]
            else:
                alpha, beta = fit_two(w, g)
                d_own = th * (chan(alpha) * n[i] + chan(beta) * msg[i])
            if i == 0:
                d_in = self.b_current(a, b)
            elif exact:
                d_in = sum(chan(w[j]) * layer_current(net, i, sub[j]) for j in range(m))
            elif self.variant == "tf_count_end":
                d_in = layer_current(net, i, n)
            elif self.variant == "tf_count_uniform":
                d_in = chan(w.mean(0)) * layer_current(net, i, n)
            else:
                d_in = chan(alpha) * layer_current(net, i, n) + chan(beta) * layer_current(net, i, msg)
            prev = self.states[i]
            v = d_in - d_own if prev is None else chan(torch.exp(-(b - a) / tau)) * prev + d_in - d_own
            u = v + th * sub[-1][i]                                     # before the last sub-step's own reset
            news.append(v)
            upres.append(u)
            demands.append(torch.floor(torch.clamp(u, min=0.0) / th))
        idx = clip.select(a, b)
        self.finish(idx, ((clip.t[idx] - a) % 50) / 50.0, n, news, upres, demands)


def disagreement(run, ref):
    out = []
    for s, r in zip(run.states, ref.states):
        s, r = s.double().flatten(), r.double().flatten()
        rel = float((s - r).norm() / (r.norm() + 1e-12))
        cos = float(torch.dot(s, r) / (s.norm() * r.norm() + 1e-12))
        out.append((rel, cos))
    return out


def counts(prob, label, mask, threshold=0.9):
    pred = prob[mask] >= threshold
    lab = label[mask]
    return np.array([np.sum(pred & lab), np.sum(pred & ~lab), np.sum(~pred & lab)], np.int64)


def selfcheck(net, clip, cfg, variant):
    """The probe's 50 ms partition must reproduce net.forward window by window (in every variant)."""
    run = Run("A50", "A", partition("uniform", 50, 400, None), clip, net, variant)
    states = None
    worst = 0.0
    for a, b in run.pieces:
        idx = clip.select(a, b)
        x = clip.encode(idx, a, b)
        set_dt(net, 50)
        logits, states, _ = net(x, clip.events(idx, (clip.t[idx] - a) / 50.0), states)
        run.step(a, b)
        if len(idx):
            worst = max(worst, float(np.abs(torch.sigmoid(logits).float().cpu().numpy() - run.prob[idx]).max()))
    print("SELFCHECK %s (%s): max |p_forward - p_probe| = %.3g over the first 400 ms" % (
        "PASS" if worst < 1e-5 else "FAIL", variant, worst), flush=True)
    return worst < 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--variant", default="base", choices=VARIANTS)
    args = ap.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = dict(ckpt["config"])
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    net = build_net(cfg, device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    if not bool(net.gain_calibrated):
        raise RuntimeError("checkpoint gains are not calibrated")
    if int(cfg["window_ms"]) != 50:
        raise ValueError("the probe assumes a model trained with 50 ms windows")
    if cfg["neuron"] != "lif" or (args.variant != "base" and cfg["norm"] != "none"):
        raise ValueError("variants need LIF neurons and norm none (currents linear in the spike counts)")
    taus = [blk.neuron.tau().detach().cpu().numpy() for blk in (net.enc1, net.enc2, net.enc3, net.enc4, net.dec3,
                                                                 net.dec2, net.dec1)]
    print("learned tau (ms) min / median / max per layer: " + " | ".join(
        "%s %.0f/%.0f/%.0f" % (n, t.min(), np.median(t), t.max()) for n, t in zip(LAYERS, taus)), flush=True)
    names = sorted(n for n in os.listdir(os.path.join(args.root, args.split)) if n.endswith(".npz"))
    if args.max_clips:
        names = names[:args.max_clips]
    specs = [("A25", "A", "uniform", 25), ("A50", "A", "uniform", 50), ("A100", "A", "uniform", 100),
             ("A200", "A", "uniform", 200), ("Arand", "A", "randA", None), ("B50", "B", "uniform", 50),
             ("B100", "B", "uniform", 100), ("B200", "B", "uniform", 200), ("Brand", "B", "randB", None)]
    forced = args.variant in TF_VARIANTS
    if forced:
        specs = [s for s in specs if s[0] == "A50" or s[1] == "B"]      # oracles are defined on B pieces only
    elif args.variant != "base":
        specs = [s for s in specs if s[0] != "A25"]                     # cap 1, no correction: identical to base
    totals = {s[0]: {"all": np.zeros(3, np.int64), "fair": np.zeros(3, np.int64), "dp": 0.0, "flips": 0,
                     "nfair": 0, "rel": np.zeros(7), "cos": np.zeros(7), "nck": 0, "spikes": np.zeros(7),
                     "sat": np.zeros(7), "demand2": np.zeros(7), "deficit": np.zeros(7),
                     "joint": np.zeros((7, 5, QMAX + 1), np.int64), "steps": 0, "seconds": 0.0} for s in specs}
    neurons = None
    t0 = time.time()
    with torch.no_grad():
        for ci, name in enumerate(names):
            seq = load_npz_events(os.path.join(args.root, args.split, name), int(cfg["height"]), int(cfg["width_px"]),
                                  int(cfg["window_ms"]), int(cfg["n_windows"]), int(cfg["time_bins"]))
            clip = Clip(seq, cfg, q99, device, next(net.parameters()).dtype)
            if args.selfcheck and ci == 0 and not selfcheck(net, clip, cfg, args.variant):
                raise RuntimeError("self-check failed")
            rng = np.random.RandomState(zlib.crc32(name.encode("utf-8")) & 0x7fffffff)
            ref = Run("A50", "A", partition("uniform", 50, clip.total_ms, rng), clip, net, args.variant,
                      record=forced)
            runs = [ref]                                                # first: forced runs read its trace
            for n, m, k, v in (("A25", "A", "uniform", 25), ("A100", "A", "uniform", 100),
                               ("A200", "A", "uniform", 200), ("Arand", "A", "randA", None),
                               ("B100", "B", "uniform", 100), ("B200", "B", "uniform", 200),
                               ("Brand", "B", "randB", None)):
                pieces = partition(k, v, clip.total_ms, rng)            # drawn in a fixed order for every variant
                if n in totals:
                    runs.append(Run(n, m, pieces, clip, net, args.variant, teacher=ref if forced else None))
            for ck in range(BLOCK_MS, clip.total_ms + 1, BLOCK_MS):
                for r in runs:
                    r.advance_to(ck)
                q_ref = ref.take_block()
                for i in range(7):
                    totals["A50"]["joint"][i] += joint_hist(q_ref[i], q_ref[i])
                for r in runs:
                    if r is ref:
                        continue
                    q = r.take_block()
                    tot = totals[r.name]
                    for i in range(7):
                        tot["joint"][i] += joint_hist(q[i], q_ref[i])
                    d = disagreement(r, ref)
                    tot["rel"] += np.array([x[0] for x in d])
                    tot["cos"] += np.array([x[1] for x in d])
                    tot["nck"] += 1
            if neurons is None:
                neurons = np.array([float(s[0].numel()) for s in ref.states])
            tmod = clip.t % BLOCK_MS
            for r in runs:
                tot = totals[r.name]
                fair = tmod >= (BLOCK_MS - (25 if r.mode == "A" else 50))
                fair_ref = tmod >= (BLOCK_MS - 50) if r.mode == "B" else fair
                tot["all"] += counts(r.prob, clip.label, np.ones(clip.n, bool))
                tot["fair"] += counts(r.prob, clip.label, fair)
                if r is not ref:
                    m = fair_ref
                    tot["dp"] += float(np.abs(r.prob[m] - ref.prob[m]).sum())
                    tot["flips"] += int(np.sum((r.prob[m] >= 0.9) != (ref.prob[m] >= 0.9)))
                    tot["nfair"] += int(m.sum())
                tot["spikes"] += r.spikes
                tot["sat"] += r.sat
                tot["demand2"] += r.demand2
                tot["deficit"] += r.deficit
                tot["steps"] += r.steps
                tot["seconds"] += clip.total_ms / 1000.0
            b50 = totals["B50"]                                         # B with 50 ms pieces = the reference
            b50["all"] += counts(ref.prob, clip.label, np.ones(clip.n, bool))
            b50["fair"] += counts(ref.prob, clip.label, tmod >= BLOCK_MS - 50)
            b50["spikes"] += ref.spikes
            b50["sat"] += ref.sat
            b50["demand2"] += ref.demand2
            b50["deficit"] += ref.deficit
            b50["steps"] += ref.steps
            b50["seconds"] += clip.total_ms / 1000.0
            print("clip %d/%d %s done (%.0f s)" % (ci + 1, len(names), name, time.time() - t0), flush=True)

    def iou(c):
        return float(c[0] / max(c.sum(), 1))

    totals["B50"]["joint"] = totals["A50"]["joint"]
    q_axis = np.arange(QMAX + 1, dtype=np.float64)

    def block_stats(h):
        """h [5, QMAX + 1] -> count ratio sum q / sum q_ref, P(q = q_ref | q_ref >= 1), mean q | q_ref = 4."""
        h = h.astype(np.float64)
        ratio = float((h.sum(0) * q_axis).sum() / max((h.sum(1) * q_axis[:5]).sum(), 1.0))
        match = float(sum(h[r, r] for r in range(1, 5)) / max(h[1:].sum(), 1.0))
        q4 = float((h[4] * q_axis).sum() / max(h[4].sum(), 1.0))
        return ratio, match, q4

    report = {}
    print("\nvariant %s" % args.variant, flush=True)
    print("%-6s %8s %7s %7s | %8s %9s %7s | %-34s | %-34s | %s" % (
        "part", "IoU_all", "recall", "FP", "IoU_fair", "mean|dp|", "flips%", "state rel. diff (enc1..dec1)",
        "spikes/neuron/s (enc1..dec1)", "|u|>10th"), flush=True)
    for n, mode, kind, value in specs:
        tot = totals[n]
        c, f = tot["all"], tot["fair"]
        row = {"mode": mode, "partition": kind if value is None else "%d ms" % value,
               "iou_all": iou(c), "recall_all": float(c[0] / max(c[0] + c[2], 1)), "false_events": int(c[1]),
               "iou_fair": iou(f),
               "mean_abs_dp_fair": tot["dp"] / max(tot["nfair"], 1),
               "flip_rate_fair": tot["flips"] / max(tot["nfair"], 1),
               "state_rel_diff": (tot["rel"] / max(tot["nck"], 1)).tolist(),
               "state_cos": (tot["cos"] / max(tot["nck"], 1)).tolist(),
               "spikes_per_neuron_per_s": (tot["spikes"] / neurons / max(tot["seconds"], 1e-9)).tolist(),
               "frac_u_above_10th": (tot["sat"] / max(tot["steps"], 1)).tolist(), "steps": tot["steps"],
               "frac_demand_ge2": (tot["demand2"] / max(tot["steps"], 1)).tolist(),
               "mean_deficit": (tot["deficit"] / max(tot["steps"], 1)).tolist(),
               "block_joint_ref_vs_run": tot["joint"].tolist()}
        stats = [block_stats(tot["joint"][i]) for i in range(7)]
        row["block_count_ratio"] = [s[0] for s in stats]
        row["block_match_active"] = [s[1] for s in stats]
        row["block_mean_q_given_ref4"] = [s[2] for s in stats]
        report[n] = row
        print("%-6s %8.4f %7.4f %7d | %8.4f %9.4f %7.2f | %-34s | %-34s | %s" % (
            n, row["iou_all"], row["recall_all"], row["false_events"], row["iou_fair"], row["mean_abs_dp_fair"],
            100 * row["flip_rate_fair"], " ".join("%.2f" % v for v in row["state_rel_diff"]),
            " ".join("%.3g" % v for v in row["spikes_per_neuron_per_s"]),
            " ".join("%.3f" % v for v in row["frac_u_above_10th"])), flush=True)

    marg = totals["A50"]["joint"].sum(2).astype(np.float64)                 # [7, 5] reference counts per block
    mult = []
    print("\nreference (50 ms) spikes per neuron per 200 ms block")
    print("%-5s %8s | %7s %7s %7s %7s | %8s" % ("layer", "P(q>=1)", "q=1|act", "q=2|act", "q=3|act", "q=4|act",
                                                "P(q>=2)"))
    for i, layer in enumerate(LAYERS):
        p = marg[i] / max(marg[i].sum(), 1.0)
        act = p[1:] / max(p[1:].sum(), 1e-12)
        mult.append({"p_q": p.tolist(), "p_q_given_active": act.tolist(), "p_q_ge2": float(p[2:].sum())})
        print("%-5s %8.4f | %7.3f %7.3f %7.3f %7.3f | %8.4f" % (layer, p[1:].sum(), act[0], act[1], act[2], act[3],
                                                                p[2:].sum()))
    print("\nblock spike counts vs the reference, per layer enc1..dec1")
    print("%-6s | %-41s | %-41s | %s" % ("part", "count ratio sum(q) / sum(q_ref)", "P(q = q_ref | q_ref >= 1)",
                                         "mean q given q_ref = 4"))
    for n, _, _, _ in specs:
        row = report[n]
        print("%-6s | %-41s | %-41s | %s" % (n, " ".join("%.3f" % v for v in row["block_count_ratio"]),
                                             " ".join("%.3f" % v for v in row["block_match_active"]),
                                             " ".join("%.2f" % v for v in row["block_mean_q_given_ref4"])))
    print("\nlatent demand floor(u_pre / v_th) per neuron-step, per layer enc1..dec1")
    print("%-6s | %-41s | %s" % ("part", "P(demand >= 2)", "mean (demand - emitted)"))
    for n, _, _, _ in specs:
        row = report[n]
        print("%-6s | %-41s | %s" % (n, " ".join("%.3f" % v for v in row["frac_demand_ge2"]),
                                     " ".join("%.3f" % v for v in row["mean_deficit"])))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "partition_probe.json"), "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint), "split": args.split, "clips": names,
                   "variant": args.variant, "layers": LAYERS, "reference_multiplicity": mult,
                   "results": report}, fh, indent=2)
    print("PROBE FINISHED (%s): %d clips, %.0f s -> %s" % (args.variant, len(names), time.time() - t0, args.out),
          flush=True)


if __name__ == "__main__":
    main()
