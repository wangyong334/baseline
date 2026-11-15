"""Per-event predictions of a competitor (EV-SpSegNet or PACT) from its official code.

The official test.py of both repositories keeps the predictions in memory only. This script runs the same pipeline
(the repository's dataset class, collate, model and p2v mapping, every event of the recording, eval mode, no grad)
on the recordings of one split sorted by name and writes each recording's per-event probabilities to
<out>/<name>.npz (key "prob", the event order of the original npz). For the train split the repositories' random
event subsampling (max_events_num) is switched off, so every event gets a prediction.
Memory: a recording is processed whole (the 8 s input of both methods). Only if that runs out of GPU memory it is
split at its middle time and each half processed alone, recursively (at most 8 parts); the number of split
recordings is reported in the last line (0 = every recording had its full 8 s context).
Metrics: IoU / ACC / Pd / Fa at threshold 0.9 with the baseline's evaluator (EV-SpSegNet utils/eval.py) for both
methods; PACT's copy of it counts false-alarm components inside the per-event loop (indentation), which inflates Fa.
Usage (any dir; the repository's compiled ops and deps must be importable):
    python tools/vis/dump_competitor.py --repo <official repo dir> --model evspsegnet|PACT --config <yaml> \
        --ckpt <weights .pt> --out <dir> [--split test] [--eval-repo <EV-Seg dir>]
Last line: "DUMP COMPETITOR FINISHED ...".
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np


def predict(net, dataset, item, torch, depth=0):
    """Probabilities of every event of one recording (dict of per-event arrays); split in time on CUDA OOM."""
    n = len(item["ev_loc"])
    try:
        batch = dataset.custom_collate([item])
        with torch.no_grad():
            preds = net(batch["voxel_ev"])[0]
            prob = preds[batch["p2v_map"].long().cuda()].reshape(-1).float().cpu().numpy()
        if len(prob) != n:
            raise RuntimeError("%d predictions for %d events" % (len(prob), n))
        return prob, 0
    except RuntimeError as e:
        if "out of memory" not in str(e) or depth >= 3:
            raise
    torch.cuda.empty_cache()                               # the failed attempt's tensors are released here
    t = item["ev_loc"][:, 2]
    mid = 0.5 * (float(t.min()) + float(t.max()))
    prob = np.zeros(n, np.float32)
    for m in (t < mid, t >= mid):
        sub = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in item.items()}
        prob[m], _ = predict(net, dataset, sub, torch, depth + 1)
    return prob, 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True, choices=("evspsegnet", "PACT"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-repo", default=None, help="EV-Seg repository whose utils/eval.py gives the metrics "
                                                      "(default: --repo)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    out = os.path.abspath(args.out)
    eval_py = os.path.join(os.path.abspath(args.eval_repo or args.repo), "utils", "eval.py")
    sys.argv = [sys.argv[0], "--config", os.path.abspath(args.config)]   # the repositories parse argv on import
    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch
    from configs.configs import cfg
    from dataset.ev_uav import EvUAV
    if args.model == "evspsegnet":
        from model.evspsegnet import evspsegnet as Net
    else:
        from model.PACT import PACT as Net
    spec = importlib.util.spec_from_file_location("baseline_eval", eval_py)
    baseline_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_eval)

    cfg.max_events_num = 10 ** 12                          # no random subsampling, also for the train split
    net = Net(cfg).eval().cuda()
    net.load_state_dict(torch.load(args.ckpt, map_location="cuda:0"))
    dataset = EvUAV(cfg, mode=args.split)
    dataset.file_list = sorted(dataset.file_list)
    evaluter = baseline_eval.evalute(cfg)
    os.makedirs(out, exist_ok=True)
    t0, n_split = time.time(), 0
    for sample, name in enumerate(dataset.file_list):
        item = dataset[sample]
        lab_key = "seg_label" if "seg_label" in item else "seg_label_ev"
        prob, was_split = predict(net, dataset, item, torch)
        n_split += was_split
        np.savez_compressed(os.path.join(out, name), prob=prob.astype(np.float32))
        loc = np.asarray(item["ev_loc"], np.float32)
        label = torch.from_numpy(np.asarray(item[lab_key], np.float32))
        pred = torch.from_numpy(prob)
        evaluter.matches[str(sample)] = {"seg_pred": pred, "seg_gt": label}
        ev_locs = torch.from_numpy(np.hstack([np.zeros((len(loc), 1), np.float32), loc]))   # (b, x, y, t)
        evaluter.roc_update(ev_locs[:, 3], pred.clone(), np.asarray(item["idx"]), label, ev_locs)
        print("%s %3d/%d %s: %d events%s (%.0f s)" % (args.split, sample + 1, len(dataset.file_list), name, len(prob),
              ", split in time (GPU memory)" if was_split else "", time.time() - t0), flush=True)
        torch.cuda.empty_cache()
    iou = float(evaluter.evaluate_semantic_segmantation_miou())
    acc = float(evaluter.evaluate_semantic_segmantation_accuracy())
    pd, fa = evaluter.cal_roc()
    print("DUMP COMPETITOR FINISHED %s %s: IoU %.4f ACC %.4f Pd %.4f Fa %.3e (baseline evaluator, threshold 0.9), "
          "%d of %d recordings split in time -> %s" % (args.model, args.split, iou, acc, float(pd), float(fa), n_split,
                                                       len(dataset.file_list), out), flush=True)


if __name__ == "__main__":
    main()
