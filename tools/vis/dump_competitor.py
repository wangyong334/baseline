"""Per-event predictions of a competitor (EV-SpSegNet or PACT) from its official code, for the qualitative figure.

The official test.py of both repositories keeps the predictions in memory only. This script runs the same test loop
(same dataset class, collate, model and p2v mapping) with three changes needed to save them: the recordings are
sorted by name, batch size 1, and each recording's per-event probabilities are written to <out>/<name>.npz (key
"prob", the order of the events in the original npz). The official IoU / ACC / Pd / Fa of the saved predictions are
printed (threshold 0.9, the repositories' own evaluation code).
Usage (any dir; the repository's compiled ops and deps must be importable):
    python tools/vis/dump_competitor.py --repo <official repo dir> --model evspsegnet|PACT \
        --config <yaml> --ckpt <weights .pt> --out <dir> [--split test]
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True, choices=("evspsegnet", "PACT"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    sys.argv = [sys.argv[0], "--config", os.path.abspath(args.config)]   # the repositories parse argv on import
    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch
    from configs.configs import cfg
    from dataset.ev_uav import EvUAV
    from utils.eval import evalute
    if args.model == "evspsegnet":
        from model.evspsegnet import evspsegnet as Net
    else:
        from model.PACT import PACT as Net

    net = Net(cfg).eval().cuda()
    net.load_state_dict(torch.load(args.ckpt, map_location="cuda:0"))
    dataset = EvUAV(cfg, mode=args.split)
    dataset.file_list = sorted(dataset.file_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.custom_collate)
    evaluter = evalute(cfg)
    os.makedirs(args.out, exist_ok=True)
    for sample, ev in enumerate(loader):
        with torch.no_grad():
            x = ev['voxel_ev']
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float()
            idx = ev['idx_label']
            preds = net(x)[0]
            preds = preds[p2v_map].squeeze().cpu()
            name = dataset.file_list[sample]
            n_events = len(np.load(os.path.join(dataset.root, name))["ev_loc"])
            if preds.numel() != n_events:
                raise RuntimeError("%s: %d predictions for %d events" % (name, preds.numel(), n_events))
            np.savez_compressed(os.path.join(os.path.abspath(args.out), name), prob=preds.numpy().astype(np.float32))
            evaluter.matches[str(sample)] = {'seg_pred': preds, 'seg_gt': label}
            evaluter.roc_update(ev_locs[:, 3], preds, idx, label.cpu(), ev_locs)
        print("%s: %d events" % (name, n_events), flush=True)
    iou = float(evaluter.evaluate_semantic_segmantation_miou())
    acc = float(evaluter.evaluate_semantic_segmantation_accuracy())
    pd, fa = evaluter.cal_roc()
    print("DUMP COMPETITOR FINISHED %s %s: IoU %.4f ACC %.4f Pd %.4f Fa %.3e -> %s" % (
        args.model, args.split, iou, acc, float(pd), float(fa), args.out), flush=True)


if __name__ == "__main__":
    main()
