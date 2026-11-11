"""Paired synthetic SDU-only probe: projection grouping on/off, same weights.

Not a whole-system latency or accuracy benchmark. Excludes BIR/index/VoW and
data preparation. Tests the second iteration separately from stream caching.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst.benchmark import make_case
from tools.bst.pipeline import load_model, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=21)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model, ck = load_model(args.checkpoint, device)
    model.eval()
    result = {"scope": __doc__, "hardware": torch.cuda.get_device_name(device),
              "torch": torch.__version__, "checkpoint": args.checkpoint, "cases": []}
    for queries in [256, 1024]:
        data, meta = make_case(8192, queries, device)
        def run(fused):
            model.fused_inference = fused
            source = model.encode_sources(data["source"])
            pairs = model.prepare_pairs(source, data["indices"], data["relative"])
            state = model.make_state(data["prior"], data["polarity"])
            return model.reaudit(source, data["indices"], data["relative"], state,
                                 data["prior"], data["polarity"], 10, pair_cache=pairs)[0]
        times = {False: [], True: []}
        with torch.no_grad():
            for _ in range(3):
                run(False)
                run(True)
            for repeat in range(args.repeats):
                for fused in ([False, True] if repeat % 2 == 0 else [True, False]):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    value = run(fused)
                    torch.cuda.synchronize()
                    times[fused].append((time.perf_counter()-start)*1000)
            a, b = run(False), run(True)
            error = float((a-b).abs().max())
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        row = {"queries": queries, "sources": 8192, "neighbors": meta["total_neighbor_slots"],
               "unfused_ms": times[False], "fused_ms": times[True], "max_abs_difference": error,
               "unfused_median_ms": float(np.median(times[False])),
               "fused_median_ms": float(np.median(times[True]))}
        result["cases"].append(row)
        print(queries, row["unfused_median_ms"], row["fused_median_ms"], error, flush=True)
    save_json(args.out, result)


if __name__ == "__main__":
    main()
