"""Short real-record implementation probe; never a formal accuracy experiment."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.v15.train import (make_models, records, train_epoch, infer_record,
                            save_checkpoint, write_json, validate_config)
from train_stream_v1 import seed_everything, run_calibration
from utils.stream_common import load_flat_config
from utils.comm_budget import CommBudget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/v15/observation.yaml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunks", type=int, default=3)
    p.add_argument("--streams", type=int)
    p.add_argument("--tbptt-k", type=int)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_flat_config(args.config)
    cfg["root"] = args.data_root
    cfg["_training_scope"] = {"limit_records": None, "max_chunks": args.chunks}
    if args.streams is not None:
        cfg["streams"] = args.streams
    if args.tbptt_k is not None:
        cfg["tbptt_k"] = args.tbptt_k
    validate_config(cfg)
    torch.set_num_threads(4)
    seed_everything(cfg["seed"], cfg["deterministic"])
    device = torch.device(args.device)
    stats = json.loads((ROOT / "configs/stream_v1_train_stats.json").read_text(encoding="utf-8"))
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    pos_weight = min(float(stats["train_neg"]) / stats["train_pos"], cfg["pos_weight_cap"])
    net, head = make_models(cfg, device)
    calibration_cfg = dict(cfg, calib_sequences=2, calib_windows=8)
    run_calibration(net, calibration_cfg, q99, device, out)
    data = records(cfg, "train")
    parameters = list(net.parameters()) + ([] if head is None else list(head.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=cfg["lr"])
    comm = CommBudget.from_config(cfg, net)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial = head.output.weight.detach().clone() if head is not None else None
    result = train_epoch(net, head, data, optimizer, cfg, q99, pos_weight, device,
                         np.random.RandomState(cfg["seed"] * 1000 + 7), comm, max_chunks=args.chunks)
    if not all(torch.isfinite(p).all() for p in parameters):
        raise RuntimeError("Nonfinite parameter after short training")
    if head is not None and torch.equal(initial, head.output.weight):
        raise RuntimeError("Revision projection did not learn")
    memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    net.eval()
    if head is not None:
        head.eval()
    seq = data[0]
    pred, timing, detail = infer_record(net, head, seq, cfg, q99, device,
                                       modes=("final",), diagnostics=False)
    if detail["event_readouts"] != seq.n_events or not np.isfinite(pred["final"]).all():
        raise RuntimeError("Online main readout count/nonfinite failure")
    save_checkpoint(out / "probe.pt", net, head, optimizer, 0, -1., cfg, stats, comm)
    result = {"purpose": "implementation/gradient/memory probe, not formal model accuracy",
              "torch": torch.__version__, "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "streams": cfg["streams"], "tbptt_k": cfg["tbptt_k"], "chunks": args.chunks,
              "calibration": "2 train recordings x 8 windows; formal calibration is unchanged",
              "training": result, "peak_training_allocated_gib": memory,
              "record": seq.name, "events": seq.n_events,
              "inference_final_events": detail["counts"]["final_events"],
              "inference_event_reads": detail["event_readouts"],
              "mean_step_ms": float(np.mean(timing["step_ms"])),
              "revision_parameters": sum(p.numel() for p in head.parameters()) if head is not None else 0,
              "revision_macs": head.dense_macs(cfg["pad_height"], cfg["pad_width"]) if head is not None else 0}
    write_json(out / "probe.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
