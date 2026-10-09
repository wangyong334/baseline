"""Causal inference with a SPEED checkpoint: one result file per recording and readout, plus a run summary.

    python tools/infer.py --checkpoint runs/a/best.pt --root /path/to/EV-UAV-dataset --split test \
        --out runs/a/test [--readouts net fused_d2 pub] [--evaluate] [--device cuda:1]
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")  # limits fragmentation on large canvases

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from speed.core.accounting import EnergyStats, energy_parts  # noqa: E402
from speed.core.build import build_system, with_overrides, with_sections  # noqa: E402
from speed.core.runtime import peak_memory_gib, seed_everything, write_json  # noqa: E402
from speed.core.training import LayerMonitor, load_checkpoint  # noqa: E402
from speed.data.dataset_card import iter_split, load_card  # noqa: E402
from speed.eval.breakdown import MotionAccuracy  # noqa: E402
from speed.eval.energy import system_energy  # noqa: E402
from speed.eval.metrics import BenchmarkMetrics  # noqa: E402
from speed.eval.results import save_result  # noqa: E402
from speed.slots.publish.readouts import MotionProbe  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--card", default=None, help="defaults to the config's dataset")
    parser.add_argument("--readouts", nargs="*", default=None, help="subset of readout names to save")
    parser.add_argument("--sections", nargs="*", default=[],
                        help="YAML files whose top-level sections replace the checkpoint's (eval-time only)")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides section.key=value (eval-time only)")
    parser.add_argument("--recordings", nargs="*", default=None, help="only these recordings (e.g. test/test_003.npz)")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg = with_overrides(with_sections(ckpt["config"], args.sections), args.set)
    seed_everything(int(cfg["training"]["seed"]), bool(cfg["training"].get("deterministic", True)))
    system = build_system(cfg)
    system.network.load_state_dict(ckpt["network"])
    if not bool(system.network.backbone.gain_calibrated):
        raise RuntimeError("checkpoint gains are not calibrated")
    system.to(device)
    system.network.eval()
    card = load_card(args.card or cfg["dataset"])
    names = system.readout_names()
    keep = names if args.readouts is None else args.readouts
    unknown = sorted(set(keep) - set(names))
    if unknown:
        raise ValueError("unknown readouts %s (available %s)" % (unknown, names))
    ev = cfg["evaluation"]
    metrics = {n: BenchmarkMetrics(card["sensor"]["width"], card["sensor"]["height"], int(ev["frame_ms"] * 1000),
                                   ev["threshold"], ev["correct_thresh"]) for n in keep} if args.evaluate else {}
    monitor = LayerMonitor(system.network.v_threshold)
    energy = EnergyStats()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    probe = MotionProbe() if "motion" in system.network.heads.outputs else None
    motion_acc = MotionAccuracy(system.clock.step_ms, int(round(system.clock.step_ms * 1000))) if probe else None
    run_readouts = list(system.readouts) + ([probe] if probe else [])
    t0 = time.perf_counter()
    events = steps_total = recordings = 0
    with torch.no_grad():
        for stream in iter_split(card, args.root, args.split, args.recordings):
            results, steps = system.run_stream(stream, carry=bool(cfg["training"].get("carry", True)), monitor=monitor,
                                               energy=energy, readouts=run_readouts)
            if probe is not None:
                motion_acc.update(stream, probe.extra["motion"])
            canvas = system.canvas(stream)
            for name in keep:
                prob, publish_us = results[name]
                save_result(os.path.join(args.out, name), stream, prob, publish_us=publish_us,
                            meta={"checkpoint": os.path.abspath(args.checkpoint), "readout": name})
                if name in metrics:
                    metrics[name].update(stream, prob, publish_us=publish_us)
            events += stream.n_events
            steps_total += steps.n_steps
            recordings += 1
    seconds = time.perf_counter() - t0
    summary = {"checkpoint": os.path.abspath(args.checkpoint), "split": args.split, "recordings": recordings,
               "events": events, "steps": steps_total, "events_per_step": events / float(max(steps_total, 1)),
               "seconds": seconds, "peak_memory_gib": peak_memory_gib(device), "layers": monitor.summary(),
               "readouts": keep, "overrides": args.set, "sections": args.sections, "verifier": cfg.get("verifier")}
    stats = energy.summary(summary["layers"])
    parts = energy_parts(system, canvas[0], canvas[1], stats)
    summary["energy"] = {"stats": stats, "canvas": list(canvas),
                         "report": system_energy(parts, system.clock.step_us, card.get("clip_duration_us"))}
    tot = summary["energy"]["report"]["total"]["x10"]
    print("energy (exp/log = 10 MAC): %.3f mJ/s%s | %s" % (
        tot["mj_per_s"], " (%.2f mJ per clip)" % tot["mj_per_clip"] if "mj_per_clip" in tot else "",
        "  ".join("%s %.3f" % (k, v["x10"]["mj_per_s"]) for k, v in summary["energy"]["report"]["parts"].items())),
        flush=True)
    for name, m in metrics.items():
        r = m.result()
        summary.setdefault("metrics", {})[name] = {k: v for k, v in r.items() if k != "per_recording"}
        fmt = lambda v, spec: "-" if v is None else spec % v  # noqa: E731  (no detections -> no latency)
        print("%-10s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | publish mean %s ms | first det. median %s ms" % (
            name, r["iou"], r["acc"], r["pd"], r["fa"], fmt(r["publish_latency"]["mean_ms"], "%.1f"),
            fmt(r["first_detection"]["median_ms"], "%.1f")), flush=True)
    if motion_acc is not None:
        summary["motion_eval"] = ma = motion_acc.result()
        if ma:
            fmt = lambda v: "-" if v is None else "%.2f" % v  # noqa: E731
            print("learned motion on target events (px/step): angle median %s deg | speed ratio %s | error median %s"
                  " | inside sigma points %s" % (fmt(ma["angle_median"]), fmt(ma["speed_ratio_median"]),
                                                 fmt(ma["error_median"]), fmt(ma["inside_sigma_points"])), flush=True)
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "infer_summary.json"), summary)
    print("INFER FINISHED: %d recordings, %.0f s -> %s" % (recordings, seconds, args.out), flush=True)


if __name__ == "__main__":
    main()
