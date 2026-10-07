"""Stage-1 gate for the data layer.

EV-UAV: every recording read by SPEED must equal the V1-V3 loader (dataset/stream_windows.load_npz_events) bit for bit,
including the 50 ms / 160-window partition. Sources: split directories (--evuav-root) or the original zip files.
EV-Flying: every recording must load and validate; counts are compared with an optional reference json.

    python bridge/check_readers.py --evuav-zip train.zip val.zip test.zip --evflying-root /path/to/evflying
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # legacy V1-V3 code
sys.path.insert(0, os.path.join(ROOT, "SPEED"))

from dataset.stream_windows import load_npz_events  # noqa: E402
from speed.data.dataset_card import list_recordings, load_card  # noqa: E402
from speed.data.readers import read_evuav_npz, read_recording  # noqa: E402


def compare_evuav(new, old):
    """Names of fields that differ between the SPEED stream and the legacy StreamSequence."""
    diffs = []
    pairs = (("x", new.x, old.x), ("y", new.y, old.y), ("t", new.t, old.t.astype(np.int64) * 1000),
             ("p", new.p, old.p), ("label", new.label, old.label), ("target_id", new.target_id, old.target_id))
    for name, a, b in pairs:
        if a.shape != b.shape or not np.array_equal(a.astype(np.float64), np.asarray(b, dtype=np.float64)):
            diffs.append(name)
    order, bounds = new.window_partition(50000, 160)
    if not np.array_equal(order, old.order):
        diffs.append("order")
    if not np.array_equal(bounds, old.bounds):
        diffs.append("bounds")
    return diffs


def evuav_sources(args):
    if args.evuav_root:
        card = load_card("evuav")
        for split in ("train", "val", "test"):
            for name, path in list_recordings(card, args.evuav_root, split):
                with open(path, "rb") as handle:
                    yield name, handle.read()
    for archive in args.evuav_zip or []:
        with zipfile.ZipFile(archive) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".npz")):
                yield name, zf.read(name)


def check_evuav(args):
    card = load_card("evuav")
    total = failed = 0
    for name, blob in evuav_sources(args):
        new = read_evuav_npz(io.BytesIO(blob), card, name)
        old = load_npz_events(io.BytesIO(blob), 260, 346, 50, 160, 5)
        diffs = compare_evuav(new, old)
        total += 1
        if diffs:
            failed += 1
            print("MISMATCH %s: %s" % (name, ", ".join(diffs)))
    print("EV-UAV: %d recordings, %d mismatches" % (total, failed))
    return failed == 0 and total > 0


def check_evflying(args):
    card = load_card("evflying")
    reference = {}
    if args.evflying_reference:
        with open(args.evflying_reference, "r", encoding="utf-8") as handle:
            for r in json.load(handle)["sequences"]:
                reference["%s/%s/%s.npy" % (r["split"], r["id"], r["id"])] = r
    splits = ["train", "val", "test"] + (["extra_drone"] if args.with_drone else [])
    ok = True
    for split in splits:
        for name, path in list_recordings(card, args.evflying_root, split):
            stream = read_recording(card, path, name)
            s = stream.summary()
            line = "%-6s %-18s events %10d  target events %9d  targets %4d" % (
                split, name, s["n_events"], s["n_target_events"], s["n_targets"])
            ref = reference.get(name)
            if ref is not None:
                expected_target = int(round(ref["label_share"] * ref["n_events"]))
                same = (ref["n_events"] == s["n_events"] and ref.get("n_objects") == s["n_targets"]
                        and abs(expected_target - s["n_target_events"]) <= 1)
                line += "  reference %s" % ("ok" if same else "MISMATCH")
                ok &= same
            print(line)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evuav-root", default=None)
    parser.add_argument("--evuav-zip", nargs="*", default=None)
    parser.add_argument("--evflying-root", default=None)
    parser.add_argument("--evflying-reference", default=None)
    parser.add_argument("--with-drone", action="store_true")
    args = parser.parse_args()
    results = []
    if args.evuav_root or args.evuav_zip:
        results.append(check_evuav(args))
    if args.evflying_root:
        results.append(check_evflying(args))
    print("GATE", "PASS" if results and all(results) else "FAIL")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
