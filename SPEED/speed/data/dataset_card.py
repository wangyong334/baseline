import fnmatch
import json
import os

from speed.data.readers import read_recording

CARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards")


def load_card(name_or_path):
    path = name_or_path
    if not os.path.isfile(path):
        path = os.path.join(CARD_DIR, "%s.json" % name_or_path)
    with open(path, "r", encoding="utf-8") as handle:
        card = json.load(handle)
    card["_path"] = os.path.abspath(path)
    return card


def stats_path(card):
    return os.path.splitext(card["_path"])[0] + ".stats.json"


def list_recordings(card, root, split):
    """[(name, absolute path)] of one split; a split is either an explicit file list or a directory pattern."""
    spec = card["splits"][split]
    if "files" in spec:
        rel = list(spec["files"])
    else:
        directory = os.path.join(root, spec["dir"])
        rel = [os.path.join(spec["dir"], n) for n in sorted(os.listdir(directory))
               if fnmatch.fnmatch(n, spec.get("pattern", "*"))]
    if not rel:
        raise RuntimeError("%s: split %s is empty under %s" % (card["name"], split, root))
    items = []
    for r in rel:
        path = os.path.join(root, r)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        items.append((r.replace("\\", "/"), path))
    return items


def iter_split(card, root, split):
    for name, path in list_recordings(card, root, split):
        yield read_recording(card, path, name)
