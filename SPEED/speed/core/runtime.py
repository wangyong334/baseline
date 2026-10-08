"""Seeding and deterministic execution. Entry points set CUBLAS_WORKSPACE_CONFIG before importing torch."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def peak_memory_gib(device):
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def memory_line(device, label):
    """Allocated / reserved / peak GPU memory (GiB) after releasing cached blocks; '' on CPU."""
    if device.type != "cuda":
        return ""
    torch.cuda.empty_cache()
    gib = 2 ** 30
    return "[memory] %-18s allocated %.2f reserved %.2f peak %.2f GiB" % (
        label, torch.cuda.memory_allocated(device) / gib, torch.cuda.memory_reserved(device) / gib,
        torch.cuda.max_memory_allocated(device) / gib)

