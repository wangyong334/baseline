"""Keep GPU memory reserved for a job pool (11-14): every few seconds read the target (GB) from a control file that the
pool writes (HOLD_GB minus JOB_GB per running job) and allocate or free 1 GiB blocks to match it; a block that does
not fit is retried later. Exits, freeing everything, when the stop file exists.
Usage (the pool starts it): CUDA_VISIBLE_DEVICES=<id> python tools/v18/hold_gpu.py <control file> <stop file>
"""
import os
import sys
import time

import torch

GIB = 1 << 28                                      # float32 elements in 1 GiB


def main(ctl, stop, period=5.0):
    dev = torch.device("cuda:0")
    blocks, last = [], None
    while not os.path.exists(stop):
        try:
            with open(ctl) as fh:
                want = max(0, int(float(fh.read().strip())))
        except (OSError, ValueError):
            want = len(blocks)                     # control file being rewritten: keep the current size
        while len(blocks) > want:
            blocks.pop()
            torch.cuda.empty_cache()
        while len(blocks) < want:
            try:
                blocks.append(torch.empty(GIB, dtype=torch.float32, device=dev))
            except RuntimeError:                   # the card is full: try again next round
                torch.cuda.empty_cache()
                break
        if len(blocks) != last:
            last = len(blocks)
            print("%s holding %d GiB (target %d)" % (time.strftime("%m-%d %H:%M:%S"), last, want), flush=True)
        time.sleep(period)
    blocks.clear()
    torch.cuda.empty_cache()
    print("%s released, stop file seen" % time.strftime("%m-%d %H:%M:%S"), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
