"""Elastic GPU placeholder (11-10): keeps a pool GPU ours for as long as its runner lives.

The fixed placeholder of gpu_queue.sh (HOLD_GB GB, only while the queue is empty) left the card open while a job ran
(our jobs use about 5 GB of 24 GB), and after the job its 20 GB request failed once another program had come in.
This process instead holds GPU memory in chunks all the time and keeps only a margin free:
    queue idle (no job in <queue>/running/)                         free <= --idle-free-gb
    a job runs                                                      free ~ --job-free-gb (room for our job to grow)
    the first --warmup-min minutes after a job starts, or after a
    new process of ours (cwd under the code dir) appears on the GPU  free ~ --start-free-gb
It only takes memory that is free and gives chunks back as soon as free memory drops below the target, so it never
evicts a running program; a new program of another user starts only if it fits into the free margin.
Free memory comes from nvidia-smi (torch 1.9 has no mem_get_info).
Stop: touch <queue>/HOLD_STOP (the chunks are released and the process exits); pid in <queue>/hold.pid.
Usage (code dir, after `conda activate evuav`; gpu_queue.sh starts it itself when HOLD_GB=elastic):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nohup python -u tools/final/gpu_hold.py --gpu 1 \
        --queue run/queue/gpu1 >> run/queue/gpu1/hold_elastic.log 2>&1 < /dev/null &
"""
import argparse
import glob
import os
import subprocess
import time

MIB = 2 ** 20


def ts():
    return time.strftime("%m-%d %H:%M:%S")


def log(msg):
    print("[%s] %s" % (ts(), msg), flush=True)


def smi(args):
    return subprocess.run(["nvidia-smi"] + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          universal_newlines=True, timeout=30).stdout


def gpu_mem(gpu):
    """(total MiB, free MiB, PCI bus id) of physical GPU `gpu`."""
    out = smi(["--query-gpu=memory.total,memory.free,pci.bus_id", "--format=csv,noheader,nounits", "-i", str(gpu)])
    total, free, bus = [x.strip() for x in out.strip().split(",")]
    return int(total), int(free), bus


def our_pids(bus, code_dir, me):
    """PIDs on the GPU whose working directory lies under the code dir (our jobs), this process excluded."""
    pids = set()
    for line in smi(["--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"]).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or parts[0] != bus or not parts[1].isdigit() or int(parts[1]) == me:
            continue
        try:
            cwd = os.path.realpath(os.readlink("/proc/%s/cwd" % parts[1]))
        except OSError:
            continue
        if cwd == code_dir or cwd.startswith(code_dir + os.sep):
            pids.add(int(parts[1]))
    return pids


def target_free_mib(job_running, now, warm_until, idle_gb, job_gb, start_gb):
    """Free memory to leave on the card (MiB)."""
    if not job_running:
        return int(idle_gb * 1024)
    return int((start_gb if now < warm_until else job_gb) * 1024)


def plan(free_mib, target_mib, held, chunk_mib, max_step):
    """Chunks to allocate (> 0) or release (< 0): release whenever free memory is below the target, allocate only
    while more than one chunk above it, at most max_step chunks per poll."""
    if free_mib < target_mib:
        return -min(held, -(-(target_mib - free_mib) // chunk_mib))
    if free_mib > target_mib + chunk_mib:
        return min((free_mib - target_mib) // chunk_mib, max_step)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, required=True, help="physical GPU id (nvidia-smi index)")
    ap.add_argument("--queue", required=True, help="runner queue dir, e.g. run/queue/gpu1")
    ap.add_argument("--code-dir", default=".", help="processes with their cwd under this dir count as ours")
    ap.add_argument("--idle-free-gb", type=float, default=float(os.environ.get("HOLD_IDLE_FREE_GB", 1)))
    ap.add_argument("--job-free-gb", type=float, default=float(os.environ.get("HOLD_JOB_FREE_GB", 4)))
    ap.add_argument("--start-free-gb", type=float, default=float(os.environ.get("HOLD_START_FREE_GB", 8)))
    ap.add_argument("--warmup-min", type=float, default=float(os.environ.get("HOLD_WARMUP_MIN", 5)))
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--max-step-gb", type=float, default=4.0, help="largest growth per poll")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between memory checks")
    ap.add_argument("--pid-poll", type=float, default=10.0, help="seconds between checks for new processes of ours")
    a = ap.parse_args()
    import torch
    me = os.getpid()
    code_dir = os.path.realpath(a.code_dir)
    stop = os.path.join(a.queue, "HOLD_STOP")
    with open(os.path.join(a.queue, "hold.pid"), "w") as f:
        f.write(str(me))
    total, free, bus = gpu_mem(a.gpu)
    max_step = max(1, int(a.max_step_gb * 1024) // a.chunk_mb)
    log("elastic hold on GPU %d (%s, %d MiB, pid %d): free kept at %.1f GB idle, %.1f GB during a job, %.1f GB for "
        "%.0f min after a job or a new process of ours starts" % (a.gpu, bus, total, me, a.idle_free_gb,
                                                                  a.job_free_gb, a.start_free_gb, a.warmup_min))
    held, seen, last_job, warm_until, next_pid_check, last_report = [], set(), None, 0.0, 0.0, None
    while True:
        if os.path.exists(stop):
            os.remove(stop)
            del held[:]
            torch.cuda.empty_cache()
            log("HOLD_STOP found: released, exiting")
            return
        now = time.time()
        running = sorted(glob.glob(os.path.join(a.queue, "running", "*.sh")))
        job = os.path.basename(running[0]) if running else None
        if job != last_job:
            if job:
                warm_until = max(warm_until, now + a.warmup_min * 60)
                log("job %s running: keep %.1f GB free for %.0f min" % (job, a.start_free_gb, a.warmup_min))
            else:
                log("queue idle")
            last_job = job
        if now >= next_pid_check:
            try:
                pids = our_pids(bus, code_dir, me)
            except (subprocess.SubprocessError, OSError):
                pids = seen
            if job and pids - seen:
                warm_until = max(warm_until, now + a.warmup_min * 60)
                log("new process of ours on the GPU (pid %s): keep %.1f GB free for %.0f min"
                    % (",".join(map(str, sorted(pids - seen))), a.start_free_gb, a.warmup_min))
            seen, next_pid_check = pids, now + a.pid_poll
        try:
            total, free, _ = gpu_mem(a.gpu)
        except (subprocess.SubprocessError, OSError, ValueError):
            time.sleep(a.poll)
            continue
        target = target_free_mib(job is not None, now, warm_until, a.idle_free_gb, a.job_free_gb, a.start_free_gb)
        n = plan(free, target, len(held), a.chunk_mb, max_step)
        if n < 0:
            del held[n:]
            torch.cuda.empty_cache()
        elif n > 0:
            try:
                for _ in range(n):
                    held.append(torch.empty(a.chunk_mb * MIB // 4, dtype=torch.float32, device="cuda:0"))
            except RuntimeError:                  # someone took the memory meanwhile: keep what we got
                torch.cuda.empty_cache()
        report = (len(held) * a.chunk_mb // 1024, target // 1024)
        if report != last_report:
            log("holding %.1f GB, free %.1f GB, target free %.1f GB" % (len(held) * a.chunk_mb / 1024, free / 1024,
                                                                       target / 1024))
            last_report = report
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
