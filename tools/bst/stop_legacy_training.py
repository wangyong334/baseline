"""Inspect or SIGTERM one precisely identified legacy BST trainer on Linux.

Default is read-only. This tool never signals the GPU pool, queue worker, an
evaluation/policy-fitting process, a different user, or a different checkout.
"""
import argparse
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = "log/bst_shared_v5s37_s37_20261111"


class StopRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Process:
    pid: int
    real_uid: int
    effective_uid: int
    start_ticks: int
    state: str
    cwd: object
    argv: tuple


def stat_identity(raw):
    # comm may contain spaces and parentheses; fields after its last ')' begin
    # with state (field 3). starttime is field 22, hence offset 19 here.
    fields = raw[raw.rfind(")") + 1:].split()
    if len(fields) < 20 or len(fields[0]) != 1:
        raise StopRefused("Cannot parse process start time")
    return fields[0], int(fields[19])


def read_process(pid, proc_root=Path("/proc")):
    folder = Path(proc_root)/str(pid)
    try:
        status = (folder/"status").read_text()
        uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
        uids = [int(value) for value in uid_line.split()[1:]]
        state, start = stat_identity((folder/"stat").read_text())
        if state == "Z":
            return Process(pid, uids[0], uids[1], start, state, None, ())
        cwd = Path(os.readlink(str(folder/"cwd"))).resolve()
        raw = (folder/"cmdline").read_bytes()
        argv = tuple(os.fsdecode(part) for part in raw.rstrip(b"\0").split(b"\0")) if raw else ()
        return Process(pid, uids[0], uids[1], start, state, cwd, argv)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, StopIteration, ValueError, IndexError) as error:
        raise StopRefused("Cannot inspect PID %d completely: %s" % (pid, error))


def owned_pids(uid, proc_root=Path("/proc")):
    for path in Path(proc_root).iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid == uid:
                yield int(path.name)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise StopRefused("Cannot inspect process ownership: %s" % error)


def resolve_output(root, value):
    path = Path(value)
    resolved = (path if path.is_absolute() else root/path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise StopRefused("Requested output must stay inside this checkout")
    if not relative.parts:
        raise StopRefused("Checkout root is not a specific training output")
    return resolved


def python_script_index(argv):
    if not argv or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", Path(argv[0]).name, re.I):
        return None
    position = 1
    simple = {"-u", "-B", "-O", "-OO", "-E", "-s", "-S", "-I", "-q", "-d", "-v", "-b", "-bb", "-i", "-P"}
    while position < len(argv):
        word = argv[position]
        if word == "--":
            return position + 1 if position + 1 < len(argv) else None
        if word in ("-c", "-m"):
            return None
        if word in simple:
            position += 1
        elif word in ("-W", "-X"):
            position += 2
        elif word.startswith("-W") or word.startswith("-X"):
            position += 1
        elif word.startswith("-"):
            return None
        else:
            return position
    return None


def matches(process, root, output, uid):
    if process is None or process.state == "Z" or process.real_uid != uid or process.effective_uid != uid:
        return False
    if process.cwd != root:
        return False
    position = python_script_index(process.argv)
    if position is None:
        return False
    script = Path(process.argv[position])
    if (script if script.is_absolute() else root/script).resolve() != root/"tools/bst/train.py":
        return False
    arguments, outputs = process.argv[position + 1:], []
    for index, argument in enumerate(arguments):
        if argument == "--out":
            if index + 1 >= len(arguments) or not arguments[index + 1] or arguments[index + 1].startswith("--"):
                raise StopRefused("Ambiguous --out for trainer PID %d" % process.pid)
            outputs.append(arguments[index + 1])
        elif argument.startswith("--out="):
            outputs.append(argument[len("--out="):])
    if len(outputs) != 1 or not outputs[0]:
        raise StopRefused("Expected exactly one --out for trainer PID %d" % process.pid)
    actual = Path(outputs[0])
    actual = (actual if actual.is_absolute() else root/actual).resolve()
    return actual == output


def find_matches(root, output, uid, reader=read_process):
    found = []
    for pid in owned_pids(uid):
        process = reader(pid)
        if matches(process, root, output, uid):
            found.append(process)
    if len(found) > 1:
        raise StopRefused("Multiple matching trainers; refusing to guess: %s" % [p.pid for p in found])
    return found


def same_identity(left, right):
    return (left.pid, left.real_uid, left.effective_uid, left.start_ticks, left.cwd, left.argv) == (
        right.pid, right.real_uid, right.effective_uid, right.start_ticks, right.cwd, right.argv)


def _pidfd_available():
    return callable(getattr(os, "pidfd_open", None)) and callable(getattr(signal, "pidfd_send_signal", None))


def send_term_checked(expected, root, output, uid, reader=read_process):
    handle = None
    try:
        if _pidfd_available():
            try:
                handle = os.pidfd_open(expected.pid, 0)
            except OSError as error:
                if error.errno not in (errno.ENOSYS, errno.EINVAL):
                    raise
        current = reader(expected.pid)
        if current is None or current.state == "Z":
            return False
        if not same_identity(expected, current) or not matches(current, root, output, uid):
            raise StopRefused("PID identity/start time changed before SIGTERM; no signal sent")
        if handle is not None:
            signal.pidfd_send_signal(handle, signal.SIGTERM, None, 0)
        else:
            # Python 3.8 compatibility: the full /proc identity was rechecked
            # immediately above. Never reuse a PID captured in a screenshot.
            os.kill(expected.pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False
    except OSError as error:
        raise StopRefused("Could not send SIGTERM: %s" % error)
    finally:
        if handle is not None:
            os.close(handle)


def stop_one(expected, root, output, uid, timeout=20, reader=read_process,
             clock=time.monotonic, sleeper=time.sleep):
    if not 0 < timeout <= 20:
        raise StopRefused("Exit confirmation timeout must be in (0, 20] seconds")
    if not send_term_checked(expected, root, output, uid, reader):
        return "ALREADY_EXITED: no signal was necessary"
    deadline = clock() + timeout
    while True:
        try:
            current = reader(expected.pid)
        except StopRefused as error:
            raise StopRefused("SIGTERM sent, but exit inspection failed: %s" % error)
        if current is None or current.state == "Z":
            return "EXIT_CONFIRMED: trainer exited after SIGTERM"
        if current.start_ticks != expected.start_ticks:
            return "EXIT_CONFIRMED: original trainer exited; PID now belongs to another process"
        if clock() >= deadline:
            raise StopRefused("SIGTERM sent, but exit was not confirmed within %.1f seconds; no SIGKILL sent" % timeout)
        sleeper(min(.2, max(0., deadline - clock())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="send SIGTERM after a second identity check")
    mode.add_argument("--dry-run", action="store_true", help="explicit read-only mode (the default)")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("Run this inspector on the Linux experiment server")
    try:
        if not 0 < args.timeout <= 20:
            raise StopRefused("--timeout must be in (0, 20] seconds")
        root, uid = ROOT.resolve(), os.geteuid()
        output = resolve_output(root, args.out)
        found = find_matches(root, output, uid)
        if not found:
            print("NO_MATCH: no matching live trainer; no signal sent. This does not prove the whole queue job ended.")
            return 0
        process = found[0]
        print(json.dumps({"pid": process.pid, "uid": uid, "start_ticks": process.start_ticks,
                          "cwd": str(process.cwd), "argv": list(process.argv), "out": str(output)}, indent=2))
        if not args.execute:
            print("DRY_RUN: no signal sent. Add --execute only to stop this matching trainer.")
            return 0
        print(stop_one(process, root, output, uid, args.timeout))
        return 0
    except StopRefused as error:
        print("REFUSED: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
