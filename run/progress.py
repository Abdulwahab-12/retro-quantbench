#!/usr/bin/env python3
"""Live progress of a running sweep. Reads only files, touches no GPU.

    python progress.py                 # current directory as the work dir
    python progress.py /path/to/work
    python progress.py . 1000          # with a target, to get an ETA
    watch -n 30 python progress.py . 1000

Why this exists: meta.json is written when a worker FINISHES, so during a run
there is nothing to read there. But every molecule is appended to its shard
file and fsync'd immediately, so the shards carry live progress, live
throughput, and the error count. nvidia-smi is read for memory, with the
caveat that it reports RESERVED memory (PyTorch's allocator cache included),
which is always larger than the resident figure the paper reports.
"""
import glob, json, os, subprocess, sys

work = sys.argv[1] if len(sys.argv) > 1 else "."
target = int(sys.argv[2]) if len(sys.argv) > 2 else 0

shards = sorted(glob.glob(os.path.join(work, "shards", "*.s*.jsonl")))
if not shards:
    sys.exit(f"no shards under {os.path.abspath(work)}/shards -- wrong directory, "
             "or the run has not written its first molecule yet")

by_level = {}
for p in shards:
    lvl = os.path.basename(p).rsplit(".s", 1)[0]
    by_level.setdefault(lvl, []).append(p)

print(f"{'level':<20}{'done':>8}{'workers':>9}{'s/mol':>9}{'elapsed':>10}"
      f"{'eta':>9}   notes")
for lvl, ps in sorted(by_level.items()):
    n = t = err = empty = 0
    for p in ps:
        try:
            fh = open(p, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                n += 1
                try:
                    r = json.loads(line)
                except Exception:
                    continue                      # half-written final line
                t += r.get("seconds", 0) or 0
                if str(r.get("stop_reason", "")).startswith("error"):
                    err += 1
                if not (r.get("candidates") or []):
                    empty += 1
    mean = t / n if n else 0.0
    # wall-clock elapsed is per worker, so divide the summed generation time
    elapsed = t / max(len(ps), 1) / 3600
    eta = ((target - n) * mean / max(len(ps), 1) / 3600) if (target and mean) else 0
    notes = []
    if err:
        notes.append(f"{err} errors")
    if empty:
        notes.append(f"{empty} empty")
    print(f"{lvl:<20}{n:>8}{len(ps):>9}{mean:>9.1f}{elapsed:>9.1f}h"
          f"{(f'{eta:.1f}h' if eta else '-'):>9}   {', '.join(notes)}")

# GPU memory, best effort. Reserved, not resident -- see the docstring.
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    if out.returncode == 0 and out.stdout.strip():
        print("\nGPU memory (reserved, includes allocator cache):")
        for line in out.stdout.strip().splitlines():
            print("   " + line)
except Exception:
    pass

# The tail of each log is where a dying worker leaves its traceback.
logs = sorted(glob.glob(os.path.join(work, "logs", "*.log")))
if logs:
    print("\nlast line of each worker log:")
    for p in logs[:8]:
        try:
            tail = [l.rstrip() for l in open(p, errors="replace") if l.strip()]
            print(f"   {os.path.basename(p):<24} {tail[-1][:90] if tail else '(empty)'}")
        except OSError:
            pass
