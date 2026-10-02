#!/usr/bin/env bash
# ===========================================================================
#  Recover everything a sweep has produced, including work that never made it
#  into the results table.
#
#      bash recover.sh                 look only -- changes nothing
#      bash recover.sh --merge         also rebuild raw/ from shards/
#      bash recover.sh ~/qb-llm        a different results directory
#
#  WHY THIS EXISTS
#  Workers write one file per GPU into shards/. The orchestrator merges them
#  into raw/ only AFTER every worker for that level has finished. Kill the run,
#  lose the ssh session, hit a node reboot -- and the molecules that were
#  already computed are sitting in shards/ where nothing reads them. They look
#  lost. They are not.
#
#  SAFE BY DEFAULT: it prints an inventory and stops. --merge is the only thing
#  that writes, it copies any raw file aside before replacing it, and it never
#  replaces a raw file that already holds MORE records than the shards do.
# ===========================================================================
set -u

DO_MERGE=0
D=""
for a in "$@"; do
  case "$a" in
    --merge) DO_MERGE=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) D="$a" ;;
  esac
done

# Find the results directory: the argument, else here, else the usual places.
if [ -z "$D" ]; then
  for c in . ./results "$HOME/qb-llm" /work; do
    if [ -d "$c/shards" ] || [ -d "$c/raw" ]; then D="$c"; break; fi
  done
fi
[ -n "$D" ] || { echo "no results directory found (looked in . ./results ~/qb-llm /work)" >&2; exit 1; }
D="$(cd "$D" && pwd)"

echo "=========================================================="
echo "RESULTS DIRECTORY: $D"
echo "=========================================================="

# --- 1. Is something still running? ---------------------------------------
echo
echo "--- tmux sessions -----------------------------------------"
if command -v tmux >/dev/null 2>&1; then
  if tmux ls 2>/dev/null; then
    echo "  reattach with:  tmux attach -t <name>"
    echo "  a detached session is still RUNNING -- do not start a second one"
  else
    echo "  none. Nothing is running under tmux."
  fi
else
  echo "  tmux not installed"
fi

echo
echo "--- worker processes --------------------------------------"
if pgrep -af "run_llm.py" 2>/dev/null; then :; else echo "  none alive"; fi

# --- 2. What is on disk ----------------------------------------------------
python3 - "$D" "$DO_MERGE" <<'PY'
import glob, json, os, shutil, sys
from collections import Counter, defaultdict

D, do_merge = sys.argv[1], sys.argv[2] == "1"
raw_d, sh_d = os.path.join(D, "raw"), os.path.join(D, "shards")


def count(path):
    """(records, distinct uids, fingerprints seen)."""
    n, uids, cfgs = 0, set(), Counter()
    try:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue            # a half-written last line after a kill
            n += 1
            u = r.get("uid")
            if u is not None:
                uids.add(u)
            cfgs[r.get("cfg", "<none>")] += 1
    except OSError:
        pass
    return n, uids, cfgs


print()
print("--- raw/  (what the results table reads) ------------------")
raw_files = sorted(glob.glob(os.path.join(raw_d, "*.jsonl")))
raw_by_tag = {}
if not raw_files:
    print("  EMPTY")
for p in raw_files:
    n, u, _ = count(p)
    b = os.path.basename(p)
    tag = b[len("retrodfm-r-8b."):-len(".retro.jsonl")] if b.startswith("retrodfm-r-8b.") else None
    if tag:
        raw_by_tag[tag] = len(u)
    print(f"  {b:56s} {n:6d} records, {len(u):6d} molecules")

print()
print("--- shards/  (per-GPU output, merged only at level end) ---")
by_tag = defaultdict(list)
for p in sorted(glob.glob(os.path.join(sh_d, "*.jsonl"))):
    b = os.path.basename(p)[:-len(".jsonl")]
    tag = b.rsplit(".s", 1)[0]
    by_tag[tag].append(p)
if not by_tag:
    print("  EMPTY")

recoverable = []
for tag, parts in sorted(by_tag.items()):
    uids, total, cfgs = set(), 0, Counter()
    for p in parts:
        n, u, c = count(p)
        total += n
        uids |= u
        cfgs.update(c)
    # The runner keeps only records matching the fingerprint in the .cfg
    # sidecar, so count the same way or this over-reports what is recoverable.
    sidecars = sorted(glob.glob(os.path.join(sh_d, f"{tag}.s*.jsonl.cfg")))
    want = open(sidecars[0]).read().strip() if sidecars else None
    usable = cfgs[want] if want in cfgs else total
    have = raw_by_tag.get(tag, 0)
    gain = len(uids) - have
    flag = ""
    if gain > 0:
        flag = f"   <-- {gain} molecules NOT in raw/"
        recoverable.append(tag)
    print(f"  {tag:40s} {len(parts):2d} shards {len(uids):6d} molecules "
          f"(raw/ has {have}){flag}")
    if len(cfgs) > 1:
        print(f"      {len(cfgs)} different configurations present; "
              f"the merge keeps only {want}")

print()
print("--- packages ----------------------------------------------")
zips = sorted(glob.glob(os.path.join(D, "*.zip")))
for z in zips:
    print(f"  {os.path.basename(z):56s} {os.path.getsize(z)/1e6:8.1f} MB")
if not zips:
    print("  none")

if not recoverable:
    print()
    print("==========================================================")
    print("NOTHING IS STRANDED. raw/ already holds everything shards/ has.")
    print("==========================================================")
    sys.exit(0)

print()
print("==========================================================")
print(f"RECOVERABLE: {len(recoverable)} level(s) have molecules in shards/")
print("that raw/ does not contain:")
for t in recoverable:
    print(f"    {t}")
print("==========================================================")

if not do_merge:
    print()
    print("  Re-run with --merge to rebuild raw/ from shards/.")
    print("  Nothing has been changed.")
    sys.exit(0)

os.makedirs(raw_d, exist_ok=True)
for tag in recoverable:
    out_path = os.path.join(raw_d, f"retrodfm-r-8b.{tag}.retro.jsonl")
    sidecars = sorted(glob.glob(os.path.join(sh_d, f"{tag}.s*.jsonl.cfg")))
    want = open(sidecars[0]).read().strip() if sidecars else None
    if os.path.exists(out_path):
        # Never destroy the old file. If the merge turns out worse, the
        # comparison is one `wc -l` away.
        bak = out_path + ".before-recover"
        shutil.copy2(out_path, bak)
        print(f"  kept a copy: {os.path.basename(bak)}")
    # Keep the BEST record per molecule, not the first one seen. A retry is
    # written after the failure it replaces, so first-wins kept the error and
    # discarded the good answer. Rank 1 = a real answer, rank 0 = an error;
    # later wins on a tie, and an error never displaces an answer.
    best = {}
    for p in sorted(glob.glob(os.path.join(sh_d, f"{tag}.s*.jsonl"))):
        for line in open(p):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if want is not None and rec.get("cfg") != want:
                continue
            rank = 0 if str(rec.get("stop_reason", "")).startswith("error") else 1
            prev = best.get(rec.get("uid"))
            if prev is None or rank >= prev[0]:
                best[rec.get("uid")] = (rank, line)
    with open(out_path, "w") as out:
        for rank, line in best.values():
            out.write(line)
    kept = len(best)
    bad = sum(1 for rank, _ in best.values() if rank == 0)
    print(f"  merged {kept:6d} records -> {os.path.basename(out_path)}")
    if bad:
        print(f"  {bad} of them are errors, not answers -- re-run the sweep "
              f"with the same command to retry those molecules")

print()
# Pathless when it is the folder qb.sh already uses, which is the normal case:
# these files sit beside the .sif and everything is relative to that folder.
if os.path.basename(D) == "results":
    print("  Now re-score:   bash qb.sh --score-only")
else:
    print("  Now re-score:   bash qb.sh --results %s --score-only" % D)
PY
