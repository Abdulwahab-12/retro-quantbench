#!/usr/bin/env bash
# ===========================================================================
#  How far along is the run, and how fast is it going?
#
#      bash qb-watch.sh                  once
#      bash qb-watch.sh 5005             once, with an ETA against a target
#      watch -n 60 bash qb-watch.sh 5005 refreshing
#
#  Safe to run at any time, including while the sweep is running. It only
#  reads files -- it never touches the GPU, never loads the model, and cannot
#  disturb the job.
#
#  WHY IT READS SHARDS AND NOT THE MERGED OUTPUT
#  The merged raw/ files are written when a LEVEL finishes, and meta.json when
#  a WORKER finishes, so during a run neither exists. Every molecule, however,
#  is appended to its shard file and fsync'd as it completes -- so the shards
#  carry live progress, live throughput and the live error count.
#
#  Uses the container's python if the host has none, so a machine with only a
#  GPU driver and singularity can still watch its own job.
# ===========================================================================
set -u

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${QB_WATCH_DIR:-$D/results}"
TARGET="${1:-0}"

[ -d "$WORK" ] || { echo "no results directory at $WORK" >&2; exit 1; }

# throughput.sh is pure bash + awk, so it always runs on the host.
TP="$D/throughput.sh"
PG="$D/progress.py"

# progress.py needs a python. Prefer the host's; fall back to the image's,
# which is guaranteed to have one.
PY="$(command -v python3 || command -v python || true)"
SING="$(command -v singularity || command -v apptainer || true)"
SIF=""
for c in "$D"/*.sif; do [ -f "$c" ] && SIF="$c" && break; done

echo "=============================== progress ==============================="
if [ -f "$PG" ] && [ -n "$PY" ]; then
  "$PY" "$PG" "$WORK" "$TARGET"
elif [ -f "$PG" ] && [ -n "$SING" ] && [ -n "$SIF" ]; then
  "$SING" exec --bind "$WORK:/work" --bind "$PG:/tmp/progress.py" "$SIF" \
      python /tmp/progress.py /work "$TARGET"
else
  echo "  progress.py or a python interpreter not found; counting lines instead"
  for f in "$WORK"/shards/*.jsonl; do
    [ -f "$f" ] || continue
    printf '  %-34s %s\n' "$(basename "$f")" "$(wc -l < "$f")"
  done
fi

echo
echo "============================== throughput =============================="
if [ -f "$TP" ]; then
  bash "$TP" "$WORK" 2>&1
else
  echo "  throughput.sh not found next to this script"
fi

echo
echo "GPU memory now (RESERVED -- includes the allocator cache, so larger than"
echo "the resident figure the paper reports):"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /' \
  || echo "  nvidia-smi unavailable"
