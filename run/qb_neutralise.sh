#!/usr/bin/env bash
# qb_neutralise.sh -- how much does charge standardisation change top-k?
#
#   bash qb_neutralise.sh
#
# That is the whole command. No arguments. It finds the image beside itself,
# finds the results folder beside itself, re-scores every raw file twice -- once
# exactly as score_sweep.py does, once with rdMolStandardize.Uncharger applied
# to every candidate AND to the ground truth -- and prints both columns, the
# difference, and every between-level gap before and after.
#
# Nothing is re-run on the GPU. No model is loaded. This reads the .jsonl files
# that are already on disk, so it works on a login node while a sweep has the
# card.
#
# Optional, only to save time -- uncharging every fragment of a 20x10x10 file at
# n=5005 takes a few minutes, an unaugmented file takes seconds:
#   --levels bf16,nf4     only these levels
#   --tag k20x10x10       only this k-setting
#   --tag none            only the unaugmented (1x1x1) run
#   --results DIR         a results folder somewhere else
#
# The report is also written to <results>/neutralise.txt so it can be handed on
# without copying it out of a terminal.
set -u

QB_NEU_REV="2026-09-27.1"

if [ "$#" -ge 1 ] && [ -d "$1" ] && [ "${1#--}" = "$1" ]; then
  D="$(cd "$1" && pwd)"; shift
else
  D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

WORK=""; LV=""; TAG=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --results) WORK="$2"; shift 2 ;;
    --levels)  LV="$2";   shift 2 ;;
    --tag)     TAG="$2";  shift 2 ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2
       echo "  bash qb_neutralise.sh [--levels a,b] [--tag k20x10x10|none] [--results DIR]" >&2
       exit 2 ;;
  esac
done

# The analysis itself lives beside this script, the same way score_sweep.py
# lives beside qb.sh. Say which file is missing rather than failing inside the
# container on a path nobody typed.
PY="$D/neutralise_sweep.py"
if [ ! -f "$PY" ]; then
  echo "cannot find neutralise_sweep.py beside this script" >&2
  echo "  looked for : $PY" >&2
  echo "  folder has : $(ls -1 "$D" | tr '\n' ' ')" >&2
  exit 1
fi

[ -n "$WORK" ] || WORK="${QB_RESULTS:-$D/results}"
if [ ! -d "$WORK" ]; then
  echo "results folder does not exist: $WORK" >&2
  echo "  give one with:  bash qb_neutralise.sh --results /path/to/results" >&2
  exit 1
fi
WORK="$(cd "$WORK" && pwd)"

if [ ! -d "$WORK/raw" ]; then
  echo "no raw/ inside $WORK -- there is nothing to score" >&2
  echo "  $WORK contains: $(ls -1 "$WORK" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi
NF="$(ls -1 "$WORK"/raw/*.jsonl 2>/dev/null | wc -l | tr -d ' ')"

# Any .sif beside the script, exactly as qb.sh picks it, so a rebuild under a
# new name needs no edit here. No image is not an error: rdkit is all this
# needs, and a plain python3 that has it will do.
SIF=""; NSIF=0
for cand in "$D"/*.sif; do
  [ -f "$cand" ] || continue
  NSIF=$((NSIF + 1)); SIF="$cand"
done
SING="$(command -v singularity || command -v apptainer || true)"

echo "qb_neutralise.sh  rev $QB_NEU_REV"
echo "folder  : $D"
echo "results : $WORK   ($NF raw file(s))"
[ -n "$LV" ]  && echo "levels  : $LV"
[ -n "$TAG" ] && echo "tag     : $TAG"

OUT="$WORK/neutralise.txt"

if [ "$NSIF" -ge 1 ] && [ -n "$SING" ]; then
  [ "$NSIF" -gt 1 ] && { echo "more than one .sif in $D -- keep only the one you want:" >&2
                         ls -1 "$D"/*.sif >&2; exit 1; }
  echo "image   : $SIF"
  echo
  # /work is already bound, so the analysis is copied there and run from there.
  # No new bind target has to exist inside the image, which is the one thing
  # that can fail on a cluster without overlay support.
  cp -f "$PY" "$WORK/neutralise_sweep.py"
  export SINGULARITYENV_QB_NEU_LEVELS="$LV" APPTAINERENV_QB_NEU_LEVELS="$LV"
  export SINGULARITYENV_QB_NEU_TAG="$TAG"   APPTAINERENV_QB_NEU_TAG="$TAG"
  "$SING" exec --bind "$WORK:/work" "$SIF" \
    python -u /work/neutralise_sweep.py /work/raw 2>&1 | tee "$OUT"
  rc=${PIPESTATUS[0]}
else
  if [ "$NSIF" -eq 0 ]; then
    echo "image   : none found in $D -- using the python on this machine"
  else
    echo "image   : $SIF found, but no singularity/apptainer -- using local python"
  fi
  PYBIN="$(command -v python3 || command -v python || true)"
  [ -n "$PYBIN" ] || { echo "no python found either" >&2; exit 1; }
  if ! "$PYBIN" -c 'import rdkit' 2>/dev/null; then
    echo "this python has no rdkit, and there is no image to fall back on:" >&2
    echo "  $PYBIN" >&2
    echo "  run this beside the .sif, or point it at one:" >&2
    echo "      bash qb_neutralise.sh /path/to/folder/with/the/sif" >&2
    exit 1
  fi
  echo
  QB_NEU_LEVELS="$LV" QB_NEU_TAG="$TAG" \
    "$PYBIN" -u "$PY" "$WORK/raw" 2>&1 | tee "$OUT"
  rc=${PIPESTATUS[0]}
fi

echo
echo "written : $OUT"
exit "$rc"
