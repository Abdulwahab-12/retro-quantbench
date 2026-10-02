#!/usr/bin/env bash
# ===========================================================================
#  Tokens per second per level, from prediction files already on disk.
#
#      bash throughput.sh                 # current directory
#      bash throughput.sh quant3
#      bash throughput.sh quant3 bf16     # add ratios against a reference
#
#  Pure shell and awk. No Python, no container, no GPU, no reruns -- so it
#  works on a machine where the python in PATH is missing or wrong, and it
#  works identically inside the image:
#
#      singularity exec --bind ~/qb-llm:/work IMAGE.sif bash throughput.sh /work
#
#  Looks in <dir>/raw/*.retro.jsonl, then <dir>/shards/*.jsonl, then <dir>/*.jsonl.
#  Shards matter because a run that crashed or is still going never produces
#  the merged raw/ files, while per-molecule records are written and fsync'd
#  as they happen.
#
#  WHY TOKENS PER SECOND
#  Seconds per molecule mixes how fast the model computes with how much it
#  chose to write. A reasoning model's answer length varies a lot, so a level
#  producing longer reasoning looks slower even when every token costs the
#  same. Report both: seconds per molecule is what a user waits for, tokens
#  per second is what the arithmetic costs.
# ===========================================================================
set -u

DIR="${1:-.}"
REF="${2:-}"

[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }

# Pick the first location that has files. MODE tells awk how to derive the
# level name, because the two layouts name things differently. The glob must
# be unquoted to expand, and the kind cannot be folded into the pattern -- an
# earlier version wrote "raw:$DIR/raw"/*.jsonl, which the shell tried to match
# against a directory literally called "raw:" and so matched nothing.
MODE=""; FILES=""
pick() {
  local kind="$1"; shift
  local found="" f
  for f in "$@"; do
    [ -e "$f" ] || continue
    case "$f" in *.meta.json|*.stale-*) continue ;; esac
    found="$found $f"
  done
  [ -n "${found// /}" ] || return 1
  MODE="$kind"; FILES="$found"; return 0
}

pick raw    "$DIR"/raw/*.retro.jsonl \
  || pick shards "$DIR"/shards/*.jsonl \
  || pick here   "$DIR"/*.jsonl \
  || true

if [ -z "${FILES// /}" ]; then
  echo "no prediction files under $(cd "$DIR" && pwd)" >&2
  echo "  (looked in raw/, shards/, and the directory itself)" >&2
  exit 1
fi

# shellcheck disable=SC2086
awk -v mode="$MODE" -v ref="$REF" -v dir="$DIR" '
BEGIN {
  KNOWN = " fp32 fp16 bf16 int8_ao int4_ao w8a8 w4a8 w4a4 w4a8_intx" \
          " fp8_ao fp8_w8a8 fp8_w4a8 nvfp4 nvfp4_w mxfp8 mxfp4" \
          " nf4 nf4dq fp4 bnb_int8 bnb_nf4 f16" \
          " q2_k q3_k_m q4_0 q4_k_m q5_k_m q6_k q8_0 "
}
# Filenames are <model>.<level>[.<suffix>].retro.jsonl, and the level was found
# by deleting everything up to the FIRST dot. That breaks whenever the MODEL
# name contains a dot: chemdfm-v1.5-8b.q2_k was reported as level "5-8b.q2_k".
# A mislabelled row is worse than a missing one, so find the first component
# that is actually a known level and take the name from there.
function level(f,   a, k, b, parts, i, j, out) {
  k = split(f, a, "/"); b = a[k]
  sub(/\.retro\.jsonl$/, "", b); sub(/\.jsonl$/, "", b)
  if (mode == "shards") { sub(/\.s[0-9]+$/, "", b); return b }
  k = split(b, parts, ".")
  for (i = 1; i <= k; i++) {
    if (index(KNOWN, " " parts[i] " ") > 0) {
      out = parts[i]
      for (j = i + 1; j <= k; j++) out = out "." parts[j]   # keep .k20x1x2 etc
      return out
    }
  }
  if (b ~ /\./) { sub(/^[^.]*\./, "", b) }   # unknown level: old behaviour
  return b
}
function num(line, key,   v) {
  if (match(line, "\"" key "\": *[0-9.]+")) {
    v = substr(line, RSTART, RLENGTH); sub(/.*: */, "", v); return v + 0
  }
  return 0
}
{
  L = level(FILENAME)
  seen[L] = 1
  if ($0 ~ /"stop_reason": *"error/) { err[L]++; next }
  t = num($0, "n_gen_tokens")
  if (t == 0) { notok[L]++; next }
  n[L]++; TOK[L] += t; SEC[L] += num($0, "seconds")
}
END {
  if (ref != "" && !(ref in n)) {
    printf "reference level %s has no usable records; ignoring\n\n", ref
    ref = ""
  }
  printf "source: %s/  under %s\n\n", mode, dir
  printf "%-22s%6s%9s%10s%9s", "level", "n", "s/mol", "tok/mol", "tok/s"
  if (ref != "") printf "%9s%9s", "s/mol x", "tok/s x"
  printf "   notes\n"

  if (ref != "") { bs = SEC[ref]/n[ref]; bt = TOK[ref]/SEC[ref] }

  # sort the level names without relying on gawk-only asort
  m = 0; for (L in seen) { keys[++m] = L }
  for (i = 1; i < m; i++) for (j = i+1; j <= m; j++)
    if (keys[j] < keys[i]) { tmp = keys[i]; keys[i] = keys[j]; keys[j] = tmp }

  for (i = 1; i <= m; i++) {
    L = keys[i]
    if (!(L in n)) continue                     # level with no usable records
    spm = SEC[L]/n[L]; tpm = TOK[L]/n[L]; tps = (SEC[L] > 0 ? TOK[L]/SEC[L] : 0)
    printf "%-22s%6d%9.1f%10.0f%9.1f", L, n[L], spm, tpm, tps
    if (ref != "") printf "%9.2f%9.2f", spm/bs, tps/bt
    note = ""
    if (L in err)   note = note sprintf("%d errored", err[L])
    if (L in notok) note = note (note != "" ? ", " : "") sprintf("%d without token counts", notok[L])
    printf "   %s\n", note
  }
  if (ref != "") {
    printf "\nratios against %s. Slower per molecule but equal per token means\n", ref
    printf "the level wrote more, not that it computed more slowly.\n"
  }
  printf "errored molecules are excluded: a failure has no meaningful duration.\n"
}
' $FILES
