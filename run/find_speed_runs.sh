#!/usr/bin/env bash
# ===========================================================================
#  Find every prediction file under a directory tree and report tokens/second.
#
#      bash find_speed_runs.sh ~            search a whole home directory
#      bash find_speed_runs.sh ~/qb-llm     search one results tree
#      bash find_speed_runs.sh ~ 60         only runs with <= 60 records
#
#  throughput.sh looks in three fixed places (raw/, shards/, the directory
#  itself). This one RECURSES, because a dedicated speed run is usually in
#  some directory nobody remembers the name of, and finding it by content is
#  more reliable than guessing the path.
#
#  Reports per FILE rather than merging by level, so two runs of the same
#  level in different directories stay distinguishable -- which is the whole
#  point when you are trying to work out which run produced a published
#  number.
#
#  Pure bash + awk. No python, no GPU, no container.
# ===========================================================================
set -u

ROOT="${1:-.}"
MAXN="${2:-0}"          # 0 = no limit; otherwise only files with <= MAXN records

[ -d "$ROOT" ] || { echo "no such directory: $ROOT" >&2; exit 1; }

printf '%-52s %6s %9s %9s %9s   %s\n' \
       "file" "n" "s/mol" "tok/mol" "tok/s" "notes"

found=0
while IFS= read -r f; do
  case "$f" in *.meta.json|*.cfg|*subset*|*.stale-*) continue ;; esac
  # Must look like predictions: a first line carrying a token count.
  head -1 "$f" 2>/dev/null | grep -q '"n_gen_tokens"' || continue
  n=$(wc -l < "$f")
  [ "$MAXN" -gt 0 ] && [ "$n" -gt "$MAXN" ] && continue
  found=$((found + 1))
  awk -v F="$f" '
    function num(line, key,   v) {
      if (match(line, "\"" key "\": *[0-9.]+")) {
        v = substr(line, RSTART, RLENGTH); sub(/.*: */, "", v); return v + 0
      }
      return 0
    }
    /"stop_reason": *"error/ { err++; next }
    {
      t = num($0, "n_gen_tokens")
      if (t == 0) { notok++; next }
      n++; TOK += t; SEC += num($0, "seconds")
    }
    END {
      if (n == 0) { printf "%-52s %6s %9s %9s %9s   no usable records\n",
                           substr(F, length(F)-51), "-", "-", "-", "-"; exit }
      note = ""
      if (err)   note = note sprintf("%d errored", err)
      if (notok) note = note (note ? ", " : "") sprintf("%d untimed", notok)
      # show the tail of the path: the leading directories are usually shared
      p = F; if (length(p) > 52) p = "..." substr(p, length(p) - 48)
      printf "%-52s %6d %9.1f %9.0f %9.1f   %s\n",
             p, n, SEC/n, TOK/n, (SEC > 0 ? TOK/SEC : 0), note
    }' "$f"
done < <(find "$ROOT" -type f -name '*.jsonl' -size -500M 2>/dev/null | sort)

if [ "$found" -eq 0 ]; then
  echo
  echo "no files with per-record token counts under $ROOT"
  echo "  (looked for *.jsonl whose first line contains \"n_gen_tokens\")"
  exit 1
fi

echo
echo "tok/s = total generated tokens / total generation seconds, errored"
echo "molecules excluded. Compare rows only within one machine and one run"
echo "condition: a shared GPU inflates s/mol without changing tok/mol."
