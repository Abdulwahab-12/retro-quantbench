#!/usr/bin/env bash
# Print the arithmetic from whatever 1x1x1 and 20x10x10 runs are on disk.
# No GPU, no container, seconds. Safe to run any time, including mid-sweep.
set -u
if [ "$#" -ge 1 ] && [ -d "$1" ]; then RUN="$(cd "$1" && pwd)"
else RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "no python on PATH"; exit 1; }
exec "$PY" - "$RUN/results/raw" <<'PYEOF'
import glob, json, os, re, sys
root = sys.argv[1]
rows = {}
for fp in sorted(glob.glob(os.path.join(root, "*.retro.jsonl"))):
    tok = sec = 0.0; n = 0
    for line in open(fp):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get("error") or str(r.get("stop_reason", "")).startswith("error"):
            continue
        t, s = r.get("n_gen_tokens"), r.get("seconds")
        if not isinstance(t, int) or not isinstance(s, (int, float)) or s <= 0:
            continue
        tok += t; sec += s; n += 1
        lvl = r.get("quant") or "?"
        bud = "%sx%sx%s" % (r.get("k_a"), r.get("k_s"), r.get("k_b"))
    if n:
        rows[(lvl, bud)] = (n, tok / n, sec / n, tok / sec)

if not rows:
    sys.exit("no usable records under " + root)
print("=" * 74)
print("MEASURED, one card, per molecule")
print("=" * 74)
print(f"{'level':8s} {'budget':10s} {'n':>4s} {'tok/mol':>11s} {'s/mol':>10s} {'tok/s':>8s}")
for (lvl, bud), (n, t, s, r) in sorted(rows.items()):
    print(f"{lvl:8s} {bud:10s} {n:4d} {t:11,.0f} {s:10.1f} {r:8.1f}")

print()
print("=" * 74)
print("DOES IT CLOSE?   time ratio  ==  token ratio / throughput ratio")
print("=" * 74)
levels = sorted({l for l, _ in rows})
any_pair = False
for l in levels:
    a = rows.get((l, "1x1x1")); b = rows.get((l, "20x10x10"))
    if not (a and b): continue
    any_pair = True
    tr, rr, sr = b[1] / a[1], b[3] / a[3], b[2] / a[2]
    print(f"\n  {l}")
    print(f"    tokens      {a[1]:>10,.0f} -> {b[1]:>10,.0f}   x{tr:8.1f}")
    print(f"    throughput  {a[3]:>10.1f} -> {b[3]:>10.1f}   x{rr:8.2f}   (batching)")
    print(f"    predicted time ratio = {tr:.1f} / {rr:.2f} = {tr/rr:.1f}x")
    print(f"    MEASURED  time ratio = {b[2]:.1f} / {a[2]:.1f} = {sr:.1f}x")
    d = 100 * abs(tr/rr - sr) / sr
    print(f"    agreement: {d:.1f}%  ->  {'CLOSES' if d < 5 else 'DOES NOT CLOSE -- investigate'}")
if not any_pair:
    print("\n  no level has BOTH a 1x1x1 and a 20x10x10 run yet.")
print()
print("  2000 predictions cost 'token ratio' times the work, not 2000x,")
print("  because only k_a*k_s = 200 of them involve a reasoning trace.")
print("  The rest of the gap is batching: one token costs one full read of")
print("  the weights however many sequences are being generated at once.")
PYEOF
