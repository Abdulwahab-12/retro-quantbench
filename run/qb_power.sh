#!/usr/bin/env bash
# qb_power.sh -- energy per molecule, every level, BOTH settings, one command.
#
#     bash qb_power.sh
#
# Put this file next to qb.sh and the .sif, point it at an empty folder, and it
# measures watts, joules per molecule and seconds at full GPU load for every
# quantization level in both the unaugmented (1x1x1) and the augmented
# (20x10x10) setting, then prints the change against bf16 for each.
#
# ---------------------------------------------------------------------------
# 1. IT MUST NOT RESUME. THIS IS THE WHOLE POINT.
# ---------------------------------------------------------------------------
# The runner is resumable: a molecule already on disk with a matching
# fingerprint is skipped. All 5005 test molecules have already been run, so
# pointing an energy measurement at an existing results folder would skip every
# molecule, finish in seconds, and report the energy of a model load as if it
# were the energy of N molecules -- a number that looks like a measurement and
# is not one.
#
# So this script writes into its OWN fresh folder, one per pass, and then CHECKS
# that each pass produced the records it asked for. A pass that produced fewer
# is reported as unusable instead of being turned into an energy figure. Use an
# empty output folder and nothing can resume at all.
#
# ---------------------------------------------------------------------------
# 2. WHY TWO PASSES PER LEVEL PER SETTING
# ---------------------------------------------------------------------------
# A run is two phases with opposite power profiles:
#
#   load     read 16 GB of bf16 weights, and for nf4/fp4/nf4dq/int8_ao quantize
#            them layer by layer. Minutes long, PCIe- and CPU-bound, SMs idle.
#   generate the actual work, SMs near 100%.
#
# Average watts over a whole run therefore depends mostly on how long the model
# took to build. A level that quantizes slowly reports LOWER average watts while
# doing identical work -- which is exactly how one concludes "nf4dq uses half the
# power" from an artefact.
#
# So each level runs twice per setting, with n1 and n2 molecules, and the
# per-molecule energy is the DIFFERENCE:
#
#     E(n) = E_load + n * e_mol       ->      e_mol = (E(n2)-E(n1)) / (n2-n1)
#
# The load term cancels exactly: no phase detection, no log alignment, no
# assumption about when generation began. E_load falls out as a by-product and is
# reported, because "what does building a 4-bit model cost" is its own question.
#
# ---------------------------------------------------------------------------
# 3. WHAT IS REPORTED
# ---------------------------------------------------------------------------
#   J/mol            total energy the card drew per molecule
#   J/mol above idle idle draw, measured on this card before anything runs,
#                    times the time, subtracted. A card at 40 W spends that
#                    whether or not this job exists; CO2 accounting wants only
#                    the attributable part.
#   W gen            J/mol divided by load-free s/mol -- the number "does 4-bit
#                    pull fewer watts" is actually about.
#   s/mol            load-free, from the runner's own wall_seconds.
#   100% s/mol       seconds at >= 95% utilization per molecule, which is the
#                    closest nvidia-smi gets to "seconds at 100% load".
#   J/token          levels emit different numbers of tokens for the same
#                    molecules, and J/mol alone cannot separate "cheaper per
#                    token" from "wrote less".
#
# Energy comes from the driver's total-energy counter when this driver exposes it
# (exact, immune to sampling), otherwise from integrating power samples. When
# both exist both are computed and the report says whether they agree.
#
# ---------------------------------------------------------------------------
# OPTIONS -- all optional
#   --levels a,b,c   default bf16,int8_ao,nf4,fp4,nf4dq
#   --settings a,b   default 1x1x1,20x10x10   (k_a x k_s x k_b)
#   --n1 N --n2 N    molecules per pass, unaugmented settings (default 1, 4)
#   --an1 N --an2 N  molecules per pass, augmented settings   (default 1, 2)
#   --extra "..."    passed to qb.sh for augmented settings only, e.g.
#                    --extra "--ans-chunk 2" on a card under ~16 GB
#   --gpu N          physical card to use and to measure (default 0)
#   --idle-s N       idle baseline before anything runs (default 30)
#   --interval-ms N  sampling period (default 200)
#   --out DIR        where to write (default ./power_<timestamp>)
#   --qb PATH        qb.sh, if it is not beside this file
#   --force          run even if another process is already on the card
#   --plan           print the plan and the time estimate, run nothing
set -u

QB_POWER_REV="2026-09-27.2  (both settings in one run)"

if [ "$#" -ge 1 ] && [ -d "$1" ] && [ "${1#--}" = "$1" ]; then
  D="$(cd "$1" && pwd)"; shift
else
  D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

LEVELS="bf16,int8_ao,nf4,fp4,nf4dq"
SETTINGS="1x1x1,20x10x10"
N1=1; N2=4; AN1=1; AN2=2; GPU=0; IDLE_S=30; IVAL=200
OUT=""; QBSH=""; EXTRA=""; FORCE=0; PLAN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --levels)      LEVELS="$2"; shift 2 ;;
    --settings)    SETTINGS="$2"; shift 2 ;;
    --n1)          N1="$2"; shift 2 ;;
    --n2)          N2="$2"; shift 2 ;;
    --an1)         AN1="$2"; shift 2 ;;
    --an2)         AN2="$2"; shift 2 ;;
    --extra)       EXTRA="$2"; shift 2 ;;
    --gpu)         GPU="$2"; shift 2 ;;
    --idle-s)      IDLE_S="$2"; shift 2 ;;
    --interval-ms) IVAL="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --qb)          QBSH="$2"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --plan)        PLAN=1; shift ;;
    -h|--help)     sed -n '2,86p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

LEVELS="$(echo "$LEVELS" | tr ' ,' '\n\n' | sed '/^$/d')"
SETTINGS="$(echo "$SETTINGS" | tr ' ,' '\n\n' | sed '/^$/d')"

[ -n "$QBSH" ] || QBSH="$D/qb.sh"
if [ ! -f "$QBSH" ]; then
  echo "cannot find qb.sh" >&2
  echo "  looked for : $QBSH" >&2
  echo "  folder has : $(ls -1 "$D" | tr '\n' ' ')" >&2
  echo "  point at it with:  bash qb_power.sh --qb /path/to/qb.sh" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi not on PATH -- there is no way to read the card's power" >&2
  exit 1; }

# Every setting is checked before anything runs: a typo in one of them should not
# surface forty minutes in, and n2 must exceed n1 or e_mol is 0/0.
for S in $SETTINGS; do
  case "$S" in
    *x*x*) : ;;
    *) echo "setting '$S' is not k_a x k_s x k_b, e.g. 1x1x1 or 20x10x10" >&2
       exit 2 ;;
  esac
done
if [ "$N2" -le "$N1" ] || [ "$AN2" -le "$AN1" ]; then
  echo "n2 must be greater than n1: the per-molecule energy is the difference" >&2
  echo "  between the two passes divided by (n2 - n1)." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
[ -n "$OUT" ] || OUT="$D/power_$STAMP"
mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 1; }
OUT="$(cd "$OUT" && pwd)"
TRACE="$OUT/trace.csv"; MARKS="$OUT/marks.tsv"
REPORT="$OUT/power_report.txt"; RUNLOG="$OUT/runs.log"

# What this driver is willing to tell us. Probed, not assumed: the energy counter
# exists only on some GPUs and driver versions, and --loop-ms is
# not in every nvidia-smi.
QF="timestamp,power.draw,utilization.gpu,clocks.sm,temperature.gpu,memory.used"
HAVE_E=0
if nvidia-smi --help-query-gpu 2>/dev/null | grep -q 'total_energy_consumption'; then
  if nvidia-smi --query-gpu=total_energy_consumption --format=csv,noheader,nounits \
       -i "$GPU" >/dev/null 2>&1; then
    QF="$QF,total_energy_consumption"; HAVE_E=1
  fi
fi
LOOPOPT="--loop-ms=$IVAL"
nvidia-smi --help 2>/dev/null | grep -q -- '--loop-ms' || LOOPOPT="-l 1"

GPUNAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" 2>/dev/null)"
[ -n "$GPUNAME" ] || { echo "no GPU $GPU on this machine" >&2
                       nvidia-smi --query-gpu=index,name --format=csv >&2; exit 1; }
PLIMIT="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)"

# Someone else's job on the same card would be counted as ours. Refuse rather
# than publish a contaminated number.
BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" 2>/dev/null)"
if [ -n "$BUSY" ] && [ "$FORCE" != 1 ]; then
  echo "another process is already using GPU $GPU:" >&2
  echo "$BUSY" | sed 's/^/    /' >&2
  echo "  its power would be counted as ours. Use a free card with --gpu N," >&2
  echo "  or --force if you know that process is idle." >&2
  exit 1
fi

# Build the full pass list first, so the plan, the run and the reducer all work
# from one list and cannot disagree. 90 s per load, 30 s/mol unaugmented and
# 350 s/mol at 20x10x10 are deliberately rough -- an order of magnitude, not a
# prediction.
PASSES=""; COMBOS=""; EST=0
for S in $SETTINGS; do
  KA="${S%%x*}"; REST="${S#*x}"; KS="${REST%%x*}"; KB="${REST#*x}"
  if [ "$S" = "1x1x1" ]; then A="$N1"; B="$N2"; SPM=30; else A="$AN1"; B="$AN2"; SPM=350; fi
  for LV in $LEVELS; do
    for N in "$A" "$B"; do
      PASSES="$PASSES $LV|$S|$KA|$KS|$KB|$N"
      EST=$(( EST + 90 + N * SPM ))
    done
    COMBOS="$COMBOS,$LV@$S@$A@$B"
  done
done
COMBOS="${COMBOS#,}"
NPASS="$(echo $PASSES | wc -w | tr -d ' ')"

echo "qb_power.sh  rev $QB_POWER_REV"
echo "card    : GPU $GPU -- $GPUNAME  (power limit ${PLIMIT:-?} W)"
echo "levels  : $(echo "$LEVELS" | tr '\n' ' ')"
echo "settings: $(echo "$SETTINGS" | tr '\n' ' ')"
echo "passes  : $NPASS  ($N1 and $N2 molecules unaugmented, $AN1 and $AN2 augmented)"
echo "energy  : $([ "$HAVE_E" = 1 ] && echo "driver counter + integrated samples (cross-checked)" || echo "integrated power samples (this driver has no energy counter)")"
echo "sampling: every ${IVAL} ms  ($LOOPOPT)"
echo "out     : $OUT"
echo "estimate: very roughly $(( EST / 60 )) min, plus $IDLE_S s of idle baseline"
echo

if [ "$PLAN" = 1 ]; then
  echo "plan only -- nothing was run. Each pass gets its own fresh folder:"
  for P in $PASSES; do
    IFS='|' read -r LV S KA KS KB N <<< "$P"
    echo "  $LV  $S  n=$N  ->  $OUT/$LV.$S.n$N"
  done
  exit 0
fi

# Pin the card for everything inside the container. The runner sets
# CUDA_VISIBLE_DEVICES=0 per worker, which selects the first card VISIBLE to it,
# so masking from out here makes that card the physical one we are metering.
if [ "$GPU" != 0 ]; then
  export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU"
fi

mark() { printf '%s\t%s\n' "$1" "$(date '+%Y/%m/%d %H:%M:%S.%N')" >> "$MARKS"; }

: > "$MARKS"; : > "$RUNLOG"
printf '%s\n' "$QF" > "$TRACE"
nvidia-smi --query-gpu="$QF" --format=csv,noheader,nounits -i "$GPU" \
  $LOOPOPT >> "$TRACE" 2>>"$OUT/sampler.err" &
SPID=$!
cleanup() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }
trap 'cleanup; echo; echo "interrupted -- partial trace in $TRACE" >&2; exit 130' INT TERM
sleep 2
if ! kill -0 "$SPID" 2>/dev/null; then
  echo "the sampler died immediately. nvidia-smi said:" >&2
  sed 's/^/    /' "$OUT/sampler.err" >&2
  exit 1
fi

echo "idle baseline: $IDLE_S s with nothing running ..."
mark "idle.start"; sleep "$IDLE_S"; mark "idle.end"

FAILED=""; I=0
for P in $PASSES; do
  IFS='|' read -r LV S KA KS KB N <<< "$P"
  I=$((I + 1))
  W="$OUT/$LV.$S.n$N"
  # A FRESH folder per pass. Pass 2 must regenerate the molecules pass 1 did, so
  # it cannot share a results directory with it or it would resume from them and
  # measure nothing.
  mkdir -p "$W"
  # Deterministic decoding for the unaugmented arm, matching the published
  # 1x1x1 run; the augmented arm samples, and takes --extra if the card needs it.
  if [ "$S" = "1x1x1" ]; then OPT="--greedy"; else OPT="$EXTRA"; fi
  echo "== [$I/$NPASS] $LV  $S  n=$N =="
  echo "=== $LV $S n=$N ===" >> "$RUNLOG"
  mark "$LV.$S.n$N.start"
  # No `set -e` anywhere in this script: a level that cannot run on this card
  # (one whose kernels this torch build lacks) must leave
  # the others measured rather than abort the sweep.
  bash "$QBSH" --results "$W" --levels "$LV" --n "$N" \
       --ka "$KA" --ks "$KS" --kb "$KB" --gpus 1 $OPT >> "$RUNLOG" 2>&1
  RC=$?
  mark "$LV.$S.n$N.end"
  GOT=0
  for f in "$W"/shards/*.jsonl; do
    [ -f "$f" ] || continue
    GOT=$((GOT + $(grep -c '"uid"' "$f" 2>/dev/null || echo 0)))
  done
  echo "   exit $RC, $GOT record(s) written"
  if [ "$GOT" -lt "$N" ]; then
    echo "   !! asked for $N molecules, got $GOT -- this pass is not usable"
    FAILED="$FAILED $LV/$S"
  fi
done

mark "all.end"; sleep 2; cleanup; trap - INT TERM
echo
echo "samples : $(( $(wc -l < "$TRACE") - 1 ))"
echo

PY=""
for cand in python3 python; do
  command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
done
RED="$OUT/reduce_power.py"
cat > "$RED" <<'PYEOF'
#!/usr/bin/env python3
"""Turn the nvidia-smi trace into energy per molecule. Called by qb_power.sh.

Two passes per level per setting, n1 and n2 molecules:

    E(n) = E_load + n * e_mol      ->      e_mol = (E(n2)-E(n1)) / (n2-n1)

so the model-load energy cancels instead of being averaged into the result --
which matters because building a 4-bit model takes minutes at low SM load and
would otherwise drag that level's average watts down for free.

Seconds per molecule are taken the same way, from the runner's own
wall_seconds (which starts AFTER the load), so watts = joules / seconds is
load-free on both sides.

    python3 reduce_power.py <out dir> <combos>
    combos: lv@tag@n1@n2,lv@tag@n1@n2,...        tag is 1x1x1, 20x10x10, ...
"""
import json, os, sys, glob, datetime as dt

OUT = sys.argv[1]
COMBOS = []
for spec in sys.argv[2].split(","):
    if not spec.strip():
        continue
    lv, tag, n1, n2 = spec.split("@")
    COMBOS.append((lv, tag, int(n1), int(n2)))
UTIL_BUSY = 95.0


def parse_ts(s):
    s = s.strip()
    # nvidia-smi: "2026/09/27 20:11:33.123"; date +%N gives 9 digits, so both
    # are cut to microseconds and parsed by the same code in the same timezone.
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:26], fmt)
        except ValueError:
            pass
    return None


with open(os.path.join(OUT, "trace.csv")) as f:
    header = [h.strip() for h in f.readline().split(",")]
    rows = []
    for line in f:
        bits = [b.strip() for b in line.split(",")]
        if len(bits) != len(header):
            continue                      # a truncated final line
        t = parse_ts(bits[0])
        if t is None:
            continue
        rec = {"t": t}
        for h, b in zip(header[1:], bits[1:]):
            try:
                rec[h] = float(b)
            except ValueError:
                rec[h] = None
        rows.append(rec)
rows.sort(key=lambda r: r["t"])
HAVE_E = "total_energy_consumption" in header

marks = {}
with open(os.path.join(OUT, "marks.tsv")) as f:
    for line in f:
        if "\t" not in line:
            continue
        k, v = line.rstrip("\n").split("\t", 1)
        t = parse_ts(v)
        if t is not None:
            marks[k] = t


def window(a, b):
    if a not in marks or b not in marks:
        return None
    seg = [r for r in rows if marks[a] <= r["t"] <= marks[b]]
    if len(seg) < 2:
        return None
    pw = [r["power.draw"] for r in seg if r["power.draw"] is not None]
    # Trapezoid over the actual sample times, not count * interval: nvidia-smi
    # does not promise an even cadence and a dropped sample would bias a
    # rectangle sum.
    e_int = 0.0
    busy = 0.0
    for x, y in zip(seg, seg[1:]):
        gap = (y["t"] - x["t"]).total_seconds()
        if x["power.draw"] is not None and y["power.draw"] is not None:
            e_int += 0.5 * (x["power.draw"] + y["power.draw"]) * gap
        if x["utilization.gpu"] is not None and x["utilization.gpu"] >= UTIL_BUSY:
            busy += gap
    e_cnt = None
    if HAVE_E:
        ev = [r["total_energy_consumption"] for r in seg
              if r["total_energy_consumption"] is not None]
        if len(ev) >= 2:
            e_cnt = (ev[-1] - ev[0]) / 1000.0        # mJ -> J
    util = [r["utilization.gpu"] for r in seg if r["utilization.gpu"] is not None]
    return dict(dur=(seg[-1]["t"] - seg[0]["t"]).total_seconds(), n=len(seg),
                w_mean=(sum(pw) / len(pw)) if pw else 0.0,
                w_max=max(pw) if pw else 0.0,
                e_int=e_int, e_cnt=e_cnt, busy=busy,
                u_mean=(sum(util) / len(util)) if util else 0.0)


def energy(w):
    """Prefer the driver's counter; fall back to the integral."""
    return w["e_cnt"] if (w and w["e_cnt"] is not None) else (w["e_int"] if w else None)


def read_pass(lv, tag, n):
    """load_seconds, generation wall_seconds, molecules and tokens actually done."""
    d = os.path.join(OUT, f"{lv}.{tag}.n{n}", "shards")
    load_s = wall_s = 0.0
    mols = toks = 0
    for m in glob.glob(os.path.join(d, "*.meta.json")):
        try:
            j = json.load(open(m))
        except Exception:
            continue
        load_s += float(j.get("load_seconds") or 0)
        wall_s += float(j.get("wall_seconds") or 0)
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(p):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "uid" not in r:
                continue
            if str(r.get("stop_reason", "")).startswith("error"):
                continue
            mols += 1
            toks += int(r.get("n_gen_tokens") or 0)
    return dict(load_s=load_s, wall_s=wall_s, mols=mols, toks=toks)


idle = window("idle.start", "idle.end")
P_IDLE = idle["w_mean"] if idle else 0.0

print("=" * 112)
print("GPU ENERGY PER MOLECULE -- measured, model-load energy removed by differencing two passes")
print("=" * 112)
if idle:
    print(f"  idle baseline : {P_IDLE:6.1f} W over {idle['dur']:.0f} s "
          f"({idle['n']} samples, mean util {idle['u_mean']:.1f}%)")
print(f"  energy source : {'driver total-energy counter' if HAVE_E else 'integrated power samples'}")
print()

ALL = {}
for lv, tag, n1, n2 in COMBOS:
    wa = window(f"{lv}.{tag}.n{n1}.start", f"{lv}.{tag}.n{n1}.end")
    wb = window(f"{lv}.{tag}.n{n2}.start", f"{lv}.{tag}.n{n2}.end")
    pa, pb = read_pass(lv, tag, n1), read_pass(lv, tag, n2)
    note = None
    if wa is None or wb is None:
        note = "no usable trace for one of its passes"
    elif pa["mols"] < n1 or pb["mols"] < n2:
        note = (f"produced {pa['mols']}/{n1} and {pb['mols']}/{n2} molecules "
                f"-- NOT USABLE, the passes did not do the work")
    if note:
        ALL.setdefault(tag, []).append(dict(lv=lv, note=note))
        continue
    ea, eb = energy(wa), energy(wb)
    dn = pb["mols"] - pa["mols"]
    if dn <= 0:
        ALL.setdefault(tag, []).append(
            dict(lv=lv, note="both passes did the same number of molecules"))
        continue
    e_mol = (eb - ea) / dn
    s_mol = (pb["wall_s"] - pa["wall_s"]) / dn
    t_mol = (pb["toks"] - pa["toks"]) / dn
    # Seconds at full load, differenced like everything else so the loading
    # phase does not contribute. This is Igor's "seconds with 100% GPU load",
    # per molecule; it should come out close to s/mol, and if it is much lower
    # the card is waiting on something rather than computing.
    busy_mol = (wb["busy"] - wa["busy"]) / dn
    drift = None
    if wb["e_cnt"] is not None and wb["e_int"] > 0:
        drift = 100.0 * (wb["e_cnt"] - wb["e_int"]) / wb["e_int"]
    ALL.setdefault(tag, []).append(dict(
        lv=lv, note=None, e_mol=e_mol, s_mol=s_mol, t_mol=t_mol,
        w_gen=(e_mol / s_mol) if s_mol > 0 else 0.0,
        e_net=e_mol - P_IDLE * s_mol,
        e_load=ea - pa["mols"] * e_mol,
        busy_mol=busy_mol, drift=drift,
        w_peak=max(wa["w_max"], wb["w_max"]),
        ok=(e_mol > 0 and s_mol > 0)))

if not any(r.get("note") is None for g in ALL.values() for r in g):
    print("  nothing usable was measured. The per-pass notes say why:")
    for tag, g in ALL.items():
        for r in g:
            print(f"    {tag:<10} {r['lv']:<9} {r.get('note')}")
    sys.exit(3)

for tag, g in ALL.items():
    good = [r for r in g if r["note"] is None]
    print("-" * 112)
    print(f"setting {tag}" + ("   (unaugmented)" if tag == "1x1x1" else ""))
    print("-" * 112)
    for r in g:
        if r["note"]:
            print(f"  {r['lv']:<9} {r['note']}")
    if not good:
        print()
        continue
    print(f"  {'level':<9}{'W peak':>8}{'W gen':>8}{'s/mol':>9}{'J/mol':>10}"
          f"{'J/mol-idle':>12}{'tok/mol':>9}{'J/tok':>8}{'Wh/1k mol':>11}"
          f"{'load kJ':>9}{'100% s/mol':>12}")
    for r in good:
        print(f"  {r['lv']:<9}{r['w_peak']:8.0f}{r['w_gen']:8.0f}{r['s_mol']:9.1f}"
              f"{r['e_mol']:10.0f}{r['e_net']:12.0f}{r['t_mol']:9.0f}"
              f"{(r['e_mol']/r['t_mol'] if r['t_mol'] else 0):8.2f}"
              f"{r['e_mol']*1000/3600:11.1f}{r['e_load']/1000:9.1f}"
              f"{r['busy_mol']:12.1f}")
    bad = [r["lv"] for r in good if not r["ok"]]
    if bad:
        print(f"  !! {', '.join(bad)}: energy or time did not increase with more")
        print("     molecules. That is noise swamping the difference, not a")
        print("     result -- rerun that setting with a larger --n2.")
    ref = next((r for r in good if r["lv"] == "bf16" and r["ok"]), None)
    if ref:
        print()
        print(f"  against bf16:   {'level':<9}{'W gen':>12}{'s/mol':>12}"
              f"{'J/mol':>12}{'J/mol above idle':>20}")
        for r in good:
            if r["lv"] == "bf16" or not r["ok"]:
                continue
            def pct(a, b):
                return f"{100.0 * (a - b) / b:+.1f}%" if b else "n/a"
            print(f"  {'':<16}{r['lv']:<9}{pct(r['w_gen'], ref['w_gen']):>12}"
                  f"{pct(r['s_mol'], ref['s_mol']):>12}"
                  f"{pct(r['e_mol'], ref['e_mol']):>12}"
                  f"{pct(r['e_net'], ref['e_net']):>20}")
    print()

print("  W gen = J/mol divided by load-free s/mol; s/mol is the runner's own")
print("  wall_seconds, differenced the same way as the energy, so both sides of")
print("  the division exclude the model load.")
print("  Energy per molecule is watts times seconds. A level can pull fewer watts")
print("  and still cost MORE energy if it is slower -- J/mol is the column that")
print("  answers the CO2 question, and the two before it say which factor moved.")

dr = [r["drift"] for g in ALL.values() for r in g
      if r.get("drift") is not None]
if dr:
    worst = max(dr, key=abs)
    print()
    print(f"  counter vs integrated samples: worst disagreement {worst:+.2f}%"
          + ("  -- the two methods agree" if abs(worst) < 5 else
             "  -- THEY DISAGREE; trust neither until this is explained"))

json.dump(dict(gpu_idle_w=P_IDLE, have_counter=HAVE_E, settings=ALL),
          open(os.path.join(OUT, "power.json"), "w"), indent=2, default=str)
print()
print(f"  machine-readable copy: {os.path.join(OUT, 'power.json')}")
PYEOF

if [ -z "$PY" ]; then
  # No host python: the image has one, and /work is the folder we just wrote.
  SIF=""; for c in "$D"/*.sif; do [ -f "$c" ] && SIF="$c"; done
  SING="$(command -v singularity || command -v apptainer || true)"
  if [ -n "$SIF" ] && [ -n "$SING" ]; then
    "$SING" exec --bind "$OUT:/work" "$SIF" \
      python -u /work/reduce_power.py /work "$COMBOS" | tee "$REPORT"
  else
    echo "no python on this machine and no image to borrow one from." >&2
    echo "  the trace is complete: $TRACE" >&2
    echo "  run the reducer anywhere with python3:" >&2
    echo "      python3 $RED $OUT $COMBOS" >&2
    exit 1
  fi
else
  "$PY" -u "$RED" "$OUT" "$COMBOS" | tee "$REPORT"
fi

echo
[ -n "$FAILED" ] && echo "passes that did no work:$FAILED  (see $RUNLOG)"
echo "report  : $REPORT"
echo "trace   : $TRACE   (every sample, for re-analysis)"
