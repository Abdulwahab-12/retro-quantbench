#!/usr/bin/env bash
# qb_aug.sh -- does the k_a augmentation-mode comparison end to end. One command.
#
#     bash qb_aug.sh
#
# Runs the permute arm at k_a=20, k_s=10, k_b=10 on the full 5005 molecules,
# then scores it, neutralises it, and pairs it against the canonical arm that is
# already on disk.
#
# ---------------------------------------------------------------------------
# WHAT IS BEING COMPARED, AND WHY
# ---------------------------------------------------------------------------
# k_a asks the model the same retrosynthesis question in k_a different SMILES
# spellings of the product. There are two ways to make those spellings, and the
# project has always used the first:
#
#   canonical  MolToSmiles(mol, rootedAtAtom=r, canonical=True)
#              the ROOT ATOM varies, the traversal order does not. This is the
#              default (QB_AUG=canonical) and produced every augmented number we
#              have.
#   permute    RenumberAtoms with a seeded permutation, then canonical=False
#              root atom AND traversal order vary -- which is what the paper's
#              wording describes.
#
# run_llm.py's own docstring predicts the consequence: fewer distinct prompts
# means fewer distinct reasoning paths, which "barely moves top-1 ... but starves
# the tail". Our gap against the published numbers is top-1 and top-10 short with
# top-3 and top-5 reproducing, so this is worth measuring rather than arguing
# about. If the distinct-candidate counts come out equal, the spelling generator
# is not the cause and the next suspect is elsewhere.
#
# ---------------------------------------------------------------------------
# IT CANNOT OVERWRITE THE CANONICAL RESULTS
# ---------------------------------------------------------------------------
# Checked in run_llm_cluster.sh before writing this:
#
#     if [ "${QB_AUG:-canonical}" != canonical ]; then KTAG="$KTAG.aug$QB_AUG"; fi
#
# so the permute arm writes to files with .aug<mode> in the name and the two arms
# land side by side in the same results folder. The mode is also part of the
# record fingerprint when k_a > 1, so the two can never merge even if a filename
# were forced to collide. Running this in an existing results folder is safe.
#
# ---------------------------------------------------------------------------
# THE PROBE
# ---------------------------------------------------------------------------
# QB_AUG is read inside the container, so it has to cross the singularity
# boundary as SINGULARITYENV_QB_AUG. If that failed silently the run would
# quietly repeat the canonical arm for thirty hours and produce a file that looks
# right. So a one-molecule probe runs first and the script refuses to continue
# unless the probe's own output filename contains .aug<mode> -- the filename is
# proof that the variable arrived. --skip-probe turns it off.
#
# OPTIONS -- all optional
#   --mode permute    the augmentation mode to test (default permute)
#   --levels bf16     levels to run (default bf16; the question is not
#                     level-specific, so one level answers it)
#   --ka/--ks/--kb    default 20 10 10
#   --n 5005          molecules (default 5005)
#   --gpus 16         cards (default 16)
#   --results DIR     default ./results, beside this script
#   --skip-probe      skip the one-molecule env check
#   --score-only      run nothing; score, neutralise and pair what is on disk
#   --plan            print what would run and stop
set -u

QB_AUG_REV="2026-09-27.1"

if [ "$#" -ge 1 ] && [ -d "$1" ] && [ "${1#--}" = "$1" ]; then
  D="$(cd "$1" && pwd)"; shift
else
  D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODE="permute"; LEVELS="bf16"; KA=20; KS=10; KB=10; N=5005; GPUS=16
RES=""; QBSH=""; SKIP_PROBE=0; SCORE_ONLY=0; PLAN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)       MODE="$2"; shift 2 ;;
    --levels)     LEVELS="$2"; shift 2 ;;
    --ka)         KA="$2"; shift 2 ;;
    --ks)         KS="$2"; shift 2 ;;
    --kb)         KB="$2"; shift 2 ;;
    --n)          N="$2"; shift 2 ;;
    --gpus)       GPUS="$2"; shift 2 ;;
    --results)    RES="$2"; shift 2 ;;
    --qb)         QBSH="$2"; shift 2 ;;
    --skip-probe) SKIP_PROBE=1; shift ;;
    --score-only) SCORE_ONLY=1; shift ;;
    --plan)       PLAN=1; shift ;;
    -h|--help)    sed -n '2,66p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ "$MODE" = canonical ]; then
  echo "--mode canonical is the DEFAULT arm, which is already on disk." >&2
  echo "  Running it again would add nothing and would share filenames with it." >&2
  echo "  This script exists to run the OTHER arm: --mode permute." >&2
  exit 2
fi

[ -n "$QBSH" ] || QBSH="$D/qb.sh"
[ -f "$QBSH" ] || { echo "cannot find qb.sh at $QBSH" >&2
                    echo "  folder has: $(ls -1 "$D" | tr '\n' ' ')" >&2; exit 1; }
[ -n "$RES" ] || RES="$D/results"
mkdir -p "$RES" 2>/dev/null || true
[ -d "$RES" ] || { echo "results folder does not exist: $RES" >&2; exit 1; }
RES="$(cd "$RES" && pwd)"

LEVELS="$(echo "$LEVELS" | tr ' ,' '\n\n' | sed '/^$/d')"
KTAG="k${KA}x${KS}x${KB}"
LOG="$RES/aug_$MODE.log"

echo "qb_aug.sh  rev $QB_AUG_REV"
echo "mode    : $MODE   (canonical arm stays untouched)"
echo "levels  : $(echo "$LEVELS" | tr '\n' ' ')"
echo "k       : ${KA}x${KS}x${KB}   n=$N   gpus=$GPUS"
echo "results : $RES"
for LV in $LEVELS; do
  echo "  canonical : $RES/raw/retrodfm-r-8b.$LV.$KTAG.retro.jsonl"
  echo "  $MODE     : $RES/raw/retrodfm-r-8b.$LV.$KTAG.aug$MODE.retro.jsonl"
done
echo

MISSING=""
for LV in $LEVELS; do
  [ -s "$RES/raw/retrodfm-r-8b.$LV.$KTAG.retro.jsonl" ] || MISSING="$MISSING $LV"
done
if [ -n "$MISSING" ]; then
  echo "note: no canonical arm on disk for:$MISSING"
  echo "  the $MODE arm will still run, but there is nothing to pair it against"
  echo "  until that file exists. To produce it:"
  echo "      bash qb.sh --levels$MISSING --ka $KA --ks $KS --kb $KB --n $N --gpus $GPUS"
  echo
fi

if [ "$PLAN" = 1 ]; then
  echo "plan only -- nothing was run."
  exit 0
fi

# QB_AUG is read INSIDE the container; qb.sh just execs singularity, so the
# variable has to be handed over with the SINGULARITYENV_ / APPTAINERENV_ prefix.
export SINGULARITYENV_QB_AUG="$MODE" APPTAINERENV_QB_AUG="$MODE"

if [ "$SCORE_ONLY" != 1 ]; then
  if [ "$SKIP_PROBE" != 1 ]; then
    PROBE="$RES/_augprobe"
    LV1="$(echo "$LEVELS" | head -1)"
    echo "probe: one molecule at k_a=2, to prove QB_AUG=$MODE crossed into the"
    echo "       container before committing the real run to the queue ..."
    rm -rf "$PROBE"; mkdir -p "$PROBE"
    bash "$QBSH" --results "$PROBE" --levels "$LV1" --ka 2 --ks 1 --kb 1 \
         --n 1 --gpus 1 > "$RES/aug_probe.log" 2>&1
    HIT="$(ls -1 "$PROBE"/shards/*aug$MODE* 2>/dev/null | head -1)"
    if [ -z "$HIT" ]; then
      echo "  FAILED. The probe wrote:"
      ls -1 "$PROBE"/shards/ 2>/dev/null | sed 's/^/      /'
      echo "  No .aug$MODE in those names means QB_AUG did not reach the runner,"
      echo "  so the real run would silently repeat the canonical arm. Stopping."
      echo "  Log: $RES/aug_probe.log"
      exit 1
    fi
    echo "  ok -- $(basename "$HIT")"
    echo "  (throwaway; delete $PROBE whenever you like)"
    echo
  fi

  echo "running the $MODE arm. This resumes: rerun the same command after any"
  echo "interruption and it picks up the molecules that have no record yet."
  echo "log: $LOG"
  echo
  # tr alone leaves a trailing comma, which the runner reads as an empty level
  # and folds into the output tag -- "bf16,.k20x10x10". Strip it.
  bash "$QBSH" --results "$RES" \
       --levels "$(echo "$LEVELS" | tr '\n' ',' | sed 's/,$//')" \
       --ka "$KA" --ks "$KS" --kb "$KB" --n "$N" --gpus "$GPUS" 2>&1 | tee "$LOG"
  echo
fi

# Score everything in the folder: both arms, one scorer, one pass.
echo "scoring both arms ..."
bash "$QBSH" --results "$RES" --score-only 2>&1 | tail -40
echo

if [ -f "$D/qb_neutralise.sh" ]; then
  echo "neutralisation, both arms ..."
  bash "$D/qb_neutralise.sh" "$D" --results "$RES" --tag "$KTAG" 2>&1 | tail -30
  bash "$D/qb_neutralise.sh" "$D" --results "$RES" --tag "$KTAG.aug$MODE" 2>&1 | tail -30
  echo
else
  echo "note: qb_neutralise.sh is not beside this script, so the neutralised"
  echo "      columns are skipped. Put it in $D to get them."
  echo
fi

PAIR="$RES/aug_paired.py"
cat > "$PAIR" <<'PYEOF'
#!/usr/bin/env python3
"""Pair the two augmentation arms on the same molecules. Called by qb_aug.sh.

    python3 aug_paired.py <canonical raw.jsonl> <permute raw.jsonl>

The two arms differ in ONE thing: how the k_a product spellings are generated.

  canonical  MolToSmiles(mol, rootedAtAtom=r, canonical=True)
             the root atom varies, the traversal order does not.
  permute    RenumberAtoms with a seeded permutation, then canonical=False
             root atom AND traversal order vary.

Because both arms see the same molecules, the right test is paired: McNemar on
the discordant molecules, not two independent confidence intervals. At n=5005 a
paired test resolves differences a CI comparison would call a tie.

Everything is reported twice, as-scored and with rdMolStandardize.Uncharger
applied to both the candidates and the truth, so this table can sit beside the
neutralisation report without anyone having to ask which convention it used.

The mechanism to watch is the distinct-candidate count. The claim behind
permute is that more distinct spellings produce more distinct reasoning paths
and therefore more distinct candidates, which matters for the TAIL of the
ranked list (top-10) far more than for top-1. If the distinct counts come out
the same, the two arms will score the same and there is nothing here.
"""
import json, math, os, sys
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')

A_PATH, B_PATH = sys.argv[1], sys.argv[2]
A_NAME = os.path.basename(A_PATH)
B_NAME = os.path.basename(B_PATH)
KS = (1, 3, 5, 10)
_U = rdMolStandardize.Uncharger()
_ck = {}


def key(smi, neutral=False):
    """Canonical, order-free identity of one reactant set -- score_sweep.py's
    rule, plus optional uncharging. None for anything unparseable, so an
    invalid prediction is discarded rather than occupying a rank."""
    if smi is None:
        return None
    s = smi.strip()
    if not s or any(c.isspace() for c in s):
        return None
    ck = (s, neutral)
    if ck in _ck:
        return _ck[ck]
    parts = []
    for frag in s.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            _ck[ck] = None
            return None
        if neutral:
            try:
                m = _U.uncharge(m)
            except Exception:
                pass
        parts.append(Chem.MolToSmiles(m))
    v = tuple(sorted(parts))
    _ck[ck] = v
    return v


def answer_of(t):
    if "<answer>" in t and "</answer>" in t:
        return t.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return ""


def candidates_of(r):
    c = r.get("candidates")
    if isinstance(c, list) and c:
        return c
    raw = r.get("raw")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        a = answer_of(raw)
        if a:
            return [a]
    return []


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def mcnemar(b, c):
    """Two-sided exact binomial p over the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
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
            out[r["uid"]] = r          # a later success replaces an earlier one
    return out


A, B = load(A_PATH), load(B_PATH)
shared = sorted(set(A) & set(B))
print("=" * 100)
print("AUGMENTATION MODE, PAIRED ON THE SAME MOLECULES")
print("=" * 100)
print(f"  arm A (canonical) : {A_NAME}   {len(A)} molecules")
print(f"  arm B (permute)   : {B_NAME}   {len(B)} molecules")
print(f"  scored on the {len(shared)} molecules present in BOTH")
if not shared:
    print("\n  nothing in common -- are these the two arms of the same run?")
    sys.exit(3)
only_a, only_b = len(A) - len(shared), len(B) - len(shared)
if only_a or only_b:
    print(f"  ({only_a} only in A, {only_b} only in B -- excluded, so the "
          f"comparison stays paired)")
print()

for neutral in (False, True):
    label = "NEUTRALISED (Uncharger on candidates and truth)" if neutral \
            else "AS SCORED (score_sweep.py convention)"
    hits = {"A": {k: 0 for k in KS}, "B": {k: 0 for k in KS}}
    disc = {k: [0, 0] for k in KS}          # [A hit & B miss, B hit & A miss]
    nd = {"A": [], "B": []}
    for uid in shared:
        for tag, src in (("A", A), ("B", B)):
            r = src[uid]
            cands = dedup([x for x in (key(v, neutral)
                                       for v in candidates_of(r)) if x is not None])
            nd[tag].append(len(cands))
            t = key(r["truth_key"], neutral)
            for k in KS:
                if t is not None and t in cands[:k]:
                    hits[tag][k] += 1
            src[uid]["_c"] = cands
            src[uid]["_t"] = t
        for k in KS:
            ta, tb = A[uid]["_t"], B[uid]["_t"]
            ha = ta is not None and ta in A[uid]["_c"][:k]
            hb = tb is not None and tb in B[uid]["_c"][:k]
            if ha and not hb:
                disc[k][0] += 1
            elif hb and not ha:
                disc[k][1] += 1

    n = len(shared)
    print("-" * 100)
    print(label)
    print("-" * 100)
    print(f"  {'k':>3}  {'canonical':>10}  {'permute':>10}  {'diff':>7}"
          f"  {'A only':>7}  {'B only':>7}  {'McNemar p':>11}")
    for k in KS:
        a = 100.0 * hits["A"][k] / n
        b = 100.0 * hits["B"][k] / n
        p = mcnemar(disc[k][0], disc[k][1])
        star = "  *" if p < 0.05 else ""
        print(f"  {k:>3}  {a:10.2f}  {b:10.2f}  {b - a:+7.2f}"
              f"  {disc[k][0]:7d}  {disc[k][1]:7d}  {p:11.4g}{star}")

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else 0
    print()
    print(f"  distinct candidates per molecule -- canonical median "
          f"{med(nd['A'])}, mean {sum(nd['A'])/n:.2f}   |   permute median "
          f"{med(nd['B'])}, mean {sum(nd['B'])/n:.2f}")
    lo = sum(1 for x in nd["A"] if x < 10), sum(1 for x in nd["B"] if x < 10)
    print(f"  molecules with fewer than 10 distinct candidates -- canonical "
          f"{lo[0]} ({100.0*lo[0]/n:.1f}%), permute {lo[1]} ({100.0*lo[1]/n:.1f}%)")
    print()

print("  * marks p < 0.05. A only / B only are the discordant molecules: the")
print("  ones one arm gets right and the other does not. A difference in the")
print("  percentages with almost no discordant pairs is not a real difference;")
print("  McNemar tests exactly that and is the correct test for two arms run on")
print("  the same molecules.")
print()
print("  If the distinct-candidate counts are equal, the spelling generator is")
print("  not the source of the gap against the published numbers, whatever the")
print("  top-k columns do -- and the next suspect is elsewhere.")
PYEOF

PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
SIF=""; for c in "$D"/*.sif; do [ -f "$c" ] && SIF="$c"; done
SING="$(command -v singularity || command -v apptainer || true)"

RC=0
for LV in $LEVELS; do
  A="$RES/raw/retrodfm-r-8b.$LV.$KTAG.retro.jsonl"
  B="$RES/raw/retrodfm-r-8b.$LV.$KTAG.aug$MODE.retro.jsonl"
  OUTF="$RES/aug_paired.$LV.txt"
  if [ ! -s "$A" ] || [ ! -s "$B" ]; then
    echo "$LV: cannot pair -- missing $( [ -s "$A" ] || echo "$(basename "$A")" ) $( [ -s "$B" ] || echo "$(basename "$B")" )"
    RC=1
    continue
  fi
  echo "pairing $LV ..."
  # rdkit lives in the image; use it unless this machine has its own.
  if [ -n "$PY" ] && "$PY" -c 'import rdkit' 2>/dev/null; then
    "$PY" -u "$PAIR" "$A" "$B" | tee "$OUTF"
  elif [ -n "$SIF" ] && [ -n "$SING" ]; then
    "$SING" exec --bind "$RES:/work" "$SIF" python -u /work/aug_paired.py \
      "/work/raw/$(basename "$A")" "/work/raw/$(basename "$B")" | tee "$OUTF"
  else
    echo "  no rdkit here and no image to borrow one from; run by hand:" >&2
    echo "      python3 $PAIR $A $B" >&2
    RC=1
  fi
  echo
done

echo "written:"
echo "  $RES/scored.json                  both arms, score_sweep"
ls -1 "$RES"/aug_paired.*.txt 2>/dev/null | sed 's/^/  /'
[ -f "$RES/neutralise.txt" ] && echo "  $RES/neutralise.txt               (last neutralisation pass)"
exit "$RC"
