#!/usr/bin/env python3
"""How much does charge standardisation change top-k? -- measured, per level.

WHY THIS EXISTS
---------------
score_sweep.py compares a predicted reactant set to the ground truth by
canonicalising each fragment with RDKit and sorting the fragments. It does NOT
neutralise formal charges. So a prediction of `N` against a truth of `[NH4+]`
is scored as a miss, even though the two describe the same reagent at a
different pH.

Syntheseus best practice S2 (the one this project follows) prescribes four
things: discard invalid molecules, remove duplicate reactions, report
stereochemistry both ways, report inference time. It says nothing about charge
or tautomer standardisation. So neither choice violates it -- which is exactly
why the size of the difference has to be measured rather than argued about.

WHAT IT MEASURES
----------------
For every <model>.<level>.<task>.jsonl in the raw folder, twice:

  as-scored    identical to score_sweep.py
  neutralised  the same, with rdMolStandardize.Uncharger applied to every
               fragment of every candidate AND of the ground truth, before
               canonicalisation

Both passes use the same record filtering, the same invalid-molecule discard,
the same deduplication and the same truncation, so the only difference between
the two columns is the uncharging step. Deduplication is re-run AFTER
neutralisation, because neutralising can collapse two candidates into one and a
collapsed entry must not occupy a rank.

It also reports:

  charged truth   the share of ground-truth reactant sets that carry a formal
                  charge at all. This is a property of the TEST SET, not of any
                  model, so it is the same number for every level and it is the
                  ceiling on what neutralisation can possibly move.
  charged subset  top-k on just those molecules, both ways. All of the movement
                  lives here; the whole-set delta is this times the share.

Finally it prints every between-level gap before and after. If the gaps do not
move, neutralisation changes no conclusion in the paper, and that is the claim
the footer either supports or refutes.

    python3 neutralise_sweep.py <raw folder>
"""
import json, math, os, sys, glob
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw"

# Optional filters. Uncharging every fragment of every candidate is the cost
# here, so a 20x10x10 file at n=5005 takes minutes while an unaugmented one
# takes seconds. Both are empty by default: `bash qb_neutralise.sh` with no
# arguments does every file it finds.
#   QB_NEU_LEVELS=bf16,nf4      only these base levels
#   QB_NEU_TAG=k20x10x10        only this k-setting;  none -> unaugmented only
ONLY_LEVELS = [x for x in os.environ.get("QB_NEU_LEVELS", "").replace(
    " ", "").split(",") if x]
ONLY_TAG = os.environ.get("QB_NEU_TAG", "").strip()

_UNCHARGER = rdMolStandardize.Uncharger()

# Same level vocabulary as score_sweep.py, for the same reason: a MODEL name
# can contain a dot, so the level cannot be found by counting dots.
BASE_LEVELS = {
    "fp32", "fp16", "bf16", "int8_ao", "int4_ao", "w8a8", "w4a8", "w4a4",
    "w4a8_intx", "fp8_ao", "fp8_w8a8", "fp8_w4a8", "nvfp4", "nvfp4_w",
    "mxfp8", "mxfp4", "nf4", "nf4dq", "fp4",
    "q2_k", "q3_k_m", "q3_k_s", "q4_0", "q4_1", "q4_k_m", "q4_k_s",
    "q5_0", "q5_k_m", "q5_k_s", "q6_k", "q8_0",
}


def split_name(stem):
    bits = stem.split(".")
    if len(bits) < 3:
        return None
    for i in range(1, len(bits) - 1):
        if bits[i] in BASE_LEVELS:
            return ".".join(bits[:i]), ".".join(bits[i:-1])
    return bits[0], ".".join(bits[1:-1])


_ck = {}
def key(smi, stereo=True, neutral=False):
    """Canonical, order-free identity of one reactant set.

    neutral=True runs rdMolStandardize.Uncharger over each fragment first.
    Returns None for anything unparseable, exactly as score_sweep.py does, so
    an invalid prediction is discarded rather than occupying a rank.
    """
    if smi is None:
        return None
    s = smi.strip()
    if not s or any(c.isspace() for c in s):
        return None
    ck = (s, stereo, neutral)
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
                m = _UNCHARGER.uncharge(m)
            except Exception:
                pass                      # leave it charged rather than drop it
        parts.append(Chem.MolToSmiles(m, isomericSmiles=stereo))
    v = tuple(sorted(parts))
    _ck[ck] = v
    return v


def has_charge(smi):
    """Does this reactant set carry any non-zero formal charge?"""
    if smi is None:
        return False
    for frag in smi.strip().split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            continue
        for a in m.GetAtoms():
            if a.GetFormalCharge() != 0:
                return True
    return False


def answer_of(text):
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return ""


def candidates_of(r):
    """The ranked candidate list, whichever way the run wrote it.
    Copied from score_sweep.py: three record formats exist on disk."""
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


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


KS = (1, 3, 5, 10)
rows = []
FILES = sorted(glob.glob(os.path.join(RAW, "*.jsonl")))
if not FILES:
    print("NOTHING TO SCORE")
    print(f"  looked in : {os.path.abspath(RAW)}")
    print(f"  exists    : {os.path.isdir(RAW)}")
    if os.path.isdir(RAW):
        here = sorted(os.listdir(RAW))
        print(f"  contains  : {len(here)} entries"
              + (f" -- {', '.join(here[:8])}" if here else " (empty)"))
    print()
    print("  Expected  <model>.<level>.<task>.jsonl, e.g.")
    print("      retrodfm-r-8b.bf16.retro.jsonl            (unaugmented, 1x1x1)")
    print("      retrodfm-r-8b.bf16.k20x10x10.retro.jsonl  (augmented)")
    sys.exit(3)

for path in FILES:
    base = os.path.basename(path)
    stem = base[:-len(".jsonl")] if base.endswith(".jsonl") else base
    parsed = split_name(stem)
    if parsed is None:
        continue
    model, level = parsed
    bits = level.split(".", 1)
    base, tag = bits[0], (bits[1] if len(bits) > 1 else "")
    if ONLY_LEVELS and base not in ONLY_LEVELS:
        continue
    if ONLY_TAG and not (tag == ONLY_TAG
                         or (ONLY_TAG.lower() == "none" and tag == "")):
        continue
    print(f"  ... {base:<9} {tag or '1x1x1':<12} reading {base}",
          file=sys.stderr, flush=True)
    all_rows = [json.loads(l) for l in open(path) if l.strip()]
    # PROJECT RULE: errored records leave the denominator with a visible count.
    keep = [r for r in all_rows
            if not str(r.get("stop_reason", "")).startswith("error")]
    n_err = len(all_rows) - len(keep)

    n = n_chg = 0
    hit = {k: 0 for k in KS}          # as-scored
    hitn = {k: 0 for k in KS}         # neutralised
    hit_c = {k: 0 for k in KS}        # as-scored, charged-truth subset
    hitn_c = {k: 0 for k in KS}       # neutralised, charged-truth subset
    flips = []                        # miss -> hit at top-1, for the examples
    for r in keep:
        raw = candidates_of(r)
        if not raw and "candidates" not in r:
            continue                  # unreadable, not a wrong answer
        n += 1
        truth_s = r["truth_key"]
        chg = has_charge(truth_s)
        if chg:
            n_chg += 1
        # SYNTHESEUS S2, both halves, in order: discard invalid, then dedup.
        # Dedup runs separately on each pass because neutralising can collapse
        # two distinct candidates into one.
        c_a = dedup([x for x in (key(v) for v in raw) if x is not None])
        c_n = dedup([x for x in (key(v, neutral=True) for v in raw)
                     if x is not None])
        t_a = key(truth_s)
        t_n = key(truth_s, neutral=True)
        for k in KS:
            a = t_a is not None and t_a in c_a[:k]
            b = t_n is not None and t_n in c_n[:k]
            if a:
                hit[k] += 1
                if chg: hit_c[k] += 1
            if b:
                hitn[k] += 1
                if chg: hitn_c[k] += 1
            if k == 1 and b and not a and len(flips) < 6:
                flips.append((c_a[0] if c_a else "", truth_s))
    rows.append(dict(model=model, level=level, n=n, n_err=n_err, n_chg=n_chg,
                     hit=hit, hitn=hitn, hit_c=hit_c, hitn_c=hitn_c,
                     flips=flips))

# An empty table below this line would print headers and no rows, which reads
# as "every level scored zero". score_sweep.py makes the same point about the
# same failure mode. Say plainly that nothing matched instead.
if not rows:
    print("NO FILE MATCHED THE FILTERS")
    print(f"  looked in     : {os.path.abspath(RAW)}")
    print(f"  files present : {len(FILES)}")
    for p in FILES:
        print(f"      {os.path.basename(p)}")
    if ONLY_LEVELS:
        print(f"  QB_NEU_LEVELS : {','.join(ONLY_LEVELS)}")
    if ONLY_TAG:
        print(f"  QB_NEU_TAG    : {ONLY_TAG}"
              + ("   (means files with NO k-tag, i.e. the 1x1x1 run)"
                 if ONLY_TAG.lower() == "none" else ""))
    print()
    print("  An unaugmented run is named without a k-tag:")
    print("      retrodfm-r-8b.bf16.retro.jsonl")
    print("  If that file is not in the list above, the 1x1x1 results are not")
    print("  in this folder -- they are wherever that run wrote them.")
    sys.exit(3)

W = 104
def pc(h, n):
    return 100.0 * h / n if n else 0.0

print("=" * W)
print("CHARGE STANDARDISATION -- what neutralising costs or buys, per level")
print("=" * W)
print("  as-scored    = score_sweep.py, no uncharging  (the numbers in the paper)")
print("  neutralised  = rdMolStandardize.Uncharger on every candidate AND the truth")
print()

for r in rows:
    tag = f"{r['model']}.{r['level']}"
    print("-" * W)
    print(f"{tag}    n = {r['n']}" + (f"   ({r['n_err']} errored, excluded)"
                                      if r['n_err'] else ""))
    print(f"  ground-truth reactant sets carrying a formal charge: "
          f"{r['n_chg']} of {r['n']} = {pc(r['n_chg'], r['n']):.2f}%"
          "   <- ceiling on the effect")
    print()
    print(f"  {'k':>3}  {'as-scored':>22}  {'neutralised':>22}  {'delta':>7}"
          f"   {'charged-truth subset':>24}")
    for k in KS:
        a, b = pc(r['hit'][k], r['n']), pc(r['hitn'][k], r['n'])
        lo, hi = wilson(a / 100, r['n'])
        ca = pc(r['hit_c'][k], r['n_chg'])
        cb = pc(r['hitn_c'][k], r['n_chg'])
        print(f"  {k:>3}  {a:7.2f}  [{lo:5.1f},{hi:5.1f}]  {b:21.2f}  "
              f"{b - a:+7.2f}   {ca:9.2f} -> {cb:6.2f}")
    if r['flips']:
        print()
        print("  examples rescued at top-1 (predicted  vs  truth):")
        for p, t in r['flips']:
            ps = ".".join(p) if isinstance(p, tuple) else str(p)
            print(f"    {ps[:44]:<44}  vs  {t[:44]}")
    print()

# The claim this whole script exists to test: does neutralising move any
# BETWEEN-LEVEL gap? Absolute accuracy shifting by half a point changes no
# conclusion; a gap between two levels shifting would change several.
print("=" * W)
print("BETWEEN-LEVEL GAPS -- the thing that would actually change a conclusion")
print("=" * W)
by_task = {}
for r in rows:
    task = r['level'].split(".", 1)[1] if "." in r['level'] else "1x1x1"
    by_task.setdefault(task, []).append(r)
for task, group in sorted(by_task.items()):
    if len(group) < 2:
        continue
    ref = group[0]
    print(f"\n  setting: {task}   reference level: {ref['level']}")
    print(f"  {'level':<14}" + "".join(f"{'top-'+str(k):>20}" for k in KS))
    for r in group[1:]:
        cells = []
        for k in KS:
            ga = pc(r['hit'][k], r['n']) - pc(ref['hit'][k], ref['n'])
            gb = pc(r['hitn'][k], r['n']) - pc(ref['hitn'][k], ref['n'])
            cells.append(f"{ga:+6.2f} -> {gb:+6.2f}".rjust(20))
        print(f"  {r['level']:<14}" + "".join(cells))
print()
print("  Read the arrows: left is the gap as the paper reports it, right is the")
print("  gap after neutralising both sides. If they agree, the ranking of the")
print("  quantization levels does not depend on this choice.")
print()
