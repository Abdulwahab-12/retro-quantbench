#!/usr/bin/env python3
"""Score the quantization sweep.

Three things, in this order of importance:

  1. BASELINE GATE. Every fp32 row is checked against what we already know.
     Chemformer measured 51.0% on these same 200 molecules in the previous
     runs. If the new runner disagrees, nothing downstream is trusted.
  2. FIDELITY. How many top-1 predictions changed against that model's own
     fp32 reference. This is the primary metric: it is a paired count, so it
     is far tighter at n=200 than absolute accuracy.
  3. ACCURACY. top-1 / top-3 / top-5 / top-10 against ground truth, with a
     Wilson interval so nobody over-reads a 2-point difference.

Comparison is canonical and order-free: reactant fragments are canonicalised
with RDKit and sorted. Raw string comparison would report differences that are
not there -- the same failure mode as the ' + ' separator bug.

    python3 scripts/score_sweep.py results_sweep/raw
"""
import json, math, os, sys, glob
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')

RAW = sys.argv[1] if len(sys.argv) > 1 else "results_sweep/raw"
OUT = os.path.join(os.path.dirname(RAW.rstrip("/")), "scored.json")

# ---- CHARGE STANDARDISATION -----------------------------------------------
#
# Standardisation is part of the statistics, so it is computed here rather than
# in a separate script nobody remembers to run. The main table is unchanged: a
# second table below reports the same top-k with protonation normalised, so the
# size of that choice is visible instead of being argued about.
#
# Syntheseus best practice S2 prescribes four things -- discard invalid
# molecules, remove duplicate reactions, report stereochemistry both ways,
# report inference time -- and says nothing about charge. So neither convention
# violates it, which is exactly why the difference has to be measured.
#
# It costs a second canonicalisation of every candidate. On 5005 molecules at
# k_a x k_s x k_b = 20x10x10 that is a few minutes. QB_NEUTRALISE=0 turns it
# off for a quick re-score.
NEUTRALISE = os.environ.get("QB_NEUTRALISE", "1") not in ("0", "", "no")
_UNCHARGER = rdMolStandardize.Uncharger()

_cache = {}
def key(smi, stereo=True, neutral=False):
    """Canonical, order-free identity of one reactant set.

    stereo=False strips stereochemistry. Syntheseus §2.1.2 notes that in USPTO
    "stereochemistry is often unlabelled or mislabelled", so an exact-stereo
    match penalises models that predict the right disconnection with different
    or absent stereo annotation. Both variants are therefore reported.

    neutral=True additionally runs rdMolStandardize.Uncharger over each
    fragment, so a prediction of `N` matches a truth of `[NH4+]`. Permanent
    charges (nitro, azide, N-oxide) are correctly left alone by the Uncharger;
    only protonation states move.
    """
    if smi is None:
        return None
    s = smi.strip()
    if not s or any(c.isspace() for c in s):
        return None                      # interior whitespace -> reject
    ck = (s, stereo, neutral)
    if ck in _cache:
        return _cache[ck]
    parts = []
    for frag in s.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            _cache[ck] = None
            return None
        if neutral:
            try:
                m = _UNCHARGER.uncharge(m)
            except Exception:
                pass                     # leave it charged rather than drop it
        parts.append(Chem.MolToSmiles(m, isomericSmiles=stereo))
    v = tuple(sorted(parts))
    _cache[ck] = v
    return v

# Levels are matched against this list, not guessed by counting dots, because
# a MODEL name can contain a dot: chemdfm-v1.5-8b.q2_k.retro.jsonl split on the
# first dot gave model "chemdfm-v1" and level "5-8b.q2_k". The rows were still
# printed, under a model that does not exist, so nothing looked wrong.
BASE_LEVELS = {
    "fp32", "fp16", "bf16", "int8_ao", "int4_ao", "w8a8", "w4a8", "w4a4",
    "w4a8_intx", "fp8_ao", "fp8_w8a8", "fp8_w4a8", "nvfp4", "nvfp4_w",
    "mxfp8", "mxfp4", "nf4", "nf4dq", "fp4",
    "q2_k", "q3_k_m", "q3_k_s", "q4_0", "q4_1", "q4_k_m", "q4_k_s",
    "q5_0", "q5_k_m", "q5_k_s", "q6_k", "q8_0",
}


def split_name(stem):
    """<model>.<level>.<task> -> (model, level), tolerating dots in the model."""
    bits = stem.split(".")
    if len(bits) < 3:
        return None
    for i in range(1, len(bits) - 1):
        if bits[i] in BASE_LEVELS:
            return ".".join(bits[:i]), ".".join(bits[i:-1])
    return bits[0], ".".join(bits[1:-1])          # unknown level: old behaviour


def answer_of(text):
    """The <answer>...</answer> block, exactly as run_llm.py extracts it."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return ""


def candidates_of(r):
    """The ranked candidate list, whichever way the run wrote it.

    THIS IS THE BUG THAT MADE THE WHOLE GGUF LADDER READ 0.0%.

    Three record formats exist on disk:
      1. "candidates": [...]            what run_llm.py writes today
      2. "raw": "[\"CCO\", ...]"        a JSON list as a string
      3. "raw": "<think>...</think>\n<answer>\nCCO\n</answer>"
         the llama.cpp/GGUF runs -- NO "candidates" key at all

    The scorer only ever read key 1, so every format-3 record contributed an
    empty list, was scored as a miss, and the table reported 0.0% top-1 across
    every k-quant of RetroDFM-R -- 2087 correct-format predictions read as
    zero. A crash would have been kinder: 0.0% in a formatted table looks like
    a measurement.
    """
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
    """Order-preserving deduplication. First occurrence keeps its rank, which
    is what a search program would do: it takes the highest-ranked instance of
    a reaction and discards the rest."""
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

def mcnemar_exact(b, c):
    """Two-sided exact binomial p for discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

cells = {}
skipped = []          # files whose names this script could not interpret
unreadable = []       # files with records this script could not read
FILES = sorted(glob.glob(os.path.join(RAW, "*.jsonl")))
if not FILES:
    print("=" * 100)
    print("NOTHING TO SCORE")
    print("=" * 100)
    print(f"  looked in : {os.path.abspath(RAW)}")
    print(f"  exists    : {os.path.isdir(RAW)}")
    if os.path.isdir(RAW):
        here = sorted(os.listdir(RAW))
        print(f"  contains  : {len(here)} entries"
              + (f" -- {', '.join(here[:8])}" + (" ..." if len(here) > 8 else "")
                 if here else " (empty)"))
    print()
    print("  Expected files named  <model>.<level>.<task>.jsonl, e.g.")
    print("      retrodfm-r-8b.nf4dq.retro.jsonl")
    print("      retrodfm-r-8b.bf16.k20x1x10.retro.jsonl")
    print()
    print("  To re-score without re-running anything:")
    print("      bash qb.sh --score-only")
    print("  Results from an earlier sweep kept elsewhere:")
    print("      bash qb.sh --results <that folder> --score-only")
    print()
    print("  Empty tables below this line would be printed with no rows, which")
    print("  looks like every level scoring zero. It is not -- there is no data.")
    sys.exit(3)

for path in FILES:
    base = os.path.basename(path)
    # Filenames are model.LEVEL.task.jsonl, and LEVEL may itself contain dots:
    # fp16.greedy, fp16.seed999, fp16.k5x1x2. Taking split(".")[1] truncated
    # every variant to its base level, so three different runs all landed in
    # the "fp16" cell and the last one read won -- a 1000-molecule control
    # silently replaced the 5005-molecule reference in the table.
    #
    # A file that does NOT follow model.level.task.jsonl used to end the whole
    # run on this line with
    #     ValueError: not enough values to unpack (expected 2, got 1)
    # which names neither the file nor the convention. One stray
    # "predictions.jsonl" in a results directory therefore threw away the
    # scoring of every valid file beside it. Skip it, remember it, say so at
    # the end.
    stem = base[:-len(".jsonl")] if base.endswith(".jsonl") else base
    parsed = split_name(stem)
    if parsed is None:
        skipped.append((base, "expected <model>.<level>.<task>.jsonl"))
        continue
    model, level = parsed
    all_rows = [json.loads(l) for l in open(path) if l.strip()]
    # PROJECT RULE: records that ERRORED are excluded from the denominator with
    # a visible count -- never scored as wrong answers. LocalRetro OOM'd on 64
    # of 200 batches; counting those as failures read 31.5% instead of 46.3%
    # and would have looked like a broken model rather than a memory problem.
    rows = [r for r in all_rows if not str(r.get("stop_reason", "")).startswith("error")]
    n_err = len(all_rows) - len(rows)
    per = {}
    n_valid = n_raw = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    hits_ns = {1: 0, 3: 0, 5: 0, 10: 0}      # stereo-insensitive
    hits_nu = {1: 0, 3: 0, 5: 0, 10: 0}      # charge-standardised
    n_charged = 0                            # truths the Uncharger actually moves
    n_unreadable = 0
    for r in rows:
        raw = candidates_of(r)
        # A record whose writer INTENDED a candidate list and produced an empty
        # one is a genuine miss. A record with no candidates key at all is one
        # this script cannot read, and must not be counted as a wrong answer.
        if not raw and "candidates" not in r:
            n_unreadable += 1
            continue
        n_raw += len(raw)
        # SYNTHESEUS BEST PRACTICE S2 (§2.1.2): "best practice should be to only
        # consider valid molecules when computing top-k accuracy" -- a
        # well-engineered CASP program discards invalid outputs rather than
        # letting them occupy a slot. Previously an invalid prediction consumed
        # a rank and silently depressed top-k. Filter FIRST, then truncate.
        # SYNTHESEUS BEST PRACTICE S2 (§2.1.2), both halves, in order:
        #   1. drop INVALID molecules -- "a well-engineered CASP program would
        #      discard invalid molecules instead of considering them", so an
        #      unparseable prediction must not occupy a top-k slot;
        #   2. then DEDUPLICATE -- "a well-engineered CASP program would remove
        #      duplicate reactions, because they are redundant for the search".
        #      The paper reports GLN's published top-5 rising by 5.8% from
        #      deduplication alone.
        #
        # Deduplication was not needed at beam 10 (measured 0% duplicates), but
        # this study now requests 50 candidates from every model, and the
        # stereo-insensitive list duplicates by construction: two stereoisomers
        # of the same disconnection collapse to one key. Without this, those
        # collapsed entries would consume ranks and depress top-k.
        cands    = dedup([c for c in (key(x) for x in raw) if c is not None])
        cands_ns = dedup([c for c in (key(x, stereo=False) for x in raw) if c is not None])
        n_valid += len(cands)
        truth = key(r["truth_key"])
        truth_ns = key(r["truth_key"], stereo=False)
        # Store the RANK of the truth alongside the top-1 string. The old value
        # was the top-1 string alone, which supports "did the prediction
        # change" and nothing else -- so McNemar, the only correct test for
        # arms paired on the same molecules, could not be computed at any k and
        # was hard-coded to None a few lines below.
        rank = (cands.index(truth) + 1
                if truth is not None and truth in cands else None)
        per[r["uid"]] = (cands[0] if cands else None, rank)
        for k_ in hits:
            if truth is not None and truth in cands[:k_]:
                hits[k_] += 1
            if truth_ns is not None and truth_ns in cands_ns[:k_]:
                hits_ns[k_] += 1
        if NEUTRALISE:
            # Deduplication is re-run on the neutralised list, because
            # uncharging can collapse two candidates into one and a collapsed
            # entry must not go on occupying a rank.
            cands_nu = dedup([c for c in (key(x, neutral=True) for x in raw)
                              if c is not None])
            truth_nu = key(r["truth_key"], neutral=True)
            # The ceiling on the effect is not "truths carrying any charge" --
            # nitro and azide groups are permanently charged and can never be
            # rescued. It is the truths whose canonical form the Uncharger
            # actually changes.
            if truth is not None and truth_nu is not None and truth != truth_nu:
                n_charged += 1
            for k_ in hits_nu:
                if truth_nu is not None and truth_nu in cands_nu[:k_]:
                    hits_nu[k_] += 1
    rows = [r for r in rows
            if candidates_of(r) or "candidates" in r]   # drop the unreadable
    n = len(rows) or 1
    n_attempted = len(all_rows)
    if n_unreadable:
        unreadable.append((base, n_unreadable, len(all_rows)))
    cells[(model, level)] = {
        "n": n, "top1": hits[1] / n, "top3": hits[3] / n,
        "top5": hits[5] / n, "top10": hits[10] / n,
        "top1_nostereo": hits_ns[1] / n, "top10_nostereo": hits_ns[10] / n,
        "validity": n_valid / max(1, n_raw), "errors": n_err,
        "mean_candidates": n_raw / n, "mean_valid_candidates": n_valid / n,
        "n_attempted": n_attempted, "n_unreadable": n_unreadable, "per": per,
        "top1_neutral": (hits_nu[1] / n) if NEUTRALISE else None,
        "top3_neutral": (hits_nu[3] / n) if NEUTRALISE else None,
        "top5_neutral": (hits_nu[5] / n) if NEUTRALISE else None,
        "top10_neutral": (hits_nu[10] / n) if NEUTRALISE else None,
        "n_charged": n_charged if NEUTRALISE else None,
    }

# ---- fidelity and significance against each model's full-precision arm ----
#
# TWO BUGS LIVED HERE.
#
#  1. The reference was hard-coded to level "fp32". RetroDFM-R-8B has no fp32
#     arm -- its full-precision reference is bf16 -- so `ref` was None on every
#     RetroDFM row and BOTH columns came out blank. That is why the table
#     circulated on 2026-09-13 had an empty `changed` column on every line.
#  2. mcnemar_p was assigned None unconditionally, three lines after the
#     discordant counters were initialised. The test was never computed, for
#     any model, ever.
#
# This is worse than a blank column. These arms are PAIRED on the same
# molecules, so reading two overlapping 95% CIs as "not significantly
# different" is the wrong test applied to the wrong data: McNemar looks only at
# the molecules where the two arms disagree and can return a significant
# difference while the CIs overlap almost completely.
def _ref_for(model, level):
    """The full-precision arm AT THE SAME BUDGET.

    Levels carry their budget, so `nf4.k20x10x10` must pair with
    `bf16.k20x10x10` and never with `bf16.k20x1x2` -- that would charge a 50x
    compute difference to the quantizer.
    """
    base, _, suffix = level.partition(".")
    if base in ("fp32", "bf16", "fp16"):
        return None, None                     # this row IS a reference
    for cand in ("fp32", "bf16", "fp16"):
        k = (model, cand + ("." + suffix if suffix else ""))
        if k in cells:
            return cells[k], k[1]
    return None, None


for (model, level), c in cells.items():
    ref, ref_name = _ref_for(model, level)
    c["reference"] = ref_name
    if ref is None:
        c["changed"] = c["mcnemar_p"] = c["mcnemar"] = None
        continue
    shared = [u for u in c["per"] if u in ref["per"]]
    c["n_paired"] = len(shared)
    c["changed"] = sum(1 for u in shared
                       if c["per"][u][0] != ref["per"][u][0])

    def _hit(cell, uid, k_):
        rank = cell["per"][uid][1]
        return rank is not None and rank <= k_

    mc = {}
    for k_ in (1, 3, 5, 10):
        b = sum(1 for u in shared if _hit(c, u, k_) and not _hit(ref, u, k_))
        d = sum(1 for u in shared if _hit(ref, u, k_) and not _hit(c, u, k_))
        mc[f"top{k_}"] = {"quant_wins": b, "ref_wins": d,
                          "discordant": b + d, "p": mcnemar_exact(b, d)}
    c["mcnemar"] = mc
    c["mcnemar_p"] = mc["top1"]["p"]      # the old key, for anything reading it

W = 100
if skipped:
    print("=" * W)
    print("FILES SKIPPED -- name not understood (they were NOT scored)")
    print("=" * W)
    for b, why in skipped:
        print(f"  {b:50s} {why}")
    print()
if unreadable:
    print("=" * W)
    print("RECORDS THIS SCRIPT COULD NOT READ -- excluded, NOT counted as wrong")
    print("=" * W)
    print("  No 'candidates' key and no <answer> block in 'raw'. These rows are")
    print("  removed from the denominator. They are NOT scored as misses --")
    print("  doing that is what made a whole ladder read 0.0%.")
    for b, k, tot in unreadable:
        print(f"  {b:52s} {k:5d} of {tot:5d} unreadable")
    print()
if not cells:
    print("=" * W)
    print("NOTHING TO SCORE -- every file in this directory was skipped above.")
    print("=" * W)
    print("  Rename them to <model>.<level>.<task>.jsonl, e.g.")
    print("      predictions.jsonl  ->  retrodfm-r-8b.nf4dq.retro.jsonl")
    sys.exit(3)

print("=" * W)
print("BASELINE GATE -- fp32 rows must reproduce what we already know")
print("=" * W)
KNOWN = {"chemformer": 51.0}
gate_ok = True
for (model, level), c in sorted(cells.items()):
    if level != "fp32":
        continue
    lo, hi = wilson(c["top1"], c["n"])
    note = ""
    if model in KNOWN:
        exp = KNOWN[model]
        inside = lo <= exp <= hi
        note = f"   expected {exp:.1f}  ->  {'MATCH' if inside else 'MISMATCH'}"
        gate_ok &= inside
    errs = f"  ({c['errors']} errors excluded of {c['n_attempted']})" if c["errors"] else ""
    print(f"  {model:18s} top-1 {100*c['top1']:5.1f}%  [{lo:4.1f}, {hi:4.1f}]  "
          f"n={c['n']:3d}  valid {100*c['validity']:5.1f}%{note}{errs}")
print(f"\n  GATE: {'PASS' if gate_ok else 'FAIL -- do not trust deltas yet'}\n")

print("=" * W)
print("CANDIDATE BUDGET -- must be equal across models to compare top-k")
print("=" * W)
print("Syntheseus §2.1.2: inconsistent post-processing distorts comparisons.")
print("A model returning 2 candidates cannot be compared at top-10 with one")
print("returning 10.\n")
print(f"  {'model':18s} {'n':>6s} {'mean returned':>14s} {'mean valid':>11s}")
for model in sorted({m for m, _ in cells}):
    # fp32 is the natural row to quote the budget from, but the 8B LLM has no
    # fp32 cell -- 32 GB of weights do not fit a 32 GB card -- so this section
    # printed nothing at all for it. Fall back to whatever level exists.
    c = cells.get((model, "fp32"))
    if c is None:
        for _lvl in ("bf16", "fp16"):
            c = cells.get((model, _lvl))
            if c:
                break
    if c is None:
        _avail = sorted(l for m, l in cells if m == model)
        c = cells.get((model, _avail[0])) if _avail else None
    if c:
        flag = "  <-- UNDER BUDGET" if c["mean_candidates"] < 8 else ""
        print(f"  {model:18s} {c['n']:6d} {c['mean_candidates']:14.2f} "
              f"{c['mean_valid_candidates']:11.2f}{flag}")

dropped = []
print("\n" + "=" * W)
print("FULL TABLE   (top-k over VALID candidates only, per Syntheseus S2)")
print("=" * W)
# top-3 is computed but was never printed. The RetroDFM-R paper's Table 2 has
# columns 1 / 3 / 5 / 10, so omitting top-3 meant our table could not be laid
# beside theirs without going back to scored.json for the missing column.
# n sits immediately after the level because it qualifies every number to its
# right. Without it a 10-molecule smoke test and a 5005-molecule run print rows
# of identical shape, and the reader cannot tell a result from a rehearsal --
# which is exactly what happened when a k20x1x2 test on 10 molecules was read
# as a finished measurement.
print(f"{'model':18s} {'level':16s} {'n':>6s} {'top1':>6s} {'95% CI':>14s} "
      f"{'top3':>6s} {'top5':>6s} {'top10':>6s} {'t1 nostereo':>11s} "
      f"{'t10 nostereo':>12s} {'changed':>9s} {'excluded':>9s}")
for model in sorted({m for m, _ in cells}):
    # A hard-coded order silently DROPS any level not on the list. Once levels
    # gained suffixes (fp16.greedy, fp16.seed999, fp16.k5x1x2) and bitsandbytes
    # arms appeared, those rows would simply not have been printed.
    known = ["fp32", "bf16", "fp16", "int8_ao", "int4_ao",
             "w8a8", "w4a8", "w4a4", "nf4", "nf4dq", "fp4"]
    present = [l for m, l in cells if m == model]
    order = ([l for l in known if l in present]
             + sorted(l for l in present if l not in known))
    for level in order:
        c = cells.get((model, level))
        if not c:
            continue
        lo, hi = wilson(c["top1"], c["n"])
        ch = "" if c["changed"] is None else f"{c['changed']}/{c['n']}"
        # n is what was SCORED. n_attempted is what the file held. The gap is
        # errored or unreadable records, dropped on purpose -- never counted as
        # wrong answers. But dropping them silently is how "merged 30 records"
        # becomes "n 21" with nothing explaining the other nine.
        _exc = c["n_attempted"] - c["n"]
        exc = f"{_exc} of {c['n_attempted']}" if _exc else ""
        if _exc:
            dropped.append((model, level, _exc, c["n_attempted"]))
        print(f"{model:18s} {level:16s} {c['n']:>6d} {100*c['top1']:5.1f}% "
              f"[{lo:5.1f},{hi:5.1f}] "
              f"{100*c['top3']:5.1f}% {100*c['top5']:5.1f}% {100*c['top10']:5.1f}% "
              f"{100*c['top1_nostereo']:10.1f}% {100*c['top10_nostereo']:11.1f}% "
              f"{ch:>9s} {exc:>9s}")
    print()

if dropped:
    print("=" * W)
    print("EXCLUDED RECORDS -- read this before quoting any row above")
    print("=" * W)
    for m, l, k, tot in dropped:
        frac = 100.0 * k / max(1, tot)
        flag = "   <-- OVER 10%: treat this row as provisional" if frac > 10 else ""
        print(f"  {m}.{l}: {k} of {tot} dropped ({frac:.0f}%){flag}")
    print()
    print("  Errored records are excluded rather than scored as wrong answers,")
    print("  which is correct -- an out-of-memory failure is not a bad")
    print("  prediction. But the molecules that fail are not a random subset:")
    print("  on a memory-constrained card the LONGEST reasoning traces are the")
    print("  ones that run out, and long traces are the hard molecules. So the")
    print("  surviving sample is easier than the full set and the accuracy")
    print("  above is biased UPWARD. Check the reasons with:")
    print("     grep -o '\"stop_reason\": \"[^\"]*\"' <the .jsonl> | sort | uniq -c")
    print()

# ---- charge standardisation, printed ---------------------------------------
_neu = [(m, l, c) for (m, l), c in sorted(cells.items())
        if c.get("top1_neutral") is not None]
if _neu:
    print("=" * W)
    print("CHARGE STANDARDISATION -- the table above with protonation normalised")
    print("=" * W)
    print("  Above, a prediction of N against a truth of [NH4+] is a miss.")
    print("  Here rdMolStandardize.Uncharger is applied to every candidate AND")
    print("  to the ground truth before canonicalisation, with the same")
    print("  invalid-discard, deduplication and truncation. Syntheseus S2 is")
    print("  silent on charge, so neither convention violates it -- which is")
    print("  why the size of the difference is reported rather than assumed.")
    print()
    print("  'moved' counts the ground truths the Uncharger actually changes;")
    print("  permanent charges (nitro, azide, N-oxide) are left alone and can")
    print("  never be rescued, so that count, not the number of charged truths,")
    print("  is the ceiling on the effect.")
    print()
    print(f"  {'model.level':<32}{'n':>6}{'moved':>7}   "
          + "".join(f"{'top-' + str(k):>19}" for k in (1, 3, 5, 10)))
    for m, l, c in _neu:
        txt = ""
        for k_ in (1, 3, 5, 10):
            a, b = 100 * c[f"top{k_}"], 100 * c[f"top{k_}_neutral"]
            txt += f"{a:6.2f}->{b:6.2f}{b - a:+5.2f}"
        print(f"  {m + '.' + l:<32}{c['n']:>6}{c['n_charged']:>7}   {txt}")
    print()
    print("  A uniform shift changes no comparison between levels; a shift that")
    print("  differs between levels does. Read the deltas against each other,")
    print("  not against zero.")
    print()
elif not NEUTRALISE:
    print("  (QB_NEUTRALISE=0 -- charge-standardised table not computed)\n")

# ---- the paired comparison, printed ---------------------------------------
paired = [(m, l, c) for (m, l), c in sorted(cells.items()) if c.get("mcnemar")]
if paired:
    print("=" * W)
    print("QUANTIZED vs FULL PRECISION -- McNemar on PAIRED molecules")
    print("=" * W)
    print("  These arms answer the SAME molecules, so this is the test, not the")
    print("  overlap of two confidence intervals. Each cell is "
          "wins/losses (p).")
    print()
    print(f"  {'model.level':<34}{'vs':<18}{'n':>6}  "
          f"{'top-1':>16}{'top-3':>16}{'top-5':>16}{'top-10':>16}")
    for m, l, c in paired:
        cells_txt = ""
        for k_ in (1, 3, 5, 10):
            e = c["mcnemar"][f"top{k_}"]
            star = "*" if e["p"] < 0.05 else " "
            cells_txt += f"{e['quant_wins']:>4}/{e['ref_wins']:<4}{e['p']:>6.3f}{star}"
        print(f"  {m + '.' + l:<34}{c['reference']:<18}{c.get('n_paired', 0):>6}  "
              f"{cells_txt}")
    print()
    print("  wins = molecules the QUANTIZED arm gets right and the reference")
    print("  does not; losses = the reverse. * marks p < 0.05. Ties are not")
    print("  counted -- that is what makes the test paired.")
    print()

json.dump({f"{m}.{l}": {k: v for k, v in c.items() if k != "per"}
           for (m, l), c in cells.items()}, open(OUT, "w"), indent=2)
print(f"written: {OUT}")
