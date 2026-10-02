#!/usr/bin/env python
"""How many DISTINCT valid candidates does each level actually return?

    python distinct_candidates.py <results>/raw

Igor's hypothesis (2026-09-26): bf16 loses at top-10 to the 4-bit levels
because it returns FEWER distinct reactant sets, so some molecules cannot fill
ten ranks at all. Top-10 is then capped by the candidate list, not by chemistry.

This reproduces score_sweep.py's pipeline EXACTLY -- same candidates_of(), same
key(), same dedup(), same Syntheseus S2 order (drop invalid FIRST, then
deduplicate) -- so the counts below are the same lists the accuracy table ranks
the truth against. Anything else would answer a different question.

Errored records are excluded, as in scoring: a failure has no candidate list.
"""
import glob, json, os, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw"

_cache = {}
def key(smi, stereo=True):
    if smi is None:
        return None
    s = smi.strip()
    if not s or any(c.isspace() for c in s):
        return None
    ck = (s, stereo)
    if ck in _cache:
        return _cache[ck]
    parts = []
    for frag in s.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            _cache[ck] = None
            return None
        parts.append(Chem.MolToSmiles(m, isomericSmiles=stereo))
    v = tuple(sorted(parts))
    _cache[ck] = v
    return v

def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

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
    return []

def median(v):
    if not v: return 0.0
    v = sorted(v); m = len(v) // 2
    return float(v[m]) if len(v) % 2 else (v[m-1] + v[m]) / 2.0

files = sorted(glob.glob(os.path.join(RAW, "*.retro.jsonl")))
if not files:
    sys.exit(f"no *.retro.jsonl under {RAW}")

rows = []
for fp in files:
    base = os.path.basename(fp)[:-len(".retro.jsonl")]
    model, _, level = base.partition(".")
    d, dns, n_raw, n_err, n_skip = [], [], 0, 0, 0
    for line in open(fp):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        sr = str(r.get("stop_reason", ""))
        if r.get("error") or sr.startswith("error"):
            n_err += 1; continue
        c = candidates_of(r)
        if not c:
            n_skip += 1
        n_raw += len(c)
        d.append(len(dedup([k for k in (key(x) for x in c) if k is not None])))
        dns.append(len(dedup([k for k in (key(x, stereo=False) for x in c)
                              if k is not None])))
    if not d: continue
    n = len(d)
    rows.append(dict(level=level, n=n, err=n_err, empty=n_skip,
                     raw=n_raw / n, mean=sum(d) / n, med=median(d),
                     mean_ns=sum(dns) / n,
                     lt3=sum(1 for x in d if x < 3),
                     lt5=sum(1 for x in d if x < 5),
                     lt10=sum(1 for x in d if x < 10),
                     mn=min(d), mx=max(d)))

W = max(len(r["level"]) for r in rows) + 1
print()
print("=" * (W + 84))
print("DISTINCT VALID CANDIDATES PER MOLECULE  (score_sweep pipeline: invalid "
      "dropped, then deduped)")
print("=" * (W + 84))
print(f"{'level':{W}s} {'n':>5s} {'returned':>9s} {'distinct':>9s} "
      f"{'median':>7s} {'min':>4s} {'max':>4s} {'<3':>10s} {'<5':>10s} {'<10':>10s}")
for r in rows:
    def pc(k): return f"{r[k]:4d} {100*r[k]/r['n']:4.1f}%"
    print(f"{r['level']:{W}s} {r['n']:5d} {r['raw']:9.2f} {r['mean']:9.2f} "
          f"{r['med']:7.1f} {r['mn']:4d} {r['mx']:4d} "
          f"{pc('lt3'):>10s} {pc('lt5'):>10s} {pc('lt10'):>10s}")
print()
print("  returned = raw predictions before validity filtering and dedup")
print("  distinct = mean size of the list top-k is computed over")
print("  <k       = molecules whose distinct list is SHORTER than k, so top-k")
print("             cannot be reached however good the chemistry is")
print()
print(f"{'level':{W}s} {'distinct (no stereo)':>21s}   errors  empty-candidate records")
for r in rows:
    print(f"{r['level']:{W}s} {r['mean_ns']:21.2f}   {r['err']:6d}  {r['empty']:6d}")
print()
