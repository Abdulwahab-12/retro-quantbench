#!/usr/bin/env python3
"""Top-k accuracy of every prediction file in a folder.

    python3 score_sweep.py results/raw

A prediction counts as correct when its reactant set equals the ground truth
after both have been charge-standardised with RDKit's Uncharger and
canonicalised with RDKit (fragments in any order). Invalid SMILES are removed
and duplicates merged before the ranks are counted. Records that stopped with
an error are left out of n.

Prints one table (top-1 with its Wilson 95% interval, top-3, top-5, top-10)
and writes scored.json next to the folder.
"""
import glob
import json
import math
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

KS = (1, 3, 5, 10)
_UNCHARGER = rdMolStandardize.Uncharger()
_cache = {}

# Level names, so that a model name containing a dot is not split wrongly.
BASE_LEVELS = {
    "fp32", "fp16", "bf16", "int8_ao", "int4_ao", "fp8_ao", "int2_ao",
    "w8a8", "w4a8", "w4a4", "w4a8_intx", "w2a8_intx", "fp8_w8a8", "fp8_w4a8",
    "nvfp4", "nvfp4_w", "mxfp8", "mxfp4", "nf4", "nf4dq", "fp4",
}
ORDER = ["fp32", "bf16", "fp16", "int8_ao", "int4_ao", "nf4", "nf4dq", "fp4"]


def key(smi):
    """Canonical, order-free, charge-standardised identity of a reactant set."""
    if smi is None:
        return None
    s = smi.strip()
    if not s or any(c.isspace() for c in s):
        return None
    if s in _cache:
        return _cache[s]
    parts = []
    for frag in s.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            _cache[s] = None
            return None
        try:
            m = _UNCHARGER.uncharge(m)
        except Exception:
            pass
        parts.append(Chem.MolToSmiles(m))
    v = tuple(sorted(parts))
    _cache[s] = v
    return v


def split_name(stem):
    """<model>.<level>.<task> -> (model, level); None if the name does not fit."""
    bits = stem.split(".")
    if len(bits) < 3:
        return None
    for i in range(1, len(bits) - 1):
        if bits[i] in BASE_LEVELS:
            return ".".join(bits[:i]), ".".join(bits[i:-1])
    return bits[0], ".".join(bits[1:-1])


def candidates_of(r):
    """The ranked candidate list of one record."""
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
        if "<answer>" in raw and "</answer>" in raw:
            a = raw.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
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
        return 0.0, 0.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def score_file(path):
    """Per-molecule result of one file: {uid: (rank of the truth or None,
    number of distinct valid candidates)}, plus the number of records."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    per = {}
    for r in rows:
        if str(r.get("stop_reason", "")).startswith("error"):
            continue
        raw = candidates_of(r)
        if not raw and "candidates" not in r:
            continue                      # unreadable record, not a wrong answer
        cands = dedup([c for c in (key(x) for x in raw) if c is not None])
        truth = key(r["truth_key"])
        rank = cands.index(truth) + 1 if truth is not None and truth in cands else None
        per[r["uid"]] = (rank, len(cands))
    return per, len(rows)


def score_folder(raw_dir):
    """{(model, level): per-molecule results} for every file in raw_dir."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        stem = os.path.basename(path)[:-len(".jsonl")]
        parsed = split_name(stem)
        if parsed is None:
            print(f"skipped {os.path.basename(path)}: expected "
                  "<model>.<level>.<task>.jsonl", file=sys.stderr)
            continue
        per, n_records = score_file(path)
        cells[parsed] = {"per": per, "n_records": n_records}
    return cells


def topk(per, k):
    n = len(per)
    return sum(1 for rank, _ in per.values() if rank is not None and rank <= k) / n if n else 0.0


def main(raw_dir):
    cells = score_folder(raw_dir)
    if not cells:
        print(f"nothing to score in {os.path.abspath(raw_dir)}")
        sys.exit(3)

    print(f"{'model':16s} {'level':22s} {'n':>5s} {'top1':>6s} {'95% CI':>14s} "
          f"{'top3':>6s} {'top5':>6s} {'top10':>6s}")
    scored = {}
    for model in sorted({m for m, _ in cells}):
        levels = [l for m, l in cells if m == model]
        levels.sort(key=lambda l: (ORDER.index(l.split(".")[0])
                                   if l.split(".")[0] in ORDER else len(ORDER), l))
        for level in levels:
            c = cells[(model, level)]
            n = len(c["per"])
            if n == 0:
                continue                  # every record failed: nothing to score
            acc = {k: topk(c["per"], k) for k in KS}
            lo, hi = wilson(acc[1], n)
            print(f"{model:16s} {level:22s} {n:5d} {100 * acc[1]:5.1f}% "
                  f"[{lo:5.1f},{hi:5.1f}] {100 * acc[3]:5.1f}% "
                  f"{100 * acc[5]:5.1f}% {100 * acc[10]:5.1f}%")
            scored[f"{model}.{level}"] = {
                "n": n, "errors": c["n_records"] - n,
                **{f"top{k}": acc[k] for k in KS}}

    out = os.path.join(os.path.dirname(os.path.abspath(raw_dir)), "scored.json")
    with open(out, "w") as f:
        json.dump(scored, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
