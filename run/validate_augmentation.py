#!/usr/bin/env python3
"""Is OUR SMILES augmentation the same METHOD as the authors'?

    python validate_augmentation.py <their_test_file> [n_molecules]

WHY THIS EXISTS, AND WHY IT IS NOT CHEATING
-------------------------------------------
An independent reproduction must not consume the authors' generated inputs --
feeding their augmented SMILES into our pipeline and then reporting agreement
would be circular. So we reimplement the method and generate our own strings.

But a reimplementation has to be checked against something. This script uses
their released file ONCE, as a unit test of our generator: for the same product,
does our code produce the same set of spellings theirs did? Their data never
enters the results. It only answers "did we implement the method correctly".

WHAT IT REPORTS
    exact set match      our 20 strings == their 20 strings
    overlap              how many of theirs we also produce
    count                do we even produce as many as they do
    character            if the sets differ, are the strings the same KIND of
                         string (same length distribution, same root atoms)

READING THE RESULT
    high exact match  -> our generator is faithful; the augmentation is NOT the
                         cause of any accuracy gap, look elsewhere
    high overlap, low exact match
                      -> same method, different random draw. Fine: augmentation
                         is a sampling procedure, not a fixed list.
    low overlap       -> different method. The diagnostic output shows how.

Needs only rdkit and the authors' test file. No GPU, no model, seconds to run.
"""
import collections
import json
import re
import os
import random
import statistics as st
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SRC = sys.argv[1] if len(sys.argv) > 1 else ""
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
K = int(os.environ.get("QB_KA", "20"))
AUG = os.environ.get("QB_AUG", "random")


# --- our generator, copied verbatim from run_llm.py so the test tests the
# --- shipping code path and not a paraphrase of it
def roots_of(smi, k, seed):
    if k <= 1:
        return [smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    out = [Chem.MolToSmiles(mol)]
    seen = set(out)
    rng = random.Random(seed)
    order = list(range(mol.GetNumAtoms()))
    rng.shuffle(order)
    if AUG == "canonical":
        for r in order:
            if len(out) >= k:
                break
            try:
                s = Chem.MolToSmiles(mol, rootedAtAtom=int(r), canonical=True)
            except Exception:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:k]
    n_at = mol.GetNumAtoms()
    tries, limit = 0, max(k * 8, 64)
    while len(out) < k and tries < limit:
        tries += 1
        perm = list(range(n_at))
        rng.shuffle(perm)
        try:
            s = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        except Exception:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:k]


def unwrap(s):
    """Their rows hold the whole prompt, not a bare SMILES.

    The inference prompt is
        <SMILES> {product} </SMILES> Given the product SMILES, ...
    so pull the product out of the tag when it is there, and fall back to the
    raw string when it is not. Without this the column detector finds no
    SMILES-like column and the script gives up on a perfectly good file.
    """
    if not isinstance(s, str):
        return None
    m = re.search(r"<SMILES>\s*(.+?)\s*</SMILES>", s, re.S)
    return m.group(1) if m else s.strip()


def canon(s):
    s = unwrap(s)
    if not s or len(s) > 600:          # a paragraph is not a SMILES
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def raw_smiles(s):
    """The spelling AS WRITTEN, which is what we are comparing."""
    return unwrap(s)


def load_theirs(path):
    """Group their rows by canonical product -> list of spellings.

    Accepts jsonl, json, csv or parquet, and guesses the SMILES column, because
    the released layout is not documented and should not have to be.
    """
    rows = []
    if path.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    elif path.endswith(".csv") or path.endswith(".tsv"):
        import csv
        sep = "\t" if path.endswith(".tsv") else ","
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter=sep))
    else:
        with open(path) as f:
            head = f.read(2048)
            f.seek(0)
            if head.lstrip().startswith("["):
                rows = json.load(f)
            else:
                rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        sys.exit("no rows read")

    # Find the column holding the PRODUCT smiles: the one whose values parse as
    # molecules and that repeats ~k times per distinct canonical form.
    keys = [k for k in rows[0] if isinstance(rows[0][k], str)]
    best, best_score = None, -1
    for key in keys:
        vals = [r[key] for r in rows[:400] if isinstance(r.get(key), str)]
        cs = [canon(v) for v in vals]
        ok = [c for c in cs if c]
        if len(ok) < len(vals) * 0.9:
            continue
        rep = len(ok) / max(1, len(set(ok)))      # rows per distinct molecule
        if rep > best_score:
            best, best_score = key, rep
    if best is None:
        sys.exit(f"no SMILES-like column found. columns: {list(rows[0])}")
    print(f"  using column {best!r} (~{best_score:.1f} rows per distinct molecule)")

    g = collections.defaultdict(list)
    for r in rows:
        v = r.get(best)
        if not isinstance(v, str):
            continue
        c = canon(v)
        if c:
            g[c].append(raw_smiles(v))
    return g


if not SRC or not os.path.exists(SRC):
    sys.exit(__doc__.strip() + "\n\nGive the path to the authors' test file.")

theirs = load_theirs(SRC)
print(f"  {len(theirs)} distinct products, "
      f"{st.mean(len(v) for v in theirs.values()):.1f} spellings each\n")

exact = overlap = 0
ours_n, theirs_n, jac = [], [], []
picked = list(theirs.items())[:LIMIT]
for c, their_list in picked:
    our_list = roots_of(c, K, 1234)
    a, b = set(our_list), set(their_list)
    ours_n.append(len(a))
    theirs_n.append(len(b))
    if a == b:
        exact += 1
    overlap += len(a & b)
    jac.append(len(a & b) / max(1, len(a | b)))

n = len(picked)
print(f"compared {n} products, k_a={K}, QB_AUG={AUG}\n")
print(f"  spellings we produce   : {st.mean(ours_n):.1f}")
print(f"  spellings they produced: {st.mean(theirs_n):.1f}")
print(f"  exact set match        : {exact}/{n} ({100*exact/n:.1f}%)")
print(f"  mean overlap           : {overlap/n:.1f} strings per product")
print(f"  mean Jaccard           : {st.mean(jac):.3f}")

# If the sets differ, are they at least the same KIND of string?
ol = [len(s) for _, v in picked for s in v]
al = [len(s) for c, _ in picked for s in roots_of(c, K, 1234)]
print(f"\n  string length  ours {st.mean(al):.1f} +/- {st.pstdev(al):.1f}"
      f"   theirs {st.mean(ol):.1f} +/- {st.pstdev(ol):.1f}")
print("\n  Same mean length with low exact match means the same method with a")
print("  different random draw, which is expected and fine. A large length or")
print("  count difference means the methods differ.")
