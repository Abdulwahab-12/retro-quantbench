#!/usr/bin/env python3
"""Rebuild the accuracy results of the paper from the stored predictions.

    python make_tables.py

Reads results/predictions/*.retro.jsonl.gz, scores them with
run/score_sweep.py, counts distinct candidates with run/distinct_candidates.py,
and writes

    results/tables/table1.csv               Table 1 of the paper
    results/tables/accuracy.csv             every level, setting and k, with
                                            95% Wilson intervals
    results/tables/distinct_candidates.csv  distinct candidates per molecule
                                            in the 20x10x10 setting

All accuracies are computed after RDKit's Uncharger has been applied to both
the predictions and the ground truth, over the reactions present in each file;
a file with fewer reactions than the test set is named after the table. No GPU or model weights are needed; the run
takes a few minutes.
"""
import csv
import gzip
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "results" / "predictions"
TABLES = ROOT / "results" / "tables"
SCRIPTS = ROOT / "run"
MODEL = "retrodfm-r-8b"
LEVELS = ["bf16", "int8_ao", "nf4", "fp4", "nf4dq"]
DEFAULT = ""                 # 1x1x1 files carry no setting in their name
AUGMENTED = ".k20x10x10"


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def run(script, folder):
    r = subprocess.run([sys.executable, str(SCRIPTS / script), str(folder)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{script} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def distinct_rows(text):
    """Rows of the first table printed by distinct_candidates.py."""
    rows = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) == 13 and t[1].isdigit():
            rows[t[0]] = {"n": int(t[1]), "mean": float(t[3]),
                          "median": float(t[4]), "below_10": int(t[11])}
    return rows


def fmt(x):
    return "" if x is None else f"{x:.1f}"


def main():
    files = sorted(PREDICTIONS.glob("*.retro.jsonl.gz"))
    if not files:
        sys.exit(f"no predictions in {PREDICTIONS}")
    TABLES.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        raw.mkdir()
        for f in files:
            with gzip.open(f, "rb") as src, open(raw / f.name[:-3], "wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"scoring {len(files)} prediction files with run/score_sweep.py")
        run("score_sweep.py", raw)
        scored = json.loads((Path(tmp) / "scored.json").read_text())
        print("counting distinct candidates with run/distinct_candidates.py")
        distinct = distinct_rows(run("distinct_candidates.py", raw))

    # accuracy.csv: one row per level, setting and k
    acc, ns = {}, {}
    with open(TABLES / "accuracy.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["level", "setting", "n", "k", "accuracy", "ci_low",
                    "ci_high"])
        for level in LEVELS:
            for tag, setting, ks in ((DEFAULT, "1x1x1", (1,)),
                                     (AUGMENTED, "20x10x10", (1, 3, 5, 10))):
                s = scored.get(f"{MODEL}.{level}{tag}")
                if s is None:
                    continue
                if s["errors"]:
                    print(f"  note: {level} {setting} has {s['errors']} errored records")
                n = s["n"]
                ns[(level, setting)] = n
                for k in ks:
                    p = s[f"top{k}_neutral"]
                    lo, hi = wilson(p, n)
                    acc[(level, setting, k)] = 100 * p
                    w.writerow([level, setting, n, k, f"{100 * p:.2f}",
                                f"{lo:.2f}", f"{hi:.2f}"])

    # table1.csv: the layout of Table 1
    header = ["level", "default_top1", "augmented_top1", "augmented_top3",
              "augmented_top5", "augmented_top10"]
    table = []
    for level in LEVELS:
        row = [acc.get((level, "1x1x1", 1))]
        row += [acc.get((level, "20x10x10", k)) for k in (1, 3, 5, 10)]
        table.append([level] + row)
    with open(TABLES / "table1.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in table:
            w.writerow([r[0]] + [fmt(x) for x in r[1:]])

    # distinct_candidates.csv: 20x10x10 setting
    with open(TABLES / "distinct_candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["level", "n", "mean_distinct", "median_distinct",
                    "below_10", "at_least_10_percent"])
        for level in LEVELS:
            d = distinct.get(level + AUGMENTED)
            if d is None:
                continue
            w.writerow([level, d["n"], f"{d['mean']:.2f}", f"{d['median']:.1f}",
                        d["below_10"], f"{100 * (d['n'] - d['below_10']) / d['n']:.1f}"])

    print()
    print("Table 1. Top-k accuracy (%), USPTO-50K test split, n = 5005")
    print()
    print(f"{'':10s}{'default':>9s}   augmented")
    print(f"{'level':10s}{'top-1':>9s}   {'top-1':>6s}{'top-3':>7s}{'top-5':>7s}{'top-10':>8s}")
    for r in table:
        v = ["-" if x is None else f"{x:.1f}" for x in r[1:]]
        print(f"{r[0]:10s}{v[0]:>9s}   {v[1]:>6s}{v[2]:>7s}{v[3]:>7s}{v[4]:>8s}")
    n_test = sum(1 for line in open(ROOT / "data" / "subset_full.jsonl") if line.strip())
    for (level, setting), n in ns.items():
        if n != n_test:
            print(f"{level} {setting}: calculated using {n} of the {n_test} reactions")
    print()
    print(f"written to {TABLES.relative_to(ROOT)}/: table1.csv, accuracy.csv, "
          "distinct_candidates.csv")


if __name__ == "__main__":
    main()
