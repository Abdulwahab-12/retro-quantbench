#!/usr/bin/env python3
"""Rebuild the accuracy results of the paper from the stored predictions.

    python make_tables.py

Reads results/predictions/*.retro.jsonl.gz, scores them with
run/score_sweep.py, compares them with bf16 with run/paired_stats.py, counts
distinct candidates with run/distinct_candidates.py, and writes to
results/tables/

    table2.csv               Table 2 of the paper
    accuracy.csv             every prediction file and k, with 95% Wilson
                             intervals
    paired_differences.csv   Table S2 of the paper, for every k: differences
                             from bf16, McNemar and Holm-corrected p
    top10_restricted.csv     Top-k on the reactions for which bf16, fp4, nf4
                             and nf4dq all returned at least ten candidates
    seeds.csv                bf16 runs with three seeds
    distinct_candidates.csv  distinct candidates per reaction in the
                             augmented setting

All accuracies are computed after RDKit's Uncharger has been applied to both
the predictions and the ground truth. No GPU or model weights are needed; the
run takes a few minutes.
"""
import csv
import gzip
import json
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
LEVELS = ["bf16", "int8_ao", "fp4", "nf4", "nf4dq"]
AUGMENTED = ".k20x10x10"

sys.path.insert(0, str(SCRIPTS))
from score_sweep import wilson  # noqa: E402


def run(script, *args):
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
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


def setting_of(run_name):
    """bf16 -> 1x1x1, nf4.k20x10x10 -> 20x10x10, bf16.seed0695 -> 1x1x1."""
    for part in run_name.split(".")[1:]:
        if part[:1] == "k" and part[1:2].isdigit():
            return part[1:]
    return "1x1x1"


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
        print(f"scoring {len(files)} prediction files")
        print(run("score_sweep.py", raw))
        scored = json.loads((Path(tmp) / "scored.json").read_text())
        run("paired_stats.py", raw, TABLES)
        distinct = distinct_rows(run("distinct_candidates.py", raw))

    # accuracy.csv: every file; Top-3/5/10 only where several candidates exist
    runs = sorted((name[len(MODEL) + 1:] for name in scored),
                  key=lambda r: (setting_of(r) != "1x1x1", setting_of(r),
                                 LEVELS.index(r.split(".")[0])
                                 if r.split(".")[0] in LEVELS else 99, r))
    acc = {}
    with open(TABLES / "accuracy.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["run", "setting", "n", "k", "accuracy", "ci_low", "ci_high"])
        for r in runs:
            s = scored[f"{MODEL}.{r}"]
            ks = (1,) if setting_of(r) == "1x1x1" else (1, 3, 5, 10)
            for k in ks:
                p = s[f"top{k}"]
                lo, hi = wilson(p, s["n"])
                acc[(r, k)] = 100 * p
                w.writerow([r, setting_of(r), s["n"], k, f"{100 * p:.2f}",
                            f"{lo:.2f}", f"{hi:.2f}"])

    # table2.csv: the layout of Table 2
    with open(TABLES / "table2.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["level", "default_top1", "augmented_top1", "augmented_top3",
                    "augmented_top5", "augmented_top10"])
        for level in LEVELS:
            vals = [acc.get((level, 1))] + [acc.get((level + AUGMENTED, k))
                                            for k in (1, 3, 5, 10)]
            w.writerow([level] + ["" if v is None else f"{v:.1f}" for v in vals])

    # distinct_candidates.csv: augmented setting
    with open(TABLES / "distinct_candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["level", "n", "mean_distinct", "median_distinct",
                    "below_10", "at_least_10_percent"])
        for level in LEVELS:
            d = distinct.get(level + AUGMENTED)
            if d is None:
                continue
            w.writerow([level, d["n"], f"{d['mean']:.2f}", f"{d['median']:.1f}",
                        d["below_10"],
                        f"{100 * (d['n'] - d['below_10']) / d['n']:.1f}"])

    print(f"written to {TABLES.relative_to(ROOT)}/: table2.csv, accuracy.csv, "
          "paired_differences.csv, top10_restricted.csv, seeds.csv, "
          "distinct_candidates.csv")


if __name__ == "__main__":
    main()
