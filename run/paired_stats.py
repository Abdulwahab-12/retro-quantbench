#!/usr/bin/env python3
"""Paired comparisons with bf16, as reported in the paper.

    python3 paired_stats.py <folder with .jsonl predictions> <output folder>

Every prediction is scored as in score_sweep.py (charge-standardised). Writes

  paired_differences.csv  int8_ao, nf4, nf4dq and fp4 against bf16 on the same
                          reactions: b = reactions only the quantized model got
                          right, c = reactions only bf16 got right, difference
                          in percentage points with its 95% interval (and the
                          90% interval used for the equivalence test), exact
                          McNemar p and Holm-corrected p. The Holm family is
                          the 4 Top-1 tests in the default setting and the 16
                          tests (Top-1, 3, 5, 10) in the augmented setting.
  top10_restricted.csv    Top-k in the augmented setting on the reactions for
                          which bf16, fp4, nf4 and nf4dq all returned at least
                          ten distinct candidates.
  seeds.csv               Top-1 of the bf16 runs with other seeds, and the
                          number of reactions whose result changed against the
                          bf16 run with seed 1234 (default setting).
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_sweep import score_folder, topk  # noqa: E402

KS = (1, 3, 5, 10)
QUANT = ["int8_ao", "nf4", "nf4dq", "fp4"]
SETTINGS = [("default", "", (1,)), ("augmented", ".k20x10x10", KS)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired_ci(b, c, n, z):
    """Difference of paired proportions with a Wald interval, in points."""
    d = (b - c) / n
    se = math.sqrt(max(0.0, (b + c) / n - d * d) / n)
    return 100 * d, 100 * (d - z * se), 100 * (d + z * se)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def hit(per, uid, k):
    rank = per[uid][0]
    return rank is not None and rank <= k


def discordant(q, ref, k):
    shared = [u for u in q if u in ref]
    b = sum(1 for u in shared if hit(q, u, k) and not hit(ref, u, k))
    c = sum(1 for u in shared if hit(ref, u, k) and not hit(q, u, k))
    return len(shared), b, c


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main(raw_dir, out_dir):
    cells = {level: c["per"] for (_, level), c in score_folder(raw_dir).items()}
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for setting, tag, ks in SETTINGS:
        ref = cells.get("bf16" + tag)
        if ref is None:
            continue
        block = []
        for level in QUANT:
            q = cells.get(level + tag)
            if q is None:
                continue
            for k in ks:
                n, b, c = discordant(q, ref, k)
                d, lo, hi = paired_ci(b, c, n, 1.96)
                _, lo90, hi90 = paired_ci(b, c, n, 1.645)
                block.append([level, setting, k, n, b, c, d, lo, hi, lo90, hi90,
                              mcnemar_exact(b, c)])
        for r, p_holm in zip(block, holm([r[-1] for r in block])):
            level, setting_, k, n, b, c, d, lo, hi, lo90, hi90, p = r
            rows.append([level, setting_, k, n, b, c, f"{d:+.2f}", f"{lo:+.2f}",
                         f"{hi:+.2f}", f"{lo90:+.2f}", f"{hi90:+.2f}",
                         f"{p:.2e}", f"{p_holm:.2e}"])
    write(os.path.join(out_dir, "paired_differences.csv"),
          ["level", "setting", "k", "n", "b_only_level_correct",
           "c_only_bf16_correct", "difference_pp", "ci95_low", "ci95_high",
           "ci90_low", "ci90_high", "mcnemar_p", "holm_p"], rows)

    rows = []
    group = [l + ".k20x10x10" for l in ("bf16", "fp4", "nf4", "nf4dq")]
    if all(g in cells for g in group):
        common = set.intersection(*[{u for u, (_, nd) in cells[g].items() if nd >= 10}
                                    for g in group])
        for g in group:
            sub = {u: cells[g][u] for u in common}
            rows.append([g.split(".")[0], len(common)]
                        + [f"{100 * topk(sub, k):.2f}" for k in KS])
    write(os.path.join(out_dir, "top10_restricted.csv"),
          ["level", "n"] + [f"top{k}" for k in KS], rows)

    rows = []
    ref = cells.get("bf16")
    if ref is not None:
        runs = ["bf16"] + sorted(l for l in cells if l.startswith("bf16.seed"))
        for level in runs:
            n, b, c = discordant(cells[level], ref, 1)
            seed = level.split("seed")[1] if "seed" in level else "1234"
            rows.append([level, int(seed), len(cells[level]),
                         f"{100 * topk(cells[level], 1):.2f}",
                         "" if level == "bf16" else b, "" if level == "bf16" else c,
                         "" if level == "bf16" else b + c])
    write(os.path.join(out_dir, "seeds.csv"),
          ["run", "seed", "n", "top1", "gained_vs_seed1234",
           "lost_vs_seed1234", "changed_vs_seed1234"], rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
