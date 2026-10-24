"""Paired analysis of the M38b summaries (seeds 1-8).

Headline cell fixed from the seed-0 pilot BEFORE seeds 1-8 were seen: sampling at R=16 loops,
eval noise sigma=0.4, K=32 samples; its compute-matched control is the deterministic loop at
R' = 32 * 16 = 512. Everything else is reported as secondary.
Usage: python experiments/m38_ttc/analyse.py results/m38b_noise_k_select_g30_*_summary.csv ...
"""

import csv
import sys
from math import comb

import numpy as np


def sign_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return pos, neg, p


def load(path):
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = (r["arm"], r["mode"], r["R"], r["sigma"], r["strategy"], r["K"])
        em = np.array([float(x) for x in r["em_per_seed"].split(";")])
        rows[key] = (float(r["acc_mean"]), em)
    return rows


def det(rows, arm, R):
    return rows[(arm, "deterministic", str(R), "0", "-", "1")]


def samp(rows, arm, R, sigma, strat, K=32):
    return rows[(arm, "sampled", str(R), f"{sigma:g}", strat, str(K))]


def line(name, a, b):
    pos, neg, p = sign_p(a, b)
    print(f"  {name:<62} {a.mean():.3f} vs {b.mean():.3f}  Δ {a.mean() - b.mean():+.3f}  "
          f"{pos}/{neg} p={p:.4f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for path in sys.argv[1:]:
        rows = load(path)
        print(f"\n=== {path}")
        mx, mxn = "trm_mixer", "trm_mixer_noise0.3"
        un, unn = "untied_mixer_matched", "untied_mixer_matched_noise0.3"
        ff, ffn = "ff_matched", "ff_matched_noise0.3"
        print("Deterministic EM by loops R' (mean):")
        for arm in (mx, mxn):
            print(f"  {arm:<22}", " ".join(f"{R}:{det(rows, arm, R)[1].mean():.3f}"
                                          for R in (8, 16, 32, 64, 128, 256, 512)))
        print("HEADLINE (R=16, sigma=0.4, K=32 vs deterministic R'=512), exact match:")
        for arm in (mx, mxn):
            for strat in ("violation", "confidence", "vote"):
                line(f"{arm} {strat} K32 vs det 512", samp(rows, arm, 16, 0.4, strat)[1],
                     det(rows, arm, 512)[1])
        print("Training noise (deterministic, trained depth R=8 and plateau R'=512):")
        line("mixer noise vs mixer, R=8", det(rows, mxn, 8)[1], det(rows, mx, 8)[1])
        line("mixer noise vs mixer, R'=512", det(rows, mxn, 512)[1], det(rows, mx, 512)[1])
        print("Same sampling on the controls (R=8 only for untied/ff; sigma=0.4, K=32, violation):")
        for arm in (mx, mxn, un, unn, ff, ffn):
            a = samp(rows, arm, 8, 0.4, "violation")[1]
            print(f"  {arm:<30} EM {a.mean():.3f}")
        line("tied noise vs untied noise, R=8 sampled", samp(rows, mxn, 8, 0.4, "violation")[1],
             samp(rows, unn, 8, 0.4, "violation")[1])
        line("tied vs untied, R=8 sampled", samp(rows, mx, 8, 0.4, "violation")[1],
             samp(rows, un, 8, 0.4, "violation")[1])
        print("Secondary grid, noise-trained mixer, violation / confidence EM at K=32:")
        for R in (8, 16, 32):
            print(f"  R={R:<3}", " ".join(
                f"s{s:g}:{samp(rows, mxn, R, s, 'violation')[1].mean():.3f}/"
                f"{samp(rows, mxn, R, s, 'confidence')[1].mean():.3f}" for s in (0.2, 0.4, 0.8)))
        print("K curve (noise mixer, R=16, sigma=0.4), violation / confidence:")
        print("  ", " ".join(f"K{K}:{samp(rows, mxn, 16, 0.4, 'violation', K)[1].mean():.3f}/"
                             f"{samp(rows, mxn, 16, 0.4, 'confidence', K)[1].mean():.3f}"
                             for K in (1, 2, 4, 8, 16, 32)))


if __name__ == "__main__":
    main()
