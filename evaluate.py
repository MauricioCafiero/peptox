"""Held-out evaluation of the peptox screen against PepBenchmark hemolysis data.

The thresholds in peptox.py are LOCKED before this runs. Nothing here tunes
them; this script only measures. Ablations are reported so the two criteria
have to justify themselves against simpler rules.

    python evaluate.py data/hemolytic/all.csv
"""

import argparse
import csv
import math
import sys

import peptox


def wilson(k, n, z=1.96):
    """Wilson score interval -- honest at the extremes, unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def confusion(calls, truth):
    tp = sum(c and t for c, t in zip(calls, truth))
    fp = sum(c and not t for c, t in zip(calls, truth))
    fn = sum((not c) and t for c, t in zip(calls, truth))
    tn = sum((not c) and (not t) for c, t in zip(calls, truth))
    return tp, fp, fn, tn


def report(name, calls, truth):
    tp, fp, fn, tn = confusion(calls, truth)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else float("nan")
    lo_se, hi_se = wilson(tp, tp + fn)
    lo_sp, hi_sp = wilson(tn, tn + fp)
    print(f"| {name} | {sens:.3f} [{lo_se:.2f}–{hi_se:.2f}] | "
          f"{spec:.3f} [{lo_sp:.2f}–{hi_sp:.2f}] | {ppv:.3f} | {mcc:+.3f} |")
    return mcc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="CSV with 'sequence' and binary 'label' columns")
    args = p.parse_args()

    seqs, truth = [], []
    skipped = 0
    with open(args.dataset) as fh:
        for row in csv.DictReader(fh):
            s = row["sequence"].strip().upper()
            if set(s) - set(peptox.KD) or not s:
                skipped += 1
                continue
            seqs.append(s)
            truth.append(int(float(row["label"])) == 1)

    n = len(seqs)
    pos = sum(truth)
    print(f"{args.dataset}: {n} sequences ({skipped} skipped: non-standard residues)")
    print(f"base rate: {pos}/{n} = {pos / n:.3f} hemolytic\n")

    feats = [peptox.screen(s) for s in seqs]
    q = [f["charge"] for f in feats]
    mu = [f["moment"] for f in feats]

    print("| rule | sensitivity [95% CI] | specificity [95% CI] | PPV | MCC |")
    print("|---|---|---|---:|---:|")

    # The screen as shipped, thresholds untouched.
    screen_calls = [f["verdict"] == "FLAG" for f in feats]
    report("**peptox FLAG (as shipped)**", screen_calls, truth)
    report("peptox FLAG or WATCH", [f["verdict"] != "pass" for f in feats], truth)

    # Ablations: does each criterion earn its place?
    report(f"charge only (q >= {peptox.MIN_CHARGE})",
           [x >= peptox.MIN_CHARGE for x in q], truth)
    report(f"moment only (uH >= {peptox.MIN_MOMENT})",
           [x >= peptox.MIN_MOMENT for x in mu], truth)
    report("always call hemolytic", [True] * n, truth)

    # Is the charge threshold anywhere near a sensible place? Reported, not fitted:
    # if some other cutoff is dramatically better, the shipped one is miscalibrated.
    print("\nCharge threshold sweep (diagnostic only -- thresholds stay locked):")
    print("| q cutoff | sensitivity | specificity | MCC |")
    print("|---:|---:|---:|---:|")
    for cut in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        calls = [a >= cut and b >= peptox.MIN_MOMENT for a, b in zip(q, mu)]
        tp, fp, fn, tn = confusion(calls, truth)
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / den) if den else float("nan")
        mark = "  <- shipped" if cut == peptox.MIN_CHARGE else ""
        print(f"| {cut:+.1f} | {sens:.3f} | {spec:.3f} | {mcc:+.3f} |{mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
