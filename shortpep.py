"""Do the short peptides in the corpus harm predictions at design lengths?

It is tempting to drop them as label noise. This tests whether that is
justified. Two separate worries are involved:

  labels   a dipeptide labelled hemolytic may be a mis-transcribed record.
           Unverifiable from here.
  features `muH` is computed over min(11, len) residues, so the column does not
           mean the same thing at length 2 as at length 33. True regardless of
           the labels.

Neither matters unless predictions at design lengths move. This retrains with
increasing length floors and measures, on test peptides of length >= 20 only,
whether anything changes -- plus, if DESIGNS is present, whether the ranking of
an off-distribution batch shifts under the same perturbation. That second part
is where the instability documented in the README was found. No design set is
shipped; point DESIGNS at any manifest with a `sequence` column.

    python shortpep.py
"""

import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import features

ROOT = "data/hemolytic"
DESIGNS = "designs.csv"   # optional, not shipped; see the module docstring
FLOORS = [0, 6, 11, 15]
CFG = dict(max_iter=300, learning_rate=0.08, random_state=0)


def main():
    rows = list(csv.DictReader(open(f"{ROOT}/all.csv")))
    seqs = [r["sequence"].strip().upper() for r in rows]
    y = np.array([int(float(r["label"])) for r in rows])
    X = np.array([features.featurise(s) for s in seqs], dtype=np.float32)
    L = np.array([len(s) for s in seqs])
    splits = json.load(open(f"{ROOT}/mmseqs2_split.json"))

    d_rows = (list(csv.DictReader(open(DESIGNS)))
              if os.path.exists(DESIGNS) else [])
    Xd = np.array([features.featurise(r["sequence"].strip().upper())
                   for r in d_rows], dtype=np.float32)

    print(f"corpus {len(y)} peptides; length 2-{L.max()}, median {np.median(L):.0f}")
    for f in FLOORS[1:]:
        print(f"  removing len < {f:2d} drops {(L < f).sum():4d} "
              f"({100*(L < f).mean():4.1f}%), positive rate "
              f"{y[L < f].mean():.3f} vs {y[L >= f].mean():.3f} kept")

    print("\n## Test peptides of length >= 20 only (the design-relevant range)\n")
    print("| length floor | train n | test n (len>=20) | MCC | AUC |")
    print("|---:|---:|---:|---:|---:|")
    ranks = {}
    for f in FLOORS:
        mccs, aucs, ns = [], [], []
        for s in splits.values():
            tr = np.array(s["train"] + s["valid"])
            te = np.array(s["test"])
            tr = tr[L[tr] >= f]            # floor applies to TRAINING only
            te = te[L[te] >= 20]           # test is always the design range
            m = HistGradientBoostingClassifier(**CFG).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            mccs.append(matthews_corrcoef(y[te], (p >= 0.5).astype(int)))
            aucs.append(roc_auc_score(y[te], p))
            ns.append(len(te))
        keep = (L >= f).sum()
        print(f"| {f} | {keep} | {int(np.mean(ns))} | "
              f"{np.mean(mccs):+.3f} ± {np.std(mccs):.3f} | {np.mean(aucs):.3f} |")
        if not d_rows:
            continue
        full = HistGradientBoostingClassifier(**CFG).fit(X[L >= f], y[L >= f])
        ranks[f] = full.predict_proba(Xd)[:, 1]

    if not d_rows:
        print(f"\n(no {DESIGNS}; skipping the design-stability section)")
        return 0

    print(f"\n## Effect on the {len(d_rows)} designs\n")
    print("| length floor | Spearman vs floor=0 | max |ΔP| | designs crossing 0.5 |")
    print("|---:|---:|---:|---:|")
    base = ranks[0]
    for f in FLOORS:
        rho = spearmanr(base, ranks[f]).statistic
        d = np.abs(ranks[f] - base)
        flips = int(((ranks[f] >= 0.5) != (base >= 0.5)).sum())
        print(f"| {f} | {rho:.3f} | {d.max():.3f} | {flips} |")

    worst = int(np.argmax(np.abs(ranks[15] - base)))
    label = d_rows[worst].get("peptide", d_rows[worst]["sequence"][:12])
    print(f"\nlargest shift (floor 15 vs 0): {label} "
          f"{base[worst]:.2f} -> {ranks[15][worst]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
