"""Train hemolysis classifiers on the PepBenchmark set and report honestly.

Compares three feature sets (hand descriptors, ESM-2 640-d embeddings, both)
against two baselines (the peptox rule, and majority class) across five seeds
of two split types. The random vs MMseqs2 contrast is the point: random splits
leak near-duplicate sequences between train and test, so the gap between the
two columns is the part of the random-split score that is memorisation.

    python train.py                     # all feature sets, both splits
    python train.py --split mmseqs2     # just the honest one
"""

import argparse
import csv
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import peptox

DESC = ["charge", "moment", "gravy", "f_kr", "f_phobic", "length"]


def load(root="data/hemolytic"):
    rows = list(csv.DictReader(open(f"{root}/all.csv")))
    seqs = [r["sequence"].strip().upper() for r in rows]
    y = np.array([int(float(r["label"])) for r in rows])

    feats = [peptox.screen(s) for s in seqs]
    X_desc = np.array([[f[k] for k in DESC] for f in feats], dtype=np.float32)
    rule = np.array([f["verdict"] == "FLAG" for f in feats], dtype=int)
    X_esm = np.load(f"{root}/esm2_150_embedding.npz")["data"]
    assert len(X_esm) == len(y), "embedding/label length mismatch"

    splits = {k: json.load(open(f"{root}/{k}_split.json"))
              for k in ("random", "mmseqs2")}
    return seqs, y, X_desc, X_esm, rule, splits


def models():
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0)),
        "gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              random_state=0),
    }


def run(X, y, splits, name, rule=None):
    """Train/test over all five seeds of one split type; return (mcc, auc) arrays."""
    out = {}
    for mname, _ in models().items():
        mccs, aucs = [], []
        for seed, idx in splits.items():
            tr = np.array(idx["train"] + idx["valid"])  # no tuning, so fold valid in
            te = np.array(idx["test"])
            m = models()[mname]
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            mccs.append(matthews_corrcoef(y[te], (p >= 0.5).astype(int)))
            aucs.append(roc_auc_score(y[te], p))
        out[mname] = (np.array(mccs), np.array(aucs))
    return out


def rule_scores(rule, y, splits):
    mccs = []
    for seed, idx in splits.items():
        te = np.array(idx["test"])
        mccs.append(matthews_corrcoef(y[te], rule[te]))
    return np.array(mccs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "mmseqs2", "both"], default="both")
    ap.add_argument("--root", default="data/hemolytic")
    args = ap.parse_args()

    seqs, y, X_desc, X_esm, rule, splits = load(args.root)
    X_both = np.hstack([StandardScaler().fit_transform(X_desc), X_esm])
    print(f"{len(y)} peptides, {y.mean():.3f} positive, "
          f"descriptors {X_desc.shape[1]}-d, ESM-2 {X_esm.shape[1]}-d\n")

    which = ["random", "mmseqs2"] if args.split == "both" else [args.split]
    feature_sets = [("descriptors (6)", X_desc), ("ESM-2 (640)", X_esm),
                    ("ESM-2 + descriptors", X_both)]

    print("| features | model | " + " | ".join(
        f"{s} MCC | {s} AUC" for s in which) + " |")
    print("|---|---|" + "---:|" * (2 * len(which)))

    results = {}
    for fname, X in feature_sets:
        per_split = {s: run(X, y, splits[s], fname) for s in which}
        for mname in models():
            cells = []
            for s in which:
                mcc, auc = per_split[s][mname]
                cells.append(f"{mcc.mean():+.3f} ± {mcc.std():.3f}")
                cells.append(f"{auc.mean():.3f}")
            results[(fname, mname)] = per_split
            print(f"| {fname} | {mname} | " + " | ".join(cells) + " |")

    cells = []
    for s in which:
        r = rule_scores(rule, y, splits[s])
        cells.append(f"{r.mean():+.3f} ± {r.std():.3f}")
        cells.append("—")
    print(f"| **peptox rule (untrained)** | — | " + " | ".join(cells) + " |")

    if len(which) == 2:
        print("\nThe random-minus-MMseqs2 gap is the memorisation. A model that "
              "only scores well in the left column has learned the corpus's "
              "near-duplicates, not hemolysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
