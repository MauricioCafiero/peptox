"""Does the model get worse on test peptides unlike its training set?

Sets the applicability gate by measurement instead of assertion. For each test
peptide in the MMseqs2 splits, compute k-mer containment against that fold's
training set, bin by it, and report accuracy per bin. A flat curve means the
gate is unnecessary at these similarities; a falling one says where to put it.

    python domain.py
"""

import csv
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef

K = 3


def kmers(s, k=K):
    """k-mer set, for the alignment-free similarity proxy."""
    return {s[i:i + k] for i in range(len(s) - k + 1)} or {s}

ROOT = "data/hemolytic"
BINS = [0.0, 0.15, 0.25, 0.40, 0.60, 1.01]


def main():
    rows = list(csv.DictReader(open(f"{ROOT}/all.csv")))
    seqs = [r["sequence"].strip().upper() for r in rows]
    y = np.array([int(float(r["label"])) for r in rows])
    X = np.load(f"{ROOT}/esm2_150_embedding.npz")["data"]
    ks = [kmers(s) for s in seqs]
    splits = json.load(open(f"{ROOT}/mmseqs2_split.json"))

    sims, correct, labels, preds = [], [], [], []
    for seed, idx in splits.items():
        tr = np.array(idx["train"] + idx["valid"])
        te = np.array(idx["test"])
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           random_state=0).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        tr_k = [ks[i] for i in tr]
        for j, i in enumerate(te):
            q = ks[i]
            sims.append(max(len(q & t) / len(q) for t in tr_k))
            correct.append(int(pred[j] == y[i]))
            labels.append(y[i])
            preds.append(pred[j])

    sims = np.array(sims)
    correct = np.array(correct)
    print(f"{len(sims)} test predictions across {len(splits)} MMseqs2 folds\n")
    labels = np.array(labels)
    preds = np.array(preds)
    # Accuracy per bin is uninterpretable here: similarity is strongly confounded
    # with the label (hemolytic peptides cluster into families, so the unlike-
    # anything bin is mostly negatives). MCC and balanced accuracy are not.
    print("| k-mer containment to train | n | base rate | accuracy | "
          "balanced acc | MCC |")
    print("|---|---:|---:|---:|---:|---:|")
    for lo, hi in zip(BINS, BINS[1:]):
        m = (sims >= lo) & (sims < hi)
        if m.sum() == 0:
            continue
        yt, yp = labels[m], preds[m]
        tp = ((yp == 1) & (yt == 1)).sum(); tn = ((yp == 0) & (yt == 0)).sum()
        fp = ((yp == 1) & (yt == 0)).sum(); fn = ((yp == 0) & (yt == 1)).sum()
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        bal = (sens + spec) / 2
        mcc = matthews_corrcoef(yt, yp) if len(set(yt)) > 1 else float("nan")
        print(f"| {lo:.2f}–{hi:.2f} | {m.sum()} | {yt.mean():.3f} | "
              f"{correct[m].mean():.3f} | {bal:.3f} | {mcc:+.3f} |")
    print(f"\noverall accuracy {correct.mean():.3f}, "
          f"MCC {matthews_corrcoef(labels, preds):+.3f}")


if __name__ == "__main__":
    main()
