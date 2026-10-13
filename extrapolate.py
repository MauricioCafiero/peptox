"""Do mechanistic descriptors extrapolate where ESM-2 embeddings do not?

Two evaluations of the same models:

  in-family      the MMseqs2 splits -- held out by sequence similarity, but
                 train and test are drawn from the same AMP population.
  extrapolation  train on corpus peptides with net charge >= 0, test on those
                 with net charge < 0. The test set is the anionic tail, the
                 regime occupied by the design set that motivated this work
                 (median charge -1.0). Nothing like it appears in training.

The second is the one that matters. A feature set that holds up there has
learned something transferable; one that collapses has learned the corpus.

An unlabelled direction check then runs over DESIGNS, if that file is present.
No design set is shipped; point it at any manifest with a `sequence` column.

    python extrapolate.py
"""

import csv
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

import features

ROOT = "data/hemolytic"
DESIGNS = "designs.csv"   # optional, not shipped; see the module docstring


def load():
    rows = list(csv.DictReader(open(f"{ROOT}/all.csv")))
    seqs = [r["sequence"].strip().upper() for r in rows]
    y = np.array([int(float(r["label"])) for r in rows])
    Xd = np.array([features.featurise(s) for s in seqs], dtype=np.float32)
    Xe = np.load(f"{ROOT}/esm2_150_embedding.npz")["data"]
    return seqs, y, Xd, Xe


def fit_eval(X, y, tr, te):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                       random_state=0).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y[te], pred, labels=[0, 1])
    return (matthews_corrcoef(y[te], pred), roc_auc_score(y[te], p),
            balanced_accuracy_score(y[te], pred), cm)


def show_cm(name, cm):
    (tn, fp), (fn, tp) = cm
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    print(f"\n{name}")
    print("|  | pred non-hemolytic | pred hemolytic |  |")
    print("|---|---:|---:|---|")
    print(f"| **true non-hemolytic** | {tn} (TN) | {fp} (FP) | spec {spec:.3f} |")
    print(f"| **true hemolytic** | {fn} (FN) | {tp} (TP) | sens {sens:.3f} |")
    print(f"|  | NPV {npv:.3f} | PPV {ppv:.3f} |  |")


def main():
    seqs, y, Xd, Xe = load()
    q = Xd[:, features.NAMES.index("charge")]
    sets = [("descriptors (18)", Xd), ("ESM-2 (640)", Xe),
            ("ESM-2 + descriptors", np.hstack([Xd, Xe]))]

    print(f"{len(y)} peptides, {y.mean():.3f} positive, "
          f"{Xd.shape[1]} descriptors\n")

    print("## In-family (MMseqs2 splits, 5 seeds)\n")
    print("| features | MCC | AUC | balanced acc |")
    print("|---|---:|---:|---:|")
    splits = json.load(open(f"{ROOT}/mmseqs2_split.json"))
    infam = {}
    for name, X in sets:
        res = [fit_eval(X, y, np.array(s["train"] + s["valid"]),
                        np.array(s["test"])) for s in splits.values()]
        r = np.array([x[:3] for x in res])
        infam[name] = (r.mean(0), sum(x[3] for x in res))
        print(f"| {name} | {r[:,0].mean():+.3f} ± {r[:,0].std():.3f} | "
              f"{r[:,1].mean():.3f} | {r[:,2].mean():.3f} |")

    tr = np.where(q >= 0)[0]
    te = np.where(q < 0)[0]
    print(f"\n## Extrapolation (train q >= 0, n={len(tr)}; "
          f"test q < 0, n={len(te)}, {y[te].mean():.3f} positive)\n")
    print("| features | MCC | AUC | balanced acc | vs in-family MCC |")
    print("|---|---:|---:|---:|---:|")
    extra = {}
    for name, X in sets:
        mcc, auc, bal, cm = fit_eval(X, y, tr, te)
        extra[name] = cm
        drop = mcc - infam[name][0][0]
        print(f"| {name} | {mcc:+.3f} | {auc:.3f} | {bal:.3f} | {drop:+.3f} |")

    print("\n### Confusion matrices, in-family (summed over 5 MMseqs2 folds)")
    for name, _ in sets:
        show_cm(name, infam[name][1])
    print("\n### Confusion matrices, extrapolation (anionic test set)")
    for name, _ in sets:
        show_cm(name, extra[name])

    # Does the descriptor model behave mechanistically on a design batch? No
    # labels exist for one, so the only available check is whether its output
    # tracks amphipathicity in the direction biophysics requires.
    if not os.path.exists(DESIGNS):
        print(f"\n(no {DESIGNS}; skipping the unlabelled direction check)")
        return 0
    print(f"\n## Sanity check on {DESIGNS} (no labels; direction only)\n")
    mu_i = features.NAMES.index("muH_helix_max")
    d_rows = list(csv.DictReader(open(DESIGNS)))
    d_seqs = [r["sequence"].strip().upper() for r in d_rows]
    Xd_d = np.array([features.featurise(s) for s in d_seqs], dtype=np.float32)
    print("| model | corr(P, amphipathicity) | corr(P, charge) |")
    print("|---|---:|---:|")
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                       random_state=0).fit(Xd, y)
    p = m.predict_proba(Xd_d)[:, 1]
    print(f"| descriptors | {np.corrcoef(p, Xd_d[:, mu_i])[0,1]:+.3f} | "
          f"{np.corrcoef(p, Xd_d[:, features.NAMES.index('charge')])[0,1]:+.3f} |")
    print("| ESM-2 (archive/predict.py) | -0.326 | -0.089 |")

    for r, pi in sorted(zip(d_rows, p), key=lambda t: -t[1])[:5]:
        print(f"\n  {pi:.2f}  {r.get('peptide', r['sequence'][:12])} "
              f"({r.get('kind', '')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
