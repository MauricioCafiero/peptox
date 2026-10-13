"""Hyperparameter sweep for the ranking model, selected on the validation folds.

The first evaluations here used one hand-set configuration and folded
`valid` into training. That avoided leakage but never checked whether the defaults were any
good. This does it properly: for each MMseqs2 seed, every configuration is
fitted on `train`, scored on `valid`, and only the winner is scored on `test`.
Selection and reporting never touch the same data.

The same sweep is then repeated for the extrapolation split, selecting on a
held-out slice of the cationic training set and testing on the anionic tail.

    python sweep.py
"""

import csv
import itertools
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

import features

ROOT = "data/hemolytic"

# max_iter is swept rather than delegated to early stopping. Setting
# early_stopping=True carves 15% out of train for an internal stop and costs
# ~0.13 MCC on the extrapolation split against a fixed 300 iterations -- a
# protocol difference larger than the spread across this entire grid.
GRID = dict(
    learning_rate=[0.03, 0.08, 0.15],
    max_leaf_nodes=[15, 31, 63],
    min_samples_leaf=[10, 20, 40],
    l2_regularization=[0.0, 1.0],
    max_iter=[300, 600],
)
# Exactly what infer.py ships.
DEFAULT = dict(learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
               l2_regularization=0.0, max_iter=300)


def configs():
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, vals))


def fit(cfg, X, y, tr):
    # No early stopping: the shipped model does not use it, and mixing the two
    # protocols makes the grid uninterpretable.
    m = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **cfg)
    return m.fit(X[tr], y[tr])


def select(X, y, tr, va):
    best, best_s = None, -2.0
    for cfg in configs():
        s = matthews_corrcoef(y[va], fit(cfg, X, y, tr).predict(X[va]))
        if s > best_s:
            best, best_s = cfg, s
    return best, best_s


def score(model, X, y, te):
    p = model.predict_proba(X[te])[:, 1]
    return matthews_corrcoef(y[te], (p >= 0.5).astype(int)), roc_auc_score(y[te], p)


def main():
    rows = list(csv.DictReader(open(f"{ROOT}/all.csv")))
    seqs = [r["sequence"].strip().upper() for r in rows]
    y = np.array([int(float(r["label"])) for r in rows])
    X = np.array([features.featurise(s) for s in seqs], dtype=np.float32)
    n_cfg = len(list(configs()))
    print(f"{n_cfg} configurations, 18 descriptors, {len(y)} peptides\n")

    print("## In-family (MMseqs2): select on `valid`, report on `test`\n")
    print("| seed | chosen config | valid MCC | test MCC | default test MCC |")
    print("|---|---|---:|---:|---:|")
    tuned, base = [], []
    chosen = []
    for seed, s in json.load(open(f"{ROOT}/mmseqs2_split.json")).items():
        tr, va, te = (np.array(s["train"]), np.array(s["valid"]),
                      np.array(s["test"]))
        cfg, vs = select(X, y, tr, va)
        t_mcc, _ = score(fit(cfg, X, y, np.concatenate([tr, va])), X, y, te)
        d_mcc, _ = score(fit(DEFAULT, X, y, np.concatenate([tr, va])), X, y, te)
        tuned.append(t_mcc)
        base.append(d_mcc)
        chosen.append(cfg)
        short = f"lr{cfg['learning_rate']} leaf{cfg['max_leaf_nodes']} " \
                f"msl{cfg['min_samples_leaf']} l2={cfg['l2_regularization']:g} " \
                f"it{cfg['max_iter']}"
        print(f"| {seed} | {short} | {vs:+.3f} | {t_mcc:+.3f} | {d_mcc:+.3f} |")
    tuned, base = np.array(tuned), np.array(base)
    print(f"\ntuned  {tuned.mean():+.3f} ± {tuned.std():.3f}")
    print(f"default {base.mean():+.3f} ± {base.std():.3f}")
    print(f"gain   {tuned.mean() - base.mean():+.3f}")

    print("\n## Extrapolation: select inside the cationic set, test on anionic\n")
    q = X[:, features.NAMES.index("charge")]
    pool = np.where(q >= 0)[0]
    te = np.where(q < 0)[0]
    tr, va = train_test_split(pool, test_size=0.15, random_state=0,
                              stratify=y[pool])
    cfg, vs = select(X, y, tr, va)
    t_mcc, t_auc = score(fit(cfg, X, y, pool), X, y, te)
    d_mcc, d_auc = score(fit(DEFAULT, X, y, pool), X, y, te)
    print(f"chosen: {cfg}  (valid MCC {vs:+.3f})")
    print("\n| config | test MCC | test AUC |")
    print("|---|---:|---:|")
    print(f"| tuned | {t_mcc:+.3f} | {t_auc:.3f} |")
    print(f"| default | {d_mcc:+.3f} | {d_auc:.3f} |")
    print(f"\ngain {t_mcc - d_mcc:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
