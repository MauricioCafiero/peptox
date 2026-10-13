"""Score peptides with a hemolysis model trained on the full PepBenchmark set,
reporting each score with the reliability measured at that sequence distance.

The model is MCC +0.64 overall on similarity-aware splits, but that number is an
average over a range: domain.py measures MCC +0.77 for test peptides close to
the training set and +0.48 for those far from it. A bare probability hides which
of those you are getting, so every score here carries its tier.

    python archive/predict.py validation.csv
    python archive/predict.py mydesigns.csv --csv out.csv
"""

import argparse
import csv
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import peptox

ROOT = "data/hemolytic"
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"  # 640-d, matches the corpus npz
K = 3

# Measured in domain.py over 2,160 held-out predictions in five MMseqs2 folds.
# Bins are (lower bound, balanced accuracy, MCC, label).
#
# READ THIS BEFORE USING THEM. These numbers describe peptides that are far from
# the training split but still drawn from the same corpus -- i.e. still AMPs.
# They do NOT transfer to a sequence from outside that population altogether.
# Evidence: on this project's anionic Gly/Ser-rich designs the model's output
# anti-correlates with amphipathicity (r = -0.33) and charge (r = -0.09),
# inverting the best-established structure-activity relation in the field. The
# low-similarity tier promises balanced accuracy 0.71 on such peptides and does
# not deliver it, because k-mer distance within one family is not a measure of
# distance to a different family. See README.md "Why the model cannot be applied".
TIERS = [
    (0.60, 0.891, 0.772, "close"),
    (0.40, 0.783, 0.569, "moderate"),
    (0.25, 0.729, 0.470, "far"),
    (0.15, 0.722, 0.459, "far"),
    (0.00, 0.710, 0.483, "very far"),
]

# Corpus 5th-95th percentiles, for context in the output. Deliberately NOT used
# as a gate: only 7 of this project's 21 designs fall outside this box, because
# the corpus does contain some anionic peptides, so it does not separate the
# populations either. No cheap composition test found so far does. That is the
# finding, not a gap to paper over with a better threshold.
CORPUS_CHARGE_P5, CORPUS_CHARGE_P95 = -1.08, 9.44
CORPUS_MOMENT_P5, CORPUS_MOMENT_P95 = 0.22, 0.90


def kmers(s, k=K):
    return {s[i:i + k] for i in range(len(s) - k + 1)} or {s}


def max_similarity(seq, train_kmers):
    """Largest k-mer containment against any training sequence, in [0, 1].

    A proxy for identity, not identity: cheap, alignment-free, enough to place a
    peptide in a reliability tier. It over-reads on low-complexity sequences --
    a poly-alanine design has few distinct 3-mers, so containment is inflated.
    """
    q = kmers(seq)
    return max((len(q & t) / len(q) for t in train_kmers), default=0.0)


def tier_for(sim):
    for lo, bal, mcc, label in TIERS:
        if sim >= lo:
            return bal, mcc, label
    return TIERS[-1][1:]


def embed(seqs):
    """Mean-pooled ESM-2 150M representations, matching the corpus embeddings."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ESM_MODEL)
    mdl = AutoModel.from_pretrained(ESM_MODEL).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 16):
            batch = seqs[i:i + 16]
            enc = tok(batch, return_tensors="pt", padding=True)
            h = mdl(**enc).last_hidden_state
            # Pool over every unpadded token, BOS and EOS included. Verified
            # against data/hemolytic/esm2_150_embedding.npz: this convention
            # reproduces the corpus embeddings exactly (cosine 1.000000), while
            # dropping BOS/EOS does not (0.9993). The model is trained on those
            # vectors, so the convention has to match or the features are shifted.
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="CSV with a 'sequence' column")
    ap.add_argument("--csv", help="also write results here")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(f"{ROOT}/all.csv")))
    tr_seqs = [r["sequence"].strip().upper() for r in rows]
    y = np.array([int(float(r["label"])) for r in rows])
    X = np.load(f"{ROOT}/esm2_150_embedding.npz")["data"]

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           random_state=0).fit(X, y)
    train_kmers = [kmers(s) for s in tr_seqs]

    targets = list(csv.DictReader(open(args.manifest)))
    seqs = [r["sequence"].strip().upper() for r in targets]
    probs = model.predict_proba(embed(seqs))[:, 1]

    print("| peptide | kind | len | P(hemolytic) | k-mer sim | distance | "
          "bal. acc there | rule |")
    print("|---|---|---:|---:|---:|---|---:|---|")
    results = []
    for r, s, p in zip(targets, seqs, probs):
        sim = max_similarity(s, train_kmers)
        bal, mcc, label = tier_for(sim)
        v = peptox.screen(s)["verdict"]
        name = r.get("peptide", s[:12])
        call = "**hemolytic**" if p >= 0.5 else "non-hemolytic"
        print(f"| `{name}` | {r.get('kind','')} | {len(s)} | {p:.2f} {call} | "
              f"{sim:.2f} | {label} | {bal:.2f} | {v} |")
        results.append(dict(peptide=name, kind=r.get("kind", ""), sequence=s,
                            p_hemolytic=round(float(p), 4), kmer_sim=round(sim, 3),
                            distance=label, balanced_acc_at_distance=bal,
                            peptox_rule=v))

    n_pos = sum(r["p_hemolytic"] >= 0.5 for r in results)
    print(f"\n{n_pos}/{len(results)} predicted hemolytic.")

    # Sanity check every run, because the failure mode is silent. Hemolysis is
    # driven by cationic amphipathicity; if the scores run the other way on this
    # batch, the model is outside its population and the numbers are noise.
    import numpy as _np
    pr = _np.array([r["p_hemolytic"] for r in results])
    mu = _np.array([peptox.screen(r["sequence"])["moment"] for r in results])
    if len(pr) > 4 and _np.std(pr) > 0 and _np.std(mu) > 0:
        r_mu = float(_np.corrcoef(pr, mu)[0, 1])
        print(f"\ncorr(P_hemolytic, amphipathicity) = {r_mu:+.3f}")
        if r_mu < 0.1:
            print("*** DO NOT USE THESE SCORES. ***\n"
                  "Predictions do not increase with amphipathicity, which "
                  "contradicts the mechanism. The model is being asked about a "
                  "population it was not trained on and is returning noise. "
                  "See README.md \"Why the model cannot be applied\".")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
