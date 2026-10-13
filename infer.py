"""Screen designed peptides for membrane-activity liability.

Runs the two checks this project has evidence for, and labels what each one is
good for. They answer different questions and are meant to be read together:

  1. DRIFT CHECK (rule, no training)
     "Has the design pipeline wandered into antimicrobial-peptide space?"
     Validated: fires on 73% of a 4,306-peptide AMP corpus and 14% of this
     project's designs. Reliable, because it is a statement about a
     distribution rather than a prediction about biology.

  2. MEMBRANE-ACTIVITY RANKING (18-descriptor model, ensembled)
     "Which of these designs should I look at first?"
     Balanced accuracy 0.737 extrapolating to an anionic test set unlike its
     training data, where an ESM-2 model collapses to 0.717 with half the
     sensitivity. Good for ordering candidates. NOT a hemolysis measurement
     and not grounds for clearing a peptide.

     Reported as a band, not a number. A single fit gives point estimates that
     move by up to 0.53 when 2.5% of training rows are dropped (shortpep.py),
     because these designs sit off the training distribution. The band is the
     10th-90th percentile over N_ENSEMBLE bootstrap resamples; a verdict is
     only given where the whole band falls on one side of 0.5.

Neither check covers aggregation, protease lability or immunogenicity, which
for Gly/Ser-rich anionic designs are plausibly the larger risks.

    python infer.py validation.csv
    python infer.py mydesigns.csv --csv out.csv
    python infer.py --seq GIGAVLKVLTTGLPALISWIKRKRQQ
"""

import argparse
import csv
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import features
import peptox

CORPUS = "data/hemolytic/all.csv"
# Corpus percentiles of max helical moment, for the "is this amphipathic by AMP
# standards" column. The rule's own 0.35 threshold sits near the corpus 25th
# percentile: it was calibrated against this project's designs, not against
# real AMPs, so it is a good drift line and a poor absolute scale.
CORPUS_MOMENT_Q = (0.41, 0.54, 0.68)  # p25, p50, p75, measured on CORPUS
# Bootstrap resamples behind each band. 12 keeps the run near 20 s; the band
# width is stable past about 10.
N_ENSEMBLE = 12


def predict_band(Xd):
    """Median and 10-90% band of P(membrane-active) over bootstrap resamples.

    Refit on every run rather than pickled: it takes ~20 s on 18 features, and a
    stale artifact that silently disagrees with the data beside it is a worse
    failure than the wait. Short peptides are deliberately NOT filtered out --
    shortpep.py shows removing them does not change accuracy at design lengths.
    """
    rows = list(csv.DictReader(open(CORPUS)))
    seqs = [r["sequence"].strip().upper() for r in rows]
    y = np.array([int(float(r["label"])) for r in rows])
    X = np.array([features.featurise(s) for s in seqs], dtype=np.float32)
    rng = np.random.default_rng(0)
    preds = []
    for b in range(N_ENSEMBLE):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(set(y[idx])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           random_state=b).fit(X[idx], y[idx])
        preds.append(m.predict_proba(Xd)[:, 1])
    P = np.array(preds)
    return (np.median(P, axis=0), np.percentile(P, 10, axis=0),
            np.percentile(P, 90, axis=0))


def amphipathy_note(mu):
    p25, p50, p75 = CORPUS_MOMENT_Q
    if mu >= p75:
        return "high (>p75 of AMPs)"
    if mu >= p50:
        return "mid"
    if mu >= p25:
        return "low (<p50 of AMPs)"
    return "very low"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="?", help="CSV with a 'sequence' column")
    ap.add_argument("--seq", help="screen a single sequence instead")
    ap.add_argument("--csv", help="also write results here")
    args = ap.parse_args()

    if args.seq:
        targets = [{"peptide": args.seq[:12], "kind": "", "sequence": args.seq}]
    elif args.manifest:
        targets = list(csv.DictReader(open(args.manifest)))
    else:
        ap.error("give a manifest CSV or --seq")

    seqs = [t["sequence"].strip().upper() for t in targets]
    Xd = np.array([features.featurise(s) for s in seqs], dtype=np.float32)
    prob, lo, hi = predict_band(Xd)
    scr = [peptox.screen(s) for s in seqs]

    print("=" * 76)
    print("1. DRIFT CHECK  --  has the design drifted into AMP sequence space?")
    print("   Validated use. A flag means LOOK AT THIS ONE, not discard, and")
    print("   not toxic. A pass is not a safety claim.")
    print("=" * 76)
    print("\n| peptide | kind | len | charge | <uH> helix | <uH> sheet | verdict |")
    print("|---|---|---:|---:|---:|---:|---|")
    mu_h = features.NAMES.index("muH_helix_max")
    mu_s = features.NAMES.index("muH_sheet_max")
    for t, s, x in zip(targets, scr, Xd):
        mark = {"FLAG": "**FLAG**", "WATCH": "WATCH", "pass": "pass"}[s["verdict"]]
        print(f"| `{t.get('peptide', s['seq'][:12])}` | {t.get('kind','')} | "
              f"{s['length']} | {s['charge']:+.1f} | {x[mu_h]:.2f} | "
              f"{x[mu_s]:.2f} | {mark} |")
    n_flag = sum(s["verdict"] == "FLAG" for s in scr)
    print(f"\n{n_flag}/{len(scr)} drifted toward AMP space.")

    print("\n" + "=" * 76)
    print("2. MEMBRANE-ACTIVITY RANKING  --  which to look at first?")
    print("   18-descriptor model, trained on 4,306 peptides with measured")
    print("   hemolysis. Ordering signal only: balanced accuracy 0.737 when")
    print("   extrapolating off its training distribution. NOT a measurement,")
    print("   NOT grounds for clearing a peptide.")
    print("=" * 76)
    print(f"\n| rank | peptide | kind | median P | 10-90% band "
          f"({N_ENSEMBLE} resamples) | call | amphipathicity | drift |")
    print("|---:|---|---|---:|---|---|---|---|")
    order = sorted(range(len(seqs)), key=lambda i: -prob[i])
    for rank, i in enumerate(order, 1):
        t, s = targets[i], scr[i]
        if lo[i] > 0.5:
            call = "**elevated**"
        elif hi[i] < 0.5:
            call = "low"
        else:
            call = "*unresolved*"
        print(f"| {rank} | `{t.get('peptide', s['seq'][:12])}` | "
              f"{t.get('kind','')} | {prob[i]:.2f} | {lo[i]:.2f}-{hi[i]:.2f} | "
              f"{call} | {amphipathy_note(Xd[i][mu_h])} | {s['verdict']} |")
    n_un = sum(1 for i in range(len(seqs)) if lo[i] <= 0.5 <= hi[i])
    print(f"\n{n_un}/{len(seqs)} unresolved -- the band straddles 0.5, so the "
          f"model does not separate them. Median band width "
          f"{np.median(hi - lo):.2f}.")

    # Direction check. If the model's scores do not rise with amphipathicity on
    # this batch, it is being asked about a population it cannot speak to --
    # exactly how the archived ESM-2 model failed (r = -0.33 on these designs).
    if len(seqs) > 4 and np.std(prob) > 0:
        r = float(np.corrcoef(prob, Xd[:, mu_h])[0, 1])
        print(f"\ncorr(P, amphipathicity) = {r:+.3f}", end="  ")
        print("-- consistent with the mechanism." if r >= 0.1 else
              "\n*** SUSPECT. Scores do not rise with amphipathicity; treat the "
              "ranking as unreliable for this batch. ***")

    agree = [i for i in range(len(seqs))
             if scr[i]["verdict"] == "FLAG" and lo[i] > 0.5]
    if agree:
        print("\nBoth checks agree on: " +
              ", ".join(f"`{targets[i].get('peptide', seqs[i][:12])}`"
                        for i in agree) + " -- the strongest signal available here.")

    print("\nNot covered by either check: aggregation, protease lability, "
          "immunogenicity.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["peptide", "kind", "sequence", "length", "charge",
                        "muH_helix", "muH_sheet", "drift_verdict",
                        "p_median", "p_lo10", "p_hi90"])
            for i, t in enumerate(targets):
                w.writerow([t.get("peptide", seqs[i][:12]), t.get("kind", ""),
                            seqs[i], scr[i]["length"], round(scr[i]["charge"], 2),
                            round(float(Xd[i][mu_h]), 3),
                            round(float(Xd[i][mu_s]), 3),
                            scr[i]["verdict"], round(float(prob[i]), 4),
                            round(float(lo[i]), 4), round(float(hi[i]), 4)])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
