"""Drift detector for designed peptides: flags sequences that have wandered
into antimicrobial-peptide space (cationic + amphipathic).

NOT A HEMOLYSIS PREDICTOR. On 4,306 held-out peptides with measured hemolysis
labels this rule scores MCC +0.09, barely above chance, because the negatives in
that corpus are themselves cationic amphipathic peptides. Run evaluate.py to
reproduce. What it does reliably is measure distance from AMP space: it fires on
73% of that corpus and on 14% of this project's designs. Use it as a tripwire on
generated sequences, never to clear a peptide. See README.md.

    python peptox.py validation.csv         # table over a manifest
    python peptox.py --seq RGERKEKSKLIFII   # one sequence
"""

import argparse
import csv
import math
import sys

# Kyte-Doolittle, for GRAVY.
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)

# Eisenberg consensus, for the hydrophobic moment. A different scale on purpose:
# GRAVY asks how hydrophobic the peptide is, the moment asks how the hydrophobicity
# is arranged around a helix, and Eisenberg is the scale that question was posed on.
EIS = dict(A=0.62, R=-2.53, N=-0.78, D=-0.90, C=0.29, Q=-0.85, E=-0.74, G=0.48,
           H=-0.40, I=1.38, L=1.06, K=-1.50, M=0.64, F=1.19, P=0.12, S=-0.18,
           T=-0.05, W=0.81, Y=0.26, V=1.08)

# Wimley-White interfacial scale: dG of transfer from water to a POPC interface,
# kcal/mol, negative = favourable. This, not GRAVY, is what gates the screen.
# Kyte-Doolittle was built to find transmembrane segments and scores Trp at -0.9;
# on the interface Trp is the single most favourable residue there is, and KD also
# over-penalises Arg. Using it let indolicidin and LL-37 -- both hemolytic -- past
# the hydrophobicity criterion. It matters doubly here because this design family
# places W and Y deliberately, to stack against the UV filters' aromatic rings.
WW = dict(A=0.17, R=0.81, N=0.42, D=1.23, C=-0.24, Q=0.58, E=2.02, G=0.01,
          H=0.17, I=-0.31, L=-0.56, K=0.99, M=-0.23, F=-1.13, P=0.45, S=0.13,
          T=0.14, W=-1.85, Y=-0.94, V=0.07)

HYDROPHOBIC = set("AILMFWVY")

# Side-chain pKa values. Charge is reported at pH 7.4 rather than as an integer
# count of K+R-D-E so that His and the termini are not silently dropped; on a
# 25-mer the difference is a third of a charge unit, which matters at the cutoff.
PKA_POS = dict(K=10.5, R=12.5, H=6.0)
PKA_NEG = dict(D=3.9, E=4.3, C=8.3, Y=10.1)
PKA_NTERM, PKA_CTERM = 9.0, 2.3

# The screen fires on two properties, not three: charge to find the anionic
# headgroups, and a segregated hydrophobic face to insert. A third criterion on
# bulk hydrophobicity was tried and dropped -- see README.md "What was dropped".
# On validation.csv it let indolicidin and LL-37 through, and no hydrophobicity
# scale tested (Kyte-Doolittle, Wimley-White interfacial, windowed Eisenberg)
# separated the hemolytic peptides from the rest: LL-37 and pexiganan score at
# the bottom of every one of them. Charge and moment alone catch 5/5 hemolytic
# references at 4/5 specificity.
#
# Thresholds sit at the low edge of the active range reported for helical AMPs,
# so the screen errs toward flagging.
MIN_CHARGE = 2.0
MIN_MOMENT = 0.35
NEAR_CHARGE = 1.0    # borderline band, reported as WATCH rather than passed
NEAR_MOMENT = 0.30
WINDOW = 11          # residues, ~3 helical turns -- the shortest lytic segment
HELIX_ANGLE = 100.0  # degrees per residue in an alpha helix


def net_charge(seq, pH=7.4):
    q = 1 / (1 + 10 ** (pH - PKA_NTERM)) - 1 / (1 + 10 ** (PKA_CTERM - pH))
    for a in seq:
        if a in PKA_POS:
            q += 1 / (1 + 10 ** (pH - PKA_POS[a]))
        if a in PKA_NEG:
            q -= 1 / (1 + 10 ** (PKA_NEG[a] - pH))
    return q


def hydrophobic_moment(seq):
    """Largest Eisenberg moment over any WINDOW-residue stretch, helical geometry.

    Windowed rather than whole-sequence because a lytic face only needs three
    turns: a 33-mer with one amphipathic end and a disordered tail is a real
    liability, and its whole-sequence moment would average that away. Returns
    the moment and the window start, so a flag can say where the face is.
    """
    d = math.radians(HELIX_ANGLE)
    w = min(WINDOW, len(seq))
    best, at = 0.0, 0
    for s in range(len(seq) - w + 1):
        sx = sy = 0.0
        for i, a in enumerate(seq[s:s + w]):
            h = EIS.get(a, 0.0)
            sx += h * math.cos(i * d)
            sy += h * math.sin(i * d)
        m = math.hypot(sx, sy) / w
        if m > best:
            best, at = m, s
    return best, at


def screen(seq):
    seq = seq.strip().upper()
    unknown = set(seq) - set(KD)
    if unknown or not seq:
        raise ValueError(f"not a peptide sequence: {''.join(sorted(unknown)) or 'empty'}")

    q = net_charge(seq)
    gravy = sum(KD[a] for a in seq) / len(seq)
    moment, at = hydrophobic_moment(seq)
    f_kr = sum(a in "KR" for a in seq) / len(seq)
    f_phobic = sum(a in HYDROPHOBIC for a in seq) / len(seq)

    cationic, amphipathic = q >= MIN_CHARGE, moment >= MIN_MOMENT
    w = min(WINDOW, len(seq))
    face = f"{seq[at:at + w]} (residues {at + 1}-{at + w})"

    if cationic and amphipathic:
        verdict = "FLAG"
        reason = f"cationic amphipathic: q={q:+.1f}, <uH>={moment:.2f} over {face}"
    elif (cationic and moment >= NEAR_MOMENT) or (amphipathic and q >= NEAR_CHARGE):
        verdict = "WATCH"
        near = "moment" if cationic else "charge"
        reason = (f"borderline on {near}: q={q:+.1f}, <uH>={moment:.2f} over {face}")
    else:
        verdict = "pass"
        if amphipathic:
            reason = f"amphipathic (<uH>={moment:.2f}) but not cationic (q={q:+.1f})"
        elif cationic:
            reason = f"cationic (q={q:+.1f}) but no hydrophobic face (<uH>={moment:.2f})"
        else:
            reason = f"neither cationic nor amphipathic (q={q:+.1f}, <uH>={moment:.2f})"

    return dict(seq=seq, length=len(seq), charge=q, gravy=gravy, moment=moment,
                f_kr=f_kr, f_phobic=f_phobic, verdict=verdict, reason=reason)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", nargs="?",
                   help="CSV with a 'sequence' column; 'peptide'/'kind' used if present")
    p.add_argument("--seq", help="screen a single sequence instead")
    args = p.parse_args()

    if args.seq:
        r = screen(args.seq)
        print(f"{r['verdict']}: {r['reason']}")
        return 0 if r["verdict"] == "pass" else 1

    if not args.manifest:
        p.error("give a manifest CSV or --seq")

    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "sequence" not in rows[0]:
        p.error(f"{args.manifest} has no 'sequence' column")

    print("| peptide | kind | len | q(7.4) | GRAVY | <uH> | %K+R | verdict |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    flagged = []
    for row in rows:
        r = screen(row["sequence"])
        name = row.get("peptide") or r["seq"][:12]
        kind = row.get("kind", "")
        mark = {"FLAG": "**FLAG**", "WATCH": "WATCH", "pass": "pass"}[r["verdict"]]
        print(f"| `{name}` | {kind} | {r['length']} | {r['charge']:+.2f} | "
              f"{r['gravy']:+.2f} | {r['moment']:.2f} | {100 * r['f_kr']:.1f} | {mark} |")
        if r["verdict"] != "pass":
            flagged.append((name, r))

    print(f"\n{len(rows)} screened, {sum(1 for _, r in flagged if r['verdict'] == 'FLAG')} "
          f"flagged, {sum(1 for _, r in flagged if r['verdict'] == 'WATCH')} to watch.")
    for name, r in flagged:
        print(f"\n{r['verdict']} {name}\n  {r['seq']}\n  {r['reason']}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
