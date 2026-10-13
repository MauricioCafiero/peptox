"""Mechanistic descriptors for membrane activity.

Every feature here is a property of composition or of how composition is
arranged along the chain. None of them encodes sequence identity, which is the
point: a quantity like net charge means the same thing in a conotoxin and in a
designed glycine-rich binder, whereas a language-model embedding places a
sequence relative to families it has seen. That is why these are expected to
extrapolate off-distribution where ESM-2 does not -- a claim extrapolate.py
tests rather than assumes.

Both helical (100 deg/residue) and sheet (180 deg) periodicities are computed.
An earlier feature set assumed a helix everywhere, which is wrong for any
peptide that folds to a sheet.
"""

import math

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)

EIS = dict(A=0.62, R=-2.53, N=-0.78, D=-0.90, C=0.29, Q=-0.85, E=-0.74, G=0.48,
           H=-0.40, I=1.38, L=1.06, K=-1.50, M=0.64, F=1.19, P=0.12, S=-0.18,
           T=-0.05, W=0.81, Y=0.26, V=1.08)

# Wimley-White water -> POPC interface, kcal/mol, negative = favourable.
WW = dict(A=0.17, R=0.81, N=0.42, D=1.23, C=-0.24, Q=0.58, E=2.02, G=0.01,
          H=0.17, I=-0.31, L=-0.56, K=0.99, M=-0.23, F=-1.13, P=0.45, S=0.13,
          T=0.14, W=-1.85, Y=-0.94, V=0.07)

# Boman: potential to bind other proteins; high values track promiscuity.
BOMAN = dict(A=-0.5, R=2.5, N=0.2, D=3.0, C=-1.0, Q=0.2, E=3.0, G=0.0, H=-0.5,
             I=-1.8, L=-1.8, K=3.0, M=-1.3, F=-2.5, P=0.0, S=0.3, T=-0.4,
             W=-3.4, Y=-2.3, V=-1.5)

PKA_POS = dict(K=10.5, R=12.5, H=6.0)
PKA_NEG = dict(D=3.9, E=4.3, C=8.3, Y=10.1)
PKA_NTERM, PKA_CTERM = 9.0, 2.3

HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
WINDOW = 11

NAMES = [
    "length", "charge", "charge_density", "frac_KR", "frac_DE",
    "muH_helix_max", "muH_helix_mean", "muH_sheet_max",
    "gravy", "ww_mean", "ww_window_max",
    "frac_hydrophobic", "frac_aromatic", "frac_GSP", "frac_C",
    "longest_hydrophobic_run", "aliphatic_index", "boman",
]


def net_charge(seq, pH=7.4):
    q = 1 / (1 + 10 ** (pH - PKA_NTERM)) - 1 / (1 + 10 ** (PKA_CTERM - pH))
    for a in seq:
        if a in PKA_POS:
            q += 1 / (1 + 10 ** (pH - PKA_POS[a]))
        if a in PKA_NEG:
            q -= 1 / (1 + 10 ** (PKA_NEG[a] - pH))
    return q


def moments(seq, degrees):
    """(max, mean) Eisenberg moment over sliding windows at a given periodicity."""
    d = math.radians(degrees)
    w = min(WINDOW, len(seq))
    vals = []
    for s in range(len(seq) - w + 1):
        sx = sy = 0.0
        for i, a in enumerate(seq[s:s + w]):
            h = EIS.get(a, 0.0)
            sx += h * math.cos(i * d)
            sy += h * math.sin(i * d)
        vals.append(math.hypot(sx, sy) / w)
    return (max(vals), sum(vals) / len(vals)) if vals else (0.0, 0.0)


def longest_run(seq, alphabet):
    best = cur = 0
    for a in seq:
        cur = cur + 1 if a in alphabet else 0
        best = max(best, cur)
    return best


def featurise(seq):
    seq = seq.strip().upper()
    n = len(seq)
    f = {}
    f["length"] = n
    f["charge"] = net_charge(seq)
    f["charge_density"] = f["charge"] / n
    f["frac_KR"] = sum(a in "KR" for a in seq) / n
    f["frac_DE"] = sum(a in "DE" for a in seq) / n
    f["muH_helix_max"], f["muH_helix_mean"] = moments(seq, 100.0)
    f["muH_sheet_max"], _ = moments(seq, 180.0)
    f["gravy"] = sum(KD[a] for a in seq) / n
    f["ww_mean"] = -sum(WW[a] for a in seq) / n
    w = min(WINDOW, n)
    f["ww_window_max"] = max(
        -sum(WW[a] for a in seq[s:s + w]) / w for s in range(n - w + 1))
    f["frac_hydrophobic"] = sum(a in HYDROPHOBIC for a in seq) / n
    f["frac_aromatic"] = sum(a in AROMATIC for a in seq) / n
    f["frac_GSP"] = sum(a in "GSP" for a in seq) / n
    f["frac_C"] = seq.count("C") / n
    f["longest_hydrophobic_run"] = longest_run(seq, HYDROPHOBIC)
    # Aliphatic index: relative volume of aliphatic side chains, a thermostability
    # proxy that also tracks the bulk available to pack into a bilayer core.
    f["aliphatic_index"] = (100 * (seq.count("A") + 2.9 * seq.count("V")
                            + 3.9 * (seq.count("I") + seq.count("L"))) / n)
    f["boman"] = sum(BOMAN[a] for a in seq) / n
    return [f[k] for k in NAMES]
