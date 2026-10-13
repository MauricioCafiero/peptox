# peptox

A hemolysis model trained on PepBenchmark, a descriptor heuristic, and the
evaluations that say what each of them can and cannot be used for.

**Short answer: neither predicts toxicity for peptides unlike the training
corpus.** The trained model is good — MCC +0.64, AUC 0.89 on similarity-aware
splits — and it could not be applied to the design set it was built for, which
is demonstrated rather than assumed. The heuristic is near-chance as a
hemolysis predictor but works as a drift detector. Details below, including two
wrong turns kept on the record because the way they were wrong is the useful
part.

This project was built to screen an unpublished set of 21 designed peptides.
**Those sequences are not in this repository** and the results reported for
them are anonymised: the designs are referred to by a label and their
composition (`cationic-1`, `anionic-2`, and so on). Everything needed to run
the same analysis on your own manifest is here.

## Use it

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_data.py                     # <- downloads the corpus
.venv/bin/python infer.py validation.csv           # <- the entry point
.venv/bin/python infer.py --seq GIGAVLKVLTTGLPAL   # one sequence
```

`infer.py` takes any CSV with a `sequence` column (and optionally `peptide` and
`kind`, used for labelling). `validation.csv` ships as a worked example: ten
canonical literature peptides, which also serve as the in-distribution control
in [the band section](#the-control-do-the-bands-track-distance-from-the-data).

It runs both checks and labels what each is for: a **drift check** (validated —
has the pipeline wandered into AMP space?) and a **membrane-activity ranking**
(ordering signal only — which candidate to look at first). It trains the
descriptor model from `data/` on each run rather than carrying a pickle that
can drift from the data beside it.

The corpus is fetched, not vendored: the upstream release states no licence,
so redistributing it here would not be appropriate. See
[Training data](#training-data).

How both checks were fitted, what was and was not tuned, and how well their
scores are calibrated: [How the two shipped checks were built](#how-the-two-shipped-checks-were-built).

Reproducing the evaluations:

```sh
.venv/bin/python fetch_data.py --all # extrapolate.py and domain.py need the
                                     # ESM-2 embeddings and the extra splits
.venv/bin/python train.py            # feature sets x split types
.venv/bin/python extrapolate.py      # descriptors vs ESM-2 off-distribution
.venv/bin/python domain.py           # accuracy vs sequence distance
.venv/bin/python sweep.py            # 108-config hyperparameter grid
.venv/bin/python shortpep.py         # length-floor ablation
python evaluate.py data/hemolytic/all.csv   # the heuristic, held out (stdlib)
```

`extrapolate.py` and `shortpep.py` finish with an unlabelled check on a design
batch, which they skip unless you put a `designs.csv` beside them.

## The model

`data/hemolytic/` is PepBenchmark's hemolysis set: 4,306 peptides, 50.6%
hemolytic, from Hemolytik/DBAASP, with precomputed ESM-2 150M embeddings and
both random and MMseqs2 splits (5 seeds each). Mean ± sd test MCC:

| features | model | random MCC | random AUC | mmseqs2 MCC | mmseqs2 AUC |
|---|---|---:|---:|---:|---:|
| descriptors (6) | logreg | +0.255 ± 0.045 | 0.637 | +0.224 ± 0.028 | 0.640 |
| descriptors (6) | gbm | +0.547 ± 0.014 | 0.845 | +0.457 ± 0.048 | 0.798 |
| ESM-2 (640) | logreg | +0.642 ± 0.031 | 0.895 | +0.591 ± 0.041 | 0.879 |
| ESM-2 (640) | gbm | +0.700 ± 0.021 | 0.920 | +0.642 ± 0.034 | 0.893 |
| **ESM-2 + descriptors** | **gbm** | +0.686 ± 0.015 | 0.919 | **+0.650 ± 0.026** | **0.895** |
| peptox rule (untrained) | — | +0.105 ± 0.011 | — | +0.044 ± 0.046 | — |

The random→MMseqs2 drop is small (+0.700 → +0.642), so this is not memorised
near-duplicates. For peptides from this population the model is genuinely good.

**The heuristic's failure was the rule, not the features.** A GBM on the *same
six descriptors* scores +0.457 where the hand-tuned rule scores +0.044 — ten
times better from identical inputs. Two hard thresholds ANDed together throw
away almost everything six continuous features contain. The mechanistic story
justified the features; it did not justify the functional form, and that
distinction was not checked before shipping it.

## Why the ESM-2 model cannot be applied off-family

This section is about the **ESM-2 model**, now in `archive/` — not the
descriptor model that `infer.py` ships. It is the better model in-family and
the reason the repository does not use it.

Scoring the 21 designs produces numbers that are confidently backwards:

| design | P(hemolytic) |
|---|---|
| `cationic-1` (q +5.0, ⟨µH⟩ 0.40, cationic amphipathic) | **0.03** |
| `cationic-2` (q +4.0, ⟨µH⟩ 0.43, cationic amphipathic) | **0.00** |
| `anionic-1` (q −3.0, Gly/Ser-rich) | **0.95** |
| `anionic-2` (q −1.0, Gly/Ser-rich) | **0.84** |

Across all 21, **corr(P, amphipathicity) = −0.33** and corr(P, charge) = −0.09.
The model ranks anionic glycine-rich chains as more hemolytic than cationic
amphipathic ones, inverting the best-established structure–activity relation in
membrane biophysics. On in-domain batches the same correlation is positive
(+0.34 on canonical references, +0.21 on 60 random corpus peptides), so this is
specific to the designs, not a property of the model everywhere.

Independent corroboration: glucagon, a non-hemolytic hormone at k-mer
containment 0.07 from the corpus, scores 0.77.

The mechanism of the failure is visible in the corpus itself:
**corr(hemolytic label, net charge) = +0.067.** Within a corpus where both
classes are AMPs, charge carries almost no signal, so the model never learned a
mechanistic axis. It learned to recognise AMP families. Given a sequence from no
family it has seen, it has nothing to fall back on and returns noise.

Both scoring paths now compute corr(P, amphipathicity) on every batch and warn
if it is not positive (`infer.py`, and `archive/predict.py` which refuses
outright). That check is a tripwire for this failure mode, not a guarantee
against it.

### The applicability tiers were also wrong

`domain.py` bins 2,160 held-out MMseqs2 predictions by k-mer containment to
their training fold:

| containment to train | n | base rate | balanced acc | MCC |
|---|---:|---:|---:|---:|
| 0.00–0.15 | 461 | 0.132 | 0.710 | +0.483 |
| 0.15–0.25 | 549 | 0.191 | 0.722 | +0.459 |
| 0.25–0.40 | 445 | 0.342 | 0.729 | +0.470 |
| 0.40–0.60 | 241 | 0.490 | 0.783 | +0.569 |
| 0.60–1.01 | 464 | 0.662 | 0.891 | +0.772 |

Read naively this says the model keeps useful skill (MCC +0.48) even on the most
dissimilar peptides, and the designs sit at 0.10–0.25, so they should be scored
with balanced accuracy ≈ 0.71. **That inference is invalid and the predictions
above are what invalidates it.** Every peptide in the lowest bin is still an AMP
from the same corpus; it is far from its training *fold*, not far from the
*population*. The designs are far in a different sense that this measurement
never sampled. A within-family distance cannot bound out-of-family error, and
k-mer containment does not distinguish the two cases.

(Note also that accuracy per bin is confounded: the low-similarity bins have base
rate 0.13 and the high ones 0.66, so raw accuracy appears U-shaped while MCC
falls monotonically. The first version of `domain.py` reported accuracy only and
read as "no degradation at all".)

## Descriptor model vs ESM-2: the extrapolation test

`extrapolate.py` tests the hypothesis that mechanistic descriptors transfer
where embeddings do not. `features.py` supplies 18 of them — charge and charge
density, helical *and* sheet hydrophobic moments, Kyte-Doolittle and
Wimley-White hydrophobicity, composition fractions, longest hydrophobic run,
aliphatic index, Boman index. None encodes sequence identity.

Two evaluations. **In-family** is the MMseqs2 split: held out by similarity, but
train and test are both AMPs. **Extrapolation** trains on corpus peptides with
net charge ≥ 0 and tests on the 430 with charge < 0 — the anionic regime this
project's designs occupy (median −1.0), with nothing like it in training.

| features | in-family MCC | extrapolation MCC | degradation |
|---|---:|---:|---:|
| descriptors (18) | +0.553 ± 0.025 | **+0.474** | **−0.079** |
| ESM-2 (640) | **+0.642 ± 0.034** | +0.482 | −0.160 |
| ESM-2 + descriptors | +0.641 ± 0.028 | +0.471 | −0.170 |

ESM-2 wins in-family by a clear margin and loses twice as much skill going
off-distribution. On MCC the two end up tied — but MCC hides what actually
happens, and the confusion matrices do not.

### Confusion matrices on the anionic extrapolation set (n=430, 43.0% positive)

**descriptors (18)**

|  | pred non-hemolytic | pred hemolytic |  |
|---|---:|---:|---|
| **true non-hemolytic** | 189 (TN) | 56 (FP) | spec 0.771 |
| **true hemolytic** | 55 (FN) | 130 (TP) | **sens 0.703** |
|  | NPV 0.775 | PPV 0.699 | bal acc **0.737** |

**ESM-2 (640)**

|  | pred non-hemolytic | pred hemolytic |  |
|---|---:|---:|---|
| **true non-hemolytic** | 224 (TN) | 21 (FP) | spec 0.914 |
| **true hemolytic** | **89 (FN)** | 96 (TP) | **sens 0.519** |
|  | NPV 0.716 | PPV 0.821 | bal acc 0.717 |

ESM-2 does not degrade symmetrically — it collapses toward the majority call,
**missing 89 of 185 hemolytic peptides**. Its equal MCC is bought with
specificity that is worthless for a safety screen. The descriptor model misses
55. For this purpose sensitivity is the column that matters, and it is 0.703
against 0.519.

For reference, in-family the ordering reverses: ESM-2 reaches spec 0.899 /
PPV 0.792 against the descriptors' 0.825 / 0.688, at near-identical sensitivity.

### Direction check on the 21 designs

| model | corr(P, amphipathicity) | corr(P, charge) |
|---|---:|---:|
| **descriptors** | **+0.455** | −0.092 |
| ESM-2 | **−0.326** | −0.089 |

The descriptor model moves the right way; the embedding model is inverted. That
is the practical result: **use the descriptor model here.**

### What it keys on, and where it still disagrees

Permutation importance on the anionic test set: `muH_helix_mean` (+0.102),
`ww_window_max` (+0.075), `frac_C` (+0.073), `boman` (+0.045),
`muH_sheet_max` (+0.032). Mechanistic quantities, which is why they travel.

Two cautions. `frac_C` ranks third because disulfide-rich peptides (defensins)
are non-hemolytic — and every design here has zero cysteine, so that feature is
constant across the batch and contributes nothing where it is needed.

And the model disagrees with the heuristic on the two cationic designs, scoring
`cationic-1` 0.05 and `cationic-2` 0.06 despite charges of +5 and +4. The
disagreement is explainable: their ⟨µH⟩ of 0.40–0.43 sits near the **25th
percentile of the corpus**, whose median is 0.59. By AMP standards they are not
especially amphipathic. The heuristic's 0.35 threshold was calibrated against
the design set, not against actual AMPs, which makes it a reasonable drift
detector and a poor absolute scale. Both readings can be right: that design
lane has drifted toward AMP space, and has not arrived.

Top of the descriptor model's ranking: `helix-1` 0.97, `helix-2` 0.88,
`shuffle-1` 0.77, `sheet-1` 0.75. `helix-1` is top of both the model and the
heuristic — the one design two independent methods agree on.

## The heuristic, and what it is good for

`peptox.py` flags a peptide when net charge ≥ +2.0 **and** max ⟨µH⟩ (Eisenberg,
11-residue window) ≥ 0.35. Charge uses side-chain pKa values and both termini;
the moment slides an 11-residue window and keeps the maximum, so a 33-mer with
one amphipathic end is caught where a whole-sequence average would hide it. The
flag reports which window it found.

Held out on all 4,306 peptides (`evaluate.py`, thresholds locked):

| rule | sens | spec | PPV | MCC |
|---|---:|---:|---:|---:|
| peptox FLAG | 0.658 | 0.432 | 0.543 | +0.093 |
| charge only | 0.711 | 0.348 | 0.527 | +0.063 |
| moment only | 0.878 | 0.209 | 0.532 | +0.117 |

PPV 0.543 against a base rate of 0.506, and the conjunction is beaten by one of
its own halves. **It does not predict hemolysis.** The class distributions say
why — the negatives are themselves cationic (median q +3.01) and amphipathic
(median ⟨µH⟩ 0.49). It is AMPs versus AMPs, and "is it an AMP" cannot separate
within that.

What it does do is measure distance from AMP space, and that is well behaved: it
fires on 73% of the corpus and 14% of the designs, which sit at the **1st–10th
percentile** of corpus charge. For "has a generation stage wandered somewhere
unintended", it is reliable, because that is a statement about a distribution
rather than a prediction of a biological outcome.

### The in-sample validation that started this

`validation.csv` holds ten canonical references; the rule gets 5/5 hemolytic and
4/5 non-hemolytic. **That number is meaningless** and is kept as a record. It is
in-sample — the thresholds were fit to it, and a bulk-hydrophobicity criterion
was dropped precisely because indolicidin and LL-37 failed on it. Worse, the
negatives were insulin, glucagon, GLP-1 and poly-E: hormones and a homopolymer.
Easy negatives made a rule that separates AMPs from non-AMPs look like a rule
that separates hemolytic from non-hemolytic. Supplying the negatives that matter
— non-hemolytic AMPs — removed the apparent skill entirely.

Both wrong turns recorded here have the same shape: a test set that asked an
easier question than the one being claimed.

## Result on the design set

The 21 designs are 25–34 residues, 18 of them anionic-to-neutral and Gly/Ser-rich
(median net charge −1.0). The sequences are unpublished and not included; the
findings are reported here because the failure modes they exposed are the point
of this repository. The heuristic flags 3:

| design | source stage | q | ⟨µH⟩ | corpus q percentile |
|---|---|---:|---:|---:|
| `cationic-1` | language-model variant | +5.0 | 0.40 | 73 |
| `cationic-2` | language-model variant | +4.0 | 0.43 | 54 |
| `helix-1` | structure-generative | +3.0 | 0.81 | 47 |

Both cationic flags come from the same stage of the upstream pipeline — a step
that resamples residues under a protein language model — in two independently
seeded runs. That makes it a property of the stage, not of one design: a
language model's prior for a probable residue walks cationic and hydrophobic,
because that is what natural sequence looks like. Every design from the other
stages, and every shuffled null, sits in the bottom decile of corpus charge.

`helix-1` comes from a structure-generative stage instead: an Ala-rich helix
with arginines spaced onto one face, ⟨µH⟩ 0.81 (90th percentile). Its three
siblings from the same stage are anionic and pass.

A flag licenses *look at this one*. Not *discard*, and not *toxic*.

## Training data

Both shipped checks are built on **one** file: `data/hemolytic/all.csv` —
PepBenchmark's `hemolytic` set, 4,306 sequences, two columns (sequence, binary
label), 50.6% positive. The ranking model trains on all of it; the heuristic's
thresholds were never fitted to it and it serves only as that rule's test set.

### What is documented about it: almost nothing

The shipped files carry sequence and label and no other field. The HuggingFace
dataset card, the repository's dataset registry (`src/pepbenchmark/metadata.py`,
which records only `group=TOX, type=BINARY_CLASSIFICATION`) and the paper
abstract were all checked, and none states the source databases, the assay, the
erythrocyte species, the concentration convention, the length filter, or what
the `-50` in `PepBenchData-50` refers to. No data licence is stated anywhere.
PepBenchmark's own paper argues that preprocessing and label conventions vary
so much between hemolysis studies that cross-paper comparison is unreliable;
this dataset does not escape that criticism.

The registry also contains a **commented-out `hemolytic_strict` entry**, which
suggests the maintainers were themselves unsettled on where the label boundary
belongs.

So the characterisation below is derived from the data rather than taken from
documentation. Reproduce it by joining `data/hemolytic/all.csv` against
`data/hemolytic_hc50/all.csv` on the sequence column (both come down with
`fetch_data.py --all`).

### What the label actually means (derived)

848 sequences appear in both `hemolytic` and `hemolytic_hc50`, which allows the
binary label to be read against measured HC50:

| binary label | n joined | median HC50 | range (log10 µM) |
|---|---:|---:|---|
| 1 (hemolytic) | 838 | **43 µM** | −0.72 to 3.00 |
| 0 (non-hemolytic) | 10 | **181 µM** | 1.94 to 2.70 |

The boundary sits somewhere near 100 µM — **but it is not a clean threshold.**
220 of 838 positives have an HC50 *above* the lowest negative's. The label is
inherited from heterogeneous source annotations, not computed from a single cut,
and the two classes genuinely overlap in potency. A model trained on it is
learning a fuzzy boundary, and the 0.5 decision point inherits that fuzziness.

Note also the asymmetry: only 10 of ~2,128 negatives have an HC50 at all,
because a peptide that never reaches 50% lysis has no HC50 to report. So
`hemolytic_hc50` is effectively the positive class with quantitation attached,
not an independent dataset.

### The negatives are real, which matters

The repository ships a `NegSampler` for building assumed-negative pools, and a
corpus whose negatives were drawn that way would reproduce the failure this
project started out warning about (ToxinPred-style, where the model learns
"is this a toxin-like sequence" instead of the endpoint). These negatives are
not that:

- A logistic model on **amino-acid composition alone** separates the classes at
  **AUC 0.796**. Pool-sampled negatives would give ~0.99.
- The negatives are themselves cationic (median charge +3.01) and amphipathic
  (median ⟨µH⟩ 0.49) — they are AMPs that were tested and found not to lyse.

That is the right kind of negative, and it is also why the task is hard: the
corpus is AMPs versus AMPs throughout, which is the single fact that explains
the heuristic's near-chance MCC and the ESM-2 model's off-distribution collapse.

Composition differences between the classes are mechanistically sensible —
**Trp is 2.78× enriched** in the hemolytic class (3.97% vs 1.43%), Met is
depleted 0.28×. Tryptophan is the canonical membrane-interface anchor residue,
which is the same point that makes Kyte-Doolittle the wrong hydrophobicity
scale for this task — `features.py` carries Wimley-White alongside it for
exactly that reason.

### Hygiene, checked

| check | result |
|---|---|
| duplicate sequences | **0** |
| sequences with conflicting labels | **0** |
| non-standard residues | 2 sequences with a trailing space; stripped on load |
| length range | **2 to 50**, median 19 |
| peptides ≤ 5 residues | **109**, 35.8% labelled hemolytic |
| peptides in the design length range (25–40) | 1,134, 46.6% positive |

The short end looks alarming — the corpus contains dipeptides (`HA`, `AF`,
`FT`, `TE`) and a third of the ≤5-residue entries are labelled hemolytic. They
were measured rather than assumed to be harmful, and they are not: see
[Short peptides](#short-peptides-and-what-testing-for-them-exposed). They stay
in.

The slice that matters for this project — length 25–40 — is 1,134 peptides at
46.6% positive, so the design length range is well represented even though the
design *composition* range is not.

## Short peptides, and what testing for them exposed

The corpus contains dipeptides, and 109 peptides are <= 5 residues at 35.8%
labelled hemolytic. An earlier pass over this corpus dismissed them as noise on
no evidence beyond implausibility. `shortpep.py` tests the claim. Two concerns were
conflated and only one is checkable: whether the **labels** are wrong cannot be
verified from here, but whether the **features** are comparable can — `muH` is
computed over `min(11, len)` residues, so the column does not mean the same
thing at length 2 as at length 33, regardless of the labels.

Neither matters unless predictions move at design lengths. Training with
increasing length floors, tested only on corpus peptides of length >= 20:

| training length floor | train n | MCC (test len>=20) | AUC |
|---:|---:|---:|---:|
| 0 (all) | 4306 | +0.402 ± 0.094 | 0.806 |
| >= 6 | 4197 | +0.426 ± 0.088 | 0.808 |
| >= 11 | 3798 | +0.380 ± 0.070 | 0.814 |
| >= 15 | 2970 | +0.390 ± 0.123 | 0.800 |

**They do no measurable harm.** Every floor lands inside one standard deviation
of every other. There is no case for removing them, and they stay in.

### The instability that test uncovered

The same perturbations move the *designs* enormously. Dropping only the 109
shortest peptides — 2.5% of rows — shifts design probabilities by up to 0.376
and flips two across 0.5. At floor 15, `anionic-3` goes **0.63 to 0.10**.

Over 16 perturbed fits (4 length floors + 12 bootstrap resamples) the median
10th-90th percentile band on a design is **0.41 wide**. A point estimate from a
single fit is not meaningful at the level of an individual design, however many
decimal places it is printed to — an earlier version of this analysis reported
exactly that and should not have.

This is not a fact about short peptides. It is what being off-distribution
costs: the designs sit where the training data is thin, so the fitted surface
there is decided by a handful of points and changes when they do. Corpus-level
accuracy is unaffected, which is exactly why the earlier evaluations did not
catch it.

`infer.py` therefore reports a band over 12 bootstrap resamples instead of a
number, and gives a verdict only where the whole band falls on one side of 0.5.
On the 21 designs that leaves **1 elevated, 9 low, 11 unresolved** — an honest
accounting that a single fit disguised as 7 confident positives and 14
negatives. `helix-1` survives as the one elevated call (band 0.62–0.99), and
the tightly-bounded low calls (the cationic designs at 0.01–0.06) are the other
place the model is actually saying something.

### A caution about the data release: `nc-hemolytic` is internally inconsistent

Downloaded while investigating whether the short entries are stripped
lipopeptides. Its columns do not describe the same molecules:

| check | agreement |
|---|---:|
| FASTA length == HELM monomer count | 3376/3904 (86.5%) |
| FASTA composition == HELM composition | **69/3904 (1.8%)** |
| FASTA length == stated `length` column | 3904/3904 (100%) |

Row 1 is FASTA `AAAAMAGPKA` (10 residues) against HELM `{meA.meA.A.Me_dA.Me_dA.V.Q}`
(7 monomers); `WLKKWN` against `{F.W.C.L.W.W}`. FASTA and `length` agree with
each other and with nothing else, so the HELM/BILN/SMILES columns appear
misaligned against the FASTA/label pair. **Do not use `nc-hemolytic`** without
resolving which column carries the label.

The canonical `hemolytic` set used here has only sequence and label, so there is
nothing to cross-check and no evidence it is affected. But a column-alignment
fault in a sibling dataset from the same release is a reason to treat the whole
collection as unverified, which reinforces the point of
[Training data](#training-data): none of its provenance is documented.

### The control: do the bands track distance from the data?

A band is only useful if it widens where the model is ignorant rather than
everywhere. Running the same 12-resample ensemble over `validation.csv`, whose
references sit close to the corpus, against the designs, which do not:

| set | n | median k-mer sim to train | median band width | unresolved |
|---|---:|---:|---:|---:|
| references (`validation.csv`, shipped) | 10 | 0.94 | **0.07** | 1/10 |
| designs (not shipped) | 21 | 0.17 | **0.39** | 11/21 |

**5.6x wider on the designs**, and across all 31 peptides
corr(k-mer similarity, band width) = **−0.457**. The width is an
off-distribution effect, not general flakiness: the model is ignorant
specifically where the training data is thin.

The widest reference band is the one that demonstrates the mechanism.
**Glucagon** — k-mer similarity 0.07, the only reference genuinely far from the
corpus — comes back 0.20–0.72, median 0.50, *unresolved*. It is a non-hemolytic
hormone, and the archived ESM-2 model called it hemolytic at 0.77 with no
hedge. The band model declines to call it. The one far-from-corpus reference is
the one reference it refuses, which is the behaviour the band is for.

Everything in-domain lands correctly and tightly: indolicidin 1.00–1.00, LL-37
0.97–1.00, melittin 0.92–0.99, pexiganan 0.83–0.99. **Buforin-II** — the
cationic amphipathic translocator that is a false positive for the heuristic
and for every composition rule — is correctly **low** at 0.01–0.07. (It is in
the training corpus labelled 0, so this is recall, not generalisation.)

### A narrow band is not a correctness guarantee

`insulin-A` returns **elevated at 0.87–0.98**. The model is confidently calling
the insulin A chain hemolytic, and it is not wrong about its training data:
that sequence is in `data/hemolytic/all.csv` labelled 1. The tight band means
the prediction is well supported by the corpus — it says nothing about whether
the corpus is right.

So the two readings of a narrow band must be kept apart:

- **wide band** -> not enough training data nearby; the model does not know.
- **narrow band** -> plenty of training data nearby; the model reproduces what
  that data says, *including any error in it*.

Given that the label definition is undocumented, inconsistent at the boundary
(220 of 838 positives have an HC50 above the lowest negative's) and contains at
least one entry as questionable as insulin, a narrow band should be read as
"consistent with the corpus", never as "true".

## How the two shipped checks were built

Neither was tuned against its reported evaluation. The gap between "fitted" and
"measured" is where this project went wrong twice, so the provenance is spelled
out.

### 1. The drift check (`peptox.py`)

Not trained — a two-term rule, `net charge >= +2.0 AND max <uH> >= 0.35`.

**Threshold provenance.** Both sit at the low edge of the active range reported
for helical AMPs, chosen to err toward flagging. They were then adjusted once,
in-sample, against `validation.csv`: a third criterion (bulk hydrophobicity)
was dropped because indolicidin and LL-37 failed on it. **That makes the
thresholds fitted, not calibrated**, and the `validation.csv` scores worthless
as evidence. Everything in `evaluate.py` ran afterwards with the thresholds
frozen, which is why those numbers (MCC +0.093) count and the earlier ones do
not.

**Known miscalibration.** The 0.35 moment cutoff sits near the **25th
percentile** of the 4,306-peptide corpus, whose median is 0.54. It is a line
drawn relative to this project's designs, not to real AMPs — fine for detecting
drift, wrong as an absolute statement about amphipathicity. This is why the rule
flags the esm2 variants that the model scores at 0.05.

### 2. The ranking model (`infer.py`)

`sklearn.ensemble.HistGradientBoostingClassifier(max_iter=300,
learning_rate=0.08, random_state=0)` on the 18 features in `features.py`. No
scaling (trees do not need it); no class weighting (the corpus is 50.6%
positive).

**Hyperparameters, swept.** `sweep.py` runs a 108-configuration grid
(learning_rate x max_leaf_nodes x min_samples_leaf x l2_regularization x
max_iter), selecting on each MMseqs2 seed's `valid` fold and reporting on
`test`, so selection and reporting never touch the same data. The grid is flat:

| | tuned | shipped defaults | gain |
|---|---:|---:|---:|
| in-family (5 MMseqs2 seeds) | +0.552 ± 0.042 | **+0.553 ± 0.025** | −0.000 |
| extrapolation (anionic test) | +0.487 (AUC 0.806) | +0.474 (AUC 0.780) | +0.013 |

Tuning buys nothing in-family and +0.013 MCC extrapolating, on a single split
with no interval — inside noise. It also *raises* variance (±0.042 against
±0.025), because selecting on a ~430-peptide validation fold is noisy enough to
pick a worse configuration about as often as a better one. The defaults ship,
now because they were tested rather than because they were never questioned.

**A protocol confound found on the way.** The first version of `sweep.py` used
`early_stopping=True`, which carves 15% out of train for an internal stop. That
alone cost **~0.13 MCC** on the extrapolation split (+0.342 against +0.474) —
far more than anything in the grid, and it initially made tuning look actively
harmful. Protocol choices dominated hyperparameters here, which is worth
knowing before anyone sweeps harder.

**What it is trained on.** All 4,306 peptides in `data/hemolytic/all.csv`,
refit on every run — see [Training data](#training-data) for what that corpus
is and what is not documented about it. The reported metrics come from separate
runs on held-out splits (`extrapolate.py`, `sweep.py`, `train.py`), never from
the shipped fit, so the model you run has seen more data than the one measured
and the metrics are if anything a floor.

**Decision threshold.** 0.5, unchosen and unswept. `infer.py` ranks rather than
thresholds, so it mostly does not matter — but a 0.5 cut is not an optimised
operating point and should not be read as one.

**Calibration — measured, not assumed.** Probabilities are raw GBM outputs with
no Platt or isotonic correction. Observed frequency per predicted-probability
bin:

| predicted P | in-family: n / observed | extrapolation: n / observed |
|---|---:|---:|
| 0.0–0.2 | 1020 / **0.109** | 134 / **0.194** |
| 0.2–0.4 | 255 / 0.212 | 78 / 0.295 |
| 0.4–0.6 | 187 / 0.358 | 67 / 0.358 |
| 0.6–0.8 | 176 / 0.517 | 52 / 0.673 |
| 0.8–1.0 | 522 / 0.805 | 99 / 0.778 |
| | Brier **0.151** | Brier **0.193** |

Monotone in both regimes, so the *ordering* is trustworthy — which is what
`infer.py` uses it for. But the numbers are not probabilities you can bet on:
in-family the model is over-confident through the middle (0.6–0.8 predicted →
0.517 observed), and off-distribution the bottom bin degrades most. **A score
below 0.2 on an off-distribution peptide still corresponds to roughly a 1-in-5
chance of being hemolytic** (0.194), against 1-in-9 in-family. That is the
quantitative form of "a pass is not a safety claim".

## Known limits

- **The helix assumption is universal and unchecked.** ⟨µH⟩ uses 100°/residue;
  for a β-strand the periodicity is ~180° and the helical value is
  meaningless. One design in the set (`sheet-1`) is a three-stranded sheet by
  its predicted structure; `features.py` computes a 180° moment alongside the
  helical one, but nothing selects between them automatically.
- **Hemolysis is HC50, not a label.** The corpus binarises a concentration.
  `data/hemolytic_hc50/` (1,926 peptides, log HC50) is downloaded and unused;
  regression would at least keep the dose.
- **Corpus labels are not above question, and the model reproduces them
  confidently.** The insulin A chain appears in the corpus labelled hemolytic,
  and the ensemble returns it at 0.87–0.98. A narrow band means well-supported
  by the training data, not correct — see
  [A narrow band is not a correctness guarantee](#a-narrow-band-is-not-a-correctness-guarantee).
- **One mechanism only.** Aggregation, protease lability and immunogenicity are
  uncovered and are plausibly larger risks here: two designs (`polyA-1`,
  `polyA-2`) are near-poly-alanine 33-mers (GRAVY +2.20, +1.88) that pass both
  checks correctly and are the likeliest amyloid risks in the set.
- **A pass is not a safety claim.**

## What would actually work

Not a better threshold or a bigger model on this corpus — the obstacle is that
the training population and any non-AMP design population barely overlap. Options, in
order of how much they buy:

1. **Measure a handful.** A hemolysis assay on 3–6 designs spanning the charge
   range would do more than any further modelling, and would give the first
   in-domain labels this project has.
2. **Train on HC50 regression** (`data/hemolytic_hc50/`) with the designs'
   composition range explicitly represented, rather than classification on a
   corpus that contains almost no anionic peptides.
3. **Keep the heuristic as a drift tripwire** on generated sequences, which is
   the one job here that is validated.

## Files

| file | what it is |
|---|---|
| `infer.py` | **entry point** — drift check + ranking, use cases labelled |
| `fetch_data.py` | downloads the PepBenchmark corpora into `data/` |
| `peptox.py` | the heuristic; `screen(seq)` returns descriptors and verdict |
| `evaluate.py` | heuristic held-out evaluation and ablations; no tuning |
| `train.py` | model comparison across feature sets and split types |
| `domain.py` | accuracy vs sequence distance; sets (and limits) the tiers |
| `features.py` | 18 mechanistic descriptors, helix and sheet periodicity |
| `extrapolate.py` | descriptors vs ESM-2 off-distribution; confusion matrices |
| `sweep.py` | 108-config hyperparameter grid, selected on `valid` |
| `shortpep.py` | length-floor ablation and design-prediction stability |
| `requirements.txt` | numpy + scikit-learn; that is all the kept code needs |
| `archive/` | the retired ESM-2 path, with why — see `archive/README.md` |
| `validation.csv` | 10 canonical references; example manifest and band control |

Not in the repository:

| path | what it is |
|---|---|
| `data/hemolytic/` | PepBenchmark, 4,306 peptides, binary (+ ESM-2 embeddings) — `fetch_data.py` |
| `data/hemolytic_hc50/` | PepBenchmark, 1,926 peptides, log HC50 (unused) — `fetch_data.py --all` |
| `data/toxicity/` | PepBenchmark general toxicity set (unused) — `fetch_data.py --all` |
| `data/nc-hemolytic/` | non-canonical set — **columns inconsistent, do not use** |
| `designs.csv` | your own manifest, if you have one; `.gitignore`d by default |

The ESM-2 scoring path is in `archive/`. It is the better model *in-family*
(MCC +0.642 vs +0.553) and the wrong one here; `archive/README.md` has the
evidence and the embedding convention, which is non-obvious and worth keeping.

Data from [PepBenchmark](https://github.com/ZGCI-AI4S-Pep/PepBenchmark)
(Zhang et al., ICLR 2026) via
[HuggingFace](https://huggingface.co/datasets/jiahuizhang/PepBenchData).
