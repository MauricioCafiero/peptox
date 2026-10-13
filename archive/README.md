# archive — the ESM-2 inference path

Retired. Kept because the evidence against it is the most useful result this
project produced, and because the embedding code is correct and non-obvious.

## What is here

| file | what it is |
|---|---|
| `predict.py` | scores peptides with a GBM on ESM-2 150M embeddings |

## Why it was retired

The model itself is good: **MCC +0.642 on MMseqs2 splits**, the best in-family
result obtained here, beating the 18-descriptor model's +0.553. The problem is
only what happens off-distribution.

On the (unpublished, not included) design set this project was built for, it
produced predictions that invert the mechanism — design labels anonymised, as
in the main README:

| | P(hemolytic) |
|---|---|
| `cationic-1` (q +5.0, cationic amphipathic) | 0.03 |
| `cationic-2` (q +4.0, cationic amphipathic) | 0.00 |
| `anionic-1` (q −3.0, anionic Gly/Ser-rich) | 0.95 |

corr(P, amphipathicity) = **−0.33** across all 21, where the descriptor model
gives +0.46. On the anionic extrapolation test set it degrades asymmetrically,
collapsing to the majority call and missing 89 of 185 hemolytic peptides
(sensitivity 0.519 against the descriptor model's 0.703).

The cause: within this corpus, corr(hemolytic label, net charge) = +0.067.
Both classes are AMPs, so charge carries no signal and the model never had to
learn a mechanistic axis. It learned to recognise AMP families instead, and
has nothing to fall back on for a sequence from no family it has seen.

## When it would be the right tool

If you are scoring peptides that *are* AMPs — literature sequences, corpus-like
candidates, anything with k-mer containment above ~0.4 to
`../data/hemolytic/all.csv` — this is the better model and the descriptor one
is not. `predict.py` reports that containment per peptide.

## Running it

Needs `torch` and `transformers` (~750 MB), deliberately not in
`../requirements.txt`:

```sh
uv pip install --python ../.venv/bin/python torch transformers
cd .. && .venv/bin/python archive/predict.py validation.csv
```

It trains on `../data/hemolytic/` (run `fetch_data.py --all` first — it needs
the precomputed embeddings) and expects to be run from the parent directory. It also carries its own sanity check and will print
`*** DO NOT USE THESE SCORES ***` on a batch like the designs.

## The one part worth copying

The embedding convention. `predict.py` mean-pools `esm2_t30_150M_UR50D` over
**all unpadded tokens including BOS and EOS**, which reproduces the corpus
embeddings in `../data/hemolytic/esm2_150_embedding.npz` exactly
(cosine 1.000000). Dropping BOS/EOS — the more natural choice — gives 0.9993
and silently shifts every feature. Anything that re-embeds against that corpus
has to match it.
