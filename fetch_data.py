"""Download the PepBenchmark data this repository trains on.

The data is not vendored here. Its upstream release states no licence (see the
README's Training data section), so this fetches it from the original
HuggingFace repository rather than redistributing it.

    python fetch_data.py              # the hemolytic set -- all that infer.py needs
    python fetch_data.py --all        # also hemolytic_hc50, toxicity, embeddings

Writes into data/<set>/. Files already present are left alone. Needs only the
standard library.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

BASE = ("https://huggingface.co/datasets/jiahuizhang/PepBenchData/"
        "resolve/main/PepBenchData-50")

# The two files infer.py, train.py, sweep.py and shortpep.py actually read.
CORE = {"hemolytic": ["all.csv", "mmseqs2_split.json"]}
# extrapolate.py and domain.py want the splits and the precomputed embeddings;
# hemolytic_hc50 and toxicity are referenced by the README as unused options.
EXTRA = {
    "hemolytic": ["random_split.json", "fasta.csv", "label.csv",
                  "esm2_150_embedding.npz"],
    "hemolytic_hc50": ["all.csv", "mmseqs2_split.json"],
    "toxicity": ["all.csv", "mmseqs2_split.json"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="also fetch embeddings and the unused sets")
    args = ap.parse_args()

    wanted = {k: list(v) for k, v in CORE.items()}
    if args.all:
        for k, v in EXTRA.items():
            wanted.setdefault(k, []).extend(v)

    failed = 0
    for dataset, names in wanted.items():
        os.makedirs(f"data/{dataset}", exist_ok=True)
        for name in names:
            dest = f"data/{dataset}/{name}"
            if os.path.exists(dest):
                print(f"  have  {dest}")
                continue
            url = f"{BASE}/{dataset}/{name}"
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  got   {dest}  ({os.path.getsize(dest)/1e6:.1f} MB)")
            except urllib.error.HTTPError as e:
                # Layouts shift between releases; report rather than guess.
                print(f"  FAIL  {url} -> HTTP {e.code}")
                failed += 1

    if failed:
        print(f"\n{failed} file(s) could not be fetched. Browse "
              "https://huggingface.co/datasets/jiahuizhang/PepBenchData and "
              "place them under data/<set>/ by hand.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
