"""Download the PepBenchmark data this repository trains on, and verify it.

The data is not vendored here. Its upstream release states no licence (see the
README's Training data section), so this fetches it from the original
HuggingFace repository rather than redistributing it.

    python fetch_data.py              # the hemolytic set -- all infer.py needs
    python fetch_data.py --all        # also hemolytic_hc50, toxicity, embeddings
    python fetch_data.py --check      # re-verify what is already on disk

Every file is checked against the SHA-256 recorded below, which is the version
all the numbers in the README were measured on. A mismatch means upstream has
changed the data and those numbers no longer describe what you have; the file
is kept, not deleted, so you can diff it.

Writes into data/<set>/. Files already present are verified, not re-fetched.
Standard library only.
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

BASE = ("https://huggingface.co/datasets/jiahuizhang/PepBenchData/"
        "resolve/main/PepBenchData-50")

# path -> sha256, measured 2026-10-13. CORE is what infer.py, train.py,
# sweep.py and shortpep.py read; EXTRA adds the precomputed ESM-2 embeddings
# that extrapolate.py and domain.py want, plus the two sets the README
# discusses but does not use.
CORE = {
    "hemolytic/all.csv":
        "fb3f59115c46ee51c403497c47729b359c7cc85fef5514928b7894f38732c9ba",
    "hemolytic/mmseqs2_split.json":
        "95e16ac01cee57e32c32d26a5a7a8104656499b3aace9933c0a935d2ee8bd0d9",
}
EXTRA = {
    "hemolytic/random_split.json":
        "bed7b40003acb155ede015a3475fdb2e2a3b5942e4c7ef98e93c18674c94e491",
    "hemolytic/fasta.csv":
        "7998404f8f40eb787c638fc5db8ae1fab9c4600316d5926470ff918e188e89e8",
    "hemolytic/label.csv":
        "b8a4dd1651127a1096c73d95652a1a402af81c9be2b67f792089692463eabf15",
    "hemolytic/esm2_150_embedding.npz":
        "c68f246535cf44799eb105dd8fd6ff29ec9461e903d45b1a7a3fbc5e4690dd66",
    "hemolytic_hc50/all.csv":
        "931f56879e6582acfb882fa524caee9172cbccf03da4076dc79ae374d8eb7951",
    "hemolytic_hc50/mmseqs2_split.json":
        "5a20b758ad93a5cd97e9b5b146801ac602235e3cd5e91be49b01ddf574383a89",
    "toxicity/all.csv":
        "c4fa3153be05b1e68b3aa37aeb279d5c5fa0dafdefbe84f6f4308db1e4abb6ec",
    "toxicity/mmseqs2_split.json":
        "2e078f7e4fbaa12ebd65bfd15d0ecccf8dd4e2cfe95ead304dbe41f9853b2a1a",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="also fetch embeddings and the unused sets")
    ap.add_argument("--check", action="store_true",
                    help="verify files already on disk; download nothing")
    args = ap.parse_args()

    wanted = dict(CORE)
    if args.all or args.check:
        wanted.update(EXTRA)

    bad = []
    for rel, want in wanted.items():
        dest = f"data/{rel}"
        if not os.path.exists(dest):
            if args.check:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                urllib.request.urlretrieve(f"{BASE}/{rel}", dest)
            except urllib.error.HTTPError as e:
                # Layouts shift between releases; report rather than guess.
                print(f"  FAIL  {rel} -> HTTP {e.code}")
                bad.append(rel)
                continue
            verb = "got "
        else:
            verb = "have"

        got = sha256(dest)
        if got == want:
            print(f"  {verb}  {dest}  ({os.path.getsize(dest)/1e6:.1f} MB, sha ok)")
        else:
            print(f"  {verb}  {dest}\n        *** SHA-256 MISMATCH ***"
                  f"\n        expected {want}\n        got      {got}")
            bad.append(rel)

    if bad:
        print(f"\n{len(bad)} file(s) missing or changed upstream. The README's "
              "numbers were measured on the recorded versions and do not "
              "describe anything else. Browse\n  "
              "https://huggingface.co/datasets/jiahuizhang/PepBenchData")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
