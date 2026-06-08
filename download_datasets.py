"""
Download s1K, m1K, OpenCodeReasoning, and ARPO-SFT datasets into data/raw/.

Usage:
    python download_datasets.py [--token HF_TOKEN] [--out-dir data/raw]

Each dataset is fetched via huggingface_hub.snapshot_download (repo_type="dataset")
and stored in its own subdirectory under --out-dir, named after the dataset key
(s1k, m1k, opencodereasoning, arpo_sft).

If a repo id below turns out to be wrong/renamed on the Hub, just edit DATASET_REPOS.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_REPOS = {
    "s1k":              "simplescaling/s1K",
    "m1k":              "UCSC-VLAA/m1k-tokenized",
    "opencodereasoning": "nvidia/OpenCodeReasoning",
    "arpo_sft":         "dongguanting/ARPO-SFT-54K",
}


def download_dataset(repo_id: str, dest_dir: Path, token: str = "") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_dir),
        token=token or None,
    )
    return Path(resolved).resolve()


def main():
    p = argparse.ArgumentParser(description="Download s1K, m1K, OpenCodeReasoning, ARPO-SFT into data/raw/")
    p.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""),
                   help="Hugging Face access token (defaults to $HF_TOKEN)")
    p.add_argument("--out-dir", type=str, default="data/raw",
                   help="Root directory to download datasets into (default: data/raw)")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for key, repo_id in DATASET_REPOS.items():
        dest = out_root / key
        print(f"Downloading {repo_id} -> {dest} ...")
        try:
            path = download_dataset(repo_id, dest, token=args.token)
            print(f"  done: {path}")
        except Exception as e:
            print(f"  FAILED ({repo_id}): {e}")

    print("\nAll downloads attempted. Check messages above for any failures.")


if __name__ == "__main__":
    main()
