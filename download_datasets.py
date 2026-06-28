"""
Download s1K, m1K, OpenCodeReasoning, and ARPO-SFT datasets into data/raw/,
and/or predownload MAS model repos into a shared HF cache.

Usage:
    # datasets only (default)
    python download_datasets.py [--token HF_TOKEN] [--out-dir data/raw]

    # also predownload the Sequential-Style (Light) MAS model repos
    python download_datasets.py --models sequential_light --hf-cache-dir /p/project1/hai_1354/hf_cache

    # models only, skip datasets
    python download_datasets.py --skip-datasets --models sequential_light --hf-cache-dir /p/project1/hai_1354/hf_cache

Each dataset is fetched via huggingface_hub.snapshot_download (repo_type="dataset")
and stored in its own subdirectory under --out-dir, named after the dataset key
(s1k, m1k, opencodereasoning, arpo_sft).

Each model is fetched via huggingface_hub.snapshot_download (repo_type="model")
into --hf-cache-dir, which is also exported as HF_HOME/HF_HUB_CACHE so that
later `snapshot_download(..., local_files_only=True)` calls (used by
hf_resolver.py / modeling.py) find the cached weights.

If a repo id below turns out to be wrong/renamed on the Hub, just edit
DATASET_REPOS / MODEL_STYLE_REPOS.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
os.environ["HF_TOKEN"] = "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"  # avoid HF trying to access the network during model loading
DATASET_REPOS = {
    "s1k":              "simplescaling/s1K",
    "m1k":              "UCSC-VLAA/m1k-tokenized",
    "opencodereasoning": "nvidia/OpenCodeReasoning",
    "arpo_sft":         "dongguanting/ARPO-SFT-54K",
    "math500":          "HuggingFaceH4/MATH-500",
}

# Model repos per MAS "style", mirroring STYLE_SPECS in load_from_repo.py.
MODEL_STYLE_REPOS = {
    "sequential_light": {
        "planner": "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B",
        "critic":  "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B",
        "solver":  "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B",
        "outer":   "RecursiveMAS/Sequential-Light-Outerlinks",
    },
    "sequential_scaled": {
        "planner": "RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B",
        "critic":  "RecursiveMAS/Sequential-Scaled-Critic-Llama3.2-3B",
        "solver":  "RecursiveMAS/Sequential-Scaled-Solver-Qwen3.5-4B",
        "outer":   "RecursiveMAS/Sequential-Scaled-Outerlinks",
    },
    "mixture": {
        "math":       "RecursiveMAS/Mixture-Math-DeepSeek-R1-Distill-Qwen-1.5B",
        "code":       "RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B",
        "science":    "RecursiveMAS/Mixture-Science-BioMistral-7B",
        "summarizer": "RecursiveMAS/Mixture-Summarizer-Qwen3.5-2B",
        "outer":      "RecursiveMAS/Mixture-Outerlinks",
    },
    "distillation": {
        "expert":  "RecursiveMAS/Distillation-Expert-Qwen3.5-9B",
        "learner": "RecursiveMAS/Distillation-Learner-Qwen3.5-4B",
        "outer":   "RecursiveMAS/Distillation-Outerlinks",
    },
    "deliberation": {
        "reflector":  "RecursiveMAS/Deliberation-Reflector-Qwen3.5-4B",
        "toolcaller": "RecursiveMAS/Deliberation-Toolcaller-Qwen3.5-4B",
        "outer":      "RecursiveMAS/Deliberation-Outerlinks",
    },
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


def download_model(repo_id: str, cache_dir: Path, token: str = "") -> Path:
    resolved = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=str(cache_dir),
        token=token or None,
    )
    return Path(resolved).resolve()


def main():
    p = argparse.ArgumentParser(
        description="Download s1K, m1K, OpenCodeReasoning, ARPO-SFT into data/raw/, "
                     "and optionally predownload MAS model repos into a shared HF cache."
    )
    p.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""),
                   help="Hugging Face access token (defaults to $HF_TOKEN)")
    p.add_argument("--out-dir", type=str, default="data/raw",
                   help="Root directory to download datasets into (default: data/raw)")
    p.add_argument("--skip-datasets", action="store_true",
                   help="Skip downloading the datasets (DATASET_REPOS).")
    p.add_argument("--models", type=str, default="",
                   help="Comma-separated MAS style names to predownload models for "
                        f"(choices: {', '.join(MODEL_STYLE_REPOS)}). "
                        "Leave empty to skip model downloads.")
    p.add_argument("--hf-cache-dir", type=str,
                   default=os.environ.get("HF_HOME", "/p/project1/hai_1354/hf_cache"),
                   help="HF cache directory to download models into "
                        "(default: $HF_HOME or /p/project1/hai_1354/hf_cache). "
                        "Also exported as HF_HOME/HF_HUB_CACHE for this process.")
    args = p.parse_args()

    if not args.skip_datasets:
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

        print("\nAll dataset downloads attempted. Check messages above for any failures.")

    if args.models:
        # HF_HOME points to the parent; the actual hub cache lives in HF_HOME/hub.
        # We set HF_HOME so that huggingface_hub resolves HF_HUB_CACHE to
        # <hf_cache_dir>/hub — matching what the SLURM job sets at runtime.
        cache_dir = Path(args.hf_cache_dir)
        hub_dir = cache_dir / "hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_CACHE"] = str(hub_dir)

        styles = [s.strip() for s in args.models.split(",") if s.strip()]
        for style in styles:
            repos = MODEL_STYLE_REPOS.get(style)
            if repos is None:
                print(f"  FAILED (unknown style): {style!r} "
                      f"(choices: {', '.join(MODEL_STYLE_REPOS)})")
                continue
            print(f"\nPredownloading models for style={style!r} into {cache_dir} ...")
            for role, repo_id in repos.items():
                print(f"  Downloading {repo_id} ({role}) ...")
                try:
                    path = download_model(repo_id, cache_dir, token=args.token)
                    print(f"    done: {path}")
                except Exception as e:
                    print(f"    FAILED ({repo_id}): {e}")

        print("\nAll model downloads attempted. Check messages above for any failures.")


if __name__ == "__main__":
    main()
