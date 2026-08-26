"""
Download s1K, m1K, OpenCodeReasoning, ARPO-SFT, MATH-500, MedQA, AIME2025,
and AIME2026 datasets into data/raw/, and/or predownload MAS model repos
into a shared HF cache.

Usage:
    # datasets only (default)
    python download_datasets.py [--token HF_TOKEN] [--out-dir data/raw]

    # also predownload the Sequential-Style (Light) MAS model repos
    python download_datasets.py --models sequential_light --hf-cache-dir /p/project1/hai_1354/hf_cache

    # models only, skip datasets
    python download_datasets.py --skip-datasets --models sequential_light --hf-cache-dir /p/project1/hai_1354/hf_cache

Each dataset is fetched via huggingface_hub.snapshot_download (repo_type="dataset")
and stored in its own subdirectory under --out-dir, named after the dataset key
(s1k, m1k, opencodereasoning, arpo_sft, math500, medqa, aime2025, aime2026).

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

DATASET_REPOS = {
    "s1k":              "simplescaling/s1K",
    "m1k":              "UCSC-VLAA/m1k-tokenized",
    "opencodereasoning": "nvidia/OpenCodeReasoning",
    "arpo_sft":         "dongguanting/ARPO-SFT-54K",
    "math500":          "HuggingFaceH4/MATH-500",
    # NOTE: run.py --dataset medqa reads the local dataset/medqa.json shipped
    # with the repo; this HF copy is a predownload for offline reference.
    "medqa":            "GBaker/MedQA-USMLE-4-options",
    "aime2025":         "math-ai/aime25",
    "aime2026":         "math-ai/aime26",
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


# Eval datasets are loaded at runtime straight from the shared HF cache
# (HF_HOME=/p/project1/hai_1354/hf_cache, HF_HUB_OFFLINE=1), not from data/raw.
# Each entry mirrors the exact call made by inference_utils at eval time so the
# cache layout matches what the offline job will look for.
EVAL_DATASETS = ("gpqa", "mbppplus", "livecodebench", "math500", "aime2025", "aime2026")

# Datasets loaded at eval time via a plain load_dataset(repo, split=...).
EVAL_LOAD_DATASET_REPOS = {
    "math500":  ("HuggingFaceH4/MATH-500", "test"),
    "aime2025": ("math-ai/aime25", "test"),
    "aime2026": ("math-ai/aime26", "test"),
}

LCB_REPO = "livecodebench/code_generation_lite"
LCB_RELEASE_V6_FILES = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]


def download_eval_dataset(name: str, cache_dir: Path, token: str = "") -> None:
    from datasets import load_dataset

    from huggingface_hub import hf_hub_download

    hub_dir = cache_dir / "hub"
    datasets_dir = cache_dir / "datasets"

    if name in EVAL_LOAD_DATASET_REPOS:
        repo_id, split = EVAL_LOAD_DATASET_REPOS[name]
        ds = load_dataset(
            repo_id, split=split,
            cache_dir=str(datasets_dir), token=token or None,
        )
        print(f"    cached {repo_id} [{split}]: {len(ds)} rows")
    elif name == "gpqa":
        # inference_mas: load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        # Gated repo: the token must have accepted the terms on the Hub.
        ds = load_dataset(
            "Idavidrein/gpqa", "gpqa_diamond", split="train",
            cache_dir=str(datasets_dir), token=token or None,
        )
        print(f"    cached gpqa_diamond: {len(ds)} rows")
    elif name == "mbppplus":
        # lcb_utils.load_mbppplus_records: load_dataset("evalplus/mbppplus", split="test")
        ds = load_dataset(
            "evalplus/mbppplus", split="test",
            cache_dir=str(datasets_dir), token=token or None,
        )
        print(f"    cached mbppplus: {len(ds)} rows")
    elif name == "livecodebench":
        # lcb_utils.load_release_v6_records: hf_hub_download of each release_v6 file
        for filename in LCB_RELEASE_V6_FILES:
            path = hf_hub_download(
                repo_id=LCB_REPO, repo_type="dataset", filename=filename,
                cache_dir=str(hub_dir), token=token or None,
            )
            print(f"    cached {filename}: {path}")
    else:
        raise ValueError(f"unknown eval dataset {name!r} (choices: {', '.join(EVAL_DATASETS)})")


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
        description="Download s1K, m1K, OpenCodeReasoning, ARPO-SFT, MATH-500, MedQA, "
                     "AIME2025, AIME2026 into data/raw/, "
                     "and optionally predownload MAS model repos into a shared HF cache."
    )
    p.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""),
                   help="Hugging Face access token (defaults to $HF_TOKEN)")
    p.add_argument("--out-dir", type=str, default="data/raw",
                   help="Root directory to download datasets into (default: data/raw)")
    p.add_argument("--skip-datasets", action="store_true",
                   help="Skip downloading the datasets (DATASET_REPOS).")
    p.add_argument("--eval-datasets", type=str, default="",
                   help="Comma-separated eval datasets to predownload into the shared HF "
                        f"cache (choices: {', '.join(EVAL_DATASETS)}, or 'all'). "
                        "These go into --hf-cache-dir (not --out-dir) so offline eval "
                        "jobs find them.")
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

    if args.eval_datasets:
        cache_dir = Path(args.hf_cache_dir)
        (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
        (cache_dir / "datasets").mkdir(parents=True, exist_ok=True)

        names = (
            list(EVAL_DATASETS)
            if args.eval_datasets.strip().lower() == "all"
            else [s.strip() for s in args.eval_datasets.split(",") if s.strip()]
        )
        print(f"\nPredownloading eval datasets into {cache_dir} ...")
        for name in names:
            print(f"  Downloading {name} ...")
            try:
                download_eval_dataset(name, cache_dir, token=args.token)
                print(f"    done: {name}")
            except Exception as e:
                print(f"    FAILED ({name}): {e}")

        print("\nAll eval dataset downloads attempted. Check messages above for any failures.")

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
