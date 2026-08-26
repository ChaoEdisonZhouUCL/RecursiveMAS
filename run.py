#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from hf_resolver import (
    resolve_inner_adapter,
    resolve_medqa_dataset_arg,
    resolve_outer_paths,
    snapshot_repo,
    task_for_inner_repo,
)
from load_from_repo import DATASET_DEFAULT_SPLIT, STYLE_SPECS
from inference_utils import (
    inference_mas,
    inference_mas_deliberation,
    inference_mas_distill,
    inference_mas_mixture,
)

LATENT_STEPS_SWEEP: Tuple = (16, 32, 48)
GPQA_DEFAULT_CHOICE_OLD_PROMPT = 2
MBPPPLUS_TEMPERATURE = 0.2


class RunCapture:
    def __init__(self) -> None:
        self.stdout = io.StringIO()

    def get_text(self) -> str:
        return self.stdout.getvalue()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Release inference runner for RecursiveMAS HF checkpoints.")
    p.add_argument("--style", required=True, choices=list(STYLE_SPECS.keys()))
    p.add_argument("--dataset", required=True, default="math500", choices=["math500", "medqa", "gpqa", "mbppplus", "livecodebench", "aime2025", "aime2026"])
    p.add_argument("--dataset_split", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_seed", type=int, default=-1)
    p.add_argument("--num_recursive_rounds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--latent_length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument(
        "--greedy", action="store_true",
        help="Greedy decoding (disables sampling; temperature/top_p/top_k are ignored).",
    )
    p.add_argument(
        "--lcb_public_tests_only", action="store_true",
        help="livecodebench only: score with public tests only (default uses public+private).",
    )
    p.add_argument(
        "--num_rollouts", type=int, default=0,
        help="Rollouts per problem (pass@k). 0 = from eval_protocol.yaml, else 1.",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=-1,
        help="Max new tokens. -1 = from eval_protocol.yaml, else per-dataset built-in.",
    )
    p.add_argument("--trust_remote_code", type=int, default=1, choices=[0, 1])
    p.add_argument("--device", default=None)
    p.add_argument(
        "--ckpt_dir", type=str, default=None,
        help="Path to a locally trained outer-link checkpoint directory "
             "(outputs/checkpoints/<prefix>_<mode>_r<N>/).  Required for "
             "--style sequential_light_trained and sequential_light_shared_roae.",
    )
    p.add_argument(
        "--num_samples", type=int, default=-1,
        help="Evaluate only the first N problems (-1 = all).  Diagnostic use: a "
             "small N turns a full eval into a few minutes.",
    )
    p.add_argument(
        "--result_jsonl", type=str, default="",
        help="Write one record per problem to this path (prompt, raw output, "
             "parsed code, per-test execution result).  This is what tells you "
             "*why* a code eval scored low — parse failure vs. wrong answer vs. "
             "timeout — which the aggregate accuracy cannot.",
    )
    p.add_argument(
        "--result_json", type=str, default="",
        help="Write the final metric to this JSON path.  Compute nodes here have "
             "no internet, so eval cannot log to W&B directly; sync_eval_to_wandb.py "
             "collects these files afterwards from the login node and attaches them "
             "to the training run recorded in the checkpoint's wandb_run.json.",
    )
    return p


def infer_dataset_split(dataset: str, explicit: str) -> str:
    if explicit:
        return explicit
    return DATASET_DEFAULT_SPLIT.get(dataset.lower(), "test")


def infer_max_new_tokens(style: str, dataset: str) -> int:
    if dataset.lower() == "math500":
        if style == "sequential_light":
            return 1000
        return 2000
    return 4000


def infer_temperature(dataset: str, explicit: float) -> float:
    # Code-eval datasets use a low temperature.
    if dataset.lower() in {"mbppplus", "livecodebench"}:
        return MBPPPLUS_TEMPERATURE
    return explicit


def resolve_latent_steps(explicit: int) -> Tuple[int, ...]:
    if explicit is not None and explicit > 0:
        return (explicit,)
    return LATENT_STEPS_SWEEP


def _has_cli_flag(flag: str) -> bool:
    return flag in sys.argv[1:]


def apply_eval_protocol(args: argparse.Namespace) -> None:
    """Apply per-dataset settings from eval_protocol.yaml (paper protocol).

    Precedence: explicit CLI flags > eval_protocol.yaml > built-in defaults.
    """
    path = Path(__file__).resolve().parent / "eval_protocol.yaml"
    if not path.is_file():
        return
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[protocol] failed to read {path.name}: {e}; using built-in defaults.")
        return

    proto = dict(cfg.get("defaults") or {})
    proto.update((cfg.get("datasets") or {}).get(args.dataset.lower()) or {})
    if not proto:
        return

    field_to_flag = {
        "temperature": "--temperature",
        "top_p": "--top_p",
        "latent_length": "--latent_length",
        "max_new_tokens": "--max_new_tokens",
        "num_rollouts": "--num_rollouts",
        "greedy": "--greedy",
    }
    applied = []
    for field_name, flag in field_to_flag.items():
        if field_name not in proto or _has_cli_flag(flag):
            continue
        setattr(args, field_name, proto[field_name])
        applied.append(f"{field_name}={proto[field_name]}")
    if applied:
        print(f"[protocol] {args.dataset}: {' '.join(applied)}  (from eval_protocol.yaml; CLI flags override)")


def apply_recommended_settings(args: argparse.Namespace) -> None:
    recommended = inference_mas.get_release_recommended_settings(args.style, args.dataset)
    if recommended is None:
        return

    field_to_flag = {
        "seed": "--seed",
        "batch_size": "--batch_size",
        "latent_length": "--latent_length",
    }
    mismatches: List[str] = []
    for field_name, flag in field_to_flag.items():
        recommended_value = recommended[field_name]
        explicit = _has_cli_flag(flag)
        current_value = getattr(args, field_name)
        if not explicit:
            setattr(args, field_name, recommended_value)
            continue
        if current_value != recommended_value:
            mismatches.append(f"{field_name}={recommended_value}")

    if mismatches:
        joined = ", ".join(mismatches)
        print(
            f"[note] We recommend to use provided settings to run "
            f"{args.style} on {args.dataset}: {joined}"
        )


def resolve_style_paths(style: str, dataset: str, ckpt_dir: str = None) -> Dict[str, Path]:
    spec = STYLE_SPECS[style]
    repos = spec["repos"]
    task = task_for_inner_repo(dataset)
    out: Dict[str, Path] = {}

    def materialize(key: str) -> Path:
        return snapshot_repo(str(repos[key]))

    family = str(spec["family"])
    if family == "sequential":
        for key in ["planner", "critic", "solver", "outer"]:
            out[key] = materialize(key)
        out["planner_adapter"] = resolve_inner_adapter(out["planner"], task)
        out["critic_adapter"] = resolve_inner_adapter(out["critic"], task)
        out["solver_adapter"] = resolve_inner_adapter(out["solver"], task)
        outer_paths = resolve_outer_paths(out["outer"], task=task)
        out["outer_12"] = outer_paths["outer_12"]
        out["outer_23"] = outer_paths["outer_23"]
        out["outer_31"] = outer_paths["outer_31"]
        return out

    if family == "sequential_local":
        # Backbone models from HF cache; outer adapters from local checkpoint dir.
        if not ckpt_dir:
            raise ValueError(
                "--ckpt_dir is required for --style sequential_light_trained. "
                "Point it to outputs/checkpoints/<prefix>_original_r<N>/"
            )
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
        for key in ["planner", "critic", "solver"]:
            out[key] = materialize(key)
        out["planner_adapter"] = resolve_inner_adapter(out["planner"], task)
        out["critic_adapter"]  = resolve_inner_adapter(out["critic"],  task)
        out["solver_adapter"]  = resolve_inner_adapter(out["solver"],  task)
        outer_paths = resolve_outer_paths(ckpt_path, task=task)
        out["outer_12"] = outer_paths["outer_12"]
        out["outer_23"] = outer_paths["outer_23"]
        out["outer_31"] = outer_paths["outer_31"]
        return out

    if family == "sequential_shared_roae":
        # Backbone models from HF cache; shared link from local checkpoint dir.
        if not ckpt_dir:
            raise ValueError(
                "--ckpt_dir is required for --style sequential_light_shared_roae. "
                "Point it to outputs/checkpoints/<prefix>_shared_roae_r<N>/"
            )
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
        for key in ["planner", "critic", "solver"]:
            out[key] = materialize(key)
        out["planner_adapter"]   = resolve_inner_adapter(out["planner"], task)
        out["critic_adapter"]    = resolve_inner_adapter(out["critic"],  task)
        out["solver_adapter"]    = resolve_inner_adapter(out["solver"],  task)
        out["shared_link_path"]  = ckpt_path          # passed to inference_mas via CLI
        # Sentinel outer paths — not used (shared_link replaces them)
        out["outer_12"] = ckpt_path / "shared_recursive_link.pt"
        out["outer_23"] = ckpt_path / "shared_recursive_link.pt"
        out["outer_31"] = ckpt_path / "shared_recursive_link.pt"
        return out

    if family == "mixture":
        for key in ["math", "code", "science", "summarizer", "outer"]:
            out[key] = materialize(key)
        out["math_adapter"] = resolve_inner_adapter(out["math"], None)
        out["code_adapter"] = resolve_inner_adapter(out["code"], None)
        out["science_adapter"] = resolve_inner_adapter(out["science"], None)
        out["summarizer_adapter"] = resolve_inner_adapter(out["summarizer"], None)
        outer_paths = resolve_outer_paths(out["outer"], task=None)
        for key in ["outer_1s", "outer_2s", "outer_3s", "outer_s1", "outer_s2", "outer_s3"]:
            out[key] = outer_paths[key]
        return out

    if family == "distillation":
        for key in ["expert", "learner", "outer"]:
            out[key] = materialize(key)
        out["expert_adapter"] = resolve_inner_adapter(out["expert"], task)
        out["learner_adapter"] = resolve_inner_adapter(out["learner"], task)
        outer_paths = resolve_outer_paths(out["outer"], task=task)
        out["outer_el"] = outer_paths["outer_el"]
        out["outer_le"] = outer_paths["outer_le"]
        return out

    if family == "deliberation":
        for key in ["reflector", "toolcaller", "outer"]:
            out[key] = materialize(key)
        out["reflector_adapter"] = resolve_inner_adapter(out["reflector"], None)
        out["toolcaller_adapter"] = resolve_inner_adapter(out["toolcaller"], None)
        outer_paths = resolve_outer_paths(out["outer"], task=None)
        out["outer_rt"] = outer_paths["outer_rt"]
        out["outer_tr"] = outer_paths["outer_tr"]
        return out

    raise ValueError(f"Unsupported style family: {family}")


def build_common_cli(args: argparse.Namespace, dataset_arg: str, dataset_split: str, latent_steps: int, max_new_tokens: int) -> List[str]:
    temperature = infer_temperature(args.dataset, args.temperature)
    out = [
        "--dataset", dataset_arg,
        "--dataset_split", dataset_split,
        "--num_samples", str(args.num_samples),
        "--seed", str(args.seed),
        "--sample_seed", str(args.sample_seed),
        "--num_rollouts", str(args.num_rollouts if args.num_rollouts > 0 else 1),
        "--num_recursive_rounds", str(args.num_recursive_rounds),
        "--batch_size", str(args.batch_size),
        "--latent_steps", str(latent_steps),
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--ans_max_new_tokens", "-1",
        "--mbppplus_timeout_s", "10",
        "--mbppplus_num_prompt_tests", "3",
        "--lcb_use_private_tests", "0" if args.lcb_public_tests_only else "1",
        "--dtype", "auto",
        "--outer_dtype", "auto",
        "--trust_remote_code", str(args.trust_remote_code),
        "--enable_thinking", "0",
    ]
    if args.result_jsonl:
        # One file per latent_steps setting, so a sweep does not overwrite itself.
        jsonl = Path(args.result_jsonl)
        out.extend(["--result_jsonl",
                    str(jsonl.with_name(f"{jsonl.stem}_ls{latent_steps}{jsonl.suffix}"))])
    if args.device is not None:
        out.extend(["--device", str(args.device)])
    if not args.greedy:
        out.append("--do_sample")
    out.append("--ans")
    return out


def extract_metric(output_text: str) -> Tuple[str, float]:
    # pass@k is printed after the per-rollout accuracy lines and wins.
    pass_matches = re.findall(r"(pass@[0-9]+)=([0-9]+(?:\.[0-9]+)?)%", output_text)
    if pass_matches:
        return pass_matches[-1][0], float(pass_matches[-1][1])
    matches = re.findall(r"accuracy=([0-9]+(?:\.[0-9]+)?)%", output_text)
    if matches:
        return "accuracy", float(matches[-1])
    raise RuntimeError("Failed to parse final metric from inference output.")


def run_module(module, cli_args: List[str]) -> Tuple[str, float, str]:
    old_argv = sys.argv[:]
    capture = RunCapture()
    try:
        sys.argv = [module.__file__ or module.__name__] + cli_args
        with contextlib.redirect_stdout(capture.stdout):
            module.main()
    except Exception:
        captured = capture.get_text()
        if captured.strip():
            print(captured, file=sys.stderr, end="" if captured.endswith("\n") else "\n")
        raise
    finally:
        sys.argv = old_argv
    text = capture.get_text()
    metric_name, metric_value = extract_metric(text)
    return metric_name, metric_value, text


def build_cli_for_style(
    args: argparse.Namespace,
    family: str,
    dataset_arg: str,
    dataset_split: str,
    paths: Dict[str, Path],
    latent_steps: int,
    max_new_tokens: int,
) -> Tuple[object, List[str]]:
    common = build_common_cli(args, dataset_arg=dataset_arg, dataset_split=dataset_split, latent_steps=latent_steps, max_new_tokens=max_new_tokens)

    if family in ("sequential", "sequential_local", "sequential_shared_roae"):
        choice_old_prompt = GPQA_DEFAULT_CHOICE_OLD_PROMPT if args.dataset.lower() == "gpqa" else 0
        cli = [
            "--mas_shape", "chain",
            "--agent1_model_name_or_path", str(paths["planner"]),
            "--agent2_model_name_or_path", str(paths["critic"]),
            "--agent3_model_name_or_path", str(paths["solver"]),
            "--agent1_inner_aligner_path", str(paths["planner_adapter"]),
            "--agent2_inner_aligner_path", str(paths["critic_adapter"]),
            "--agent3_inner_aligner_path", str(paths["solver_adapter"]),
            "--outer_12_path", str(paths["outer_12"]),
            "--outer_23_path", str(paths["outer_23"]),
            "--outer_31_path", str(paths["outer_31"]),
            "--choice_old_prompt", str(choice_old_prompt),
            "--solver_pre_question", "0",
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
        ] + common
        if family == "sequential_shared_roae" and "shared_link_path" in paths:
            cli += ["--shared_link_path", str(paths["shared_link_path"])]
        return inference_mas, cli

    if family == "mixture":
        cli = [
            "--mas_shape", "hie",
            "--agent1_model_name_or_path", str(paths["math"]),
            "--agent2_model_name_or_path", str(paths["code"]),
            "--agent3_model_name_or_path", str(paths["science"]),
            "--agent4_model_name_or_path", str(paths["summarizer"]),
            "--agent1_inner_aligner_path", str(paths["math_adapter"]),
            "--agent2_inner_aligner_path", str(paths["code_adapter"]),
            "--agent3_inner_aligner_path", str(paths["science_adapter"]),
            "--agent4_inner_aligner_path", str(paths["summarizer_adapter"]),
            "--outer_1s_path", str(paths["outer_1s"]),
            "--outer_2s_path", str(paths["outer_2s"]),
            "--outer_3s_path", str(paths["outer_3s"]),
            "--outer_s1_path", str(paths["outer_s1"]),
            "--outer_s2_path", str(paths["outer_s2"]),
            "--outer_s3_path", str(paths["outer_s3"]),
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
        ] + common
        return inference_mas_mixture, cli

    if family == "distillation":
        cli = [
            "--mas_shape", "distill",
            "--expert_model_name_or_path", str(paths["expert"]),
            "--learner_model_name_or_path", str(paths["learner"]),
            "--expert_inner_aligner_path", str(paths["expert_adapter"]),
            "--learner_inner_aligner_path", str(paths["learner_adapter"]),
            "--outer_el_path", str(paths["outer_el"]),
            "--outer_le_path", str(paths["outer_le"]),
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
        ] + common
        return inference_mas_distill, cli

    if family == "deliberation":
        cli = [
            "--mas_shape", "deliberation",
            "--reflector_model_name_or_path", str(paths["reflector"]),
            "--toolcaller_model_name_or_path", str(paths["toolcaller"]),
            "--reflector_inner_aligner_path", str(paths["reflector_adapter"]),
            "--toolcaller_inner_aligner_path", str(paths["toolcaller_adapter"]),
            "--outer_rt_path", str(paths["outer_rt"]),
            "--outer_tr_path", str(paths["outer_tr"]),
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
            "--max_tool_rounds", "5",
            "--python_timeout", "10.0",
            "--python_cwd", ".",
            "--result_max_chars", "6000",
        ] + common
        cli.append("--quiet_tools")
        return inference_mas_deliberation, cli

    raise ValueError(f"Unsupported style family: {family}")


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")

    args = build_parser().parse_args()
    apply_recommended_settings(args)
    apply_eval_protocol(args)
    repo_root = Path(__file__).resolve().parent
    dataset_arg = resolve_medqa_dataset_arg(args.dataset, repo_root)
    dataset_split = infer_dataset_split(args.dataset, args.dataset_split)
    paths = resolve_style_paths(args.style, args.dataset, ckpt_dir=args.ckpt_dir)
    family = str(STYLE_SPECS[args.style]["family"])
    latent_steps_values = resolve_latent_steps(args.latent_length)

    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else infer_max_new_tokens(args.style, args.dataset)
    decoding = "greedy" if args.greedy else "sampling"
    num_rollouts = args.num_rollouts if args.num_rollouts > 0 else 1
    print(f"[run] style={args.style} dataset={args.dataset} rounds={args.num_recursive_rounds} batch_size={args.batch_size} max_new_tokens={max_new_tokens} decoding={decoding} rollouts={num_rollouts}")
    results: List[Tuple[int, str, float]] = []
    for latent_steps in latent_steps_values:
        print(f"[latent_steps={latent_steps}]", flush=True)
        module, cli = build_cli_for_style(
            args=args,
            family=family,
            dataset_arg=dataset_arg,
            dataset_split=dataset_split,
            paths=paths,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
        )
        metric_name, metric_value, _ = run_module(module, cli)
        results.append((latent_steps, metric_name, metric_value))

        print(f"  {metric_name}={metric_value:.2f}%")

    best_ls, best_metric_name, best_metric_value = max(results, key=lambda x: x[2])
    joined = ", ".join(f"ls={ls}:{name}={value:.2f}%" for ls, name, value in results)
    print(f"[result] {best_metric_name}={best_metric_value:.2f}%")

    if args.result_json:
        payload = {
            "dataset": args.dataset,
            "style": args.style,
            "ckpt_dir": args.ckpt_dir,
            "metric_name": best_metric_name,
            "metric_value": best_metric_value,
            "latent_steps": best_ls,
            "n_rounds": args.num_recursive_rounds,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "greedy": bool(args.greedy),
            "num_rollouts": num_rollouts,
            "max_new_tokens": max_new_tokens,
            "per_latent_steps": [
                {"latent_steps": ls, "metric_name": name, "metric_value": value}
                for ls, name, value in results
            ],
        }
        out = Path(args.result_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[result] written → {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
