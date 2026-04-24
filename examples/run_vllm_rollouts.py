#!/usr/bin/env python3
"""Run rollouts against a **plain OpenAI-compatible vLLM server** (no genlm / steerbot).

Examples::

    python examples/run_vllm_rollouts.py \\
      --model openai/Qwen/Qwen3-4B-Instruct-2507 \\
      --api-base http://127.0.0.1:8000/v1 \\
      --num-tasks 10 --cycle-tasks

    CUDA_VISIBLE_DEVICES=0 python examples/run_vllm_rollouts.py \\
      --model my-served-name --api-base http://localhost:8000/v1 --num-tasks 5

Supported envs:
  - textcraft (default): `platoon.textcraft.synth_rollout.run_synth_depth_aware_rollout`
  - number_search: `platoon.number_search.rollout.run_rollout`

Events: ``{output_dir}/events/*.jsonl`` (same as other rollout scripts).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys


def _consume_cuda_device_arg() -> None:
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--cuda-device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            sys.argv = sys.argv[:i] + sys.argv[i + 2 :]
            continue
        i += 1


_consume_cuda_device_arg()
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from platoon.config_defs import InferenceParams, RolloutConfig
from platoon.textcraft.synth_rollout import run_synth_depth_aware_rollout
from platoon.textcraft.synth_tasks import Difficulty, get_synth_task, get_synth_task_ids_by_difficulty
from platoon.number_search.rollout import run_rollout as run_number_search_rollout
from platoon.number_search.tasks import get_task as get_number_search_task
from platoon.number_search.tasks import get_task_ids as get_number_search_task_ids


def _parse_difficulty(s: str) -> Difficulty:
    return Difficulty[s.upper()]


def _sanitize_run_name(model: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", model.strip())
    return s[:120] if len(s) > 120 else s


def resolve_output_dir(args: argparse.Namespace) -> str:
    if getattr(args, "output_dir", None):
        return os.path.abspath(args.output_dir)
    sub = args.run_name or f"{args.env}_vllm_{_sanitize_run_name(args.model)}"
    return os.path.abspath(os.path.join(args.output_root, sub))


async def _one_rollout(
    *,
    task_id: str,
    config: RolloutConfig,
    per_agent_max_steps: int,
    max_depth: int,
    index: int,
    env: str,
) -> tuple[int, str, object]:
    label = f"[{index}] {task_id}"
    try:
        if env == "textcraft":
            out = await run_synth_depth_aware_rollout(
                task=get_synth_task(task_id),
                config=config,
                per_agent_max_steps=per_agent_max_steps,
                max_depth=max_depth,
            )
        elif env == "number_search":
            out = await run_number_search_rollout(task=get_number_search_task(task_id), config=config)
        else:
            raise ValueError(f"Unknown env {env!r}")
        return (index, label, out)
    except Exception as e:
        return (index, label, e)


async def _run_all(args: argparse.Namespace) -> list[tuple[int, str, object]]:
    if args.env == "textcraft":
        task_ids = get_synth_task_ids_by_difficulty(
            split=args.split, difficulty=_parse_difficulty(args.difficulty)
        )
    elif args.env == "number_search":
        task_ids = get_number_search_task_ids(
            split=args.split,
            num_samples_train=args.num_samples_train,
            num_samples_val=args.num_samples_val,
        )
    else:
        raise SystemExit(f"Unknown --env {args.env!r}")
    if not task_ids:
        raise SystemExit(f"No tasks for split={args.split!r} difficulty={args.difficulty!r}")

    if args.cycle_tasks:
        chosen = [task_ids[i % len(task_ids)] for i in range(args.num_tasks)]
    else:
        base = task_ids[args.task_index % len(task_ids)]
        chosen = [base] * args.num_tasks

    out_dir = resolve_output_dir(args)
    config = RolloutConfig(
        model_name=args.model,
        model_endpoint=args.api_base,
        model_api_key=args.api_key,
        output_dir=out_dir,
        inference_params=InferenceParams(
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        ),
        max_steps=10 if args.env == "number_search" else 25,
    )

    sem = asyncio.Semaphore(args.max_concurrent)

    async def _bounded(i: int, tid: str) -> tuple[int, str, object]:
        async with sem:
            return await _one_rollout(
                task_id=tid,
                config=config,
                per_agent_max_steps=args.per_agent_max_steps,
                max_depth=args.max_depth,
                index=i,
                env=args.env,
            )

    return await asyncio.gather(*[_bounded(i, tid) for i, tid in enumerate(chosen)])


def main() -> None:
    epilog = """
vLLM OpenAI server:
  Point --api-base at the OpenAI API root, e.g. http://127.0.0.1:8000/v1
  --model must match a model name served by that instance.

This script does not import genlm or steerbot; completions go straight through LiteLLM to your URL.
"""
    p = argparse.ArgumentParser(
        description="Textcraft rollouts via plain vLLM (OpenAI-compatible), no genlm steering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--env",
        choices=("textcraft", "number_search"),
        default="textcraft",
        help="Which Platoon environment to run (default: textcraft)",
    )
    p.add_argument(
        "--model",
        required=True,
        help="LiteLLM model id (e.g. openai/MODEL or the name your vLLM serves)",
    )
    p.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL (default: http://127.0.0.1:8000/v1)",
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key string if required (default: EMPTY)",
    )
    p.add_argument("--num-tasks", type=int, required=True, help="Total rollout jobs")
    p.add_argument("--split", default="val", help="Task split (default: val)")
    p.add_argument("--difficulty", default="EASY", help="Difficulty enum name (default: EASY)")
    p.add_argument(
        "--num-samples-train",
        type=int,
        default=50000,
        help="number_search: dataset size for train split (default: 50000)",
    )
    p.add_argument(
        "--num-samples-val",
        type=int,
        default=1000,
        help="number_search: dataset size for val split (default: 1000)",
    )
    p.add_argument("--task-index", type=int, default=0, help="Fixed task index if not cycling")
    p.add_argument("--cycle-tasks", action="store_true", help="Cycle task ids")
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Concurrent rollouts (default: num-tasks)",
    )
    p.add_argument("--per-agent-max-steps", type=int, default=25)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Per completion cap (Platoon InferenceParams)",
    )
    p.add_argument(
        "--output-root",
        default=os.path.join("examples", "rollout_results"),
        help="Base dir when --output-dir omitted",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Exact rollout output dir (overrides --output-root / --run-name)",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory under --output-root (default: derived from --model)",
    )
    args = p.parse_args()

    # Some Platoon envs (e.g. number_search) currently rely on LiteLLM/OpenAI env vars instead
    # of explicitly passing api_base/api_key through. For reproducibility, we overwrite here.
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
        os.environ["OPENAI_API_BASE"] = args.api_base
    # LiteLLM's OpenAI-compatible providers generally require some api_key string even if the
    # backend doesn't enforce auth (e.g. local vLLM). We set it unconditionally.
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    if args.num_tasks < 1:
        p.error("--num-tasks must be at least 1")
    if args.max_concurrent is None:
        args.max_concurrent = args.num_tasks
    if args.max_concurrent < 1:
        p.error("--max-concurrent must be at least 1")

    out_dir = resolve_output_dir(args)
    print(
        f"Rollout output_dir={out_dir} (events -> {os.path.join(out_dir, 'events')})",
        file=sys.stderr,
    )
    print(f"Env: {args.env}", file=sys.stderr)
    print(f"LLM: model={args.model!r} api_base={args.api_base!r}", file=sys.stderr)

    results = asyncio.run(_run_all(args))

    ok = 0
    for _idx, label, out in sorted(results, key=lambda x: x[0]):
        if isinstance(out, Exception):
            print(f"FAIL {label}: {out!r}", file=sys.stderr)
        else:
            ok += 1
            print(f"OK   {label}: {type(out).__name__}")

    print(
        f"\nFinished {args.num_tasks} vLLM rollouts: {ok} ok, {args.num_tasks - ok} failed",
        file=sys.stderr,
    )
    if ok < args.num_tasks:
        sys.exit(1)


if __name__ == "__main__":
    main()
