#!/usr/bin/env python3
"""TextCraft depth-aware rollouts with prompt intersection.

Platoon's default system messages stay in the base LM while the extra instruction is intersected
through a second prompted LM (see ``steerbot.utils.llm.SteeredLM``).

**Sharding (20 tasks on 4 GPUs), half-open ``[task_start, task_end)``. All shards can share the
same ``--output-dir``; Platoon writes ``events_{task_id}_{uuid}.jsonl`` per rollout.
Within each shard process, rollouts run **sequentially**.

    CUDA_VISIBLE_DEVICES=0 python examples/textcraft_multi_agent.py --task-start 0 --task-end 5
    CUDA_VISIBLE_DEVICES=1 python examples/textcraft_multi_agent.py --task-start 5 --task-end 10
    CUDA_VISIBLE_DEVICES=2 python examples/textcraft_multi_agent.py --task-start 10 --task-end 15
    CUDA_VISIBLE_DEVICES=3 python examples/textcraft_multi_agent.py --task-start 15 --task-end 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
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

from platoon.config_defs import RolloutConfig
from platoon.textcraft.synth_rollout import run_synth_depth_aware_rollout
from platoon.textcraft.synth_tasks import Difficulty, get_synth_task, get_synth_task_ids_by_difficulty
from steerbot.utils.llm import register_steered_lm

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

DEFAULT_EXTRA_SYSTEM = (
    "You are an agent that can spawn other agents to help you complete a task. "
    "Use `await launch_subagent(...)` and `await asyncio.gather(launch_subagent(...), ...)` "
    "to delegate tasks to other agents."
)


def _default_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return os.path.abspath(args.output_dir)
    sub = f"textcraft_multi_agent_{args.split}_{args.difficulty.upper()}"
    return os.path.abspath(os.path.join(args.output_root, sub))


def _parse_difficulty(s: str) -> Difficulty:
    return Difficulty[s.upper()]


async def _one_rollout(
    task_id: str,
    config: RolloutConfig,
    per_agent_max_steps: int,
    max_depth: int,
    index: int,
) -> tuple[int, str, object]:
    label = f"[{index}] {task_id}"
    try:
        out = await run_synth_depth_aware_rollout(
            task=get_synth_task(task_id),
            config=config,
            per_agent_max_steps=per_agent_max_steps,
            max_depth=max_depth,
        )
        return (index, label, out)
    except Exception as e:
        return (index, label, e)


async def _run_all(args: argparse.Namespace) -> list[tuple[int, str, object]]:
    task_ids = get_synth_task_ids_by_difficulty(
        split=args.split,
        difficulty=_parse_difficulty(args.difficulty),
    )
    lo, hi = args.task_start, args.task_end
    if lo < 0 or hi > len(task_ids) or lo >= hi:
        raise SystemExit(
            f"Invalid task range [{lo}, {hi}): need 0 <= start < end <= {len(task_ids)}"
        )
    chosen = task_ids[lo:hi]

    out_dir = _default_output_dir(args)
    config = RolloutConfig(
        model_name=f"steerbot/{args.model}",
        model_endpoint="steerbot",
        model_api_key="NONE",
        output_dir=out_dir,
        verbose=args.verbose,
    )

    results: list[tuple[int, str, object]] = []
    for i, tid in enumerate(chosen):
        results.append(
            await _one_rollout(
                tid,
                config,
                args.per_agent_max_steps,
                args.max_depth,
                i,
            )
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="TextCraft + prompt intersection (no grammar FSA).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--split", default="val")
    p.add_argument("--difficulty", default="EASY")
    p.add_argument("--task-start", type=int, default=0)
    p.add_argument("--task-end", type=int, default=20)
    p.add_argument("--per-agent-max-steps", type=int, default=25)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--n-particles", type=int, default=10)
    p.add_argument("--ess-threshold", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096 * 2)
    p.add_argument(
        "--extra-system-prompt",
        default=DEFAULT_EXTRA_SYSTEM,
        help="Prepended as extra system message in the intersecting LM (see SteeredLM).",
    )
    p.add_argument(
        "--extra-prompt-keep-original-system",
        action="store_true",
        help="Include Platoon system messages in the second LM (default: exclude them).",
    )
    p.add_argument("--output-root", default=os.path.join("examples", "rollout_results"))
    p.add_argument(
        "--output-dir",
        default=None,
        help="Shared rollout directory for all shards (default: OUTPUT_ROOT/textcraft_multi_agent_SPLIT_DIFFICULTY/)",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.n_particles < 1:
        p.error("--n-particles must be >= 1")

    register_steered_lm(
        model_name_or_path=args.model,
        potential_eff=None,
        potential_exp=None,
        extra_system_prompt=args.extra_system_prompt,
        extra_prompt_keep_original_system=args.extra_prompt_keep_original_system,
        max_tokens=args.max_tokens,
        ess_threshold=args.ess_threshold,
        engine_opts={"max_model_len": args.max_model_len},
        n_particles=args.n_particles,
    )

    out_dir = _default_output_dir(args)
    print(f"output_dir={out_dir}", file=sys.stderr)
    results = asyncio.run(_run_all(args))

    ok = sum(1 for _i, _l, o in results if not isinstance(o, Exception))
    for _idx, label, out in sorted(results, key=lambda x: x[0]):
        if isinstance(out, Exception):
            print(f"FAIL {label}: {out!r}", file=sys.stderr)
        else:
            print(f"OK   {label}: {type(out).__name__}")
    print(f"Done: {ok}/{len(results)} ok", file=sys.stderr)
    if ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
