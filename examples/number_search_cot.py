#!/usr/bin/env python3
"""Number-search rollouts: the same CFG structure constraint as ``number_search_cfg.py`` plus optional
sentence limits on the first ``<thought>``.

**Sharding (20 tasks on 2 GPUs):** half-open ``[task_start, task_end)``. Use one shared output
directory for all shards; event filenames include a per-rollout UUID (no collisions).
Within each shard process, rollouts run **sequentially**.

    CUDA_VISIBLE_DEVICES=0 python examples/number_search_cot.py --task-start 0 --task-end 10
    CUDA_VISIBLE_DEVICES=1 python examples/number_search_cot.py --task-start 10 --task-end 20
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

from genlm.control import BoolCFG

from platoon.config_defs import RolloutConfig
from platoon.number_search.rollout import run_rollout as run_number_search_rollout
from platoon.number_search.tasks import get_task as get_number_search_task
from platoon.number_search.tasks import get_task_ids as get_number_search_task_ids
from steerbot.potentials.cot_sentence_cap import COTSentenceCapPotential
from steerbot.potentials.cot_sentence_min import COTSentenceMinPotential
from steerbot.utils.llm import register_steered_lm

NUMBER_SEARCH_THOUGHT_PYTHON_CFG = r"""
start: THOUGHT_BLOCK SEP PY_BLOCK TRAILING_WS

THOUGHT_BLOCK: "<thought>" THOUGHT_BODY "</thought>"
PY_BLOCK: "<python>" WS "guess" WS "(" WS INT WS ")" WS "</python>"

THOUGHT_BODY: /[^<]*/

INT: /-?\d+/
WS: /[ \\t]*/
SEP: /[ \\t\\r\\n]*/
TRAILING_WS: /[ \\t\\r\\n]*/
"""

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _default_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return os.path.abspath(args.output_dir)
    sub = f"number_search_cot_{args.split}"
    return os.path.abspath(os.path.join(args.output_root, sub))


async def _one_rollout(task_id: str, config: RolloutConfig, index: int) -> tuple[int, str, object]:
    label = f"[{index}] {task_id}"
    try:
        out = await run_number_search_rollout(
            task=get_number_search_task(task_id),
            config=config,
        )
        return (index, label, out)
    except Exception as e:
        return (index, label, e)


async def _run_all(args: argparse.Namespace) -> list[tuple[int, str, object]]:
    all_ids = get_number_search_task_ids(
        split=args.split,
        num_samples_train=args.num_samples_train,
        num_samples_val=args.num_samples_val,
    )
    lo, hi = args.task_start, args.task_end
    if lo < 0 or hi > len(all_ids) or lo >= hi:
        raise SystemExit(
            f"Invalid task range [{lo}, {hi}): need 0 <= start < end <= {len(all_ids)}"
        )
    chosen = all_ids[lo:hi]

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
        results.append(await _one_rollout(tid, config, i))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Number-search + CFG structure constraint + CoT sentence limits.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--task-start", type=int, default=0)
    p.add_argument("--task-end", type=int, default=20)
    p.add_argument("--num-samples-train", type=int, default=50000)
    p.add_argument("--num-samples-val", type=int, default=1000)
    p.add_argument(
        "--max-sentences",
        type=int,
        default=-1,
        help="Max sentences in first <thought> (-1 disables; default: -1)",
    )
    p.add_argument(
        "--min-sentences",
        type=int,
        default=0,
        help="Minimum sentences in first <thought> (default: 0 disables)",
    )
    p.add_argument("--n-particles", type=int, default=10)
    p.add_argument("--ess-threshold", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096 * 2)
    p.add_argument("--output-root", default=os.path.join("examples", "rollout_results"))
    p.add_argument(
        "--output-dir",
        default=None,
        help="Shared rollout directory for all shards (default: OUTPUT_ROOT/number_search_cot_SPLIT/)",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.n_particles < 1:
        p.error("--n-particles must be >= 1")
    if args.min_sentences < 0:
        p.error("--min-sentences must be non-negative")
    if args.max_sentences != -1 and args.max_sentences < 0:
        p.error("--max-sentences must be >= 0, or -1 to disable")
    if args.max_sentences != -1 and args.min_sentences > args.max_sentences:
        p.error("--min-sentences must be <= --max-sentences (or disable max with -1)")

    cfg = BoolCFG.from_lark(NUMBER_SEARCH_THOUGHT_PYTHON_CFG)
    minp = COTSentenceMinPotential.aligned_with(args.min_sentences, cfg)
    cap = (
        COTSentenceCapPotential.aligned_with(args.max_sentences, cfg)
        if args.max_sentences != -1
        else None
    )
    register_steered_lm(
        model_name_or_path=args.model,
        potential_eff=(cfg * minp * cap) if cap is not None else (cfg * minp),
        potential_exp=None,
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
