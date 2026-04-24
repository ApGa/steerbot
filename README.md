# steerbot

Inference-time steering experiments for multi-agent TextCraft and Number Search rollouts using SMC.

The experiments use [platoon](https://github.com/ApGa/platoon) for the agent implementation and environments/task.

## Install

```bash
uv sync
```

## Scripts

- `examples/textcraft_multi_agent.py`: prompt-intersection TextCraft rollouts.
- `examples/number_search_cfg.py`: Number Search rollouts with a structural CFG constraint.
- `examples/number_search_cot.py`: Number Search rollouts with the same CFG plus CoT sentence constraints.
- `examples/run_vllm_rollouts.py`: baseline rollouts against a plain OpenAI-compatible vLLM server.
- `examples/run_genlm_shard_jobs.sh`: shard launcher for the genlm-based experiments above.

## Quick Start

```bash
uv run python examples/number_search_cfg.py --task-start 0 --task-end 20
```

```bash
./examples/run_genlm_shard_jobs.sh cfg --gpus 0,1 --task-start 0 --task-end 20
```