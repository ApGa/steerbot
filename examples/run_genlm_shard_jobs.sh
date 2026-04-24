#!/usr/bin/env bash
# Launch sharded genlm example rollouts (one Python process per GPU shard).
#
# Usage:
#   ./examples/run_genlm_shard_jobs.sh cfg        # number_search_cfg, 2 GPUs, tasks 0-20
#   ./examples/run_genlm_shard_jobs.sh cot        # number_search_cot, 2 GPUs
#   ./examples/run_genlm_shard_jobs.sh textcraft  # textcraft_multi_agent, 4 GPUs
#   ./examples/run_genlm_shard_jobs.sh numbers    # run cfg + cot at once
#   ./examples/run_genlm_shard_jobs.sh all        # run all three at once
#
#   REPO_ROOT=/path/to/steerbot ./examples/run_genlm_shard_jobs.sh cfg
#   ./examples/run_genlm_shard_jobs.sh cfg --gpus 0,1 --task-start 0 --task-end 20
#   ./examples/run_genlm_shard_jobs.sh textcraft --gpus 0,1,2,3 --output-dir /tmp/run1
#   ./examples/run_genlm_shard_jobs.sh all --output-dir /tmp/exp   # writes .../cfg, .../cot, .../textcraft
#
# Environment:
#   UV_RUN       Override runner (default: uv run python)
#   EXTRA_PY_ARGS  Extra args passed to every Python invocation (quoted string)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

UV_RUN="${UV_RUN:-uv run python}"
EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"

SCENARIO="${1:-}"
shift || true

if [[ -z "${SCENARIO}" ]]; then
  echo "Usage: $0 {cfg|cot|textcraft|numbers|all} [options]" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  --gpus ID[,ID...]     Physical GPU ids (single experiment only; not used with \"all\")" >&2
  echo "  --cfg-gpus CSV        For \"numbers\"/\"all\": GPUs for cfg (default: 0,1)" >&2
  echo "  --cot-gpus CSV        For \"numbers\"/\"all\": GPUs for cot (default: 2,3)" >&2
  echo "  --textcraft-gpus CSV  For \"all\": GPUs for textcraft (default: 4,5,6,7)" >&2
  echo "  --task-start N        First task index inclusive (default: 0)" >&2
  echo "  --task-end N          One past last index exclusive (default: 20)" >&2
  echo "  --split train|val     number_search + textcraft (default: val)" >&2
  echo "  --difficulty EASY     textcraft only (default: EASY)" >&2
  echo "  --output-dir PATH     Output dir (see below)" >&2
  echo "  --model MODEL         HF model id (default: script default)" >&2
  echo "" >&2
  echo "With \"numbers\", if --output-dir is set, children use DIR/cfg, DIR/cot." >&2
  echo "With \"all\", if --output-dir is set, children use DIR/cfg, DIR/cot, DIR/textcraft." >&2
  exit 1
fi

GPUS=""
CFG_GPUS="0,1"
COT_GPUS="2,3"
TEXTCRAFT_GPUS="4,5,6,7"
TASK_START=0
TASK_END=20
SPLIT="val"
DIFFICULTY="EASY"
OUTPUT_DIR=""
MODEL_ARG=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --cfg-gpus|--regex-gpus)
      CFG_GPUS="$2"
      shift 2
      ;;
    --cot-gpus)
      COT_GPUS="$2"
      shift 2
      ;;
    --textcraft-gpus)
      TEXTCRAFT_GPUS="$2"
      shift 2
      ;;
    --task-start)
      TASK_START="$2"
      shift 2
      ;;
    --task-end)
      TASK_END="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --difficulty)
      DIFFICULTY="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL_ARG=(--model "$2")
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${TASK_END}" -le "${TASK_START}" ]]; then
  echo "Need task-end > task-start (got [${TASK_START}, ${TASK_END}))" >&2
  exit 1
fi

TOTAL=$((TASK_END - TASK_START))

split_range_balanced() {
  # Print lines: "start end" for each shard (N lines), distributing remainder across early shards.
  # Uses half-open ranges [start,end). Skips empty shards (when N > TOTAL).
  local start="$1"
  local end="$2"
  local n="$3"
  local total=$((end - start))
  local base=$(( total / n ))
  local rem=$(( total % n ))
  local i s sz e
  for ((i=0; i<n; i++)); do
    sz="${base}"
    if [[ "${i}" -lt "${rem}" ]]; then
      sz=$((sz + 1))
    fi
    s=$(( start + i*base + (i < rem ? i : rem) ))
    e=$(( s + sz ))
    if [[ "${sz}" -le 0 ]]; then
      continue
    fi
    echo "${s} ${e}"
  done
}

py_cfg() {
  ${UV_RUN} examples/number_search_cfg.py \
    "${MODEL_ARG[@]}" \
    --split "${SPLIT}" \
    --task-start "$1" \
    --task-end "$2" \
    ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
    ${EXTRA_PY_ARGS}
}

py_cot() {
  ${UV_RUN} examples/number_search_cot.py \
    "${MODEL_ARG[@]}" \
    --split "${SPLIT}" \
    --task-start "$1" \
    --task-end "$2" \
    ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
    ${EXTRA_PY_ARGS}
}

py_textcraft() {
  ${UV_RUN} examples/textcraft_multi_agent.py \
    "${MODEL_ARG[@]}" \
    --split "${SPLIT}" \
    --difficulty "${DIFFICULTY}" \
    --task-start "$1" \
    --task-end "$2" \
    ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
    ${EXTRA_PY_ARGS}
}

if [[ "${SCENARIO}" == numbers ]]; then
  if [[ -n "${GPUS}" ]]; then
    echo "Use --cfg-gpus / --cot-gpus with \"numbers\", not --gpus." >&2
    exit 1
  fi
  export UV_RUN
  export EXTRA_PY_ARGS
  SELF="${BASH_SOURCE[0]}"
  echo "Launching number-search experiments in parallel: cfg GPUs=${CFG_GPUS} cot GPUs=${COT_GPUS} tasks=[${TASK_START},${TASK_END})" >&2
  OUT_CFG=() OUT_COT=()
  if [[ -n "${OUTPUT_DIR}" ]]; then
    OUT_CFG=(--output-dir "${OUTPUT_DIR}/cfg")
    OUT_COT=(--output-dir "${OUTPUT_DIR}/cot")
    echo "  output: ${OUTPUT_DIR}/{cfg,cot}" >&2
  fi
  "${SELF}" cfg --gpus "${CFG_GPUS}" --task-start "${TASK_START}" --task-end "${TASK_END}" --split "${SPLIT}" "${MODEL_ARG[@]}" "${OUT_CFG[@]}" &
  pid_cfg=$!
  "${SELF}" cot --gpus "${COT_GPUS}" --task-start "${TASK_START}" --task-end "${TASK_END}" --split "${SPLIT}" "${MODEL_ARG[@]}" "${OUT_COT[@]}" &
  pid_cot=$!
  st=0
  wait "${pid_cfg}" || st=1
  wait "${pid_cot}" || st=1
  if [[ "${st}" -ne 0 ]]; then
    echo "One or more experiments failed (exit ${st})." >&2
    exit "${st}"
  fi
  echo "Number-search experiments finished." >&2
  exit 0
fi

if [[ "${SCENARIO}" == all ]]; then
  if [[ -n "${GPUS}" ]]; then
    echo "Use --cfg-gpus / --cot-gpus / --textcraft-gpus with \"all\", not --gpus." >&2
    exit 1
  fi
  export UV_RUN
  export EXTRA_PY_ARGS
  SELF="${BASH_SOURCE[0]}"
  echo "Launching all experiments in parallel: cfg GPUs=${CFG_GPUS} cot GPUs=${COT_GPUS} textcraft GPUs=${TEXTCRAFT_GPUS} tasks=[${TASK_START},${TASK_END})" >&2
  OUT_CFG=() OUT_COT=() OUT_TEXT=()
  if [[ -n "${OUTPUT_DIR}" ]]; then
    OUT_CFG=(--output-dir "${OUTPUT_DIR}/cfg")
    OUT_COT=(--output-dir "${OUTPUT_DIR}/cot")
    OUT_TEXT=(--output-dir "${OUTPUT_DIR}/textcraft")
    echo "  output: ${OUTPUT_DIR}/{cfg,cot,textcraft}" >&2
  fi
  "${SELF}" cfg --gpus "${CFG_GPUS}" --task-start "${TASK_START}" --task-end "${TASK_END}" --split "${SPLIT}" "${MODEL_ARG[@]}" "${OUT_CFG[@]}" &
  pid_cfg=$!
  "${SELF}" cot --gpus "${COT_GPUS}" --task-start "${TASK_START}" --task-end "${TASK_END}" --split "${SPLIT}" "${MODEL_ARG[@]}" "${OUT_COT[@]}" &
  pid_cot=$!
  "${SELF}" textcraft --gpus "${TEXTCRAFT_GPUS}" --task-start "${TASK_START}" --task-end "${TASK_END}" --split "${SPLIT}" --difficulty "${DIFFICULTY}" "${MODEL_ARG[@]}" "${OUT_TEXT[@]}" &
  pid_text=$!
  st=0
  wait "${pid_cfg}" || st=1
  wait "${pid_cot}" || st=1
  wait "${pid_text}" || st=1
  if [[ "${st}" -ne 0 ]]; then
    echo "One or more experiments failed (exit ${st})." >&2
    exit "${st}"
  fi
  echo "All experiments finished." >&2
  exit 0
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"

case "${SCENARIO}" in
  cfg|regex)
    if [[ -z "${GPUS}" ]]; then
      GPU_LIST=(0 1)
    fi
    N=${#GPU_LIST[@]}
    if [[ "${N}" -lt 1 ]]; then
      echo "Need at least one GPU in --gpus" >&2
      exit 1
    fi
    echo "number_search_cfg: GPUs=${GPU_LIST[*]} tasks=[${TASK_START},${TASK_END})" >&2
    i=0
    while read -r s e; do
      if [[ "${i}" -ge "${#GPU_LIST[@]}" ]]; then
        break
      fi
      echo "  shard $i: CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} tasks [${s},${e})" >&2
      CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" py_cfg "${s}" "${e}" &
      i=$((i+1))
    done < <(split_range_balanced "${TASK_START}" "${TASK_END}" "${#GPU_LIST[@]}")
    wait
    ;;
  cot)
    if [[ -z "${GPUS}" ]]; then
      GPU_LIST=(0 1)
    fi
    N=${#GPU_LIST[@]}
    if [[ "${N}" -lt 1 ]]; then
      echo "Need at least one GPU in --gpus" >&2
      exit 1
    fi
    echo "number_search_cot: GPUs=${GPU_LIST[*]} tasks=[${TASK_START},${TASK_END})" >&2
    i=0
    while read -r s e; do
      if [[ "${i}" -ge "${#GPU_LIST[@]}" ]]; then
        break
      fi
      echo "  shard $i: CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} tasks [${s},${e})" >&2
      CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" py_cot "${s}" "${e}" &
      i=$((i+1))
    done < <(split_range_balanced "${TASK_START}" "${TASK_END}" "${#GPU_LIST[@]}")
    wait
    ;;
  textcraft)
    if [[ -z "${GPUS}" ]]; then
      GPU_LIST=(0 1 2 3)
    fi
    N=${#GPU_LIST[@]}
    if [[ "${N}" -lt 1 ]]; then
      echo "Need at least one GPU in --gpus" >&2
      exit 1
    fi
    echo "textcraft_multi_agent: GPUs=${GPU_LIST[*]} tasks=[${TASK_START},${TASK_END})" >&2
    i=0
    while read -r s e; do
      if [[ "${i}" -ge "${#GPU_LIST[@]}" ]]; then
        break
      fi
      echo "  shard $i: CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} tasks [${s},${e})" >&2
      CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" py_textcraft "${s}" "${e}" &
      i=$((i+1))
    done < <(split_range_balanced "${TASK_START}" "${TASK_END}" "${#GPU_LIST[@]}")
    wait
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO} (use cfg, cot, or textcraft)" >&2
    exit 1
    ;;
esac

echo "All shards finished." >&2
