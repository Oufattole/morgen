#!/usr/bin/env bash
set -euo pipefail

GPU_ID="0"
SAMPLE_EVERY="1.0"
DRY_RUN=0
FORCE_RERUN=0
DATASETS_CSV="mimic"
BASELINE_LR="${BASELINE_LR:-5e-4}"

usage() {
    cat <<'EOF'
Usage: launch_baseline_train_only.sh [--gpu ID] [--sample-every SECONDS] [--datasets CSV] [--force-rerun] [--dry-run]

Runs the baseline pretrain benchmark sequentially with profiling, skipping inference.
Set BASELINE_LR in the environment to override the learning rate used for the run.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --sample-every)
            SAMPLE_EVERY="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_CSV="$2"
            shift 2
            ;;
        --force-rerun)
            FORCE_RERUN=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

export PYENV_VERSION="${PYENV_VERSION:-morgen}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

IFS=',' read -r -a SELECTED_DATASETS <<< "${DATASETS_CSV}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp)] $*"
}

join_by_space() {
    local IFS=' '
    echo "$*"
}

dataset_base_dir() {
    case "$1" in
        ehrshot) echo "/storage/shared/ehr-shot/filtered_labs" ;;
        mimic) echo "/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

pretrain_batch_size() {
    case "$1" in
        ehrshot) echo "128" ;;
        mimic) echo "128" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

pretrain_max_epochs() {
    case "$1" in
        ehrshot) echo "100" ;;
        mimic) echo "100" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

pretrain_lr() {
    case "$1" in
        ehrshot|mimic) echo "${BASELINE_LR}" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

summary_exit_code() {
    local summary_path="$1"
    python - "$summary_path" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path) as f:
    print(json.load(f).get("exit_code"))
PY
}

run_profile() {
    local stage="$1"
    local dataset="$2"
    local metrics_dir="$3"
    local run_name="$4"
    shift 4

    local summary_path="${metrics_dir}/${run_name}_summary.json"
    mkdir -p "${metrics_dir}"

    if [[ -f "${summary_path}" && "${FORCE_RERUN}" -ne 1 ]]; then
        local exit_code
        exit_code="$(summary_exit_code "${summary_path}")"
        if [[ "${exit_code}" == "0" ]]; then
            log "Skipping ${run_name}; successful summary already exists at ${summary_path}"
            return
        fi
        log "Re-running ${run_name}; prior summary exists but exit_code=${exit_code}"
    fi

    if [[ -f "${summary_path}" && "${FORCE_RERUN}" -eq 1 ]]; then
        log "Force rerun enabled; overwriting previous summary at ${summary_path}"
    fi

    log "Running ${run_name}"

    local cmd=(
        morgen_profile_run
        --stage "${stage}"
        --dataset "${dataset}"
        --metrics-dir "${metrics_dir}"
        --run-name "${run_name}"
        --sample-every "${SAMPLE_EVERY}"
        --
    )
    cmd+=("$@")

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return
    fi

    "${cmd[@]}"
}

run_one_dataset() {
    local dataset="$1"

    local base_dir
    base_dir="$(dataset_base_dir "${dataset}")"
    local bench_root="${base_dir}/eic/compute_benchmark/baseline"
    local data_dir="${base_dir}/eic/final/"

    local train_batch
    train_batch="$(pretrain_batch_size "${dataset}")"
    local train_epochs
    train_epochs="$(pretrain_max_epochs "${dataset}")"
    local train_lr
    train_lr="$(pretrain_lr "${dataset}")"

    local pretrained_model_dir="${bench_root}/results/pretrained_lr_${train_lr}/"
    local profiling_dir="${bench_root}/profiling/${dataset}/"

    run_profile \
        baseline_pretrain \
        "${dataset}" \
        "${profiling_dir}" \
        "${dataset}_baseline_pretrain" \
        morgen_pretrain \
        "trainer/logger=csv" \
        "lightning_module=micro" \
        "datamodule.config.tensorized_cohort_dir=${data_dir}" \
        "output_dir=${pretrained_model_dir}" \
        "datamodule.batch_size=${train_batch}" \
        "trainer.max_epochs=${train_epochs}" \
        "lightning_module.optimizer.lr=${train_lr}"
}

log "Starting baseline pretrain benchmark"
log "Datasets: $(join_by_space "${SELECTED_DATASETS[@]}")"
log "GPU: ${CUDA_VISIBLE_DEVICES}"
log "Profiler sample interval: ${SAMPLE_EVERY}s"
log "Learning rate: ${BASELINE_LR}"
log "Force rerun: ${FORCE_RERUN}"

for dataset in "${SELECTED_DATASETS[@]}"; do
    run_one_dataset "${dataset}"
done

log "Baseline pretrain benchmark complete"
