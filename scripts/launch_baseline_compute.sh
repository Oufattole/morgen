#!/usr/bin/env bash
set -euo pipefail

GPU_ID="0"
SAMPLE_EVERY="1.0"
DRY_RUN=0
DATASETS_CSV="ehrshot,mimic"

usage() {
    cat <<'EOF'
Usage: launch_baseline_compute.sh [--gpu ID] [--sample-every SECONDS] [--datasets CSV] [--dry-run]

Runs the baseline pretrain + inference benchmark sequentially with profiling.
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

dataset_task_root_dir() {
    case "$1" in
        ehrshot) echo "/storage/shared/ehr-shot/filtered_labs/meds/labels" ;;
        mimic) echo "/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/eic/tasks" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

dataset_task_name() {
    case "$1" in
        ehrshot) echo "discharge" ;;
        mimic) echo "mimiciv/hospital_discharge/timeline_end" ;;
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
        ehrshot|mimic) echo "5e-4" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

inference_batch_size() {
    case "$1" in
        ehrshot) echo "512" ;;
        mimic) echo "512" ;;
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

    if [[ -f "${summary_path}" ]]; then
        local exit_code
        exit_code="$(summary_exit_code "${summary_path}")"
        if [[ "${exit_code}" == "0" ]]; then
            log "Skipping ${run_name}; successful summary already exists at ${summary_path}"
            return
        fi
        log "Re-running ${run_name}; prior summary exists but exit_code=${exit_code}"
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
    local task_root_dir
    task_root_dir="$(dataset_task_root_dir "${dataset}")"
    local task_name
    task_name="$(dataset_task_name "${dataset}")"
    local single_task_dir="${task_root_dir}/${task_name}/"

    local train_batch
    train_batch="$(pretrain_batch_size "${dataset}")"
    local train_epochs
    train_epochs="$(pretrain_max_epochs "${dataset}")"
    local train_lr
    train_lr="$(pretrain_lr "${dataset}")"
    local infer_batch
    infer_batch="$(inference_batch_size "${dataset}")"

    local pretrained_model_dir="${bench_root}/results/pretrained_lr_${train_lr}/"
    local generated_trajectories_dir="${bench_root}/results/generated_trajectories/micro/hospital_discharge/"
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

    run_profile \
        baseline_inference \
        "${dataset}" \
        "${profiling_dir}" \
        "${dataset}_baseline_inference" \
        morgen_generate_trajectories \
        "inference.generate_for_splits=[tuning,held_out]" \
        "inference.N_trajectories_per_task_sample=1" \
        "output_dir=${generated_trajectories_dir}" \
        "datamodule.config.tensorized_cohort_dir=${data_dir}" \
        "datamodule.config.task_labels_dir=${single_task_dir}" \
        "model_initialization_dir=${pretrained_model_dir}" \
        "datamodule.batch_size=${infer_batch}" \
        "datamodule.config.max_seq_len=128"
}

log "Starting baseline compute benchmark"
log "Datasets: $(join_by_space "${SELECTED_DATASETS[@]}")"
log "GPU: ${CUDA_VISIBLE_DEVICES}"
log "Profiler sample interval: ${SAMPLE_EVERY}s"

for dataset in "${SELECTED_DATASETS[@]}"; do
    run_one_dataset "${dataset}"
done

log "Baseline compute benchmark complete"
