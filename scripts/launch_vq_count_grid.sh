#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"

GPU_ID="0"
DRY_RUN=0
DATASET=""
MAX_COUNT="64"
OUTPUT_ROOT_NAME="vq_count"
RESOLUTIONS=(day week month quarter_annual semi_annual)

usage() {
    cat <<'EOF'
Usage: launch_vq_count_grid.sh DATASET [--gpu ID] [--max-count N] [--output-root-name NAME] [--dry-run]

Runs the vector-quantized histogram train + inference grid sequentially.

Supported datasets:
  ehrshot
  mimic
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 1
fi

DATASET="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --max-count)
            MAX_COUNT="$2"
            shift 2
            ;;
        --output-root-name)
            OUTPUT_ROOT_NAME="$2"
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

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp)] $*"
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

dataset_anchor_regex() {
    case "$1" in
        ehrshot) echo "discharge/IP" ;;
        mimic) echo "HOSPITAL_DISCHARGE//.*" ;;
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
        ehrshot) echo "ehrshot/timeline_end" ;;
        mimic) echo "mimiciv/hospital_discharge/timeline_end" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

dataset_fit_num_workers() {
    case "$1" in
        ehrshot) echo "0" ;;
        mimic) echo "8" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

dataset_fit_persistent_workers() {
    case "$1" in
        ehrshot) echo "false" ;;
        mimic) echo "true" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

dataset_train_num_workers() {
    case "$1" in
        ehrshot) echo "16" ;;
        mimic) echo "8" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

dataset_train_persistent_workers() {
    echo "true"
}

dataset_k_order() {
    case "$1" in
        ehrshot|mimic) echo "64 128 256 512" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

resolution_gap() {
    case "$1" in
        day) echo "2048" ;;
        week) echo "512" ;;
        month) echo "256" ;;
        quarter_annual) echo "400" ;;
        semi_annual) echo "200" ;;
        *)
            echo "Unknown resolution: $1" >&2
            exit 1
            ;;
    esac
}

resolution_window_days() {
    case "$1" in
        day) echo "1" ;;
        week) echo "7" ;;
        month) echo "30" ;;
        quarter_annual) echo "90" ;;
        semi_annual) echo "182" ;;
        *)
            echo "Unknown resolution: $1" >&2
            exit 1
            ;;
    esac
}

resolution_max_new_tokens() {
    case "$1" in
        day) echo "382" ;;
        week) echo "128" ;;
        month) echo "26" ;;
        quarter_annual) echo "12" ;;
        semi_annual) echo "6" ;;
        *)
            echo "Unknown resolution: $1" >&2
            exit 1
            ;;
    esac
}

train_batch_size() {
    local dataset="$1"
    local resolution="$2"

    case "${dataset}:${resolution}" in
        ehrshot:day|ehrshot:week) echo "128" ;;
        ehrshot:month|ehrshot:quarter_annual|ehrshot:semi_annual) echo "256" ;;
        mimic:day|mimic:week) echo "128" ;;
        mimic:month|mimic:quarter_annual|mimic:semi_annual) echo "256" ;;
        *)
            echo "Unknown dataset/resolution pair: ${dataset}/${resolution}" >&2
            exit 1
            ;;
    esac
}

inference_batch_size() {
    local dataset="$1"
    local resolution="$2"

    case "${dataset}:${resolution}" in
        ehrshot:day|ehrshot:week|ehrshot:month|ehrshot:quarter_annual|ehrshot:semi_annual) echo "512" ;;
        mimic:day|mimic:week|mimic:month|mimic:quarter_annual|mimic:semi_annual) echo "512" ;;
        *)
            echo "Unknown dataset/resolution pair: ${dataset}/${resolution}" >&2
            exit 1
            ;;
    esac
}

run_cmd() {
    local run_name="$1"
    shift

    log "Running ${run_name}"

    local cmd=("$@")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return
    fi

    "${cmd[@]}"
}

run_one_combo() {
    local dataset="$1"
    local k="$2"
    local resolution="$3"

    local base_dir
    base_dir="$(dataset_base_dir "${dataset}")"
    local root_dir="${base_dir}/eic/${OUTPUT_ROOT_NAME}"
    local data_dir="${base_dir}/eic/final/"
    local task_root_dir
    task_root_dir="$(dataset_task_root_dir "${dataset}")"
    local task_name
    task_name="$(dataset_task_name "${dataset}")"
    local single_task_dir="${task_root_dir}/${task_name}/"
    local anchoring_regex
    anchoring_regex="$(dataset_anchor_regex "${dataset}")"
    local fit_workers
    fit_workers="$(dataset_fit_num_workers "${dataset}")"
    local fit_persistent
    fit_persistent="$(dataset_fit_persistent_workers "${dataset}")"
    local train_workers
    train_workers="$(dataset_train_num_workers "${dataset}")"
    local train_persistent
    train_persistent="$(dataset_train_persistent_workers "${dataset}")"

    local folder_name="K${k}_${resolution}_level"
    local gap
    gap="$(resolution_gap "${resolution}")"
    local window_days
    window_days="$(resolution_window_days "${resolution}")"
    local max_new_tokens
    max_new_tokens="$(resolution_max_new_tokens "${resolution}")"
    local ar_batch
    ar_batch="$(train_batch_size "${dataset}" "${resolution}")"
    local infer_batch
    infer_batch="$(inference_batch_size "${dataset}" "${resolution}")"

    local quantizer_output_dir="${root_dir}/checkpoints/histogram_quantizer/${folder_name}/"
    local ar_output_dir="${root_dir}/checkpoints/histogram_autoregressive/${folder_name}/"
    local trajectories_output_dir="${root_dir}/trajectories/generated/${folder_name}/"
    local quantizer_checkpoint="${quantizer_output_dir}/best_quantizer.ckpt"
    local ar_checkpoint="${ar_output_dir}/best_model.ckpt"
    local first_sample_idx="0"
    local last_sample_idx="49"

    mkdir -p "${quantizer_output_dir}" "${ar_output_dir}" "${trajectories_output_dir}"

    if [[ -f "${quantizer_checkpoint}" ]]; then
        log "Skipping ${dataset}_K${k}_${resolution}_train_vq; found ${quantizer_checkpoint}"
    else
        export POLARS_MAX_THREADS=1
        run_cmd \
            "${dataset}_K${k}_${resolution}_train_vq" \
            morgen_train_quantizer \
            "trainer/logger=csv" \
            "quantizer_config.n_embeddings=${k}" \
            "quantizer_config.embedding_dim=4" \
            "quantizer_config.max_count=${MAX_COUNT}" \
            "seed=1" \
            "output_dir=${quantizer_output_dir}" \
            "datamodule.config.tensorized_cohort_dir=${data_dir}" \
            "gap_config.max_gap_length=${gap}" \
            "window_size_days=${window_days}" \
            "trainer.max_epochs=50" \
            "lightning_module.learning_rate=1e-3" \
            "do_overwrite=true" \
            "datamodule.batch_size=512" \
            "datamodule.num_workers=${fit_workers}" \
            "datamodule.anchoring_strategy=SPECIFIC_EVENT" \
            "datamodule.specific_event_regex=${anchoring_regex}" \
            "datamodule.include_anchor_token=false" \
            "datamodule.empty_window_mode=IGNORE" \
            "datamodule.persistent_workers=${fit_persistent}" \
            "trainer.val_check_interval=null"
    fi

    if [[ -f "${ar_checkpoint}" ]]; then
        log "Skipping ${dataset}_K${k}_${resolution}_train_histogram_ar; found ${ar_checkpoint}"
    else
        export POLARS_MAX_THREADS=8
        run_cmd \
            "${dataset}_K${k}_${resolution}_train_histogram_ar" \
            morgen_train_autoregressive_histogram \
            "trainer/logger=csv" \
            "datamodule.max_windows=510" \
            "datamodule.window_sampling_strategy=random" \
            "trainer.max_epochs=100" \
            "output_dir=${ar_output_dir}" \
            "gap_config.max_gap_length=${gap}" \
            "window_size_days=${window_days}" \
            "quantizer_config.n_embeddings=${k}" \
            "quantizer_config.embedding_dim=4" \
            "quantizer_config.max_count=${MAX_COUNT}" \
            "quantizer_checkpoint_path=${quantizer_checkpoint}" \
            "datamodule.config.tensorized_cohort_dir=${data_dir}" \
            "lightning_module.learning_rate=5e-4" \
            "do_overwrite=true" \
            "datamodule.batch_size=${ar_batch}" \
            "datamodule.anchoring_strategy=SPECIFIC_EVENT" \
            "datamodule.specific_event_regex=${anchoring_regex}" \
            "datamodule.empty_window_mode=SINGLE_GAP" \
            "datamodule.include_anchor_token=true" \
            "datamodule.num_workers=${train_workers}" \
            "datamodule.persistent_workers=${train_persistent}" \
            "trainer.val_check_interval=null"
    fi

    if [[ \
        -f "${trajectories_output_dir}/histogram/tuning/${first_sample_idx}.parquet" && \
        -f "${trajectories_output_dir}/histogram/tuning/${last_sample_idx}.parquet" && \
        -f "${trajectories_output_dir}/histogram/held_out/${first_sample_idx}.parquet" && \
        -f "${trajectories_output_dir}/histogram/held_out/${last_sample_idx}.parquet" && \
        -f "${trajectories_output_dir}/code/tuning/${first_sample_idx}.parquet" && \
        -f "${trajectories_output_dir}/code/tuning/${last_sample_idx}.parquet" && \
        -f "${trajectories_output_dir}/code/held_out/${first_sample_idx}.parquet" && \
        -f "${trajectories_output_dir}/code/held_out/${last_sample_idx}.parquet" \
    ]]; then
        log "Skipping ${dataset}_K${k}_${resolution}_histogram_inference; found boundary samples ${first_sample_idx} and ${last_sample_idx} for tuning and held_out"
    else
        export POLARS_MAX_THREADS=8
        run_cmd \
            "${dataset}_K${k}_${resolution}_histogram_inference" \
            morgen_generate_histogram_trajectories \
            "datamodule.max_windows=128" \
            "datamodule.window_sampling_strategy=to_end" \
            "window_size_days=${window_days}" \
            "gap_config.max_gap_length=${gap}" \
            "quantizer_config.n_embeddings=${k}" \
            "quantizer_config.embedding_dim=4" \
            "output_dir=${trajectories_output_dir}" \
            "model_initialization_dir=${ar_output_dir}" \
            "datamodule.config.tensorized_cohort_dir=${data_dir}" \
            "datamodule.config.task_labels_dir=${single_task_dir}" \
            "max_new_tokens=${max_new_tokens}" \
            "inference.generate_for_splits=[tuning,held_out]" \
            "inference.N_trajectories_per_task_sample=50" \
            "do_overwrite=false" \
            "datamodule.anchoring_strategy=END_OF_TIMELINE" \
            "datamodule.empty_window_mode=SINGLE_GAP" \
            "datamodule.include_anchor_token=true" \
            "datamodule.batch_size=${infer_batch}" \
            "datamodule.num_workers=${train_workers}" \
            "datamodule.persistent_workers=${train_persistent}"
    fi
}

log "Starting VQ histogram count grid"
log "Dataset: ${DATASET}"
log "GPU: ${CUDA_VISIBLE_DEVICES}"
log "Max count: ${MAX_COUNT}"
log "Output root: ${OUTPUT_ROOT_NAME}"

read -r -a ks <<< "$(dataset_k_order "${DATASET}")"
for priority_index in "${!ks[@]}"; do
    k="${ks[${priority_index}]}"
    log "Priority group ${priority_index}: dataset=${DATASET}, k=${k}"
    for resolution in "${RESOLUTIONS[@]}"; do
        run_one_combo "${DATASET}" "${k}" "${resolution}"
    done
done

log "VQ histogram count grid complete"
