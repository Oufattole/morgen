#!/usr/bin/env bash
set -euo pipefail

DATASET="ehrshot"
OUTPUT_ROOT_NAME="vq_count_mc2"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: materialize_vq_count_eval.sh [DATASET] [--output-root-name NAME] [--dry-run]

Finishes the evaluation cache for a VQ-count trajectory root by:
  1. ensuring the eval root contains ground_truth and minute_level references,
  2. materializing any missing TTE parquet files from trajectories/generated/K*_level,
  3. computing 30d and 730d AUROC score parquets for the appendix tasks.

Supported datasets:
  ehrshot
EOF
}

if [[ $# -gt 0 && "$1" != --* ]]; then
    DATASET="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
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

case "${DATASET}" in
     ehrshot)
        BASE_DIR="/storage/shared/ehr-shot/filtered_labs/eic"
        INDEX_DIR="/storage/shared/ehr-shot/filtered_labs/meds/labels/ehrshot/timeline_end/"
        TASKS="creatinine,death,hematocrit,hemoglobin,leukocyte,platelet,admission,timeline_end"
        HORIZONS="30,730"
        ;;
    *)
        echo "Unsupported dataset: ${DATASET}" >&2
        exit 1
        ;;
esac

export PYENV_VERSION="${PYENV_VERSION:-morgen}"

EVAL_DIR="${BASE_DIR}/${OUTPUT_ROOT_NAME}"
POSTERIOR_ANALYSIS_DIR="/home/nassim/projects/posterior_analysis"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp)] $*"
}

run_cmd() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] '
        printf '%q ' "$@"
        printf '\n'
        return
    fi
    "$@"
}

ensure_eval_layout() {
    mkdir -p "${EVAL_DIR}/ttes/tuning" "${EVAL_DIR}/ttes/held_out"

    if [[ ! -L "${EVAL_DIR}/ground_truth" && ! -e "${EVAL_DIR}/ground_truth" ]]; then
        log "Linking ground_truth into ${EVAL_DIR}"
        run_cmd ln -s "${BASE_DIR}/ground_truth" "${EVAL_DIR}/ground_truth"
    fi

    for split in tuning held_out; do
        log "Copying minute_level.parquet into ${EVAL_DIR}/ttes/${split}"
        run_cmd rsync -a \
            "${BASE_DIR}/ttes/${split}/minute_level.parquet" \
            "${EVAL_DIR}/ttes/${split}/"
    done
}

materialize_missing_ttes_for_split() {
    local split="$1"
    local generated_dir="${EVAL_DIR}/trajectories/generated"
    local existing_dir="${EVAL_DIR}/ttes/${split}"
    local -a missing_inputs=()

    while IFS= read -r model_dir; do
        local model_name
        model_name="$(basename "${model_dir}")"
        if [[ ! -f "${existing_dir}/${model_name}.parquet" ]]; then
            missing_inputs+=("${model_dir}")
        fi
    done < <(find "${generated_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

    if [[ "${#missing_inputs[@]}" -eq 0 ]]; then
        log "No missing ${split} TTE parquet files under ${existing_dir}"
        return
    fi

    local input_csv
    input_csv="$(IFS=,; echo "${missing_inputs[*]}")"
    log "Caching ${#missing_inputs[@]} missing ${split} TTE parquet files into ${existing_dir}"
    (
        cd "${POSTERIOR_ANALYSIS_DIR}"
        run_cmd python scripts/cache_ttes.py --multirun \
            hydra/launcher=joblib \
            "dataset=${DATASET}" \
            "split=${split}" \
            "input_dir=${input_csv}" \
            "index_dir=${INDEX_DIR}" \
            "output_dir=${EVAL_DIR}/ttes"
    )
}

compute_scores() {
    log "Computing score parquet files in ${EVAL_DIR}/scores"
    (
        cd "${POSTERIOR_ANALYSIS_DIR}"
        run_cmd python scripts/compute_scores.py --multirun \
            hydra/launcher=joblib \
            "dataset=${DATASET}" \
            "dataset.base_dir=${EVAL_DIR}" \
            "horizon=${HORIZONS}" \
            "task=${TASKS}"
    )
}

log "Preparing VQ-count evaluation root"
log "Dataset: ${DATASET}"
log "Eval root: ${EVAL_DIR}"

ensure_eval_layout
materialize_missing_ttes_for_split tuning
materialize_missing_ttes_for_split held_out
compute_scores

log "VQ-count evaluation materialization complete"
