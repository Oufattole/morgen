#!/bin/bash
set -e

# Help message
usage() {
    echo "Usage: $0 --dataset [ehrshot_full,ehrshot|mimic] [--k value] [--resolution value] [--train_batch_size value] [--inference_batch_size value]"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --k) K="$2"; shift ;;
        --resolution) RESOLUTION="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift ;;
        --inference_batch_size) INFERENCE_BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

MISSING_ARGS=()
for var in K RESOLUTION DATASET TRAIN_BATCH_SIZE INFERENCE_BATCH_SIZE; do
    if [[ -z "${!var:-}" ]]; then
        MISSING_ARGS+=("$var")
    fi
done
if [[ ${#MISSING_ARGS[@]} -gt 0 ]]; then
    echo "Error: The following required arguments are missing: ${MISSING_ARGS[*]}" >&2
    echo "Usage: $0 --k <val> --resolution <val> --dataset <val> --train_batch_size <val> --inference_batch_size <val>" >&2
    exit 1
fi

# 1. Check if dataset is provided and valid
if [[ "$DATASET" != "ehrshot_full" && "$DATASET" != "ehrshot" && "$DATASET" != "mimic" ]]; then
    echo "Error: --dataset must be one of [ehrshot_full, ehrshot, mimic]"
    usage
fi

# 2. Double check K is in your allowed power-of-2 list
ALLOWED_K="8 32 64 128 256 512 1024 2048 4096 8192 16384 32768"
if [[ ! $ALLOWED_K =~ (^|[[:space:]])"$K"($|[[:space:]]) ]]; then
    echo "Error: K value $K is not in the allowed list: $ALLOWED_K"
    exit 1
fi

# 3. Map Dataset Params
if [[ "$DATASET" == "ehrshot" ]]; then
    BASE_DIR="/storage/shared/ehr-shot/filtered_labs/"
    ANCHORING_REGEX="discharge/IP"
    TASK_ROOT_DIR="/storage/shared/ehr-shot/filtered_labs/meds/labels/"
    TASK_NAME="discharge"
    SINGLE_TASK_DIR="${TASK_ROOT_DIR}/${TASK_NAME}/"
elif [[ "$DATASET" == "ehrshot_full" ]]; then
    BASE_DIR="/storage/shared/ehr-shot/full/"
    ANCHORING_REGEX="discharge/IP"
    TASK_ROOT_DIR="/storage/shared/ehr-shot/full/meds/labels/"
    TASK_NAME="discharge"
    SINGLE_TASK_DIR="${TASK_ROOT_DIR}/${TASK_NAME}/"
elif [[ "$DATASET" == "mimic" ]]; then
    BASE_DIR="/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/"
    ANCHORING_REGEX="HOSPITAL_DISCHARGE//.*"
    TASK_ROOT_DIR="/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/eic/tasks/"
    TASK_NAME="mimiciv/hospital_discharge/timeline_end"
    SINGLE_TASK_DIR="${TASK_ROOT_DIR}/${TASK_NAME}/"
fi

# Output for verification
echo "Running script for $DATASET..."
echo "K: $K"
echo "Resolution: $RESOLUTION"
echo "Base Dir: $BASE_DIR"
echo "Regex: $ANCHORING_REGEX"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Inference Batch Size: $INFERENCE_BATCH_SIZE"



######## COMPUTE PARAMS ########
export POLARS_MAX_THREADS=1
NUM_WORKERS=16

######## FIXED PARAMS ########
FOLDER_NAME="K${K}_${RESOLUTION}_level"
MAX_EPOCHS=100
MAX_WINDOWS=510 # Keeps context length under 512
INPUT_CONTEXT_LENGTH=128
LR=1e-3
N_TRAJECTORIES=50
if [[ "$RESOLUTION" == "day" ]]; then
    WINDOW_SIZE_DAYS=1
    MAX_NEW_TOKENS=382
    MAX_GAP_LENGTH=2048

elif [[ "$RESOLUTION" == "week" ]]; then
    WINDOW_SIZE_DAYS=7
    MAX_NEW_TOKENS=128
    MAX_GAP_LENGTH=2048

elif [[ "$RESOLUTION" == "month" ]]; then
    WINDOW_SIZE_DAYS=30
    MAX_NEW_TOKENS=26
    MAX_GAP_LENGTH=2048

elif [[ "$RESOLUTION" == "quarter_annual" ]]; then
    WINDOW_SIZE_DAYS=90
    MAX_NEW_TOKENS=12
    MAX_GAP_LENGTH=512

elif [[ "$RESOLUTION" == "semi_annual" ]]; then
    WINDOW_SIZE_DAYS=182
    MAX_NEW_TOKENS=6
    MAX_GAP_LENGTH=256

else
    echo "Error: Invalid resolution '$RESOLUTION'. Expected: day, week, month, quarter_annual, or semi_annual." >&2
    exit 1
fi

# 1. Check if the Quantizer directory exists
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/${FOLDER_NAME}/"

# -d checks if the path is a directory
if [[ ! -d "$QUANTIZER_OUTPUT_DIR" ]]; then
    echo "Error: Directory QUANTIZER_OUTPUT_DIR does also not exist: $QUANTIZER_OUTPUT_DIR" >&2
    exit 1
fi

AR_OUTPUT_DIR="${BASE_DIR}/eic/results/em_autoregressive/${FOLDER_NAME}/"
DATA_DIR="${BASE_DIR}/eic/final/"

morgen_train_autoregressive_histogram \
    trainer/logger=wandb \
    datamodule.max_windows=${MAX_WINDOWS} \
    datamodule.window_sampling_strategy=random \
    em_params_dir="$QUANTIZER_OUTPUT_DIR" \
    output_dir="$AR_OUTPUT_DIR" \
    gap_config.max_gap_length=${MAX_GAP_LENGTH} \
    window_size_days=${WINDOW_SIZE_DAYS} \
    quantizer_config.n_embeddings=${K} \
    quantizer_checkpoint_path="$QUANTIZER_CHECKPOINT" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    trainer.max_epochs=${MAX_EPOCHS} \
    lightning_module.learning_rate=${LR} \
    do_overwrite=true \
    datamodule.batch_size=${TRAIN_BATCH_SIZE} \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex=${ANCHORING_REGEX} \
    datamodule.empty_window_mode=SINGLE_GAP \
    datamodule.include_anchor_token=true \
    datamodule.num_workers=${NUM_WORKERS} \
    datamodule.persistent_workers=true \
    trainer.val_check_interval=null


HISTOGRAM_TRAJECTORIES_DIR="${BASE_DIR}/eic/em_generated_trajectories/${FOLDER_NAME}/"
AR_CHECKPOINT="${BASE_DIR}/eic/results/em_autoregressive/${FOLDER_NAME}/"
DATA_DIR="${BASE_DIR}/eic/final/"

morgen_generate_histogram_trajectories \
    datamodule.max_windows=${INPUT_CONTEXT_LENGTH} \
    datamodule.window_sampling_strategy=to_end \
    em_params_dir="$QUANTIZER_OUTPUT_DIR" \
    window_size_days=${WINDOW_SIZE_DAYS} \
    gap_config.max_gap_length=${MAX_GAP_LENGTH} \
    quantizer_config.n_embeddings=${K} \
    output_dir="$HISTOGRAM_TRAJECTORIES_DIR" \
    model_initialization_dir="$AR_CHECKPOINT" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    datamodule.config.task_labels_dir="$SINGLE_TASK_DIR" \
    max_new_tokens=${MAX_NEW_TOKENS} \
    inference.generate_for_splits=[tuning,held_out] \
    inference.N_trajectories_per_task_sample=${N_TRAJECTORIES} \
    do_overwrite=false \
    datamodule.anchoring_strategy=END_OF_TIMELINE \
    datamodule.empty_window_mode=SINGLE_GAP \
    datamodule.include_anchor_token=true \
    datamodule.batch_size=${INFERENCE_BATCH_SIZE} \
    datamodule.num_workers=${NUM_WORKERS} \
    datamodule.persistent_workers=true
