export CUDA_VISIBLE_DEVICES=0
export POLARS_MAX_THREADS=1
K=2048
BASE_DIR=/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/debug/K${K}_day_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
CACHE_DIR="${BASE_DIR}/eic/results/fit_em_cache/day_level/"

morgen_fit_em \
    cache_dir=${CACHE_DIR} \
    quantizer_config.n_embeddings=${K} \
    seed=1 \
    output_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    gap_config.max_gap_length=2048 \
    window_size_days=1 \
    trainer.max_epochs=50 \
    lightning_module.learning_rate=1e-3 \
    do_overwrite=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=8 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="HOSPITAL_DISCHARGE//.*" \
    datamodule.include_anchor_token=false \
    datamodule.empty_window_mode=IGNORE \
    datamodule.persistent_workers=true \
    trainer.val_check_interval=null

export CUDA_VISIBLE_DEVICES=0
export POLARS_MAX_THREADS=8
BASE_DIR=/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/
AR_OUTPUT_DIR="${BASE_DIR}/eic/results/em_autoregressive/day_level/"
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/day_level/"
DATA_DIR="${BASE_DIR}/eic/final/"

morgen_train_autoregressive_histogram \
    trainer/logger=wandb \
    datamodule.max_windows=510 \
    datamodule.window_sampling_strategy=random \
    trainer.max_epochs=100 \
    output_dir="$AR_OUTPUT_DIR" \
    gap_config.max_gap_length=2048 \
    window_size_days=1 \
    quantizer_config.n_embeddings=256 \
    quantizer_config.embedding_dim=4 \
    em_params_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    lightning_module.learning_rate=5e-4 \
    do_overwrite=true \
    datamodule.batch_size=256 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="HOSPITAL_DISCHARGE//.*" \
    datamodule.empty_window_mode=SINGLE_GAP \
    datamodule.include_anchor_token=true \
    datamodule.num_workers=8 \
    datamodule.persistent_workers=true \
    trainer.val_check_interval=null


export CUDA_VISIBLE_DEVICES=0
export POLARS_MAX_THREADS=8
BASE_DIR=/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/
HISTOGRAM_TRAJECTORIES_DIR="${BASE_DIR}/eic/em_generated_trajectories/day_level/"
AR_CHECKPOINT="${BASE_DIR}/eic/results/em_autoregressive/day_level/"
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/day_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
TASK_ROOT_DIR="/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/eic/tasks/"
TASK_NAME="mimiciv/hospital_discharge/timeline_end"
SINGLE_TASK_DIR="${TASK_ROOT_DIR}/${TASK_NAME}/"

morgen_generate_histogram_trajectories \
    datamodule.max_windows=128 \
    datamodule.window_sampling_strategy=to_end \
    em_params_dir="$QUANTIZER_OUTPUT_DIR" \
    window_size_days=1 \
    gap_config.max_gap_length=2048 \
    quantizer_config.n_embeddings=256 \
    quantizer_config.embedding_dim=4 \
    output_dir="$HISTOGRAM_TRAJECTORIES_DIR" \
    model_initialization_dir="$AR_CHECKPOINT" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    datamodule.config.task_labels_dir="$SINGLE_TASK_DIR" \
    max_new_tokens=382 \
    inference.generate_for_splits=[held_out,tuning] \
    inference.N_trajectories_per_task_sample=50 \
    do_overwrite=false \
    datamodule.anchoring_strategy=END_OF_TIMELINE \
    datamodule.empty_window_mode=SINGLE_GAP \
    datamodule.include_anchor_token=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=8 \
    datamodule.persistent_workers=true


HISTOGRAM_TRAJECTORIES_DIR="${BASE_DIR}/eic/em_generated_trajectories/day_level/histogram/"
PROCESSED_HISTOGRAM_TRAJECTORIES_DIR="${BASE_DIR}/eic/em_generated_trajectories/day_level/code/"
morgen_process_generated_histograms_soft_decoding --multirun \
    input_dir="$HISTOGRAM_TRAJECTORIES_DIR" \
    output_dir="$PROCESSED_HISTOGRAM_TRAJECTORIES_DIR" \
    num_trajectories=50 \
    split=held_out,tuning \
    num_histogram_samples=10

export CUDA_VISIBLE_DEVICES=0
export POLARS_MAX_THREADS=1
BASE_DIR=/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/week_level/"
DATA_DIR="${BASE_DIR}/eic/final/"

morgen_fit_em \
    quantizer_config.n_embeddings=256 \
    seed=1 \
    output_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    gap_config.max_gap_length=512 \
    window_size_days=7 \
    trainer.max_epochs=50 \
    lightning_module.learning_rate=1e-3 \
    do_overwrite=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=8 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="HOSPITAL_DISCHARGE//.*" \
    datamodule.include_anchor_token=false \
    datamodule.empty_window_mode=IGNORE \
    datamodule.persistent_workers=true \
    trainer.val_check_interval=null

export CUDA_VISIBLE_DEVICES=0
export POLARS_MAX_THREADS=8
BASE_DIR=/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/
AR_OUTPUT_DIR="${BASE_DIR}/eic/results/em_autoregressive/week_level/"
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/week_level/"
DATA_DIR="${BASE_DIR}/eic/final/"

morgen_train_autoregressive_histogram \
    trainer/logger=wandb \
    datamodule.max_windows=510 \
    datamodule.window_sampling_strategy=random \
    trainer.max_epochs=100 \
    output_dir="$AR_OUTPUT_DIR" \
    gap_config.max_gap_length=512 \
    window_size_days=7 \
    quantizer_config.n_embeddings=256 \
    quantizer_config.embedding_dim=4 \
    em_params_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    lightning_module.learning_rate=5e-4 \
    do_overwrite=true \
    datamodule.batch_size=256 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="HOSPITAL_DISCHARGE//.*" \
    datamodule.empty_window_mode=SINGLE_GAP \
    datamodule.include_anchor_token=true \
    datamodule.num_workers=8 \
    datamodule.persistent_workers=true \
    trainer.val_check_interval=null


export CUDA_VISIBLE_DEVICES=0
export POLARS_MAX_THREADS=8
BASE_DIR=/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/
HISTOGRAM_TRAJECTORIES_DIR="${BASE_DIR}/eic/em_generated_trajectories/week_level/"
AR_CHECKPOINT="${BASE_DIR}/eic/results/em_autoregressive/week_level/"
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/week_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
TASK_ROOT_DIR="/storage/shared/mimic-iv/meds_v0.4.0_mimicv3.1/eic/tasks/"
TASK_NAME="mimiciv/hospital_discharge/timeline_end"
SINGLE_TASK_DIR="${TASK_ROOT_DIR}/${TASK_NAME}/"

morgen_generate_histogram_trajectories \
    datamodule.max_windows=128 \
    datamodule.window_sampling_strategy=to_end \
    em_params_dir="$QUANTIZER_OUTPUT_DIR" \
    window_size_days=7 \
    gap_config.max_gap_length=512 \
    quantizer_config.n_embeddings=256 \
    quantizer_config.embedding_dim=4 \
    output_dir="$HISTOGRAM_TRAJECTORIES_DIR" \
    model_initialization_dir="$AR_CHECKPOINT" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    datamodule.config.task_labels_dir="$SINGLE_TASK_DIR" \
    max_new_tokens=128 \
    inference.generate_for_splits=[held_out,tuning] \
    inference.N_trajectories_per_task_sample=50 \
    do_overwrite=false \
    datamodule.anchoring_strategy=END_OF_TIMELINE \
    datamodule.empty_window_mode=SINGLE_GAP \
    datamodule.include_anchor_token=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=8 \
    datamodule.persistent_workers=true


HISTOGRAM_TRAJECTORIES_DIR="${BASE_DIR}/eic/em_generated_trajectories/week_level/histogram/"
PROCESSED_HISTOGRAM_TRAJECTORIES_DIR="${BASE_DIR}/eic/em_generated_trajectories/week_level/code/"
morgen_process_generated_histograms_soft_decoding --multirun \
    input_dir="$HISTOGRAM_TRAJECTORIES_DIR" \
    output_dir="$PROCESSED_HISTOGRAM_TRAJECTORIES_DIR" \
    num_trajectories=50 \
    split=held_out,tuning \
    num_histogram_samples=10
